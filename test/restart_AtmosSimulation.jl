# This test checks that:

# 1. A simulation, saved to a checkpoint, is read back identically (up to some
#   tolerance and excluding those fields that are computed during the
#   calculation of the tendencies)
# 2. A simulation, saved to a previous checkpoint, and read back and evolved to
#   the same time is identical (up to some tolerance)
# 3. ClimaAtmos can automatically detect restarts
#
# This test checks a variety of configurations and spends a long time compiling
# functions. For this reason, the test takes an optional command-line argument
# to only many tests instead of comprehensive test.

include("restart_utils.jl")

function amip_target_diagedmf(context, output_dir)
    FT = Float32
    start_date = DateTime(2010, 1, 1)

    param_dict =
        ClimaParams.create_toml_dict(
            FT;
            override_file = "toml/longrun_aquaplanet_diagedmf.toml",
        )
    params = CA.ClimaAtmosParameters(param_dict)

    deep_atmosphere = true
    cloud = CA.InteractiveCloudInRadiation()
    rayleigh_sponge = CA.RayleighSponge{FT}(;
        zd = params.zd_rayleigh,
        α_uₕ = params.alpha_rayleigh_uh,
        α_w = params.alpha_rayleigh_w,
        α_sgs_tracer = params.alpha_rayleigh_sgs_tracer,
    )
    viscous_sponge =
        CA.ViscousSponge{FT}(; zd = params.zd_viscous, κ₂ = params.kappa_2_sponge)

    diff_mode = CA.Implicit()
    hyperdiff = CA.cam_se_hyperdiffusion(FT)

    tracers = (
        "CB1", "CB2",
        "DST01", "DST02", "DST03", "DST04", "DST05",
        "OC1", "OC2",
        "SO4",
        "SSLT01", "SSLT02", "SSLT03", "SSLT04", "SSLT05",
    )
    microphysics_model = CA.EquilibriumMicrophysics0M()

    # Radiation mode
    idealized_h2o = false
    idealized_clouds = false
    add_isothermal_boundary_layer = true
    aerosol_radiation = true
    reset_rng_seed = false
    radiation_mode = CA.RRTMGPInterface.AllSkyRadiationWithClearSkyDiagnostics(
        idealized_h2o,
        idealized_clouds,
        cloud,
        add_isothermal_boundary_layer,
        aerosol_radiation,
        reset_rng_seed,
        deep_atmosphere,
    )

    insolation = CA.TimeVaryingInsolation(start_date)

    surface_setup = CA.SurfaceConditions.DefaultMoninObukhov()

    n_updrafts = 1
    prognostic_tke = true
    turbconv_model = CA.DiagnosticEDMFX{n_updrafts, prognostic_tke}(1e-5)

    edmfx_model = CA.EDMFXModel(;
        entr_model = CA.InvZEntrainment(),
        detr_model = CA.BuoyancyVelocityDetrainment(),
        sgs_mass_flux = true,
        sgs_diffusive_flux = true,
        nh_pressure = true,
        vertical_diffusion = false,
        filter = false,
        scale_blending_method = CA.SmoothMinimumBlending(),
    )
    topography = CA.EarthTopography()
    h_elem = 4
    z_elem = 15
    z_max = 60000.0
    dz_bottom = 30.0

    model = CA.AtmosModel(;
        microphysics_model,
        turbconv_model,
        edmfx_model,
        radiation_mode,
        insolation,
        rayleigh_sponge,
        viscous_sponge,
        hyperdiff,
        diff_mode,
        reproducible_restart = CA.ReproducibleRestart(),
        test_dycore_consistency = CA.TestDycoreConsistency(),
    )

    grid = CA.SphereGrid(FT; topography, h_elem, z_elem, z_max, dz_bottom, context)

    # TODO: Use jacobian flags
    approximate_linear_solve_iters = 2
    max_newton_iters_ode = 1

    newtons_method = CTS.NewtonsMethod(;
        max_iters = max_newton_iters_ode,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )

    ode_config = CTS.IMEXAlgorithm(
        CTS.ARS343(),
        newtons_method,
    )

    callback_kwargs = (;
        dt_rad = "1secs",
        dt_cloud_fraction = "1secs",
    )

    args = (; model,
        grid,
        tracers,
        dt = 1secs,
        t_end = 3secs,
        checkpoint_frequency = 1secs,
        approximate_linear_solve_iters,
        callback_kwargs,
        ode_config,
        surface_setup,
        context,
        job_id = "amip_target_diagedmf",
        output_dir)
    simulation = CA.AtmosSimulation{FT}(; args...)

    return (; simulation, args)

end

get_edmfx_model(turbconv_model) = nothing
get_edmfx_model(turbconv_model::CA.DiagnosticEDMFX) = CA.EDMFXModel(;
    entr_model = CA.InvZEntrainment(),
    detr_model = CA.BuoyancyVelocityDetrainment(),
    sgs_mass_flux = true,
    sgs_diffusive_flux = true,
    nh_pressure = true,
    vertical_diffusion = false,
    filter = false,
    scale_blending_method = CA.SmoothMinimumBlending(),
)

# Begin tests

# Disable all the @info statements that are produced when creating a simulation
Logging.disable_logging(Logging.Info)


"""
    test_restart(simulation, model, grid; job_id, comms_ctx, more_ignore = Symbol[])

Test if the restarts are consistent for a simulation.

`more_ignore` is a Vector of Symbols that identifies config-specific keys that
have to be ignored when reading a simulation.
"""
function test_restart(simulation, args; comms_ctx, more_ignore = Symbol[])
    ClimaComms.iamroot(comms_ctx) && println("job_id = $(simulation.job_id)")

    local_success = true

    CA.solve_atmos!(simulation)

    # Check re-importing the same state
    restart_dir = simulation.output_dir
    @test isfile(joinpath(restart_dir, "day0.3.hdf5"))

    # Reset random seed for RRTMGP
    Random.seed!(1234)

    ClimaComms.iamroot(comms_ctx) && println("    just reading data")
    # Recreate simulation with detect_restart_file=true
    FT = typeof(simulation.integrator.p.dt)
    simulation_restarted = CA.AtmosSimulation{FT}(;
        args...,
        context = comms_ctx,
        detect_restart_file = true,
    )

    if pkgversion(CA.RRTMGP) < v"0.22"
        # Versions of RRTMGP older than 0.22 have a bug and do not set the
        # flux_dn_dir, so that face_clear_sw_direct_flux_dn and
        # face_sw_direct_flux_dn is uninitialized and not deterministic
        rrtmgp_clear_fix =
            (:face_clear_sw_direct_flux_dn, :face_sw_direct_flux_dn)
    else
        rrtmgp_clear_fix = ()
    end

    local_success &= compare(
        simulation.integrator.u,
        simulation_restarted.integrator.u;
        name = "integrator.u",
    )
    local_success &= compare(
        axes(simulation.integrator.u.c),
        axes(simulation_restarted.integrator.u.c);
        name = "space",
    )
    local_success &= compare(
        simulation.integrator.p,
        simulation_restarted.integrator.p;
        name = "integrator.p",
        ignore = Set([
            :ghost_buffer,
            :hyperdiffusion_ghost_buffer,
            :scratch,
            :output_dir,
            :ghost_buffer,
            # Computed in tendencies (which are not computed in this case)
            :hyperdiff,
            # Precomputed scratch fields for microphysics (uninitialized until tendencies run)
            :ᶜmp_tendency, :ᶜ∂Sq_tot,
            # rc is some CUDA/CuArray internal object that we don't care about
            :rc,
            # DataHandlers contains caches, so they are stateful
            :data_handler,
            # Covariance fields are recomputed in set_precomputed_quantities!
            :ᶜT′T′, :ᶜq′q′,
            rrtmgp_clear_fix...,
            # Config-specific
            more_ignore...,
        ]),
    )

    # Check re-importing from previous state and advancing one step
    ClimaComms.iamroot(comms_ctx) && println("    reading and simulating")
    # Reset random seed for RRTMGP
    Random.seed!(1234)

    restart_file = joinpath(simulation.output_dir, "day0.2.hdf5")
    @test isfile(joinpath(restart_dir, "day0.2.hdf5"))
    # Restart from specific file
    FT = typeof(simulation.integrator.p.dt)
    simulation_restarted2 = CA.AtmosSimulation{FT}(;
        args...,
        context = comms_ctx,
        restart_file,
    )
    CA.fill_with_nans!(simulation_restarted2.integrator.p)

    CA.solve_atmos!(simulation_restarted2)
    local_success &= compare(
        simulation.integrator.u,
        simulation_restarted2.integrator.u;
        name = "integrator.u",
    )
    local_success &= compare(
        simulation.integrator.p,
        simulation_restarted2.integrator.p;
        name = "integrator.p",
        ignore = Set([
            :scratch,
            :output_dir,
            :ghost_buffer,
            :hyperdiffusion_ghost_buffer,
            :data_handler,
            :ᶜmp_tendency, :ᶜ∂Sq_tot,
            :ᶜT′T′, :ᶜq′q′,
            :rc,
            rrtmgp_clear_fix...,
        ]),
    )

    return (
        local_success,
        simulation,
        simulation_restarted,
        simulation_restarted2,
    )
end

# Let's prepare the test_dicts. TESTING is a Vector of NamedTuples, each element
# has a test_dict, a job_id, and a more_ignore

TESTING = Any[]

# Add a configuration with all the bells and whistles

FT = Float32
if MANYTESTS
    allsky_radiation =
        RRTMGPI.AllSkyRadiation(;
            idealized_h2o = false,
            idealized_clouds = false,
            cloud = CA.InteractiveCloudInRadiation(),
            add_isothermal_boundary_layer = true,
            aerosol_radiation = false,
            reset_rng_seed = false,
            deep_atmosphere = true,
        )
    diagnostic_edmfx = CA.DiagnosticEDMFX(; area_fraction = 1e-5)
    topography = CA.EarthTopography()
    if comms_ctx isa ClimaComms.SingletonCommsContext
        grids = (
            CA.SphereGrid(FT; topography, context = comms_ctx),
            CA.BoxGrid(FT; context = comms_ctx),
            CA.ColumnGrid(FT; context = comms_ctx),
        )
    else
        grids = (
            CA.SphereGrid(FT; topography, context = comms_ctx),
            CA.BoxGrid(FT; context = comms_ctx),
        )
    end

    for grid in grids
        mesh = if hasproperty(grid, :horizontal_grid)
            grid.horizontal_grid.topology.mesh
        else
            nothing
        end
        if mesh isa Meshes.EquiangularCubedSphere
            microphys_models = (CA.NonEquilibriumMicrophysics1M(),)
            topography_type = CA.EarthTopography()
            turbconv_models = (nothing, diagnostic_edmfx)
            radiation_modes = (nothing, allsky_radiation)
        else
            microphys_models = (CA.EquilibriumMicrophysics0M(),)
            topography_type = CA.NoTopography()
            turbconv_models = (diagnostic_edmfx,)
            gray_radiation = RRTMGPI.GrayRadiation(;
                add_isothermal_boundary_layer = true,
                deep_atmosphere = false,
            )
            radiation_modes = (gray_radiation, allsky_radiation)
        end

        for turbconv_model in turbconv_models
            for radiation_mode in radiation_modes
                for microphysics_model in microphys_models
                    # EDMF only supports equilibrium moisture
                    if turbconv_model isa CA.DiagnosticEDMFX &&
                       microphysics_model isa CA.NonEquilibriumMicrophysics
                        continue
                    end

                    edmfx_model = get_edmfx_model(turbconv_model)
                    model = CA.AtmosModel(;
                        radiation_mode,
                        microphysics_model,
                        turbconv_model,
                        edmfx_model,
                        insolation = CA.IdealizedInsolation(),
                        reproducible_restart = CA.ReproducibleRestart(),
                        test_dycore_consistency = CA.TestDycoreConsistency())

                    # The `enable_bubble` case is broken for ClimaCore < 0.14.6, so we
                    # hard-code this to be always false for those versions
                    bubble = pkgversion(ClimaCore) > v"0.14.5"

                    # Make sure that all MPI processes agree on the output_loc
                    output_loc =
                        ClimaComms.iamroot(comms_ctx) ? mktempdir(pwd()) :
                        ""
                    output_loc = ClimaComms.bcast(comms_ctx, output_loc)
                    # Sometimes the shared filesystem doesn't work properly
                    # and the folder is not synced across MPI processes.
                    # Let's add an additional check here.
                    maybe_wait_filesystem(comms_ctx, output_loc)

                    # Create job_id string from configuration
                    config_name =
                        mesh isa Meshes.EquiangularCubedSphere ? "sphere" :
                        mesh isa Meshes.RectilinearMesh ? "box" : "column"
                    microphysics_name =
                        microphysics_model isa CA.NonEquilibriumMicrophysics1M ?
                        "nonequil_1M" : "equil_0M"
                    topo_name = string(nameof(typeof(topography_type)))
                    rad_name =
                        isnothing(radiation_mode) ? "none" :
                        radiation_mode isa RRTMGPI.GrayRadiation ? "gray" : "allsky"
                    turbconv_name =
                        isnothing(turbconv_model) ? "none" : "diagnostic_edmfx"
                    job_id = "$(config_name)_$(microphysics_name)_$(topo_name)_$(rad_name)_$(turbconv_name)"
                    callback_kwargs = (;
                        dt_rad = "1secs",
                        dt_cloud_fraction = "1secs",
                    )
                    args = (;
                        model,
                        grid,
                        job_id,
                        callback_kwargs,
                        default_diagnostics = false,
                        dt = 1secs,
                        t_end = 3secs,
                        checkpoint_frequency = 1secs,
                    )
                    simulation = CA.AtmosSimulation{FT}(; args..., context = comms_ctx)
                    push!(
                        TESTING,
                        (; simulation, args, more_ignore = Symbol[]),
                    )
                end
            end
        end
    end
else
    output_dir = ClimaComms.iamroot(comms_ctx) ? mktempdir(pwd()) : ""
    output_dir = ClimaComms.bcast(comms_ctx, output_dir)
    # Sometimes the shared filesystem doesn't work properly and the folder is
    # not synced across MPI processes. Let's add an additional check here.
    maybe_wait_filesystem(comms_ctx, output_dir)

    push!(
        TESTING,
        (;
            amip_target_diagedmf(
                comms_ctx,
                joinpath(output_dir, "amip_target_diagedmf"),
            )...,
            more_ignore = Symbol[],
        ),
    )
end

# We know that this test is broken for old versions of ClimaCore
@test all(
    @time test_restart(
        t.simulation,
        t.args;
        comms_ctx = comms_ctx,
        more_ignore = t.more_ignore,
    )[1] for
    t in TESTING
) skip = pkgversion(ClimaCore) < v"0.14.18"
