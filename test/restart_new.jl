redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaAtmos as CA
import ClimaCore
import ClimaCore: DataLayouts, Fields, Geometry
import ClimaCore.Fields: Field, FieldVector, field_values
import ClimaCore.DataLayouts: AbstractData
import ClimaCore.Geometry: AxisTensor
import ClimaCore.Spaces: AbstractSpace
import ClimaComms
import ClimaParams
import ClimaTimeSteppers as CTS

import ClimaUtilities.OutputPathGenerator: maybe_wait_filesystem
pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import Logging
import NCDatasets
import YAML
using Test, Dates

import ClimaAtmos.RRTMGPInterface as RRTMGPI

import Random
Random.seed!(1234)

const device = ClimaComms.device()
const comms_ctx = ClimaComms.context(device)
ClimaComms.init(comms_ctx)

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

MANYTESTS = false
if length(ARGS) > 0
    if ARGS[1] == "--manytests"
        # Check if the first argument is "--manytests" (if provided), if yes, check
        # the second argument for true/false. If the second argument is not provided
        # assume true.
        second_argument = lowercase(get(ARGS, 2, "true"))
        second_argument == "true" && (MANYTESTS = true)
    else
        error("Argument $argument not recognized")
    end
end
MANYTESTS && @info "Running multiple tests"

# Technical note:
#
# Test.jl really wants to give you a stacktrace for failing tests. This seems to be
# hardcoded in the package and not easy to change without defining a whole new
# AbstractTestSet. We don't want stacktraces, we just want to know which fields are
# different.
#
# For this reason, we don't use Test but just print to screen the differences.

"""
    _error(arr1::AbstractArray, arr2::AbstractArray; ABS_TOL = 100eps(eltype(arr1)))

We compute the error in this way:
- when the absolute value is larger than ABS_TOL, we use the absolute error
- in the other cases, we compare the relative errors
"""
function _error(
    arr1::AbstractArray,
    arr2::AbstractArray;
    ABS_TOL = 100eps(eltype(arr1)),
)
    # There are some parameters, e.g. Obukhov length, for which Inf
    # is a reasonable value (implying a stability parameter in the neutral boundary layer
    # regime, for instance). We account for such instances with the `isfinite` function.
    arr1 = Array(arr1) .* isfinite.(Array(arr1))
    arr2 = Array(arr2) .* isfinite.(Array(arr2))
    diff = abs.(arr1 .- arr2)
    denominator = abs.(arr1)
    error = ifelse.(denominator .> ABS_TOL, diff ./ denominator, diff)
    return error
end


"""
    compare(v1, v2; name = "", ignore = Set([:rc]))

Return whether `v1` and `v2` are the same (up to floating point errors).

`compare` walks through all the properties in `v1` and `v2` until it finds
that there are no more properties. At that point, `compare` tries to match the
resulting objects. When such objects are arrays with floating point, `compare`
defines a notion of `error` that is the following: when the absolute value is
less than `100eps(eltype)`, `error = absolute_error`, otherwise it is relative
error. The `error` is then compared against a tolerance.

Keyword arguments
=================

- `name` is used to collect the name of the property while we go recursively
  over all the properties. You can pass a base name.
- `ignore` is a collection of `Symbol`s that identify properties that are
  ignored when walking through the tree. This is useful for properties that
  are known to be different (e.g., `output_dir`).

`:rc` is some CUDA/CuArray internal object that we don't care about
"""
function compare(
    v1::T,
    v2::T;
    name = "",
    ignore = Set([:rc]),
) where {T <: Union{FieldVector, CA.AtmosCache, AbstractSpace}}
    pass = true
    return _compare(pass, v1, v2; name, ignore)
end

function _compare(pass, v1::T, v2::T; name, ignore) where {T}
    properties = filter(x -> !(x in ignore), propertynames(v1))
    if isempty(properties)
        pass &= _compare(v1, v2; name, ignore)
    else
        # Recursive case
        for p in properties
            pass &= _compare(
                pass,
                getproperty(v1, p),
                getproperty(v2, p);
                name = "$(name).$(p)",
                ignore,
            )
        end
    end
    return pass
end

function _compare(v1::T, v2::T; name, ignore) where {T}
    return print_maybe(v1 == v2, "$name differs")
end

function _compare(
    v1::T,
    v2::T;
    name,
    ignore,
) where {T <: Union{AbstractString, Symbol}}
    # What we can safely print without filling STDOUT
    return print_maybe(v1 == v2, "$name differs: $v1 vs $v2")
end

function _compare(v1::T, v2::T; name, ignore) where {T <: Number}
    # We check with triple equal so that we also catch NaNs being equal
    return print_maybe(v1 === v2, "$name differs: $v1 vs $v2")
end

# We ignore NCDatasets. They contain a lot of state-ful information
function _compare(
    pass,
    v1::T,
    v2::T;
    name,
    ignore,
) where {T <: NCDatasets.NCDataset}
    return pass
end

function _compare(
    v1::T,
    v2::T;
    name,
    ignore,
) where {T <: Field{<:AbstractData{<:Real}}}
    return _compare(parent(v1), parent(v2); name, ignore)
end

function _compare(pass, v1::T, v2::T; name, ignore) where {T <: AbstractData}
    return pass && _compare(parent(v1), parent(v2); name, ignore)
end

# Handle views
function _compare(
    pass,
    v1::SubArray{FT},
    v2::SubArray{FT};
    name,
    ignore,
) where {FT <: AbstractFloat}
    return pass && _compare(collect(v1), collect(v2); name, ignore)
end

function _compare(
    v1::AbstractArray{FT},
    v2::AbstractArray{FT};
    name,
    ignore,
) where {FT <: AbstractFloat}
    error = maximum(_error(v1, v2))
    return print_maybe(error <= 100eps(eltype(v1)), "$name error: $error")
end

function _compare(pass, v1::T1, v2::T2; name, ignore) where {T1, T2}
    error("v1 and v2 have different types")
end

function print_maybe(exp, what)
    exp || println(what)
    return exp
end


function amip_target_diagedmf(context)
    FT = Float32
    days = 86400
    start_date = DateTime(2010, 1, 1)
    
    param_dict =
        ClimaParams.create_toml_dict(FT; override_file = "toml/longrun_aquaplanet_diagedmf.toml")
    params = CA.ClimaAtmosParameters(param_dict)
    
    deep_atmosphere = true
    cloud = CA.InteractiveCloudInRadiation()
    rayleigh_sponge = CA.RayleighSponge{FT}(;
        zd = params.zd_rayleigh,
        α_uₕ = params.alpha_rayleigh_uh,
        α_w = params.alpha_rayleigh_w,
        α_sgs_tracer = params.alpha_rayleigh_sgs_tracer,
    )
    viscous_sponge = CA.ViscousSponge{FT}(; zd = params.zd_viscous, κ₂ = params.kappa_2_sponge)
    
    diff_mode = CA.Implicit()
    hyperdiff = CA.ClimaHyperdiffusion{FT}(;
        ν₄_vorticity_coeff = 0.150 * 1.238,
        ν₄_scalar_coeff = 0.751 * 1.238,
        divergence_damping_factor = 5,
    )
    
    tracers = (
        "CB1", "CB2",
        "DST01", "DST02", "DST03", "DST04", "DST05",
        "OC1", "OC2",
        "SO4",
        "SSLT01", "SSLT02", "SSLT03", "SSLT04", "SSLT05",
    )
    microphysics_model = CA.Microphysics0Moment()
    moisture_model = CA.EquilMoistModel()
    
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
        sgs_mass_flux = Val(true),
        sgs_diffusive_flux = Val(true),
        nh_pressure = Val(true),
        vertical_diffusion = Val(false),
        filter = Val(false),
        scale_blending_method = CA.SmoothMinimumBlending(),
    )
    topography = CA.EarthTopography()
    h_elem = 16
    z_elem = 63
    z_max = 60000.0
    dz_bottom = 30.0
    
    model = CA.AtmosModel(;
        moisture_model,
        microphysics_model,
        turbconv_model,
        edmfx_model,
        radiation_mode,
        insolation,
        rayleigh_sponge,
        viscous_sponge,
        hyperdiff,
        diff_mode,
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
        dt_rad = "1hours",
        dt_cloud_fraction = "1hours",
    )
    
    simulation = CA.AtmosSimulation{FT}(;
        model,
        grid,
        tracers,
        t_end = 120days,
        checkpoint_frequency = 30days,
        approximate_linear_solve_iters,
        callback_kwargs,
        ode_config,
        surface_setup,
        context,
    )
    
    return (simulation, model, grid)
    
end    

# Begin tests

# Disable all the @info statements that are produced when creating a simulation
Logging.disable_logging(Logging.Info)


"""
    test_restart(simulation, model, grid; job_id, comms_ctx, more_ignore = Symbol[])

Test if the restarts are consistent for a simulation.

`more_ignore` is a Vector of Symbols that identifies config-specific keys that
have to be ignored when reading a simulation.
"""
function test_restart(simulation, model, grid; job_id, comms_ctx, more_ignore = Symbol[])
    ClimaComms.iamroot(comms_ctx) && println("job_id = $(job_id)")

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
        model,
        grid,
        job_id,
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
            # rc is some CUDA/CuArray internal object that we don't care about
            :rc,
            # DataHandlers contains caches, so they are stateful
            :data_handler,
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
    simulation_restarted2 = CA.AtmosSimulation{FT}(
        model = model,
        grid = grid,
        job_id = job_id,
        context = comms_ctx,
        restart_file = restart_file,
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
        RRTMGPI.AllSkyRadiation(false, false, CA.InteractiveCloudInRadiation(), true, false)
    diagnostic_edmfx = CA.DiagnosticEDMFX{1, false}(1e-5)
    if comms_ctx isa ClimaComms.SingletonCommsContext
        grids = (CA.SphereGrid(FT; context = comms_ctx), CA.BoxGrid(FT; context = comms_ctx), CA.ColumnGrid(FT; context = comms_ctx))
    else
        grids = (CA.SphereGrid(FT; context = comms_ctx), CA.BoxGrid(FT; context = comms_ctx))
    end

    for grid in grids
        if grid isa CA.SphereGrid
            moisture_models = (CA.NonEquilMoistModel(),)
            precip_models = (CA.Microphysics1Moment(),)
            topography = CA.EarthTopography()
            turbconv_models = (nothing, diagnostic_edmfx)
            radiations = (nothing, allsky_radiation)
        else
            moisture_models = (CA.EquilMoistModel(),)
            precip_models = (CA.Microphysics0Moment(),)
            topography = CA.NoWarpTopography()
            turbconv_models = (diagnostic_edmfx,)
            gray_radiation = RRTMGPI.GrayRadiation(true, false)
            radiation_modes = (gray_radiation, allsky_radiation)
        end

        for turbconv_model in turbconv_models
            for radiation_mode in radiation_modes
                for moisture_model in moisture_models
                    for precip_model in precip_models
                        # EDMF only supports equilibrium moisture
                        if turbconv_model isa CA.DiagnosticEDMFX &&
                           moisture_model isa CA.NonEquilMoistModel
                            continue
                        end

                        model = CA.AtmosModel(;
                            radiation_mode,
                            moisture_model,
                            microphysics_model = precip_model,
                            topography,
                            turbconv_model = turbconv_model,
                        )

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
                        config_name = grid isa CA.SphereGrid ? "sphere" : grid isa CA.BoxGrid ? "box" : "column"
                        moisture_name = moisture_model isa CA.NonEquilMoistModel ? "nonequil" : "equil"
                        precip_name = precip_model isa CA.Microphysics1Moment ? "1M" : "0M"
                        topo_name = topography isa CA.EarthTopography ? "Earth" : "NoWarp"
                        rad_name = isnothing(radiation_mode) ? "none" : radiation_mode isa RRTMGPI.GrayRadiation ? "gray" : "allsky"
                        turbconv_name = isnothing(turbconv_model) ? "none" : "diagnostic_edmfx"
                        job_id = "$(config_name)_$(moisture_name)_$(precip_name)_$(topo_name)_$(rad_name)_$(turbconv_name)"
                        simulation = CA.AtmosSimulation{FT}(; model, grid, job_id, context = comms_ctx)
                        push!(
                            TESTING,
                            (; simulation, model, grid, job_id, more_ignore = Symbol[]),
                        )
                    end
                end
            end
        end
    end
else
    amip_output_loc = ClimaComms.iamroot(comms_ctx) ? mktempdir(pwd()) : ""
    amip_output_loc = ClimaComms.bcast(comms_ctx, amip_output_loc)
    # Sometimes the shared filesystem doesn't work properly and the folder is
    # not synced across MPI processes. Let's add an additional check here.
    maybe_wait_filesystem(comms_ctx, amip_output_loc)

    amip_job_id = "amip_target_diagedmf"

    simulation, model, grid = amip_target_diagedmf(comms_ctx)
    push!(
        TESTING,
        (;
            simulation,
            model,
            grid,
            job_id = amip_job_id,
            more_ignore = Symbol[],
        ),
    )
end

# We know that this test is broken for old versions of ClimaCore
@test all(
    @time test_restart(t.simulation, t.model, t.grid; job_id=t.job_id, comms_ctx=comms_ctx, more_ignore=t.more_ignore)[1] for
    t in TESTING
) skip = pkgversion(ClimaCore) < v"0.14.18"
