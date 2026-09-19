module PrecompileCI

using PrecompileTools, Logging
import ClimaAtmos as CA
import ClimaComms
import ClimaParams
import ClimaTimeSteppers as CTS
import Dates

# Build the state, the cache, and the tendency function, the stages that a
# simulation shares with every job. `CTS.init` and the diagnostic writers are
# left out: they cost far more to precompile than they save.
function build_tendency(model; ode_algo = CTS.ARS343())
    (; params) = model
    ode_config = CTS.IMEXAlgorithm(
        ode_algo,
        CTS.NewtonsMethod(;
            max_iters = 1,
            update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
        ),
    )
    start_date = Dates.DateTime(2010, 1, 1)
    dt, t_start, t_end = CA.convert_time_args(60, 0, 3600, start_date)
    Y = CA.initial_state(model)
    p = CA.build_cache(Y, model, params, dt, start_date, nothing)
    return CA.args_integrator(
        Y, p, (t_start, t_end), ode_config, CTS.CallbackSet(),
        CA.ManualSparseJacobian(; approximate_solve_iters = 2), false,
        model.prescribed_flow, dt, "stage", "step",
    )
end

# The grids, the spaces, and the dry cache are precompiled in ClimaAtmos
# itself; this workload only covers what depends on a specific configuration.
@compile_workload begin
    with_logger(NullLogger()) do
        FT = Float32 # Float64?
        # TODO: compile CUDA methods as well
        context = ClimaComms.context(ClimaComms.CPUSingleThreaded())
        params = CA.ClimaAtmosParameters(FT)
        sphere_grid = CA.SphereGrid(FT; context)

        # Single-column prognostic EDMF with 1-moment microphysics, the shape of
        # most column jobs in the pipeline.
        scm_grid = CA.ColumnGrid(
            FT; context, z_max = FT(3000), z_stretch = false,
        )
        scm_model = CA.AtmosModel(
            scm_grid;
            params,
            setup = CA.Setups.Bomex(FT),
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            turbconv_model = CA.PrognosticEDMFX(; area_fraction = 1e-5),
            edmfx_model = CA.EDMFXModel(;
                entr_model = CA.InvZEntrainment(),
                detr_model = CA.BuoyancyVelocityDetrainment(),
                sgs_mass_flux = true,
                sgs_diffusive_flux = true,
                nh_pressure = true,
                vertical_diffusion = true,
                filter = true,
                scale_blending_method = CA.SmoothMinimumBlending(),
            ),
        )

        # Moist sphere without radiation, the shape of the baroclinic wave
        # and MPI jobs.
        sphere_model = CA.AtmosModel(
            sphere_grid;
            params,
            microphysics_model = CA.EquilibriumMicrophysics0M(),
        )

        build_tendency(scm_model; ode_algo = CTS.ARS222())
        build_tendency(sphere_model)
    end
end

end # module PrecompileCI
