module PrecompileCI

using PrecompileTools, Logging
import ClimaAtmos as CA
import ClimaComms
import ClimaParams
import ClimaTimeSteppers as CTS

# Build a simulation without running it. This compiles the cache, the tendency
# closures, the Jacobian, the integrator, and the diagnostic writers, which
# together dominate CI startup time.
function build_simulation(model; output_dir, ode_algo = CTS.ARS343())
    ode_config = CTS.IMEXAlgorithm(
        ode_algo,
        CTS.NewtonsMethod(;
            max_iters = 1,
            update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
        ),
    )
    return CA.AtmosSimulation(
        model;
        dt = 60,
        t_end = 3600,
        ode_config,
        jacobian = CA.ManualSparseJacobian(; approximate_solve_iters = 2),
        output_dir,
        output_dir_style = "removepreexisting",
    )
end

@compile_workload begin
    with_logger(NullLogger()) do
        FT = Float32 # Float64?
        h_elem = 6 # 16, 30?
        z_elem = 10 # 30, 31, 63?
        x_elem = y_elem = 2
        x_max = y_max = 1e8
        z_max = FT(30000.0)
        dz_bottom = FT(500)
        z_stretch = true
        bubble = true
        nh_poly = 3 # GLL{4} = nh_poly + 1
        # TODO: compile CUDA methods as well
        context = ClimaComms.context(ClimaComms.CPUSingleThreaded())
        topography = CA.NoTopography()
        params = CA.ClimaAtmosParameters(FT)
        radius = CA.Parameters.planet_radius(params)

        sphere_grid = CA.SphereGrid(
            FT;
            context,
            radius, h_elem, nh_poly,
            z_elem, z_max, z_stretch, dz_bottom,
            bubble, topography,
        )
        box_grid = CA.BoxGrid(
            FT;
            context,
            x_elem, x_max, y_elem, y_max, nh_poly, periodic_x = true, periodic_y = true,
            z_elem, z_max, z_stretch, dz_bottom,
            bubble, topography,
        )
        plane_grid = CA.PlaneGrid(
            FT;
            context,
            x_elem, x_max, nh_poly, periodic_x = true,
            z_elem, z_max, z_stretch, dz_bottom,
            topography,
        )
        column_grid = CA.ColumnGrid(
            FT; context, z_elem, z_max, z_stretch, dz_bottom,
        )
        all_grids = (sphere_grid, box_grid, plane_grid, column_grid)
        foreach(CA.get_spaces, all_grids)

        # Single-column prognostic EDMF with 1-moment microphysics, the shape of
        # most column jobs in the pipeline.
        scm_grid = CA.ColumnGrid(
            FT; context, z_elem, z_max = FT(3000), z_stretch = false,
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

        mktempdir() do dir
            build_simulation(
                scm_model;
                output_dir = joinpath(dir, "scm"),
                ode_algo = CTS.ARS222(),
            )
            build_simulation(sphere_model; output_dir = joinpath(dir, "sphere"))
        end
    end
end

end # module PrecompileCI
