using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA

# Pkg.test on Julia <1.13 forces bounds checks, so this also catches out-of-bounds indexing under `@inbounds`.
# This does not work for Julia 1.13, where Pkg.test no longer forces bounds checks.
@testset "Sphere simulation with radiation, surface fluxes, and vertical diffusion" begin
    config = CA.AtmosConfig(
        Dict(
            "config" => "sphere",
            "h_elem" => 4,
            "z_elem" => 15,
            "z_max" => 60000.0,
            "dz_bottom" => 30.0,
            "ode_algo" => "ARS343",
            "hyperdiff" => "Hyperdiffusion",
            "implicit_diffusion" => true,
            "approximate_linear_solve_iters" => 2,
            "max_newton_iters_ode" => 1,
            "surface_setup" => "DefaultMoninObukhov",
            "rad" => "allskywithclear",
            "vert_diff" => "VerticalDiffusion",
            "microphysics_model" => "0M",
            "dt" => "1secs",
            "dt_rad" => "1secs",
            "t_end" => "3secs",
            "log_progress" => false,
            "output_default_diagnostics" => false,
        );
        job_id = "checkbounds_sphere_simulation",
    )
    simulation = CA.get_simulation(config)
    sol_res = CA.solve_atmos!(simulation)
    @test sol_res.ret_code == :success
end
