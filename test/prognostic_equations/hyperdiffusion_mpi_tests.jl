# Rank consistency of the measured hyperdiffusion grid factor.
# Run with CLIMACOMMS_CONTEXT=MPI on two or more ranks; see #4673.
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
using ClimaCore.CommonSpaces

context = ClimaComms.context()
ClimaComms.init(context)

@testset "measured grid factor is rank-consistent" begin
    box = RectangleXYSpace(
        Float64;
        x_min = 0,
        x_max = 2720,
        y_min = 0,
        y_max = 2720,
        periodic_x = true,
        periodic_y = true,
        n_quad_points = 4,
        x_elem = 8,
        y_elem = 8,
        context,
    )
    sphere = CubedSphereSpace(
        Float64;
        radius = 6.371e6,
        n_quad_points = 4,
        h_elem = 6,
        context,
    )
    # Reference values from hyperdiffusion_tests.jl; see #4673.
    for (space, reference) in ((box, 4.0637), (sphere, 5.19))
        β = CA.measured_grid_factor(space)
        @test β ≈ reference rtol = 1e-2
        @test ClimaComms.allreduce(context, β, min) ==
              ClimaComms.allreduce(context, β, max)
    end
end
