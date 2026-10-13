#=
Advection operator tests for ClimaAtmos.jl

Impenetrable boundary condition: the state filter must ensure ᶠu³ = 0 at
both the surface and model top. Over sloped terrain-following coordinates
this is non-trivial because the horizontal wind projects onto the vertical
contravariant component (ᶠuₕ³ ≠ 0).

See src/cache/precomputed_quantities.jl for set_velocity_at_surface! and
set_velocity_at_top!.
=#

using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Spaces

include("../test_helpers.jl")

@testset "Impenetrable boundary condition over topography" begin
    # Uniform 10 m/s wind over a Schar mountain with linear mesh warping, so
    # coordinate surfaces are sloped up to the model top and ᶠuₕ³ ≠ 0 there.
    config = CA.AtmosConfig(
        Dict(
            "config" => "plane",
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "topography" => "Schar",
            "mesh_warp_type" => "Linear",
            "FLOAT_TYPE" => "Float64",
            "x_max" => 100e3,
            "x_elem" => 8,
            "z_max" => 21e3,
            "z_elem" => 20,
            "dz_bottom" => 500.0,
            "dt" => "1secs",
            "t_end" => "10secs",
            "output_default_diagnostics" => false,
        );
        job_id = "advection_boundary_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    FT = eltype(Y)

    # The state filter must cancel the vertical contravariant projection of
    # the horizontal wind at both boundaries, so that the total contravariant
    # velocity ᶠu³ vanishes there even over sloped coordinate surfaces.
    (; ᶠu³) = p.precomputed
    ᶠuₕ³ = Base.materialize(CA.compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ))
    top_level = Spaces.nlevels(axes(Y.c)) + CA.half
    for level in (CA.half, top_level)
        u³_boundary = Fields.level(ᶠu³.components.data.:1, level)
        uₕ³_boundary = Fields.level(ᶠuₕ³.components.data.:1, level)
        max_uₕ³ = maximum(abs, parent(uₕ³_boundary))
        # Nonzero uₕ³ at the boundary is what makes this test non-vacuous.
        @test max_uₕ³ > 0
        @test maximum(abs, parent(u³_boundary)) <= 100 * eps(FT) * max_uₕ³
    end
end
