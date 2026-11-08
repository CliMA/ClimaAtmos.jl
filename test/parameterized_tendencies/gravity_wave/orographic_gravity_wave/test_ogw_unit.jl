#=
Pure-function unit tests for the orographic gravity wave drag helpers in
`src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave.jl`.

Fast (~1 s, no simulation build) so they run in `test/runtests.jl` under the
`parameterizations` group.
=#

using Test
import ClimaAtmos as CA
import ClimaCore: CommonSpaces

@testset "Semi-implicit orographic gravity-wave drag damping" begin
    for FT in (Float32, Float64)
        u_vec = CA.Geometry.UVWVector(FT(10.0), FT(-5.0), FT(0.0))

        # Small dt d / s limit reproduces the explicit drag.
        du_small = FT(-1e-4)
        dv_small = FT(5e-5)
        d_si_small =
            CA._semi_implicit_gw_drag(du_small, dv_small, u_vec, FT(1.0))
        @test d_si_small.u ≈ du_small rtol = FT(1e-4)
        @test d_si_small.v ≈ dv_small rtol = FT(1e-4)

        # Stiff opposing drag (|D_∥| dt ≫ |u|) cannot reverse the parallel
        # wind in one step, while the perpendicular component D_⊥ is unchanged.
        dt_large = FT(360.0)
        # Purely parallel opposing drag (-2, 1) + perpendicular drag (1, 2).
        du_stiff = FT(-1.0)
        dv_stiff = FT(3.0)
        d_si_stiff =
            CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_vec, dt_large)
        u_next = FT(10.0) + dt_large * d_si_stiff.u
        v_next = FT(-5.0) + dt_large * d_si_stiff.v
        @test u_next * FT(10.0) + v_next * FT(-5.0) > 0
        @test d_si_stiff.u * FT(-5.0) - d_si_stiff.v * FT(10.0) ≈
              du_stiff * FT(-5.0) - dv_stiff * FT(10.0) rtol = FT(1e-5)

        # The same holds for a weak wind: the damped parallel deceleration is
        # below s / dt for every s > 0.
        u_weak = CA.Geometry.UVWVector(FT(1e-3), FT(-5e-4), FT(0.0))
        d_si_weak =
            CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_weak, dt_large)
        u_next = FT(1e-3) + dt_large * d_si_weak.u
        v_next = FT(-5e-4) + dt_large * d_si_weak.v
        @test u_next * FT(1e-3) + v_next * FT(-5e-4) > 0
        @test d_si_weak.u * FT(-5e-4) - d_si_weak.v * FT(1e-3) ≈
              du_stiff * FT(-5e-4) - dv_stiff * FT(1e-3) rtol = FT(1e-4)

        # Accelerating component (D ⋅ uₕ ≥ 0) is left unchanged.
        d_acc = CA._semi_implicit_gw_drag(FT(2.0), FT(-1.0), u_vec, dt_large)
        @test d_acc.u == FT(2.0)
        @test d_acc.v == FT(-1.0)

        # Zero and subnormal winds return the explicit drag, finite.
        for w in (FT(0), floatmin(FT) / 4)
            u_zero = CA.Geometry.UVWVector(w, w, FT(0.0))
            d_si_zero =
                CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_zero, dt_large)
            @test isfinite(d_si_zero.u) && isfinite(d_si_zero.v)
            @test d_si_zero.u == du_stiff
            @test d_si_zero.v == dv_stiff
        end
    end
end

@testset "Orographic non-propagating forcing guards NaN weights" begin
    for FT in (Float32, Float64)
        cspace = CommonSpaces.ColumnSpace(
            FT;
            z_elem = 8,
            z_min = 500,
            z_max = 8000,
            staggering = CommonSpaces.CellCenter(),
        )
        fspace = CA.Spaces.face_space(cspace)
        lspace = CA.Spaces.level(cspace, 1)
        flspace = CA.Spaces.level(fspace, CA.Fields.half)

        ᶜuforcing = fill(FT(1e-4), cspace)
        ᶜvforcing = fill(FT(-2e-4), cspace)
        τ_x = fill(FT(-0.5), lspace)
        τ_y = fill(FT(0.25), lspace)
        τ_l = fill(FT(1.0), lspace)
        τ_np = fill(FT(0.5), lspace)
        ᶠVτ = fill(FT(10.0), fspace)
        # Set z_pbl above model top so z_ref stays 0 and ᶠp_ref stays NaN.
        ᶠz_pbl = fill(FT(9000.0), fspace)
        ᶠz_ref = zeros(FT, flspace)
        ᶠp_ref = zeros(FT, flspace)
        ᶜmask = fill(false, cspace)
        ᶜweights = zeros(FT, cspace)
        ᶜdiff = zeros(FT, cspace)
        ᶜwtsum = zeros(FT, lspace)
        ᶠz = CA.Fields.coordinate_field(fspace).z
        ᶠdz = CA.Fields.local_geometry_field(fspace).∂x∂ξ.components.data.:1
        ᶠp = @. FT(1e5) * exp(-(ᶠz) / FT(8000))
        ᶠp_m1 = @. ᶠp * FT(1.05)
        ᶠN = fill(FT(0.01), fspace)

        CA.calc_nonpropagating_forcing!(
            ᶜuforcing,
            ᶜvforcing,
            τ_x,
            τ_y,
            τ_l,
            τ_np,
            ᶠVτ,
            ᶠz_pbl,
            ᶠz_ref,
            ᶠp_ref,
            ᶜmask,
            ᶜweights,
            ᶜdiff,
            ᶜwtsum,
            ᶠp,
            ᶠp_m1,
            ᶠN,
            ᶠz,
            ᶠdz,
            FT(9.81),
        )
        @test all(isnan, parent(ᶠp_ref))
        @test all(iszero, parent(ᶜweights))
        @test all(==(FT(1e-4)), parent(ᶜuforcing))
        @test all(==(FT(-2e-4)), parent(ᶜvforcing))
    end
end
