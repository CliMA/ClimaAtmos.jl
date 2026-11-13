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

        # Weak oblique winds: purely opposing stiff drag (D_⊥ = 0) preserves
        # forward-Euler wind sign down to s = O(1e-3); with stiff D_⊥ ≠ 0,
        # forward-Euler sign holds above the O(eps(FT) * dt * |D_⊥|) rounding
        # of uₕ + dt * D_⊥, while D_⊥ and D_∥ match at every weak speed.
        u_weak = CA.Geometry.UVWVector(FT(1e-3), FT(-5e-4), FT(0.0))
        d_si_opp = CA._semi_implicit_gw_drag(FT(-2.0), FT(1.0), u_weak, dt_large)
        u_opp_next = FT(1e-3) + dt_large * d_si_opp.u
        v_opp_next = FT(-5e-4) + dt_large * d_si_opp.v
        @test u_opp_next * FT(1e-3) + v_opp_next * FT(-5e-4) > 0

        u_mod = CA.Geometry.UVWVector(FT(0.5), FT(-0.25), FT(0.0))
        d_si_mod =
            CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_mod, dt_large)
        @test (FT(0.5) + dt_large * d_si_mod.u) * FT(0.5) +
              (FT(-0.25) + dt_large * d_si_mod.v) * FT(-0.25) > 0
        @test d_si_mod.u * FT(-0.25) - d_si_mod.v * FT(0.5) ≈
              du_stiff * FT(-0.25) - dv_stiff * FT(0.5) rtol = FT(1e-4)

        d_si_weak =
            CA._semi_implicit_gw_drag(du_stiff, dv_stiff, u_weak, dt_large)
        s_weak = hypot(FT(1e-3), FT(-5e-4))
        d_weak = -(du_stiff * FT(1e-3) + dv_stiff * FT(-5e-4)) / s_weak
        par_weak = (d_si_weak.u * FT(1e-3) + d_si_weak.v * FT(-5e-4)) / s_weak
        @test isapprox(
            par_weak,
            -(d_weak * s_weak) / (s_weak + dt_large * d_weak);
            atol = 4 * eps(FT) * hypot(du_stiff, dv_stiff),
        )
        @test d_si_weak.u * FT(-5e-4) - d_si_weak.v * FT(1e-3) ≈
              du_stiff * FT(-5e-4) - dv_stiff * FT(1e-3) rtol = FT(1e-4)

        # For s ≪ eps(FT) * dt * d, evaluating D_⊥ = (D ⋅ n̂) n̂ in the
        # orthogonal basis avoids O(eps(FT) |D|) cancellation of D against -D_∥,
        # leaving at most O(eps(FT)^2 |D|) unit-vector rounding when D_⊥ = 0.
        s_tiny = FT(1e-20)
        u_tiny = CA.Geometry.UVWVector(s_tiny, FT(0.0), FT(0.0))
        d_si_tiny =
            CA._semi_implicit_gw_drag(FT(-1.0), FT(0.25), u_tiny, dt_large)
        @test d_si_tiny.u ≈ -s_tiny / dt_large rtol = FT(1e-5)
        @test d_si_tiny.v ≈ FT(0.25) rtol = FT(1e-6)

        u_oblique =
            CA.Geometry.UVWVector(FT(6e-21), FT(8e-21), FT(0.0))
        d_si_oblique =
            CA._semi_implicit_gw_drag(FT(-0.6), FT(-0.8), u_oblique, dt_large)
        @test isapprox(
            d_si_oblique.u * FT(0.6) + d_si_oblique.v * FT(0.8),
            -FT(1e-20) / dt_large;
            rtol = FT(1e-5),
            atol = 10 * eps(FT)^2 * hypot(FT(-0.6), FT(-0.8)),
        )
        @test isapprox(
            -d_si_oblique.u * FT(0.8) + d_si_oblique.v * FT(0.6),
            FT(0);
            atol = 2 * eps(FT) * hypot(FT(-0.6), FT(-0.8)),
        )

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
        # Put the blocked-layer start above the model top so no z_ref is found and
        # ᶠp_ref stays NaN.
        ᶠz_start = fill(FT(9000.0), flspace)
        ᶠz_ref = zeros(FT, flspace)
        ᶠp_ref = zeros(FT, flspace)
        ᶜmask = fill(false, cspace)
        ᶜweights = zeros(FT, cspace)
        ᶜdiff = zeros(FT, cspace)
        ᶜwtsum = zeros(FT, lspace)
        ᶠz = CA.Fields.coordinate_field(fspace).z
        ᶠdz = CA.Fields.local_geometry_field(fspace).∂x∂ξ.components.data.:1
        ᶠp = @. FT(1e5) * exp(-(ᶠz) / FT(8000))
        ᶠN = fill(FT(0.01), fspace)

        CA.calc_nonpropagating_forcing!(
            ᶜuforcing,
            ᶜvforcing,
            τ_x,
            τ_y,
            τ_l,
            τ_np,
            ᶠVτ,
            ᶠz_start,
            ᶠz_ref,
            ᶠp_ref,
            ᶜmask,
            ᶜweights,
            ᶜdiff,
            ᶜwtsum,
            ᶠp,
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
