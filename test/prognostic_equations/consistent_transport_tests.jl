#=
Consistency of the implicit vertical transport of energy, total water, and
condensates. With central `-∂ᵥ(ᶠρ u³ ᶠinterp(e_tot) + u³ ᶠinterp(p))`, `q_tot`,
and sedimenting condensates in the Newton residual, and one post-Newton
implicit first-order dissipation plus Courant-limited high-order correction on
`e_tot - Φ`, `q_tot`, and condensates at the solved velocity, the hydrostatic
pair `(p, Φ)` stays central alongside central continuity `-∂ᵥ(ᶠρ u³)` while
`e_tot - Φ = e_int + K` shares the `I - dtγ 𝒟` dissipation solve and preserves
affine relations at uniform `(T, u)`. Checks the scalar and central velocity
Jacobian blocks, uniform-temperature and stratified columns at `C_γ > 1`,
velocity-reversal stability under `ARS343`, and the `0M` and `DryModel`
`(e_tot - Φ, Φ, p)` split.
=#
using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Geometry, Operators, Spaces
import ClimaCore: MatrixFields
import ClimaTimeSteppers as CTS
import ClimaAtmos: TD

include("../test_helpers.jl")

function _measure_allocs(f!, args...)
    f!(args...)
    return @allocated f!(args...)
end

column_config(; dt = "10secs", kwargs...) = CA.AtmosConfig(
    Dict{String, Any}(
        "config" => "column",
        "FLOAT_TYPE" => "Float64",
        "dt" => dt,
        "z_max" => 6000.0,
        "z_elem" => 60,
        "z_stretch" => false,
        "initial_condition" => "PrecipitatingColumn",
        "microphysics_model" => "1M",
        "disable_surface_flux_tendency" => true,
        "output_default_diagnostics" => false,
        (string(key) => value for (key, value) in kwargs)...,
    );
    job_id = "consistent_transport_test",
)

@testset "Affine consistency of energy and water transport" begin
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    (; dtγ) = p
    (; ᶠu³, ᶜp) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜρJ = @. Y.c.ρ * ᶜJ
    ᶠρ = @. CA.ᶠinterp(ᶜρJ) / ᶠJ
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    n_levels = Spaces.nlevels(axes(Y.c))
    z_max = Spaces.z_max(axes(Y.f))
    @test p.atmos.numerics.energy_humidity_upwinding == Val(:vanleer_limiter)

    # Zero sedimentation velocities and pressure so the material transport
    # alone is under test.
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    for ρχ_name in (:ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y.c, ρχ_name)), FT(0))
    end
    fill!(parent(ᶜp), FT(0))
    A, B, C = FT(2.5e5), FT(2.4e6), FT(-3.3e5)
    u3_arr = parent(ᶠu³.components.data.:1)
    Yₜ = similar(Y)
    Yₜ_corr = similar(Y)
    Yₜ_wired = similar(Y)

    function test_affine_fields(ρ_f, ρe_f, ρq_f, ρl_f, label)
        ρe = Array(parent(ρe_f))
        affine =
            A .* Array(parent(ρ_f)) .+ B .* Array(parent(ρq_f)) .+
            C .* Array(parent(ρl_f))
        scale = maximum(abs, affine)
        @test scale > 0 || label == "correction"
        # Subtracting two `O(A)` energy fluxes (`A = 2.5e5`) in `upwind - base`
        # leaves `O(A * eps(FT)) ≈ 1e-11` roundoff when `ρ`'s correction is 0.
        @test maximum(abs, ρe .- affine) <= FT(1e-10) * max(scale, FT(1))
    end
    test_affine(Yₜ_k, label) = test_affine_fields(
        Yₜ_k.c.ρ,
        Yₜ_k.c.ρe_tot,
        Yₜ_k.c.ρq_tot,
        Yₜ_k.c.ρq_lcl,
        label,
    )

    # Linear reconstructions commute with affine combinations of independent
    # profiles; the van Leer limiter commutes when the profiles share one shape
    # `ψ(z)` with coefficients of one sign, as across a cloud-top jump.
    ψ_step = @. ifelse(z_c < z_max / 2, FT(1), FT(0))
    ψ_smooth = @. FT(0.5) * (1 + tanh((z_max / 2 - z_c) / FT(300)))
    cases = (
        (Val(:first_order), ψ_step, ψ_smooth),
        (Val(:third_order), ψ_step, ψ_smooth),
        (Val(:vanleer_limiter), ψ_step, ψ_step),
        (Val(:vanleer_limiter), ψ_smooth, ψ_smooth),
    )
    for (upwinding, ψ_q, ψ_l) in cases,
        C_val in (FT(0.1), FT(0.2), FT(0.3), FT(0.4), FT(0.5), FT(2.5))

        @. Y.c.ρq_tot = Y.c.ρ * (FT(2e-3) + FT(8e-3) * ψ_q)
        @. Y.c.ρq_lcl = Y.c.ρ * (FT(1e-4) + FT(9e-4) * ψ_l)
        # Keep physical `p.core.ᶜΦ = g z` nonzero so that `e_tot - Φ` is affine
        # in `(1, q_tot, q_lcl)` even in a gravitational field.
        @. Y.c.ρe_tot =
            Y.c.ρ * (A + p.core.ᶜΦ) + B * Y.c.ρq_tot + C * Y.c.ρq_lcl
        u3_arr .= C_val / FT(dtγ)
        ᶜe_mat = @. CA.specific(Y.c.ρe_tot, Y.c.ρ) - p.core.ᶜΦ
        ᶜq_tot = @. CA.specific(Y.c.ρq_tot, Y.c.ρ)
        ᶜq_lcl = @. CA.specific(Y.c.ρq_lcl, Y.c.ρ)

        Yₜ .= zero(eltype(Yₜ))
        CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        ᶜρe_mat_tend =
            Base.materialize(@. Yₜ.c.ρe_tot + CA.ᶜadvdivᵥ(ᶠρ * ᶠu³ * CA.ᶠinterp(p.core.ᶜΦ)))
        test_affine_fields(
            Yₜ.c.ρ,
            ᶜρe_mat_tend,
            Yₜ.c.ρq_tot,
            Yₜ.c.ρq_lcl,
            "residual",
        )

        # Post-Newton antidiffusive correction of each transported scalar with
        # the reconstruction under test; `ρ` and central `Φ` receive none.
        Yₜ_corr .= zero(eltype(Yₜ_corr))
        for (ᶜρχₜ, ᶜχ) in (
            (Yₜ_corr.c.ρe_tot, ᶜe_mat),
            (Yₜ_corr.c.ρq_tot, ᶜq_tot),
            (Yₜ_corr.c.ρq_lcl, ᶜq_lcl),
        )
            ᶠΔχ = @. CA._c3_scalar(CA.ᶠgradᵥ(ᶜχ))
            ᶠF_anti = CA.antidiffusive_upwind1_flux(
                ᶠρ, ᶠu³, ᶜχ, ᶠΔχ, ᶜρJ, ᶠJ, dtγ, upwinding,
            )
            @. ᶜρχₜ -= CA.ᶜadvdivᵥ(ᶠF_anti)
        end
        test_affine(Yₜ_corr, "correction")

        # The full post-Newton correction (implicit first-order dissipation
        # followed by the default `energy_humidity_upwinding` antidiffusion)
        # applies one linear operator and one reconstruction to all three
        # scalars.
        if upwinding == Val(:vanleer_limiter)
            Yₜ_wired .= zero(eltype(Yₜ_wired))
            CA.add_vertical_advection_corrections!(Yₜ_wired, Y, p, dtγ)
            test_affine(Yₜ_wired, "correction")
        end
    end

    # Composite stage positivity, monotonicity, and `b < 0` affine equivariance
    # across 0 < C_γ < 1/2 as well as C_γ ≥ 1/2, on a unit step, a one-cell
    # spike, and a ramp shoulder (where Zalesak FCT clips the unconstrained
    # van Leer shoulder overshoot at every C_γ > 0), plus a stretched-grid
    # (`z_stretch = true`) column with exponential density.
    ψ_spike = similar(z_c)
    fill!(parent(ψ_spike), FT(0))
    parent(ψ_spike)[div(n_levels, 2)] = FT(1)
    ψ_shoulder = similar(z_c)
    for k in 1:n_levels
        mid = div(n_levels, 2)
        parent(ψ_shoulder)[k] =
            k < mid - 2 ? FT(0) :
            k == mid - 2 ? FT(1 / 3) :
            k == mid - 1 ? FT(2 / 3) :
            k <= mid + 6 ? FT(1) : FT(0)
    end
    Y_stage = similar(Y)
    for ψ_prof in (ψ_step, ψ_spike, ψ_shoulder),
        C_val in (FT(0.1), FT(0.2), FT(0.3), FT(0.4), FT(0.5), FT(1.0), FT(2.5))

        @. Y.c.ρ = FT(1)
        @. Y.c.ρq_tot = Y.c.ρ * ψ_prof
        @. Y.c.ρq_lcl = FT(0)
        # Negative multiplier `-B` exercises `b < 0` affine equivariance of
        # Zalesak FCT (`R̃⁺ = R⁻`, `R̃⁻ = R⁺`).
        @. Y.c.ρe_tot = Y.c.ρ * (A + p.core.ᶜΦ) - B * Y.c.ρq_tot
        @. Y.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(
                ifelse(
                    (z_f > FT(0)) & (z_f < z_max),
                    C_val * (z_max / n_levels) / FT(dtγ),
                    FT(0),
                ),
            ),
        )
        CA.set_implicit_precomputed_quantities!(Y, p, t)
        fill!(parent(p.precomputed.ᶜp), FT(0))
        Yₜ .= zero(eltype(Yₜ))
        CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        ᶠρ_stage = @. CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
        ᶜΦ_adv = Base.materialize(
            @. CA.ᶜadvdivᵥ(ᶠρ_stage * p.precomputed.ᶠu³ * CA.ᶠinterp(p.core.ᶜΦ))
        )
        @. Y_stage = Y
        @. Y_stage.c.ρq_tot = Y.c.ρq_tot + dtγ * Yₜ.c.ρq_tot
        # `Y_stage.c.ρ` is held fixed at `1` to isolate scalar transport across
        # all `C_val` (including `C_val ≥ 1`), so remove the boundary mass-flux
        # divergence `A * Yₜ.c.ρ` alongside `ᶜΦ_adv` to keep `e_tot - Φ` at
        # `A - B * q_tot` entering `add_vertical_advection_corrections!`.
        @. Y_stage.c.ρe_tot =
            Y.c.ρe_tot + dtγ * (Yₜ.c.ρe_tot + ᶜΦ_adv - A * Yₜ.c.ρ)
        Yₜ_wired .= zero(eltype(Yₜ_wired))
        CA.add_vertical_advection_corrections!(Yₜ_wired, Y_stage, p, dtγ)
        q_end =
            vec(Array(parent(Y_stage.c.ρq_tot))) .+
            FT(dtγ) .* vec(Array(parent(Yₜ_wired.c.ρq_tot)))
        e_end =
            vec(Array(parent(Y_stage.c.ρe_tot))) .+
            FT(dtγ) .* vec(Array(parent(Yₜ_wired.c.ρe_tot))) .-
            vec(Array(parent(p.core.ᶜΦ)))
        interior = q_end[4:(n_levels - 4)]
        @test minimum(interior) >= -FT(1e-13)
        @test maximum(interior) <= FT(1) + FT(1e-13)
        @test e_end[4:(n_levels - 4)] ≈
              (A .- B .* interior) rtol = FT(1e-12)
    end

    # Decelerating vs diverging, face-jump, and boundary-spike velocity profiles
    # (with updated continuity density `ρ_new = ρ + dtγ Yₜ.c.ρ`): decelerating
    # flow stays non-negative, diverging/jump/boundary cases have bounded
    # `O(10⁻²)` extrema in the low-order stage `χ*`, and Zalesak FCT never
    # widens `[min(χ*), max(χ*)]`.
    dz_u = z_max / n_levels
    for (kind, w_fn, spike_idx, update_ρ, lo_tol, hi_tol) in (
        (
            :decelerating,
            zf -> FT(1.5) * dz_u / FT(dtγ) * (FT(1.5) - zf / z_max),
            div(n_levels, 2),
            true,
            -FT(1e-13),
            FT(1) + FT(1e-13),
        ),
        (
            :diverging,
            zf -> FT(5.0) * dz_u / FT(dtγ) * (FT(0.5) + FT(0.6) * zf / z_max),
            div(n_levels, 2),
            true,
            -FT(0.05),
            FT(1.05),
        ),
        (
            :face_jump_2x,
            zf -> (zf < z_max / 2 ? FT(0.4) : FT(0.8)) * dz_u / FT(dtγ),
            div(n_levels, 2),
            false,
            -FT(0.1),
            FT(1.1),
        ),
        (
            :boundary_spike,
            zf -> FT(0.8) * dz_u / FT(dtγ),
            1,
            false,
            -FT(0.05),
            FT(1.05),
        ),
    )
        @. Y.c.ρ = FT(1)
        fill!(parent(Y.c.ρq_tot), FT(0))
        parent(Y.c.ρq_tot)[spike_idx] = FT(1)
        @. Y.c.ρq_lcl = FT(0)
        @. Y.c.ρe_tot = Y.c.ρ * (A + p.core.ᶜΦ)
        @. Y.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(
                ifelse((z_f > FT(0)) & (z_f < z_max), w_fn(z_f), FT(0)),
            ),
        )
        CA.set_implicit_precomputed_quantities!(Y, p, t)
        fill!(parent(p.precomputed.ᶜp), FT(0))
        Yₜ .= zero(eltype(Yₜ))
        CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        @. Y_stage = Y
        if update_ρ
            @. Y_stage.c.ρ = Y.c.ρ + dtγ * Yₜ.c.ρ
        end
        @. Y_stage.c.ρq_tot = Y.c.ρq_tot + dtγ * Yₜ.c.ρq_tot
        Yₜ_wired .= zero(eltype(Yₜ_wired))
        CA.add_vertical_advection_corrections!(Yₜ_wired, Y_stage, p, dtγ)
        q_end_vl =
            (
                vec(Array(parent(Y_stage.c.ρq_tot))) .+
                FT(dtγ) .* vec(Array(parent(Yₜ_wired.c.ρq_tot)))
            ) ./ vec(Array(parent(Y_stage.c.ρ)))
        # Compute the low-order dissipated profile `χ*` using the assembled
        # `ᶜupwind_dissipation_system` (`ᶜA`) and verify FCT preserves its
        # global bounds `[min(χ*), max(χ*)]` across `1:n_levels`.
        ᶜρχ_star = similar(Y_stage.c.ρq_tot)
        MatrixFields.single_field_solve!(
            p.scratch.ᶜupwind_dissipation_solver_cache,
            ᶜρχ_star,
            p.scratch.ᶜupwind_dissipation_system,
            Y_stage.c.ρq_tot,
        )
        q_star = vec(Array(parent(ᶜρχ_star))) ./ vec(Array(parent(Y_stage.c.ρ)))
        @test minimum(q_end_vl) >= minimum(q_star) - FT(1e-13)
        @test maximum(q_end_vl) <= maximum(q_star) + FT(1e-13)
        @test minimum(q_end_vl) >= lo_tol
        @test maximum(q_end_vl) <= hi_tol
        if kind === :decelerating
            @test minimum(q_end_vl[4:(n_levels - 4)]) >= -FT(1e-13)
        end
    end

    stretch_cfg = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "column",
            "FLOAT_TYPE" => "Float64",
            "z_stretch" => true,
            "z_max" => 15e3,
            "z_elem" => 24,
            "dz_bottom" => 100.0,
            "dt" => "10secs",
            "initial_condition" => "PrecipitatingColumn",
            "microphysics_model" => "1M",
            "implicit_microphysics" => false,
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "consistent_transport_stretch_fct_test",
    )
    nt_s = generate_test_simulation(stretch_cfg)
    Y_s, p_s = nt_s.Y, nt_s.p
    n_s = Spaces.nlevels(axes(Y_s.c))
    ᶜJ_s = Fields.local_geometry_field(Y_s.c).J
    ᶠJ_s = Fields.local_geometry_field(Y_s.f).J
    ᶠρ_s = Base.materialize(@. CA.ᶠinterp(Y_s.c.ρ * ᶜJ_s) / ᶠJ_s)
    z_fs = Fields.coordinate_field(Y_s.f).z
    z_max_s = Spaces.z_max(axes(Y_s.f))
    dtγ_s = FT(p_s.dtγ)
    m_ref = Array(parent(Y_s.c.ρ))[10] * Array(parent(ᶜJ_s))[10]
    Yₜ_s = similar(Y_s)
    Y_s_stage = similar(Y_s)
    Yₜ_s_corr = similar(Y_s)
    for ρχ_name in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y_s.c, ρχ_name)), FT(0))
    end
    for C_val in (FT(0.2), FT(0.5), FT(1.5))
        M0 = C_val * m_ref / dtγ_s
        # Nondivergent interior vertical mass flux `ᶠJ * ᶠρ * u³ = ᶠρ * w = M0`
        # on a stretched grid with exponential density.
        @. Y_s.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(
                ifelse(
                    (z_fs > FT(0)) & (z_fs < z_max_s),
                    M0 / ᶠρ_s,
                    FT(0),
                ),
            ),
        )
        for k in 1:n_s
            q_k =
                k < 8 ? FT(0) :
                k == 8 ? FT(1 / 3) :
                k == 9 ? FT(2 / 3) :
                k <= 16 ? FT(1) : FT(0)
            parent(Y_s.c.ρq_tot)[k] = parent(Y_s.c.ρ)[k] * q_k
        end
        CA.set_implicit_precomputed_quantities!(Y_s, p_s, FT(0))
        for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
            fill!(parent(getproperty(p_s.precomputed, w_name)), FT(0))
        end
        Yₜ_s .= zero(eltype(Yₜ_s))
        CA.implicit_vertical_advection_tendency!(Yₜ_s, Y_s, p_s, FT(0))
        @. Y_s_stage = Y_s
        @. Y_s_stage.c.ρq_tot = Y_s.c.ρq_tot + dtγ_s * Yₜ_s.c.ρq_tot
        Yₜ_s_corr .= zero(eltype(Yₜ_s_corr))
        CA.add_vertical_advection_corrections!(Yₜ_s_corr, Y_s_stage, p_s, dtγ_s)
        q_s_end =
            (
                vec(Array(parent(Y_s_stage.c.ρq_tot))) .+
                dtγ_s .* vec(Array(parent(Yₜ_s_corr.c.ρq_tot)))
            ) ./ vec(Array(parent(Y_s.c.ρ)))
        q_s_int = q_s_end[4:(n_s - 4)]
        @test minimum(q_s_int) >= -FT(1e-12)
        @test maximum(q_s_int) <= FT(1) + FT(1e-12)
    end

    # Terrain-following `u₃` and `uₕ` Jacobian actions preserve the same affine
    # relation at `ᶜp = 0`.
    topo_cfg = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "plane",
            "z_stretch" => false,
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "microphysics_model" => "1M",
            "topography" => "Schar",
            "mesh_warp_type" => "Linear",
            "FLOAT_TYPE" => "Float64",
            "x_max" => 100e3,
            "x_elem" => 4,
            "z_max" => 10e3,
            "z_elem" => 10,
            "dt" => "2secs",
            "implicit_microphysics" => false,
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "consistent_transport_topo_test",
    )
    nt_t = generate_test_simulation(topo_cfg)
    Y_t, p_t, sim_t = nt_t.Y, nt_t.p, nt_t.simulation
    mat_t = sim_t.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    z_ct = Fields.coordinate_field(Y_t.c).z
    z_ft = Fields.coordinate_field(Y_t.f).z
    x_ct = Fields.coordinate_field(Y_t.c).x
    @. Y_t.c.ρq_tot = Y_t.c.ρ * (FT(3e-3) + FT(5e-3) * exp(-z_ct / FT(2500)))
    @. Y_t.c.ρq_lcl = Y_t.c.ρ * FT(8e-4) * exp(-(z_ct - FT(2000))^2 / FT(1e6))
    for ρχ_name in (:ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y_t.c, ρχ_name)), FT(0))
    end
    @. Y_t.c.ρe_tot = Y_t.c.ρ * A + B * Y_t.c.ρq_tot + C * Y_t.c.ρq_lcl
    fill!(parent(p_t.precomputed.ᶜp), FT(0))
    dtγ_t = FT(p_t.dtγ)
    CA.update_advection_jacobian!(mat_t, Y_t, p_t, dtγ_t, CA.UseDerivative())
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        mat_t,
        Y_t,
        p_t,
        dtγ_t,
        CA.UseDerivative(),
    ) == 0

    ᶠδu₃ = @. Geometry.Covariant3Vector(
        Geometry.WVector(sin(FT(π) * z_ft / FT(10e3))),
    )
    ᶜδuₕ = @. Geometry.Covariant12Vector(
        Geometry.UVWVector(
            FT(3) * cos(FT(2π) * x_ct / FT(100e3)) * sin(z_ct / FT(2000)),
            FT(0),
            FT(0),
        ),
    )
    jact(row, col, δv) =
        Base.materialize(@. $(mat_t[row, col]) * δv)
    test_affine_fields(
        jact(MatrixFields.@name(c.ρ), MatrixFields.@name(f.u₃), ᶠδu₃),
        jact(MatrixFields.@name(c.ρe_tot), MatrixFields.@name(f.u₃), ᶠδu₃),
        jact(MatrixFields.@name(c.ρq_tot), MatrixFields.@name(f.u₃), ᶠδu₃),
        jact(MatrixFields.@name(c.ρq_lcl), MatrixFields.@name(f.u₃), ᶠδu₃),
        "J_u3",
    )
    test_affine_fields(
        jact(MatrixFields.@name(c.ρ), MatrixFields.@name(c.uₕ), ᶜδuₕ),
        jact(MatrixFields.@name(c.ρe_tot), MatrixFields.@name(c.uₕ), ᶜδuₕ),
        jact(MatrixFields.@name(c.ρq_tot), MatrixFields.@name(c.uₕ), ᶜδuₕ),
        jact(MatrixFields.@name(c.ρq_lcl), MatrixFields.@name(c.uₕ), ᶜδuₕ),
        "J_uh",
    )
end

@testset "Jacobian of the scalar and velocity transport blocks" begin
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    dtγ = FT(60)
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    z_max = Spaces.z_max(axes(Y.f))
    matrix = simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix

    # Condensates stay above the finite-difference perturbation everywhere, so
    # the `max(0, q_liq)` of the thermodynamic state is inactive.
    @. Y.c.ρq_lcl =
        Y.c.ρ * (FT(1e-5) + FT(1e-3) * exp(-(z_c - FT(2000))^2 / FT(1e6)))
    @. Y.c.ρq_rai =
        Y.c.ρ * (FT(1e-5) + FT(2e-3) * exp(-(z_c - FT(1500))^2 / FT(1e6)))
    @. Y.c.ρq_tot =
        Y.c.ρ * FT(8e-3) * (1 + FT(0.3) * sin(FT(2π) * z_c / z_max)) +
        Y.c.ρq_lcl + Y.c.ρq_rai
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(FT(2.5) * sin(FT(π) * z_f / z_max)),
    )
    CA.set_precomputed_quantities!(Y, p, t)
    # Zero sedimentation velocities so the blocks hold grid-mean advection.
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    @test _measure_allocs(CA.update_sedimentation_jacobian!, matrix, Y, p, dtγ) ==
          0

    arr(f) = Array(parent(f))
    ᶜv = @. sin(z_c / FT(700)) + FT(0.5)
    # The grid-mean advection of every scalar is central and advanced
    # explicitly by the Newton step, so with zero sedimentation velocities the
    # scalar diagonals are `-I` and the `(ρe_tot, ρq_tot)` coupling is zero.
    for (row_sym, col_sym) in (
        (:ρq_tot, :ρq_tot),
        (:ρe_tot, :ρe_tot),
        (:ρq_lcl, :ρq_lcl),
        (:ρq_rai, :ρq_rai),
    )
        block = matrix[
            MatrixFields.FieldName(:c, row_sym),
            MatrixFields.FieldName(:c, col_sym),
        ]
        @test arr(@. block * ᶜv) ≈ -arr(ᶜv)
    end
    @test maximum(
        abs,
        arr(
            @. $(matrix[MatrixFields.@name(c.ρe_tot), MatrixFields.@name(c.ρq_tot)]) *
               ᶜv
        ),
    ) == 0

    # Velocity blocks: each `(·, u₃)` block is the exact derivative of the
    # central-face flux `-ᶜadvdivᵥ((ᶠρ ᶠinterp(e_tot) + ᶠinterp(p)) ᶠu³)` or
    # `-ᶜadvdivᵥ(ᶠρ ᶠinterp(χ) ᶠu³)` with `e_tot`, `p`, and `χ` held fixed
    # (omitting `K(u₃)` dependence).
    CA.update_advection_jacobian!(matrix, Y, p, dtγ, CA.IgnoreDerivative())
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        matrix,
        Y,
        p,
        dtγ,
        CA.IgnoreDerivative(),
    ) == 0
    # Zero boundary perturbations match the impenetrable boundary condition.
    ᶠv = @. Geometry.Covariant3Vector(
        Geometry.WVector(sin(FT(π) * z_f / z_max)),
    )
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶠρ_fixed = Base.materialize(@. CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ)
    ᶜp_fixed = copy(p.precomputed.ᶜp)
    ᶜe_fixed = Base.materialize(@. CA.specific(Y.c.ρe_tot, Y.c.ρ))
    u₃_orig = copy(Y.f.u₃)
    δ_u₃ = FT(1e-3)
    ᶜflux_div = similar(Y.c.ρ)
    function fd_central_u₃_derivative(row_sym)
        tendencies = map((+1, -1)) do sgn
            ᶠu³_p = @. CA.CT3(u₃_orig + sgn * δ_u₃ * ᶠv)
            if row_sym == :ρ
                @. ᶜflux_div = -(CA.ᶜadvdivᵥ(ᶠρ_fixed * ᶠu³_p))
            elseif row_sym == :ρe_tot
                @. ᶜflux_div = -(CA.ᶜadvdivᵥ(
                    (ᶠρ_fixed * CA.ᶠinterp(ᶜe_fixed) + CA.ᶠinterp(ᶜp_fixed)) *
                    ᶠu³_p,
                ))
            else
                ᶜχ_f = @. CA.specific($(getproperty(Y.c, row_sym)), Y.c.ρ)
                @. ᶜflux_div =
                    -(CA.ᶜadvdivᵥ(ᶠρ_fixed * CA.ᶠinterp(ᶜχ_f) * ᶠu³_p))
            end
            copy(arr(ᶜflux_div))
        end
        return dtγ .* (tendencies[1] .- tendencies[2]) ./ (2 * δ_u₃)
    end
    for row_sym in (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl, :ρq_rai)
        block = matrix[
            MatrixFields.FieldName(:c, row_sym),
            MatrixFields.FieldName(:f, :u₃),
        ]
        Jv = arr(@. block * ᶠv)
        fd = fd_central_u₃_derivative(row_sym)
        @test norm(Jv .- fd) / norm(fd) < FT(1e-10)
    end
end

@testset "Temperature of moist columns transported at C_γ > 1" begin
    # 1. Hydrostatically balanced uniform-temperature column with a sharp step
    # in total water and cloud liquid under a uniform updraft at `C_γ = 2.5`.
    # With pressure work `u³ ᶠinterp(p)` kept central (avoiding the
    # `ρ Δ(R_m T) / 2` dipole of `ᶠρ (p / ρ)_upw`) and `Φ` kept central on the
    # same stencil, `e_tot - Φ = e_int + K`, `q_tot`, and `q_lcl` share one
    # affine profile shape `ψ` across the step.
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    thp = CA.CAP.thermodynamics_params(p.params)
    g = CA.CAP.grav(p.params)
    cv_d = CA.CAP.cv_d(p.params)
    R_d = CA.CAP.R_d(p.params)
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    n_levels = Spaces.nlevels(axes(Y.c))
    z_max = Spaces.z_max(axes(Y.f))
    Δz = z_max / n_levels
    dtγ = FT(p.dtγ)
    C_γ, C_euler = FT(2.5), FT(0.2)
    w = C_γ * Δz / dtγ
    dt_euler = C_euler * Δz / w
    T_0 = FT(285)
    (; ᶜΦ) = p.core
    ψ = @. ifelse(z_c < FT(1500), FT(1), FT(0))
    ᶜq_tot = @. FT(1e-3) + FT(6e-3) * ψ
    ᶜq_lcl = @. FT(5e-4) * ψ
    CA.set_precomputed_quantities!(Y, p, t)
    p_step = vec(parent(p.precomputed.ᶜp))
    qt_step = vec(parent(ᶜq_tot))
    ql_step = vec(parent(ᶜq_lcl))
    z_arr = vec(parent(z_c))
    R_m_step(k) = TD.gas_constant_air(thp, qt_step[k], ql_step[k], FT(0))
    p_step[1] = FT(1e5) * exp(-g * z_arr[1] / (R_m_step(1) * T_0))
    for k in 2:n_levels
        a = g * Δz / (2 * R_m_step(k - 1) * T_0)
        b = g * Δz / (2 * R_m_step(k) * T_0)
        p_step[k] = p_step[k - 1] * (1 - a) / (1 + b)
    end
    @. Y.c.ρ =
        p.precomputed.ᶜp /
        (TD.gas_constant_air(thp, ᶜq_tot, ᶜq_lcl, FT(0)) * T_0)
    @. Y.c.ρq_tot = Y.c.ρ * ᶜq_tot
    @. Y.c.ρq_lcl = Y.c.ρ * ᶜq_lcl
    for ρχ_name in (:ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y.c, ρχ_name)), FT(0))
    end
    @. Y.f.u₃ = Geometry.Covariant3Vector(Geometry.WVector(w + 0 * z_f))
    ᶜK = CA.compute_kinetic(Y.c.uₕ, Y.f.u₃)
    @. Y.c.ρe_tot =
        Y.c.ρ * (TD.internal_energy(thp, T_0, ᶜq_tot, ᶜq_lcl, FT(0)) + ᶜK + ᶜΦ)
    CA.set_precomputed_quantities!(Y, p, t)
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    interior = 4:(n_levels - 3)
    spread(f) = (x = vec(parent(f))[interior]; maximum(x) - minimum(x))
    spread_before = spread(p.precomputed.ᶜT)

    Yₜ = similar(Y)
    Yₜ_corr = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)
    @. Yₜ += Yₜ_corr
    for name in (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl)
        ᶜχ = getproperty(Y.c, name)
        @. ᶜχ += dt_euler * $(getproperty(Yₜ.c, name))
    end
    CA.set_implicit_precomputed_quantities!(Y, p, t)
    spread_after = spread(p.precomputed.ᶜT)
    q_vap = @. CA.specific(Y.c.ρq_tot - Y.c.ρq_lcl, Y.c.ρ)
    @test spread_before < 1e-6
    @test spread_after < spread_before + FT(1e-4)
    @test minimum(vec(parent(q_vap))[interior]) >= 0

    # 2. Smooth hydrostatically stratified moist column (`Γ = 6.5 K/km`,
    # `q_tot = 12 g/kg exp(-z / 2 km)`) with `w(z) = w_max sin(π z / z_max)`
    # peaking at `C_γ = 2.5`. At `C_γ > 1`, the implicit first-order smoothing
    # (weighted by `1 + 2 C_γ`, of which the limited antidiffusion removes
    # `1 / C_γ`) acts on `∂_z(e_tot - Φ) = ∂_z(e_int + K)`. Measured against the
    # central `h_tot` / `q_tot` transport on this 60-level 6 km column at
    # `C_euler = 0.2`, `max |T - T_central| = 2.4e-2 K` (analytic scale
    # `(1 + 2 C_γ - 1 / C_γ) C_euler R_d Γ Δz / (2 cv_d) = 0.146 K`); pinned at
    # `0.03 K`.
    Γ = FT(6.5e-3)
    @. ᶜq_tot = FT(0.012) * exp(-z_c / FT(2000))
    fill!(parent(ᶜq_lcl), FT(0))
    ᶜT_init = @. FT(290) - Γ * z_c
    p_arr = vec(parent(p.precomputed.ᶜp))
    q_arr = vec(parent(ᶜq_tot))
    T_arr = vec(parent(ᶜT_init))
    R_m1 = TD.gas_constant_air(thp, q_arr[1], FT(0), FT(0))
    p_arr[1] = FT(1e5) * exp(-g * z_arr[1] / (R_m1 * T_arr[1]))
    for k in 2:n_levels
        R_mk = TD.gas_constant_air(
            thp,
            FT(0.5) * (q_arr[k - 1] + q_arr[k]),
            FT(0),
            FT(0),
        )
        H_k = R_mk * FT(0.5) * (T_arr[k - 1] + T_arr[k]) / g
        p_arr[k] = p_arr[k - 1] * exp(-Δz / H_k)
    end
    @. Y.c.ρ =
        p.precomputed.ᶜp /
        (TD.gas_constant_air(thp, ᶜq_tot, FT(0), FT(0)) * ᶜT_init)
    @. Y.c.ρq_tot = Y.c.ρ * ᶜq_tot
    fill!(parent(Y.c.ρq_lcl), FT(0))
    @. Y.f.u₃ =
        Geometry.Covariant3Vector(Geometry.WVector(w * sin(FT(π) * z_f / z_max)))
    ᶜK = CA.compute_kinetic(Y.c.uₕ, Y.f.u₃)
    @. Y.c.ρe_tot =
        Y.c.ρ * (TD.internal_energy(thp, ᶜT_init, ᶜq_tot, FT(0), FT(0)) + ᶜK + ᶜΦ)
    CA.set_precomputed_quantities!(Y, p, t)
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end

    Y_ref = copy(Y)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    (; ᶠu³, ᶜh_tot) = p.precomputed
    ᶠρ_s = @. CA.ᶠinterp(Y_ref.c.ρ * ᶜJ) / ᶠJ
    @. Y_ref.c.ρ -= dt_euler * CA.ᶜadvdivᵥ(ᶠρ_s * ᶠu³)
    @. Y_ref.c.ρq_tot -= dt_euler * CA.ᶜadvdivᵥ(ᶠρ_s * ᶠu³ * CA.ᶠinterp(ᶜq_tot))
    @. Y_ref.c.ρe_tot -= dt_euler * CA.ᶜadvdivᵥ(ᶠρ_s * ᶠu³ * CA.ᶠinterp(ᶜh_tot))
    CA.set_implicit_precomputed_quantities!(Y_ref, p, t)
    T_central = copy(vec(parent(p.precomputed.ᶜT)))

    CA.set_precomputed_quantities!(Y, p, t)
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    Yₜ .= zero(eltype(Yₜ))
    CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)
    @. Yₜ += Yₜ_corr
    for name in (:ρ, :ρe_tot, :ρq_tot)
        ᶜχ = getproperty(Y.c, name)
        @. ᶜχ += dt_euler * $(getproperty(Yₜ.c, name))
    end
    CA.set_implicit_precomputed_quantities!(Y, p, t)
    T_split = vec(parent(p.precomputed.ᶜT))
    strat_err = maximum(abs, T_split[5:(n_levels - 4)] .- T_central[5:(n_levels - 4)])
    @test strat_err < FT(0.03)
    @test strat_err < (1 + 2 * C_γ - 1 / C_γ) * C_euler * R_d * Γ * Δz / (2 * cv_d)
end

@testset "Velocity-reversal stability under ARS343" begin
    # Thirty single-Newton `ARS343` steps on a `[0, 1]` cloud-liquid layer with
    # `ClimaAtmos`'s `(ρq_lcl, ρq_lcl)` and `(ρq_lcl, u₃)` Jacobian blocks,
    # `implicit_vertical_advection_tendency!`, and
    # `correct_implicit_advection_tendency!` at the solved velocity, while the
    # solved stage velocity reverses sign between `+A Δz / dt` and `-A Δz / dt`
    # on alternating steps:
    # (1) `stage_overshoot = true`: each implicit stage enters at
    #     `w_lin = -dir * 1e-6 * Δz / dt` across `u₃ = 0` from `w_star` (as when
    #     an explicit predictor overshoots zero before the acoustic solve). On
    #     this 60-level column setup, donor `sign(u³)` `(ρq_lcl, u₃)` blocks
    #     reach `[-2.04, 3.08]` at `A = 0.05` and `[-7.59, 8.85]` at `A = 0.08`,
    #     whereas central `ᶠinterp` velocity blocks stay within `[-0.06, 1.06]`.
    # (2) `stage_overshoot = false`: `w_lin` advances through `a_imp` across
    #     large-amplitude step reversals `A ∈ (0.3, 1.0)`.
    (; Y, p, simulation) = generate_test_simulation(column_config())
    t = simulation.integrator.t
    FT = eltype(Y)
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    z_max = Spaces.z_max(axes(Y.f))
    n_levels = Spaces.nlevels(axes(Y.c))
    Δz = z_max / n_levels
    dt = FT(simulation.integrator.dt)
    tab = simulation.integrator.alg.tableau
    a_imp = Matrix(tab.a_imp.coeffs)
    b_imp = Vector(tab.b_imp.coeffs)
    γ = a_imp[2, 2]
    dtγ = γ * dt
    matrix = simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix

    CA.set_precomputed_quantities!(Y, p, t)
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    for ρχ_name in (:ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y.c, ρχ_name)), FT(0))
    end
    q0 = FT(1e-3)
    ρ_fixed = copy(Y.c.ρ)
    Yₜ = similar(Y)

    function solve_tridiag(block, rhs)
        n = length(rhs)
        dl = [Fields.level(block, k)[].entries[1] for k in 2:n]
        d = [Fields.level(block, k)[].entries[2] for k in 1:n]
        du = [Fields.level(block, k)[].entries[3] for k in 1:(n - 1)]
        return Tridiagonal(dl, d, du) \ rhs
    end

    function run_reversal(A, stage_overshoot)
        @. Y.c.ρ = ρ_fixed
        @. Y.c.ρq_lcl =
            ρ_fixed * ifelse(FT(2000) <= z_c <= FT(4000), q0, FT(0))
        w_curr = -FT(1e-6) * Δz / dt
        T_imp = [zeros(FT, n_levels) for _ in 1:4]
        T_imp_w = zeros(FT, 4)

        for step in 1:30
            u_init = copy(vec(Array(parent(Y.c.ρq_lcl))))
            w_init = w_curr
            dir = isodd(step) ? FT(1) : FT(-1)
            w_star = dir * A * Δz / dt
            fill!(T_imp_w, FT(0))

            for s in 2:4
                temp = copy(u_init)
                w_lin = stage_overshoot ? (-dir * FT(1e-6) * Δz / dt) : w_init
                for j in 1:(s - 1)
                    if a_imp[s, j] != 0
                        temp .+= dt * a_imp[s, j] .* T_imp[j]
                        if !stage_overshoot
                            w_lin += dt * a_imp[s, j] * T_imp_w[j]
                        end
                    end
                end
                vec(parent(Y.c.ρq_lcl)) .= temp
                @. Y.f.u₃ = Geometry.Covariant3Vector(
                    Geometry.WVector(w_lin * sin(FT(π) * z_f / z_max)),
                )
                CA.set_implicit_precomputed_quantities!(Y, p, t)
                for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
                    fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
                end
                fill!(Yₜ, 0)
                CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
                R0 = dtγ .* vec(Array(parent(Yₜ.c.ρq_lcl)))

                CA.update_advection_jacobian!(
                    matrix, Y, p, dtγ, CA.IgnoreDerivative(),
                )
                CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
                J_qq = matrix[
                    MatrixFields.@name(c.ρq_lcl),
                    MatrixFields.@name(c.ρq_lcl),
                ]
                J_qu = matrix[
                    MatrixFields.@name(c.ρq_lcl),
                    MatrixFields.@name(f.u₃),
                ]
                ᶠδu₃ = @. Geometry.Covariant3Vector(
                    Geometry.WVector(
                        (w_star - w_lin) * sin(FT(π) * z_f / z_max),
                    ),
                )
                qu_δu = vec(Array(parent(@. J_qu * ᶠδu₃)))
                δρq = solve_tridiag(J_qq, -(R0 .+ qu_δu))
                vec(parent(Y.c.ρq_lcl)) .= temp .+ δρq

                @. Y.f.u₃ = Geometry.Covariant3Vector(
                    Geometry.WVector(w_star * sin(FT(π) * z_f / z_max)),
                )
                CA.set_implicit_precomputed_quantities!(Y, p, t)
                CA.correct_implicit_advection_tendency!(Yₜ, Y, p, t)
                vec(parent(Y.c.ρq_lcl)) .+=
                    dtγ .* vec(Array(parent(Yₜ.c.ρq_lcl)))

                T_imp[s] .= (vec(Array(parent(Y.c.ρq_lcl))) .- temp) ./ dtγ
                T_imp_w[s] = (w_star - w_lin) / dtγ
            end

            u_next = copy(u_init)
            w_next = w_init
            for j in 1:4
                if b_imp[j] != 0
                    u_next .+= dt * b_imp[j] .* T_imp[j]
                    w_next += dt * b_imp[j] * T_imp_w[j]
                end
            end
            vec(parent(Y.c.ρq_lcl)) .= u_next
            w_curr = w_next
        end

        return vec(Array(parent(Y.c.ρq_lcl))) ./ (vec(Array(parent(ρ_fixed))) .* q0)
    end

    # `ARS343` combines stage tendencies with `b_imp ≈ (0, 1.208, -0.644,
    # 0.436)`, so its negative stage-3 weight allows bounded O(0.1) step-level
    # undershoots across time-varying velocity reversals even when every
    # individual stage update is positive and monotone.
    for A in (FT(0.05), FT(0.08))
        q_norm = run_reversal(A, true)
        @test minimum(q_norm) > FT(-0.10)
        @test maximum(q_norm) < FT(1.10)
    end
    for A in (FT(0.3), FT(1.0))
        q_norm = run_reversal(A, false)
        @test minimum(q_norm) > FT(-0.15)
        @test maximum(q_norm) < FT(1.10)
    end
end

@testset "Equilibrium 0M unified moist transport and C_γ > 1 stability" begin
    (; Y, p, simulation) =
        generate_test_simulation(column_config(; microphysics_model = "0M"))
    t = simulation.integrator.t
    FT = eltype(Y)
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    z_max = Spaces.z_max(axes(Y.f))
    n_levels = Spaces.nlevels(axes(Y.c))
    Δz = z_max / n_levels
    dt = FT(simulation.integrator.dt)
    tab = simulation.integrator.alg.tableau
    a_imp = Matrix(tab.a_imp.coeffs)
    b_imp = Vector(tab.b_imp.coeffs)
    dtγ = FT(a_imp[2, 2]) * dt
    matrix = simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix

    @. Y.c.ρq_tot =
        Y.c.ρ * FT(6e-3) * (1 + FT(0.3) * sin(FT(2π) * z_c / z_max))
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(FT(2.0) * sin(FT(π) * z_f / z_max)),
    )
    CA.set_precomputed_quantities!(Y, p, t)
    CA.update_advection_jacobian!(matrix, Y, p, dtγ, CA.IgnoreDerivative())
    CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    @test _measure_allocs(
        CA.update_advection_jacobian!, matrix, Y, p, dtγ, CA.IgnoreDerivative(),
    ) == 0
    @test _measure_allocs(CA.update_sedimentation_jacobian!, matrix, Y, p, dtγ) ==
          0

    # Eight single-Newton `ARS343` steps on `q_tot` in nondivergent interior
    # flow at `C_γ = 1.5` (`C ≈ 3.44`): the forward-Euler central step of the
    # Newton solve (`-I` scalar diagonal) followed by the post-Newton implicit
    # dissipation and limited antidiffusion. On this 60-level column setup, the
    # central step alone with an explicit Courant-limited upwind correction
    # amplifies `q_norm` to `[-3.8e5, 3.4e5]`.
    w_const = FT(1.5) * Δz / dtγ
    @. Y.c.ρ = FT(1)
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(ifelse((z_f > FT(0)) & (z_f < z_max), w_const, FT(0))),
    )
    q0 = FT(4e-3)
    @. Y.c.ρq_tot = ifelse(FT(600) <= z_c <= FT(1800), q0, FT(0))
    q_sum_init = sum(vec(Array(parent(Y.c.ρq_tot))))
    CA.set_precomputed_quantities!(Y, p, t)
    CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    J_qq = matrix[MatrixFields.@name(c.ρq_tot), MatrixFields.@name(c.ρq_tot)]
    dl = [Fields.level(J_qq, k)[].entries[1] for k in 2:n_levels]
    d = [Fields.level(J_qq, k)[].entries[2] for k in 1:n_levels]
    du = [Fields.level(J_qq, k)[].entries[3] for k in 1:(n_levels - 1)]
    J_tridiag = Tridiagonal(dl, d, du)
    Yₜ = similar(Y)
    T_imp = [zeros(FT, n_levels) for _ in 1:4]

    for _ in 1:8
        u_init = copy(vec(Array(parent(Y.c.ρq_tot))))
        for s in 2:4
            temp = copy(u_init)
            for j in 1:(s - 1)
                a_imp[s, j] != 0 && (temp .+= dt * a_imp[s, j] .* T_imp[j])
            end
            vec(parent(Y.c.ρq_tot)) .= temp
            fill!(Yₜ, 0)
            CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
            R0 = dtγ .* vec(Array(parent(Yₜ.c.ρq_tot)))
            vec(parent(Y.c.ρq_tot)) .= temp .+ (J_tridiag \ (-R0))
            CA.correct_implicit_advection_tendency!(Yₜ, Y, p, t)
            vec(parent(Y.c.ρq_tot)) .+= dtγ .* vec(Array(parent(Yₜ.c.ρq_tot)))
            T_imp[s] .= (vec(Array(parent(Y.c.ρq_tot))) .- temp) ./ dtγ
        end
        u_next = copy(u_init)
        for j in 1:4
            b_imp[j] != 0 && (u_next .+= dt * b_imp[j] .* T_imp[j])
        end
        vec(parent(Y.c.ρq_tot)) .= u_next
    end
    q_vec = vec(Array(parent(Y.c.ρq_tot)))
    q_norm = q_vec ./ q0
    @test sum(q_vec) ≈ q_sum_init rtol = 1e-12
    # Each stage update stays in [0, 1]; the negative `b_imp[3] ≈ -0.644`
    # weight of `ARS343` produces a bounded step-level undershoot at C ≈ 3.44.
    @test minimum(q_norm) > FT(-0.30)
    @test maximum(q_norm) < FT(1.10)
end

@testset "DryModel unified geopotential-pressure split" begin
    dry_cfg(; kwargs...) = column_config(;
        initial_condition = "DecayingProfile",
        microphysics_model = "dry",
        kwargs...,
    )
    (; Y, p, simulation) = generate_test_simulation(dry_cfg())
    t = simulation.integrator.t
    FT = eltype(Y)
    z_c = Fields.coordinate_field(Y.c).z
    z_f = Fields.coordinate_field(Y.f).z
    z_max = Spaces.z_max(axes(Y.f))
    dt = FT(simulation.integrator.dt)
    dtγ = FT(p.dtγ)
    (; ᶜΦ) = p.core
    (; ᶠu³, ᶜp) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J

    @test !MatrixFields.has_field(Y, MatrixFields.@name(c.ρq_tot))
    @test simulation.integrator.sol.prob.f.T_post_imp! !== nothing

    # With `energy_humidity_upwinding = "none"`, the post-Newton correction adds
    # nothing and the central transport of the residual stands.
    nt_none = generate_test_simulation(dry_cfg(; energy_humidity_upwinding = "none"))
    Yₜ_none = similar(nt_none.Y)
    fill!(Yₜ_none, 1)
    CA.correct_implicit_advection_tendency!(
        Yₜ_none, nt_none.Y, nt_none.p, nt_none.simulation.integrator.t,
    )
    @test maximum(abs, Array(parent(Yₜ_none.c))) == 0

    # Populate sub-unity stage Courant velocity (`|u³| * dtγ ≤ 0.25 < 1`) and a
    # nonlinear temperature perturbation so both implicit and post-Newton fluxes
    # are nontrivial.
    thp = CA.CAP.thermodynamics_params(p.params)
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(
            (FT(0.25) / dtγ) * (z_max / FT(60)) * sin(FT(π) * z_f / z_max),
        ),
    )
    ᶜK = CA.compute_kinetic(Y.c.uₕ, Y.f.u₃)
    ᶜT_nl = @. FT(285) + FT(6) * exp(-(z_c - FT(2500))^2 / FT(8e5))
    @. Y.c.ρe_tot = Y.c.ρ * (TD.internal_energy_dry(thp, ᶜT_nl) + ᶜK + ᶜΦ)
    CA.set_precomputed_quantities!(Y, p, t)

    ᶜe_tot = Base.materialize(@. CA.specific(Y.c.ρe_tot, Y.c.ρ))
    ᶜe_mat = Base.materialize(CA.energy_transport_scalar(Y, p))
    @test Array(parent(ᶜe_mat)) ≈ Array(parent(@. ᶜe_tot - ᶜΦ))

    # 1. Exact residual match of `implicit_vertical_advection_tendency!` to the
    # split central flux `-∂ᵥ(ᶠρ u³ ᶠinterp(e_tot) + u³ ᶠinterp(p))`.
    Yₜ = similar(Y)
    fill!(Yₜ, 0)
    CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    ᶠρ = @. CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    ᶜρe_expected = Base.materialize(
        @. -(CA.ᶜadvdivᵥ(ᶠρ * ᶠu³ * CA.ᶠinterp(ᶜe_tot) + ᶠu³ * CA.ᶠinterp(ᶜp)))
    )
    @test Array(parent(Yₜ.c.ρe_tot)) ≈ Array(parent(ᶜρe_expected)) rtol = 1e-13

    # 2. With `dtγ = 0`, the post-Newton correction takes the central flux of
    # `e_tot - Φ` to `vanleer_limiter`, leaving mass and momentum untouched;
    # with the stage step, it is the implicit first-order dissipation plus the
    # limited antidiffusion, which conserves the column energy.
    Yₜ_corr = similar(Y)
    fill!(Yₜ_corr, 0)
    CA.add_vertical_advection_corrections!(Yₜ_corr, Y, p, FT(0))
    base_ρe = Base.materialize(
        CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜe_mat, dt, Val(:none)),
    )
    vl_ρe = Base.materialize(
        CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜe_mat, dt, Val(:vanleer_limiter)),
    )
    @test Array(parent(Yₜ_corr.c.ρe_tot)) .+ Array(parent(base_ρe)) ≈
          Array(parent(vl_ρe)) rtol = 1e-13
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)
    @test maximum(abs, Array(parent(Yₜ_corr.c.ρ))) == 0
    @test maximum(abs, Array(parent(Yₜ_corr.c.uₕ))) == 0
    @test maximum(abs, Array(parent(Yₜ_corr.f.u₃))) == 0
    col_corr = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
    Operators.column_integral_definite!(col_corr, Yₜ_corr.c.ρe_tot)
    col_e = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
    Operators.column_integral_definite!(col_e, Y.c.ρe_tot)
    @test abs(Array(parent(col_corr))[1]) * dtγ <=
          FT(1e-12) * abs(Array(parent(col_e))[1])
    @test _measure_allocs(
        CA.implicit_vertical_advection_tendency!, Yₜ, Y, p, t,
    ) == 0
    @test _measure_allocs(
        CA.correct_implicit_advection_tendency!, Yₜ_corr, Y, p, t,
    ) == 0

    # 3. Exact match of `(ρe_tot, u₃)` and terrain-following `(ρe_tot, uₕ)`
    # blocks with central face energy flux `ᶠρ ᶠinterp(ᶜe_tot) + ᶠinterp(ᶜp)`.
    matrix = simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    CA.update_advection_jacobian!(matrix, Y, p, dtγ, CA.IgnoreDerivative())
    @test _measure_allocs(
        CA.update_advection_jacobian!, matrix, Y, p, dtγ, CA.IgnoreDerivative(),
    ) == 0
    ᶠv = @. Geometry.Covariant3Vector(Geometry.WVector(sin(FT(π) * z_f / z_max)))
    u₃_orig = copy(Y.f.u₃)
    δ_u₃ = FT(1e-3)
    tend_u₃ = map((+1, -1)) do sgn
        @. p.precomputed.ᶠu³ = CA.CT3(u₃_orig + sgn * δ_u₃ * ᶠv)
        fill!(Yₜ, 0)
        CA.implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        copy(Array(parent(Yₜ.c.ρe_tot)))
    end
    CA.set_precomputed_quantities!(Y, p, t)
    fd_u₃ = dtγ .* (tend_u₃[1] .- tend_u₃[2]) ./ (2 * δ_u₃)
    Jv_u₃ = Array(
        parent(
            @. $(matrix[MatrixFields.@name(c.ρe_tot), MatrixFields.@name(f.u₃)]) *
               ᶠv
        ),
    )
    @test norm(Jv_u₃ .- fd_u₃) / norm(fd_u₃) < FT(1e-10)

    dry_topo_cfg = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "plane",
            "z_stretch" => false,
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "microphysics_model" => "dry",
            "topography" => "Schar",
            "mesh_warp_type" => "Linear",
            "FLOAT_TYPE" => "Float64",
            "x_max" => 100e3,
            "x_elem" => 4,
            "z_max" => 10e3,
            "z_elem" => 10,
            "dt" => "2secs",
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "consistent_transport_dry_topo_test",
    )
    nt_dt = generate_test_simulation(dry_topo_cfg)
    Y_dt, p_dt, sim_dt = nt_dt.Y, nt_dt.p, nt_dt.simulation
    mat_dt = sim_dt.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    dtγ_dt = FT(p_dt.dtγ)
    CA.set_precomputed_quantities!(Y_dt, p_dt, sim_dt.integrator.t)
    CA.update_advection_jacobian!(mat_dt, Y_dt, p_dt, dtγ_dt, CA.UseDerivative())
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        mat_dt,
        Y_dt,
        p_dt,
        dtγ_dt,
        CA.UseDerivative(),
    ) == 0
    x_cdt = Fields.coordinate_field(Y_dt.c).x
    z_cdt = Fields.coordinate_field(Y_dt.c).z
    ᶜδuₕ = @. Geometry.Covariant12Vector(
        Geometry.UVWVector(
            FT(3) * cos(FT(2π) * x_cdt / FT(100e3)) * sin(z_cdt / FT(2000)),
            FT(0),
            FT(0),
        ),
    )
    uₕ_orig = copy(Y_dt.c.uₕ)
    ᶜJ_dt = Fields.local_geometry_field(Y_dt.c).J
    Yₜ_dt = similar(Y_dt)
    δ_uₕ = FT(1e-3)
    tend_uₕ = map((+1, -1)) do sgn
        @. p_dt.precomputed.ᶠu³ =
            CA.ᶠwinterp(ᶜJ_dt * Y_dt.c.ρ, CA.CT3(uₕ_orig + sgn * δ_uₕ * ᶜδuₕ)) +
            CA.CT3(Y_dt.f.u₃)
        fill!(Yₜ_dt, 0)
        CA.implicit_vertical_advection_tendency!(
            Yₜ_dt, Y_dt, p_dt, sim_dt.integrator.t,
        )
        copy(Array(parent(Yₜ_dt.c.ρe_tot)))
    end
    CA.set_precomputed_quantities!(Y_dt, p_dt, sim_dt.integrator.t)
    fd_uₕ = dtγ_dt .* (tend_uₕ[1] .- tend_uₕ[2]) ./ (2 * δ_uₕ)
    Jv_uₕ = Array(
        parent(
            @. $(mat_dt[MatrixFields.@name(c.ρe_tot), MatrixFields.@name(c.uₕ)]) *
               ᶜδuₕ
        ),
    )
    @test norm(Jv_uₕ .- fd_uₕ) / norm(fd_uₕ) < FT(1e-10)

    # 4. Full IMEX steps conserve column-integrated mass and total energy.
    col_mass_0 = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
    col_energy_0 = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
    col_mass_1 = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
    col_energy_1 = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
    Operators.column_integral_definite!(col_mass_0, Y.c.ρ)
    Operators.column_integral_definite!(col_energy_0, Y.c.ρe_tot)
    for _ in 1:5
        CTS.step!(simulation.integrator)
    end
    Operators.column_integral_definite!(col_mass_1, Y.c.ρ)
    Operators.column_integral_definite!(col_energy_1, Y.c.ρe_tot)
    @test Array(parent(col_mass_1)) ≈ Array(parent(col_mass_0)) rtol = 1e-13
    @test Array(parent(col_energy_1)) ≈ Array(parent(col_energy_0)) rtol = 1e-13
end

@testset "Prognostic ρtke implicit upwind1 residual and Jacobian blocks" begin
    # 1. Dry EDOnly with explicit diffusion exercises the `DryModel` + `c.ρtke`
    # `sedimentation_jacobian_blocks` and iterative arrowhead paths without
    # diffusion blocks.
    dry_tke_cfg = column_config(;
        initial_condition = "DecayingProfile",
        microphysics_model = "dry",
        turbconv = "edonly_edmfx",
        prognostic_tke = true,
        implicit_diffusion = false,
    )
    nt_dry = generate_test_simulation(dry_tke_cfg)
    Y_d, p_d, sim_d = nt_dry.Y, nt_dry.p, nt_dry.simulation
    t_d = sim_d.integrator.t
    FT = eltype(Y_d)
    dtγ_d = FT(p_d.dtγ)
    z_cd = Fields.coordinate_field(Y_d.c).z
    z_fd = Fields.coordinate_field(Y_d.f).z
    z_max_d = Spaces.z_max(axes(Y_d.f))
    @test MatrixFields.has_field(Y_d, MatrixFields.@name(c.ρtke))
    # Sign-changing vertical velocity with interior zero crossing away from grid
    # faces exercises both donor branches of `ᶠupwind1` smoothly.
    @. Y_d.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(
            FT(1.8) *
            sin(FT(2π) * (z_fd / z_max_d - FT(0.23))) *
            sin(FT(π) * z_fd / z_max_d),
        ),
    )
    @. Y_d.c.ρtke =
        Y_d.c.ρ * (FT(0.4) + FT(0.3) * cos(FT(3π) * z_cd / z_max_d))
    CA.set_precomputed_quantities!(Y_d, p_d, t_d)

    Yₜ_d = similar(Y_d)
    fill!(Yₜ_d, 0)
    CA.implicit_vertical_advection_tendency!(Yₜ_d, Y_d, p_d, t_d)
    ᶜJ_d = Fields.local_geometry_field(Y_d.c).J
    ᶠJ_d = Fields.local_geometry_field(Y_d.f).J
    ᶠρ_d = @. CA.ᶠinterp(Y_d.c.ρ * ᶜJ_d) / ᶠJ_d
    ᶜtke_d = @. CA.specific(Y_d.c.ρtke, Y_d.c.ρ)
    ᶜρtke_exp = Base.materialize(
        @. -(CA.ᶜadvdivᵥ(ᶠρ_d * CA.ᶠupwind1(p_d.precomputed.ᶠu³, ᶜtke_d)))
    )
    @test Array(parent(Yₜ_d.c.ρtke)) ≈ Array(parent(ᶜρtke_exp)) rtol = 1e-13

    Yₜ_corr_d = similar(Y_d)
    CA.correct_implicit_advection_tendency!(Yₜ_corr_d, Y_d, p_d, t_d)
    @test maximum(abs, Array(parent(Yₜ_corr_d.c.ρtke))) == 0

    mat_d = sim_d.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    CA.update_sedimentation_jacobian!(mat_d, Y_d, p_d, dtγ_d)
    CA.update_advection_jacobian!(mat_d, Y_d, p_d, dtγ_d, CA.IgnoreDerivative())
    @test _measure_allocs(
        CA.update_sedimentation_jacobian!, mat_d, Y_d, p_d, dtγ_d,
    ) == 0
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        mat_d,
        Y_d,
        p_d,
        dtγ_d,
        CA.IgnoreDerivative(),
    ) == 0

    # Finite-difference check of `(c.ρtke, c.ρtke)` at fixed `ρ` and `u³`.
    ᶜδtke = @. FT(0.2) + FT(0.1) * sin(FT(5π) * z_cd / z_max_d)
    ρtke_orig = copy(Y_d.c.ρtke)
    δ_s = FT(1e-3)
    err_tke = map((+1, -1)) do sgn
        @. Y_d.c.ρtke = ρtke_orig + sgn * δ_s * ᶜδtke
        fill!(Yₜ_d, 0)
        CA.implicit_vertical_advection_tendency!(Yₜ_d, Y_d, p_d, t_d)
        copy(Array(parent(@. dtγ_d * Yₜ_d.c.ρtke - Y_d.c.ρtke)))
    end
    @. Y_d.c.ρtke = ρtke_orig
    fd_diag = (err_tke[1] .- err_tke[2]) ./ (2 * δ_s)
    Jv_diag = Array(
        parent(
            @. $(mat_d[MatrixFields.@name(c.ρtke), MatrixFields.@name(c.ρtke)]) *
               ᶜδtke
        ),
    )
    @test norm(Jv_diag .- fd_diag) / norm(fd_diag) < FT(1e-10)

    # Velocity blocks `(c.ρtke, f.u₃)` are omitted so `u³` sign reversals cannot
    # turn a frozen-donor upwind flux downwind.
    @test !haskey(mat_d, (MatrixFields.@name(c.ρtke), MatrixFields.@name(f.u₃)))
    CTS.step!(sim_d.integrator)
    @test all(isfinite, parent(sim_d.integrator.u.c.ρtke))

    # 2. Implicit-diffusion accumulation (`+=`, single `-I`) in dry and moist
    # EDMF (`update_diffusion_jacobian!`, which handles every closure).
    for (tc, mp) in (("edonly_edmfx", "dry"), ("prognostic_edmfx", "1M"))
        nt_imp = generate_test_simulation(
            column_config(;
                initial_condition = mp == "dry" ? "DecayingProfile" : "PrecipitatingColumn",
                microphysics_model = mp,
                turbconv = tc,
                prognostic_tke = true,
                implicit_diffusion = true,
            ),
        )
        Y_m, p_m, sim_m = nt_imp.Y, nt_imp.p, nt_imp.simulation
        t_m = sim_m.integrator.t
        dtγ_m = FT(p_m.dtγ)
        z_fm = Fields.coordinate_field(Y_m.f).z
        z_max_m = Spaces.z_max(axes(Y_m.f))
        @. Y_m.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(FT(1.2) * sin(FT(π) * z_fm / z_max_m)),
        )
        fill!(parent(Y_m.c.ρtke), FT(0))
        CA.set_precomputed_quantities!(Y_m, p_m, t_m)
        fill!(parent(p_m.precomputed.ᶠK_u), FT(0))
        j_cache = sim_m.integrator.cache.newtons_method_cache.j.cache
        mat_m = j_cache.matrix.matrix
        CA.update_sedimentation_jacobian!(mat_m, Y_m, p_m, dtγ_m)
        adv_tke_v = copy(
            Array(
                parent(
                    @. $(mat_m[MatrixFields.@name(c.ρtke), MatrixFields.@name(c.ρtke)]) *
                       ᶜδtke
                ),
            ),
        )
        eddy = CA.eddy_diffusivity_coefficients!(Y_m, p_m)
        CA.update_diffusion_jacobian!(
            mat_m, Y_m, p_m, dtγ_m, CA.UseDerivative(), eddy,
        )
        # With nonzero `u³`, `K_u = 0`, and `tke = 0`, `+=` preserves the
        # tridiagonal upwind advection block initialized by
        # `update_sedimentation_jacobian!` without subtracting `-I` twice.
        comb_tke_v = Array(
            parent(
                @. $(mat_m[MatrixFields.@name(c.ρtke), MatrixFields.@name(c.ρtke)]) *
                   ᶜδtke
            ),
        )
        @test comb_tke_v ≈ adv_tke_v rtol = 1e-13
    end

    # 3. Over Schar topography with a buoyancy perturbation reversing `u₃`
    # across a sharp TKE step, `(c.ρtke, c.uₕ)` and `(c.ρtke, f.u₃)` are
    # omitted and the implicit Newton solve preserves `ρtke >= 0`.
    tke_topo_cfg = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "plane",
            "z_stretch" => false,
            "initial_condition" => "ConstantBuoyancyFrequencyProfile",
            "microphysics_model" => "dry",
            "turbconv" => "edonly_edmfx",
            "prognostic_tke" => true,
            "implicit_diffusion" => true,
            "topography" => "Schar",
            "mesh_warp_type" => "Linear",
            "FLOAT_TYPE" => "Float64",
            "x_max" => 100e3,
            "x_elem" => 4,
            "z_max" => 10e3,
            "z_elem" => 10,
            "dt" => "2secs",
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "consistent_transport_tke_topo_test",
    )
    nt_tt = generate_test_simulation(tke_topo_cfg)
    Y_tt, p_tt, sim_tt = nt_tt.Y, nt_tt.p, nt_tt.simulation
    t_tt = sim_tt.integrator.t
    dtγ_tt = FT(p_tt.dtγ)
    z_ctt = Fields.coordinate_field(Y_tt.c).z
    z_ftt = Fields.coordinate_field(Y_tt.f).z
    @. Y_tt.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(
            FT(-6.0) *
            sin(FT(2π) * (z_ftt / FT(10e3) - FT(0.20))) *
            sin(FT(π) * z_ftt / FT(10e3)),
        ),
    )
    @. Y_tt.c.ρtke =
        ifelse((z_ctt > FT(1500)) & (z_ctt < FT(3500)), Y_tt.c.ρ * FT(4.0), FT(0))
    ρe_tt_orig = copy(Y_tt.c.ρe_tot)
    @. Y_tt.c.ρe_tot *=
        FT(1.0) + FT(0.03) * exp(-((z_ctt - FT(1500)) / FT(400))^2)
    CA.set_implicit_precomputed_quantities!(Y_tt, p_tt, t_tt)
    nm_j = sim_tt.integrator.cache.newtons_method_cache.j
    mat_tt = nm_j.cache.matrix.matrix
    @test !haskey(mat_tt, (MatrixFields.@name(c.ρtke), MatrixFields.@name(c.uₕ)))
    @test !haskey(mat_tt, (MatrixFields.@name(c.ρtke), MatrixFields.@name(f.u₃)))
    CA.update_advection_jacobian!(mat_tt, Y_tt, p_tt, dtγ_tt, CA.UseDerivative())
    @test _measure_allocs(
        CA.update_advection_jacobian!,
        mat_tt,
        Y_tt,
        p_tt,
        dtγ_tt,
        CA.UseDerivative(),
    ) == 0
    CA.update_jacobian!(nm_j.alg, nm_j.cache, Y_tt, p_tt, dtγ_tt, t_tt)
    Yₜ_tt = similar(Y_tt)
    CA.implicit_tendency!(Yₜ_tt, Y_tt, p_tt, t_tt)
    R_tt = similar(Y_tt)
    @. R_tt = dtγ_tt * Yₜ_tt
    ΔY_tt = similar(Y_tt)
    CA.invert_jacobian!(nm_j.alg, nm_j.cache, ΔY_tt, R_tt)
    ρtke_newton = Array(parent(@. Y_tt.c.ρtke - ΔY_tt.c.ρtke))
    @test all(isfinite, ρtke_newton)
    @test minimum(ρtke_newton) >= -FT(1e-12)
    @. Y_tt.c.ρe_tot = ρe_tt_orig
    CA.set_precomputed_quantities!(Y_tt, p_tt, t_tt)
    CTS.step!(sim_tt.integrator)
    @test all(isfinite, parent(sim_tt.integrator.u.c.ρtke))
end
