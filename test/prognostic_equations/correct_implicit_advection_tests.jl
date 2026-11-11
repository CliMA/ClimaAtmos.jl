#=
Regression tests for `correct_implicit_advection_tendency!` (the post-Newton
implicit first-order dissipation and Courant-limited antidiffusion of
`e_tot - Φ`, `q_tot`, and the sedimenting tracers at the solved velocity) and
the sedimentation Jacobian blocks. With `dtγ = 0`, the central transport plus the correction equals
`vertical_transport(upwinding)`; with `dtγ > 0`, the correction is the implicit
dissipation step `(I - dtγ 𝒟) ρχ* = ρχ` plus the limited antidiffusion at `χ*`.
=#
using Test
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Geometry, Operators, Spaces
import ClimaCore.MatrixFields: @name, has_field
import ClimaTimeSteppers as CTS

include("../test_helpers.jl")

function _measure_allocs(f!, args...)
    f!(args...)
    return @allocated f!(args...)
end

@testset "correct_implicit_advection_tendency! identity" begin
    config = CA.AtmosConfig(
        Dict(
            "config" => "column",
            "FLOAT_TYPE" => "Float64",
            "dt" => "10secs",
            "z_stretch" => false,
            "initial_condition" => "PrecipitatingColumn",
            "microphysics_model" => "1M",
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "correct_implicit_advection_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    Y_init = copy(Y)
    t = simulation.integrator.t
    FT = eltype(Y)
    (; dt, dtγ) = p
    (; ᶠu³, ᶜh_tot) = p.precomputed
    upwinding = p.atmos.numerics.energy_humidity_upwinding

    z_c = Fields.coordinate_field(Y.c).z
    @. Y.c.ρq_lcl = Y.c.ρ * (FT(0.001) * exp(-(z_c - FT(2000))^2 / FT(1e6)))
    @. Y.c.ρq_icl = Y.c.ρ * (FT(0.0005) * exp(-(z_c - FT(8000))^2 / FT(2e6)))
    @. Y.c.ρq_rai = Y.c.ρ * (FT(0.002) * exp(-(z_c - FT(1500))^2 / FT(1e6)))
    @. Y.c.ρq_sno = Y.c.ρ * (FT(0.001) * exp(-(z_c - FT(5000))^2 / FT(2e6)))
    u3_arr = parent(ᶠu³.components.data.:1)
    u3_arr .= (FT(0.25) / FT(dtγ)) .* sin.(range(FT(0), FT(π), length(u3_arr)))

    Yₜ_corr = similar(Y)
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)

    # With `dtγ = 0`, the correction plus central transport equals
    # `vertical_transport(upwinding)` on `e_tot - Φ`, `q_tot`, and condensates.
    Yₜ_explicit = similar(Y)
    fill!(Yₜ_explicit, 0)
    CA.add_vertical_advection_corrections!(Yₜ_explicit, Y, p, FT(0))
    ᶜe_mat = @. CA.specific(Y.c.ρe_tot, Y.c.ρ) - p.core.ᶜΦ
    vt(ᶜχ, upw) =
        parent(Base.materialize(CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜχ, dt, upw)))
    @test parent(Yₜ_explicit.c.ρe_tot) .+ vt(ᶜe_mat, Val(:none)) ≈
          vt(ᶜe_mat, upwinding)

    ᶜq_tot = @. CA.specific(Y.c.ρq_tot, Y.c.ρ)
    @test parent(Yₜ_explicit.c.ρq_tot) .+ vt(ᶜq_tot, Val(:none)) ≈
          vt(ᶜq_tot, upwinding)

    col_int = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
    col_mass = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
    for s_sym in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        ᶜρχ = getproperty(Y.c, s_sym)
        ᶜχ = @. CA.specific(ᶜρχ, Y.c.ρ)
        @test parent(getproperty(Yₜ_explicit.c, s_sym)) .+ vt(ᶜχ, Val(:none)) ≈
              vt(ᶜχ, Val(:vanleer_limiter))

        ᶜρχₜ = getproperty(Yₜ_corr.c, s_sym)
        Operators.column_integral_definite!(col_int, ᶜρχₜ)
        Operators.column_integral_definite!(col_mass, ᶜρχ)
        @test abs(parent(col_int)[1]) / parent(col_mass)[1] <= FT(1e-12)
    end

    # Implicit dissipation `(I - dtγ 𝒟) ρχ* = ρχ` plus Courant-limited
    # antidiffusion at `χ*`; snow is processed last, so `ᶜtemp_scalar`
    # holds `ρχ*`.
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶠρ = @. CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    ᶜanti_ref = similar(Y.c.ρq_sno)
    # Smooth gentle exponential profile (`Δz / H = 0.1`) with zero boundary `u³`
    # leaves Zalesak FCT inactive (`R^± = 1`) in the interior, matching
    # unconstrained van Leer.
    @. Y.c.ρq_sno = Y.c.ρ * FT(2e-3) * exp(-z_c / FT(30000))
    for C_val in (FT(0.3), FT(0.5), FT(4))
        u3_arr .= C_val / FT(dtγ)
        u3_arr[1] = FT(0)
        u3_arr[end] = FT(0)
        CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)
        ᶜρχ_star = copy(p.scratch.ᶜtemp_scalar)
        ᶜχ_star = Base.materialize(@. CA.specific(ᶜρχ_star, Y.c.ρ))
        ᶠweight = @. 1 + 2 * abs(ᶠu³.components.data.:1) * dtγ
        dissipation = Base.materialize(
            @. -(CA.ᶜadvdivᵥ(
                ᶠρ *
                ᶠweight *
                (CA.ᶠupwind1(ᶠu³, ᶜχ_star) - ᶠu³ * CA.ᶠinterp(ᶜχ_star)),
            ))
        )
        @test parent(ᶜρχ_star) .- dtγ .* parent(dissipation) ≈
              parent(Y.c.ρq_sno) rtol = 1e-10
        fill!(ᶜanti_ref, FT(0))
        CA.add_antidiffusive_upwind1_transport!(
            ᶜanti_ref,
            ᶠρ,
            ᶠu³,
            ᶜχ_star,
            Y.c.ρ,
            p,
            dtγ,
            upwinding,
        )
        @test parent(Yₜ_corr.c.ρq_sno) ≈
              parent(dissipation) .+ parent(ᶜanti_ref) rtol = 1e-10
        # Away from top/bottom boundaries, gentle monotone decay leaves Zalesak
        # FCT inactive (`R^± = 1`), matching unconstrained van Leer.
        antidiffusion =
            min(FT(1), 1 / C_val) .* (
                vt(ᶜχ_star, Val(:vanleer_limiter)) .-
                vt(ᶜχ_star, Val(:first_order))
            )
        nz_sno = size(parent(Y.c.ρq_sno), 1)
        @test vec(parent(ᶜanti_ref))[4:(nz_sno - 3)] ≈
              vec(antidiffusion)[4:(nz_sno - 3)] rtol = 1e-10
    end

    # Monotonicity, nonnegativity, and column closure at Courant numbers below
    # and above one (including 0 < C_γ < 1/2), plus diverging-velocity and
    # boundary-adjacent Zalesak FCT bound checks across all cells `1:nz`.
    matrix =
        simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    s_name = CA.MatrixFields.FieldName(:c, :ρq_rai)
    Y_fe = similar(Y)
    ρ_save = copy(Y.c.ρ)
    fill!(parent(Y.c.ρ), FT(1))

    for profile_type in (:step, :gaussian)
        for C_val in (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 4.0)
            z_f_col = Fields.coordinate_field(Y.f).z
            z_max_col = Spaces.z_max(axes(Y.f))
            @. ᶠu³ = Geometry.Contravariant3Vector(
                ifelse(
                    (z_f_col > FT(0)) & (z_f_col < z_max_col),
                    FT(C_val) / FT(dtγ),
                    FT(0),
                ),
            )
            if profile_type == :step
                # Interior plateau away from the closed bottom/top boundaries
                for k in 1:10
                    parent(Y.c.ρq_rai)[k] = (3 <= k <= 6) ? FT(0.01) : FT(0)
                end
            else
                for k in 1:10
                    z_k = (k - 0.5) * 3000.0
                    gauss_val =
                        FT(0.001) +
                        FT(0.009) * exp(-((z_k - 15000.0) / 5000.0)^2)
                    parent(Y.c.ρq_rai)[k] = gauss_val
                end
            end

            for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
                fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
            end
            # With zero sedimentation velocities, the rain diagonal block of the
            # Jacobian is `-I`: the grid-mean advection is advanced explicitly.
            CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
            block = matrix[s_name, s_name]
            @test Array(parent(@. -block * Y.c.ρq_rai)) ≈ parent(Y.c.ρq_rai)

            @. Y_fe = Y
            ᶜq_rai = @. CA.specific(Y.c.ρq_rai, Y.c.ρ)
            max_q_init = maximum(vec(Array(parent(Y.c.ρq_rai))))
            @. Y_fe.c.ρq_rai +=
                dtγ * $(CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜq_rai, dt, Val(:none)))
            fill!(parent(Yₜ_corr), FT(0))
            CA.correct_implicit_advection_tendency!(Yₜ_corr, Y_fe, p, t)

            Y_final = vec(Array(parent(Y_fe.c.ρq_rai) .+ dtγ .* parent(Yₜ_corr.c.ρq_rai)))
            min_final = minimum(Y_final[3:(end - 2)])

            # Interior nonnegativity and upper bound across all Courant numbers
            @test min_final >= -FT(1e-12) * max_q_init
            @test maximum(Y_final[3:(end - 2)]) <= max_q_init * (FT(1) + FT(1e-12))

            Operators.column_integral_definite!(col_int, Yₜ_corr.c.ρq_rai)
            Operators.column_integral_definite!(col_mass, Y_fe.c.ρq_rai)
            @test abs(parent(col_int)[1]) / parent(col_mass)[1] <= FT(1e-12)
        end
    end

    # Diverging-velocity and boundary-adjacent cells (`1:10` inclusive):
    # (a) Gently diverging interior Courant profile (< 2x jump per cell) stays
    # nonnegative across all cells `1:10`;
    # (b) Sharp Courant jump (`0` at boundary next to `0.5` interior, or `0.1`
    # to `0.3`) produces the expected small low-order `χ*` excursion, while
    # Zalesak FCT keeps `q_final[k]` within the local neighbor extrema of `χ*`
    # at every cell `1:10` (including boundary cells `1` and `10`).
    u3_div = parent(ᶠu³.components.data.:1)
    u3_div .= FT(0)
    for f_idx in 2:10
        u3_div[f_idx] = (FT(0.15) * (FT(1.15)^(f_idx - 2))) / FT(dtγ)
    end
    fill!(parent(Y.c.ρq_rai), FT(0))
    parent(Y.c.ρq_rai)[5] = FT(1)
    @. Y_fe = Y
    ᶜq_rai_div = @. CA.specific(Y.c.ρq_rai, Y.c.ρ)
    @. Y_fe.c.ρq_rai +=
        dtγ * $(CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜq_rai_div, dt, Val(:none)))
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y_fe, p, t)
    q_gentle = vec(Array(parent(Y_fe.c.ρq_rai) .+ dtγ .* parent(Yₜ_corr.c.ρq_rai)))
    @test minimum(q_gentle) >= -FT(1e-14)

    for (spike_k, c_jump) in ((1, true), (2, true), (5, false))
        u3_div .= FT(0)
        if c_jump
            u3_div[2:10] .= FT(0.5) / FT(dtγ)
        else
            u3_div[2:5] .= FT(0.1) / FT(dtγ)
            u3_div[6:10] .= FT(0.3) / FT(dtγ)
        end
        fill!(parent(Y.c.ρq_sno), FT(0))
        parent(Y.c.ρq_sno)[spike_k] = FT(1)
        @. Y_fe = Y
        ᶜq_sno_div = @. CA.specific(Y.c.ρq_sno, Y.c.ρ)
        @. Y_fe.c.ρq_sno +=
            dtγ * $(CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜq_sno_div, dt, Val(:none)))
        CA.correct_implicit_advection_tendency!(Yₜ_corr, Y_fe, p, t)
        # Snow runs last, so `p.scratch.ᶜtemp_scalar` holds its low-order `χ*`.
        χ_star_vec = vec(Array(parent(p.scratch.ᶜtemp_scalar)))
        q_end_vec =
            vec(Array(parent(Y_fe.c.ρq_sno) .+ dtγ .* parent(Yₜ_corr.c.ρq_sno)))
        for k in 1:10
            lo = min(χ_star_vec[max(1, k - 1)], χ_star_vec[k], χ_star_vec[min(10, k + 1)])
            hi = max(χ_star_vec[max(1, k - 1)], χ_star_vec[k], χ_star_vec[min(10, k + 1)])
            @test lo - FT(1e-14) <= q_end_vec[k] <= hi + FT(1e-14)
        end
    end
    @. Y = Y_init
    @. Y.c.ρ = ρ_save
    CA.set_precomputed_quantities!(Y, p, t)

    # The post-Newton high-order tracer loop covers the four condensate masses
    # (`ρtke` uses first-order upwinding directly in the Newton residual).
    @test CA.sedimenting_tracer_names(Y) ==
          (@name(ρq_lcl), @name(ρq_icl), @name(ρq_rai), @name(ρq_sno))

    # `Val(:vanleer_limiter)` passes `zero(dt)` to `ᶠlin_vanleer` so the MUSCL
    # reconstruction is Courant-independent and remains TVD at C > 1.
    vtt_vl_small_dt = Base.materialize(
        CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, FT(1), Val(:vanleer_limiter)),
    )
    vtt_vl_large_dt = Base.materialize(
        CA.vertical_transport(
            Y.c.ρ,
            ᶠu³,
            ᶜh_tot,
            FT(360),
            Val(:vanleer_limiter),
        ),
    )
    @test parent(vtt_vl_small_dt) == parent(vtt_vl_large_dt)

    # Zero-allocation checks for the vertical advection tendencies and the
    # Jacobian update.
    matrix =
        simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    dtγ = FT(120)
    @test _measure_allocs(
        CA.implicit_vertical_advection_tendency!,
        Yₜ_corr,
        Y,
        p,
        t,
    ) == 0
    @test _measure_allocs(
        CA.explicit_vertical_advection_tendency!,
        Yₜ_corr,
        Y,
        p,
        t,
    ) == 0
    @test _measure_allocs(
        CA.correct_implicit_advection_tendency!,
        Yₜ_corr,
        Y,
        p,
        t,
    ) == 0
    @test _measure_allocs(CA.update_sedimentation_jacobian!, matrix, Y, p, dtγ) ==
          0

    # With zero sedimentation velocities, the diagonal blocks of the
    # microphysics tracers are `-I`: their grid-mean advection is central and
    # advanced explicitly by the Newton step.
    z_f = Fields.coordinate_field(Y.f).z
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(FT(2.5) * sin(FT(π) * z_f / FT(30000))),
    )
    CA.set_precomputed_quantities!(Y, p, t)
    for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
        fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
    end
    CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    ᶜv = @. sin(z_c / FT(4000)) + FT(0.5)
    arr(f) = Array(parent(f))
    for s_sym in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        s_name = CA.MatrixFields.FieldName(:c, s_sym)
        block = matrix[s_name, s_name]
        @test arr(@. block * ᶜv) ≈ -arr(ᶜv)
    end

    # `compensate_explicit_scalar_advection` is `true` whenever the linear solve
    # omits the scalar advection derivative (`ManualSparseJacobian` — including
    # with `max_iters >= 2` or `LineSearch` — and `AutoSparseJacobian` in dry
    # explicit-diffusion runs, without `KrylovMethod`), and `false` otherwise.
    mk_cache(ode_algo, jac; model = p.atmos) = CA.build_cache(
        Y,
        model,
        p.params,
        p.dt,
        simulation.start_date,
        p.steady_state_velocity;
        ode_algo,
        jacobian = jac,
    )
    nm_base = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    nm_2iter = CTS.NewtonsMethod(;
        max_iters = 2,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    nm_ls = CTS.NewtonsMethod(;
        max_iters = 2,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
        line_search = CTS.LineSearch(),
    )
    nm_krylov = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
        krylov_method = CTS.KrylovMethod(;
            jacobian_free_jvp = CTS.ForwardDiffJVP(; step_adjustment = FT(1)),
            forcing_term = CTS.ConstantForcing(FT(0.1)),
        ),
    )
    @test mk_cache(CTS.IMEXAlgorithm(CTS.ARS343(), nm_base), CA.ManualSparseJacobian()).scratch.compensate_explicit_scalar_advection
    @test mk_cache(CTS.IMEXAlgorithm(CTS.ARS343(), nm_2iter), CA.ManualSparseJacobian()).scratch.compensate_explicit_scalar_advection
    @test mk_cache(CTS.IMEXAlgorithm(CTS.ARS343(), nm_ls), CA.ManualSparseJacobian()).scratch.compensate_explicit_scalar_advection
    @test !mk_cache(CTS.IMEXAlgorithm(CTS.ARS343(), nm_krylov), CA.ManualSparseJacobian()).scratch.compensate_explicit_scalar_advection
    @test !mk_cache(CTS.IMEXAlgorithm(CTS.ARS343(), nm_base), CA.AutoSparseJacobian()).scratch.compensate_explicit_scalar_advection
    @test !mk_cache(CTS.IMEXAlgorithm(CTS.ARS343(), nm_base), CA.AutoDenseJacobian()).scratch.compensate_explicit_scalar_advection
    dry_sim = generate_test_simulation(
        CA.AtmosConfig(
            Dict(
                "config" => "column",
                "FLOAT_TYPE" => "Float64",
                "dt" => "10secs",
                "z_stretch" => false,
                "moisture_model" => "dry",
                "implicit_diffusion" => false,
                "output_default_diagnostics" => false,
            );
            job_id = "compensate_dry_autosparse_test",
        ),
    )
    @test CA.build_cache(
        dry_sim.Y,
        dry_sim.p.atmos,
        dry_sim.p.params,
        dry_sim.p.dt,
        dry_sim.simulation.start_date,
        dry_sim.p.steady_state_velocity;
        ode_algo = CTS.IMEXAlgorithm(CTS.ARS343(), nm_base),
        jacobian = CA.AutoSparseJacobian(),
    ).scratch.compensate_explicit_scalar_advection
    @test !CA.build_cache(
        dry_sim.Y,
        dry_sim.p.atmos,
        dry_sim.p.params,
        dry_sim.p.dt,
        dry_sim.simulation.start_date,
        dry_sim.p.steady_state_velocity;
        ode_algo = CTS.IMEXAlgorithm(CTS.ARS343(), nm_krylov),
        jacobian = CA.AutoSparseJacobian(),
    ).scratch.compensate_explicit_scalar_advection

end
