#=
Regression tests for `correct_implicit_advection_tendency!` (the post-Newton
implicit first-order dissipation and Courant-limited antidiffusion of
`e_tot - Φ`, `q_tot`, and the sedimenting tracers at the solved velocity), the
sedimentation Jacobian blocks, and the stage-weighted precipitation surface-flux
accumulator. With `dtγ = 0`, the central transport plus the correction equals
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
    # Sampling the correction as a tendency diagnostic leaves `precip_stage_idx`
    # in phase; only `post_newton_stage_update!` advances it.
    stage_idx = p.scratch.precip_stage_idx[]
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)
    @test p.scratch.precip_stage_idx[] == stage_idx

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

    # Stage-weighted precipitation surface flux conservation across full IMEX
    # steps (`ARS343` with `T_post_imp!`).
    @. Y = Y_init
    @. Y.c.ρq_rai = Y.c.ρ * FT(1e-4) * exp(-z_c / FT(2000))
    @. Y.c.ρq_sno = Y.c.ρ * FT(5e-5) * exp(-z_c / FT(3000))
    p.scratch.precip_stage_idx[] = 0
    CA.set_precomputed_quantities!(Y, p, t)

    ρ_prev = similar(Y.c.ρ)
    ρq_prev = similar(Y.c.ρq_tot)
    ρe_prev = similar(Y.c.ρe_tot)
    col_dmass = similar(p.precomputed.surface_rain_flux)
    col_dq = similar(p.precomputed.surface_rain_flux)
    col_de = similar(p.precomputed.surface_rain_flux)

    for step in 1:4
        if step == 2
            # Negative snow cell entering the step (`specific_nonneg` clamp).
            parent(Y.c.ρq_sno)[1] = -FT(1e-6) * parent(Y.c.ρ)[1]
            CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)
        elseif step == 3
            # Column whose z₁ rain is weaker than rain aloft (w_jac,1 > w₁)
            @. Y.c.ρq_rai = Y.c.ρ * FT(1e-4) * (z_c / FT(5000) + FT(0.1))
            CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)
        elseif step == 4
            # Slightly negative rain cell at bottom level z₁ with rain aloft
            @. Y.c.ρq_rai = Y.c.ρ * FT(1e-4) * exp(-z_c / FT(2000))
            parent(Y.c.ρq_rai)[1] = -FT(1e-6) * parent(Y.c.ρ)[1]
            CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)
        end
        @. ρ_prev = Y.c.ρ
        @. ρq_prev = Y.c.ρq_tot
        @. ρe_prev = Y.c.ρe_tot

        CTS.step!(simulation.integrator)

        dt_step = FT(p.dt)
        Operators.column_integral_definite!(
            col_dmass,
            @. (Y.c.ρ - ρ_prev) / dt_step
        )
        Operators.column_integral_definite!(
            col_dq,
            @. (Y.c.ρq_tot - ρq_prev) / dt_step
        )
        Operators.column_integral_definite!(
            col_de,
            @. (Y.c.ρe_tot - ρe_prev) / dt_step
        )

        P_diag =
            Array(parent(p.precomputed.surface_rain_flux))[1] +
            Array(parent(p.precomputed.surface_snow_flux))[1]
        P_actual_q = Array(parent(col_dq))[1]
        P_actual_ρ = Array(parent(col_dmass))[1]
        E_diag =
            Array(parent(p.precomputed.col_integrated_precip_energy_tendency))[1]
        E_actual = Array(parent(col_de))[1]

        # In this inflow-only column (no net implicit sink at level 1), the
        # surface flux is downward on every step, including step 4, where the
        # bottom rain cell enters the stage negative: the Newton guess stays at
        # the stage-entering state with zero base outflow while rain inflow
        # from aloft raises `ρq_rai` at level 1 across the solve. The rain flux
        # is checked on its own so that the snow flux cannot mask its sign.
        @test P_diag < FT(0)
        @test Array(parent(p.precomputed.surface_rain_flux))[1] <= FT(0)
        @test abs(P_diag - P_actual_q) * FT(86400) < FT(1e-9)
        @test abs(P_diag - P_actual_ρ) * FT(86400) < FT(1e-7)
        @test abs(E_diag - E_actual) < FT(1e-8)
    end

    p.scratch.precip_stage_idx[] = 1
    @test _measure_allocs(CA.record_stage_precipitation_linearization!, Y, p) == 0
    @test _measure_allocs(
        CA.accumulate_stage_precipitation_surface_fluxes!,
        Y,
        p,
        p.atmos.microphysics_model,
    ) == 0
    @test _measure_allocs(
        CA.set_precipitation_surface_fluxes!,
        Y,
        p,
        p.atmos.microphysics_model,
    ) == 0

    # Empty and negative cells below a `CFL_sed > 1` rain or `Chen2022`
    # cloud-ice shaft keep the column-max Jacobian velocity, so falling
    # condensate traverses them in one stage.
    for (ρχ_sym, w_sym, w_val) in (
        (:ρq_rai, :ᶜwᵣ, FT(4)),
        (:ρq_icl, :ᶜwᵢ, FT(3)),
    )
        @. Y = Y_init
        fill!(parent(Y.f.u₃), FT(0))
        ᶜρq_sp = getproperty(Y.c, ρχ_sym)
        fill!(parent(ᶜρq_sp), FT(0))
        ρ_col = arr(Y.c.ρ)
        parent(ᶜρq_sp)[5] = FT(2e-3) * ρ_col[5]
        parent(ᶜρq_sp)[4] = -FT(1e-12) * ρ_col[4]
        parent(ᶜρq_sp)[3] = -FT(1e-12) * ρ_col[3]
        CA.set_precomputed_quantities!(Y, p, t)
        # Ensure `w_val` is active at level 5 and zero below even if the test
        # simulation uses fixed/Stokes ice velocity.
        ᶜw_sp = getproperty(p.precomputed, w_sym)
        fill!(parent(ᶜw_sp), FT(0))
        parent(ᶜw_sp)[5] = w_val
        dtγ_sed = FT(1200)
        CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ_sed)
        fill!(Yₜ_corr, 0)
        CA.implicit_vertical_advection_tendency!(Yₜ_corr, Y, p, t)
        s_sp = CA.MatrixFields.FieldName(:c, ρχ_sym)
        block_sp = matrix[s_sp, s_sp]
        nz = length(ρ_col)
        J_dense = zeros(FT, nz, nz)
        e_j = similar(ᶜρq_sp)
        for j in 1:nz
            fill!(parent(e_j), FT(0))
            parent(e_j)[j] = FT(1)
            J_dense[:, j] .= vec(arr(@. block_sp * e_j))
        end
        δρq = (-J_dense) \ (dtγ_sed .* vec(arr(getproperty(Yₜ_corr.c, ρχ_sym))))
        ρq_next = vec(arr(ᶜρq_sp)) .+ δρq
        @test J_dense[4, 4] < -FT(2)
        @test J_dense[3, 4] > FT(0.5)
        @test δρq[3] > FT(0)
        @test δρq[2] > FT(0)
        @test δρq[1] > FT(0)
        @test maximum(ρq_next) < FT(2e-3) * ρ_col[5]
    end

    # `SSP333` has an explicit-diagonal stage 1 with `b_imp[1] > 0`:
    # `imex_b_imp_stages` redistributes that weight proportionally over the
    # implicit stages so the total weight stays 1. Both `NewNewtonIteration`
    # (any `max_iters`) and `NewNewtonSolve` with `max_iters == 1` enable the
    # accumulator, whereas `(NewNewtonSolve, max_iters > 1)` disables it.
    nm_1 = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    b_ssp, n_ssp =
        @test_logs (:warn, r"SSP333 weights an explicit-diagonal") CA.imex_b_imp_stages(
            CTS.IMEXAlgorithm(CTS.SSP333(), nm_1),
            FT,
        )
    @test n_ssp == 2
    @test sum(b_ssp[1:n_ssp]) ≈ FT(1) rtol = 1e-13
    nm_stage_1 = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonSolve),
    )
    _, n_stage_1 =
        CA.imex_b_imp_stages(CTS.IMEXAlgorithm(CTS.ARS343(), nm_stage_1), FT)
    @test n_stage_1 == 3
    nm_stage_2 = CTS.NewtonsMethod(;
        max_iters = 2,
        update_j = CTS.UpdateEvery(CTS.NewNewtonSolve),
    )
    b_multi, n_multi =
        CA.imex_b_imp_stages(CTS.IMEXAlgorithm(CTS.ARS343(), nm_stage_2), FT)
    @test n_multi == 0
    @test all(iszero, b_multi)

    # `reproducible_restart = true` bypasses `surface_*_flux_accum` and computes
    # the pointwise bottom-face flux even at `precip_stage_idx == n_imp_stages`.
    rr_cfg = CA.AtmosConfig(
        Dict(
            "config" => "column",
            "FLOAT_TYPE" => "Float64",
            "dt" => "10secs",
            "z_stretch" => false,
            "initial_condition" => "PrecipitatingColumn",
            "microphysics_model" => "1M",
            "reproducible_restart" => true,
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "correct_implicit_advection_rr_test",
    )
    nt_rr = generate_test_simulation(rr_cfg)
    Y_rr, p_rr = nt_rr.Y, nt_rr.p
    @. Y_rr.c.ρq_rai = Y_rr.c.ρ * FT(2e-4)
    CA.set_precomputed_quantities!(Y_rr, p_rr, FT(0))
    fill!(p_rr.scratch.surface_rain_flux_accum, FT(-999))
    p_rr.scratch.precip_stage_idx[] = p_rr.scratch.n_imp_stages
    CA.set_precipitation_surface_fluxes!(
        Y_rr,
        p_rr,
        p_rr.atmos.microphysics_model,
    )
    @test Array(parent(p_rr.precomputed.surface_rain_flux))[1] != FT(-999)
    @test Array(parent(p_rr.precomputed.surface_rain_flux))[1] < FT(0)
end

@testset "SlabOceanSST precipitation budget and EDMF surface SGS energy" begin
    # 1. SlabOceanSST + 1M conserves column water and energy across full IMEX
    # steps on both `ARS343` and `SSP333` (`a_{11} = 0, b_1 ≠ 0`).
    for algo_name in ("ARS343", "SSP333")
        slab_cfg = CA.AtmosConfig(
            Dict(
                "config" => "column",
                "FLOAT_TYPE" => "Float64",
                "dt" => "10secs",
                "z_stretch" => false,
                "initial_condition" => "PrecipitatingColumn",
                "microphysics_model" => "1M",
                "prognostic_surface" => "SlabOceanSST",
                "ode_algo" => algo_name,
                "use_itime" => false,
                "disable_surface_flux_tendency" => true,
                "output_default_diagnostics" => false,
            );
            job_id = "slab_precip_budget_$(algo_name)",
        )
        (; Y, p, simulation) = generate_test_simulation(slab_cfg)
        FT = eltype(Y)
        z_c = Fields.coordinate_field(Y.c).z
        @. Y.c.ρq_rai = Y.c.ρ * FT(1.5e-4) * exp(-z_c / FT(2000))
        @. Y.c.ρq_sno = Y.c.ρ * FT(6e-5) * exp(-z_c / FT(3000))
        @. Y.c.ρq_tot =
            max(Y.c.ρq_tot, Y.c.ρq_lcl + Y.c.ρq_icl + Y.c.ρq_rai + Y.c.ρq_sno)
        p.scratch.precip_stage_idx[] = 0
        CA.set_precomputed_quantities!(Y, p, simulation.integrator.t)

        slab = p.atmos.surface.temperature
        C_slab = slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
        col_q = similar(Y.sfc.water)
        col_e = similar(Y.sfc.water)
        Operators.column_integral_definite!(col_q, Y.c.ρq_tot)
        Operators.column_integral_definite!(col_e, Y.c.ρe_tot)
        w0 = Array(parent(col_q))[1] + Array(parent(Y.sfc.water))[1]
        e0 = Array(parent(col_e))[1] + C_slab * Array(parent(Y.sfc.T))[1]

        integ = if algo_name == "SSP333"
            # `ClimaTimeSteppers` IMEX-SSPRK steps with plain floating-point `t`
            # rather than `ITime`.
            CTS.init(
                CTS.ODEProblem(
                    simulation.integrator.sol.prob.f,
                    Y,
                    (FT(0), FT(100)),
                    p,
                ),
                simulation.integrator.alg;
                dt = FT(p.dt),
                saveat = [FT(0), FT(100)],
            )
        else
            simulation.integrator
        end
        for _ in 1:3
            CTS.step!(integ)
            Operators.column_integral_definite!(col_q, Y.c.ρq_tot)
            Operators.column_integral_definite!(col_e, Y.c.ρe_tot)
            w_now = Array(parent(col_q))[1] + Array(parent(Y.sfc.water))[1]
            e_now = Array(parent(col_e))[1] + C_slab * Array(parent(Y.sfc.T))[1]
            @test abs(w_now - w0) <= FT(1e-12) * max(FT(1), abs(w0))
            @test abs(e_now - e0) <= FT(1e-8) * max(FT(1), abs(e0))
        end
        @test Array(parent(Y.sfc.water))[1] > FT(0)

        # Upward surface precipitation routes (a) and (b) + negative bottom cell
        # slab conservation:
        # Route (a): negative level-1 `ρq_rai^g < 0` with `w_jac > 0` (from rain
        # aloft or the 1M terminal-velocity floor) and an implicit sink
        # (`ρq_rai* < ρq_rai^g`) yields an upward stage flux
        # `F = w_jac (ρq* - ρq^g) > 0` that equals the discrete column sink and
        # removes matching water from `Y.sfc.water`.
        if algo_name == "ARS343"
            fill!(parent(Y.c.ρq_lcl), FT(0))
            fill!(parent(Y.c.ρq_icl), FT(0))
            fill!(parent(Y.c.ρq_sno), FT(0))
            fill!(parent(Y.c.ρq_rai), FT(0))
            parent(Y.c.ρq_rai)[1] = -FT(1e-4) * parent(Y.c.ρ)[1]
            parent(Y.c.ρq_rai)[8] = FT(5e-4) * parent(Y.c.ρ)[8]
            CA.set_implicit_precomputed_quantities!(Y, p, simulation.integrator.t)
            CA.set_precipitation_velocities!(
                Y,
                p,
                p.atmos.microphysics_model,
                p.atmos.turbconv_model,
            )
            p.scratch.precip_stage_idx[] = 0
            fill!(p.scratch.surface_rain_flux_accum, FT(0))
            CA.record_stage_precipitation_linearization!(Y, p)
            Y_post = copy(Y)
            # Level-1 net sink (`ρq* < ρq^g < 0`) triggers route (a) upward flux
            parent(Y_post.c.ρq_rai)[1] = -FT(2.5e-4) * parent(Y.c.ρ)[1]
            Yₜ_slab = similar(Y)
            fill!(Yₜ_slab, FT(0))
            CA.surface_precipitation_tendency!(
                Yₜ_slab,
                Y,
                p,
                simulation.integrator.t,
                slab,
                p.atmos.microphysics_model,
            )
            @test Array(parent(Yₜ_slab.sfc.water))[1] == FT(0)
            CA.accumulate_stage_precipitation_surface_fluxes!(
                Y_post,
                p,
                p.atmos.microphysics_model,
            )
            Yₜ_post_slab = similar(Y)
            fill!(Yₜ_post_slab, FT(0))
            CA.add_post_newton_surface_precipitation_tendency!(
                Yₜ_post_slab,
                Y_post,
                p,
                slab,
                p.atmos.microphysics_model,
            )
            # Advance through implicit stage 2
            # (`b_imp_stages[1] + b_imp_stages[2] ≠ 0`).
            CA.record_stage_precipitation_linearization!(Y, p)
            CA.accumulate_stage_precipitation_surface_fluxes!(
                Y_post,
                p,
                p.atmos.microphysics_model,
            )
            w_jac_val = Array(parent(p.scratch.sfc_precip_lin_state.w_rai_jac))[1]
            Δρq1 = parent(Y_post.c.ρq_rai)[1] - parent(Y.c.ρq_rai)[1]
            expected_upward_stage_flux = w_jac_val * Δρq1
            @test expected_upward_stage_flux > FT(0)
            @test Array(parent(Yₜ_post_slab.sfc.water))[1] ≈
                  -expected_upward_stage_flux rtol = FT(1e-13)
            @test Array(parent(p.scratch.surface_rain_flux_accum))[1] ≈
                  (p.scratch.b_imp_stages[1] + p.scratch.b_imp_stages[2]) *
                  expected_upward_stage_flux rtol = FT(1e-13)

            # Route (b): positive level-1 `ρq_rai^g > 0` with `w_base < |w_jac|`
            # drained below `(1 - w_base / w_jac) ρq^g` in one stage.
            parent(Y.c.ρq_rai)[1] = FT(1e-5) * parent(Y.c.ρ)[1]
            parent(Y.c.ρq_rai)[8] = FT(2e-3) * parent(Y.c.ρ)[8]
            CA.set_precipitation_velocities!(
                Y,
                p,
                p.atmos.microphysics_model,
                p.atmos.turbconv_model,
            )
            CA.record_stage_precipitation_linearization!(Y, p)
            w_jac_b = Array(parent(p.scratch.sfc_precip_lin_state.w_rai_jac))[1]
            f_guess_b = Array(parent(p.scratch.sfc_precip_lin_state.f_guess_rai))[1]
            ρq_guess_b = Array(parent(p.scratch.sfc_precip_lin_state.ρq_rai_guess))[1]
            @test abs(w_jac_b) > abs(f_guess_b / ρq_guess_b)
            Y_post_b = copy(Y)
            parent(Y_post_b.c.ρq_rai)[1] = FT(0)
            # Implicit stage 1 has `b_imp_stages[1] > 0` on `ARS343`.
            p.scratch.precip_stage_idx[] = 0
            fill!(p.scratch.surface_rain_flux_accum, FT(0))
            CA.accumulate_stage_precipitation_surface_fluxes!(
                Y_post_b,
                p,
                p.atmos.microphysics_model,
            )
            f_stage_b = f_guess_b + w_jac_b * (FT(0) - ρq_guess_b)
            @test f_stage_b > FT(0)
            @test Array(parent(p.scratch.surface_rain_flux_accum))[1] ≈
                  p.scratch.b_imp_stages[1] * f_stage_b rtol = FT(1e-13)
        end
    end

    # `_dissipation_weight` returns `1 + 2 |u³| dtγ` when `compensate == true`
    # and `1` when `compensate == false`.
    @test CA._dissipation_weight(Geometry.Contravariant3Vector(0.5), 2.0, true) == 3.0
    @test CA._dissipation_weight(Geometry.Contravariant3Vector(0.5), 2.0, false) == 1.0

    # 2. EDMF surface SGS energy path (`set_sfc_sgs_precip_energy_flux!`) and
    # negative-`qʲ` / negative-`ρaʲ` clamping in
    # `edmfx_sgs_vertical_advection_tendency!`.
    edmf_cfg = CA.AtmosConfig(
        Dict(
            "config" => "column",
            "FLOAT_TYPE" => "Float64",
            "dt" => "10secs",
            "z_stretch" => false,
            "initial_condition" => "PrecipitatingColumn",
            "microphysics_model" => "1M",
            "turbconv" => "prognostic_edmfx",
            "prognostic_surface" => "SlabOceanSST",
            "use_itime" => false,
            "disable_surface_flux_tendency" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "edmf_sfc_sgs_precip_test",
    )
    (; Y, p, simulation) = generate_test_simulation(edmf_cfg)
    FT = eltype(Y)
    t = simulation.integrator.t
    z_c = Fields.coordinate_field(Y.c).z
    @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.1)
    @. Y.c.sgsʲs.:(1).q_lcl = FT(5e-4)
    @. Y.c.sgsʲs.:(1).q_rai = FT(1e-3)
    @. Y.c.sgsʲs.:(1).q_tot = FT(1.2e-2)
    @. Y.c.sgsʲs.:(1).mse = p.core.ᶜΦ + FT(3.05e5)
    @. Y.c.ρq_lcl = Y.c.ρ * FT(2e-4)
    @. Y.c.ρq_rai = Y.c.ρ * FT(5e-4)
    CA.set_precomputed_quantities!(Y, p, t)

    Yₜ = similar(Y)
    fill!(Yₜ, 0)
    CA.vertical_advection_of_water_tendency!(Yₜ, Y, p, t)
    f_e_sgs_val = Array(parent(p.scratch.sfc_precip_lin_state.f_e_sgs))[1]
    @test isfinite(f_e_sgs_val) && f_e_sgs_val != FT(0)
    @test _measure_allocs(CA.record_stage_precipitation_linearization!, Y, p) == 0

    # Exact value check of `f_e_sgs` against `J_ratio * (ρʲ fʲ + ρ⁰ f⁰)`
    # using level-1 center-space `UVWVector` conversions.
    thp = CA.CAP.thermodynamics_params(p.params)
    J_c1 = Array(parent(Fields.local_geometry_field(Y.c).J))[1]
    J_f0 = Array(parent(Fields.local_geometry_field(Y.f).J))[1]
    J_rat = J_c1 / J_f0
    lg1 = Fields.local_geometry_field(Fields.level(Y.c, 1))[]
    uvw_gm1 = Geometry.UVWVector(Fields.level(p.precomputed.ᶜu, 1)[], lg1)
    uvw_up1 = Geometry.UVWVector(Fields.level(p.precomputed.ᶜuʲs.:(1), 1)[], lg1)
    uvw_en1 = Geometry.UVWVector(Fields.level(p.precomputed.ᶜu⁰, 1)[], lg1)
    T_gm1 = Array(parent(p.precomputed.ᶜT))[1]
    T_up1 = Array(parent(p.precomputed.ᶜTʲs.:(1)))[1]
    T_en1 = Array(parent(p.precomputed.ᶜT⁰))[1]
    ρ_up1 = Array(parent(p.precomputed.ᶜρʲs.:(1)))[1]
    ρ_en1 = Array(parent(p.scratch.sfc_precip_env_density))[1]
    ρa_up1 = Array(parent(Y.c.sgsʲs.:(1).ρa))[1]
    f_e_sgs_exact = FT(0)
    for (q_sym, ρq_sym, wʲ_sym, w_sym, e_fn) in (
        (:q_lcl, :ρq_lcl, :ᶜwₗʲs, :ᶜwₗ, CA.TD.internal_energy_liquid),
        (:q_icl, :ρq_icl, :ᶜwᵢʲs, :ᶜwᵢ, CA.TD.internal_energy_ice),
        (:q_rai, :ρq_rai, :ᶜwᵣʲs, :ᶜwᵣ, CA.TD.internal_energy_liquid),
        (:q_sno, :ρq_sno, :ᶜwₛʲs, :ᶜwₛ, CA.TD.internal_energy_ice),
    )
        qʲ_1 = Array(parent(getproperty(Y.c.sgsʲs.:(1), q_sym)))[1]
        ρq_1 = Array(parent(getproperty(Y.c, ρq_sym)))[1]
        wʲ_1 = Array(parent(getproperty(p.precomputed, wʲ_sym).:(1)))[1]
        w_1 = Array(parent(getproperty(p.precomputed, w_sym)))[1]
        f_e_sgs_exact +=
            J_rat * (
                ρ_up1 * CA.updraft_precip_energy_flux(
                    qʲ_1, wʲ_1, T_up1, uvw_up1, w_1, T_gm1, uvw_gm1, ρa_up1, ρ_up1, e_fn,
                    thp,
                ) +
                ρ_en1 * CA.environment_precip_energy_flux(
                    qʲ_1, wʲ_1, ρq_1, w_1, T_gm1, uvw_gm1, T_en1, uvw_en1, ρ_en1, ρa_up1,
                    e_fn, thp,
                )
            )
    end
    @test f_e_sgs_val ≈ f_e_sgs_exact rtol = FT(1e-13)

    # SlabOceanSST + PrognosticEDMFX + 1M column water and energy conservation
    # across a full IMEX step.
    slab_edmf = p.atmos.surface.temperature
    C_slab_edmf = slab_edmf.ρ_ocean * slab_edmf.cp_ocean * slab_edmf.depth_ocean
    col_q_edmf = similar(Y.sfc.water)
    col_e_edmf = similar(Y.sfc.water)
    p.scratch.precip_stage_idx[] = 0
    CA.set_precomputed_quantities!(Y, p, t)
    Operators.column_integral_definite!(col_q_edmf, Y.c.ρq_tot)
    Operators.column_integral_definite!(col_e_edmf, Y.c.ρe_tot)
    w0_edmf = Array(parent(col_q_edmf))[1] + Array(parent(Y.sfc.water))[1]
    e0_edmf = Array(parent(col_e_edmf))[1] + C_slab_edmf * Array(parent(Y.sfc.T))[1]
    CTS.step!(simulation.integrator)
    Operators.column_integral_definite!(col_q_edmf, Y.c.ρq_tot)
    Operators.column_integral_definite!(col_e_edmf, Y.c.ρe_tot)
    @test abs(Array(parent(col_q_edmf))[1] + Array(parent(Y.sfc.water))[1] - w0_edmf) <=
          FT(1e-12) * max(FT(1), abs(w0_edmf))
    @test abs(
        Array(parent(col_e_edmf))[1] + C_slab_edmf * Array(parent(Y.sfc.T))[1] - e0_edmf,
    ) <= FT(1e-8) * max(FT(1), abs(e0_edmf))

    # Negative updraft condensate is clamped to zero in both subdomain water
    # (`edmfx_sgs_vertical_advection_tendency!`) and energy
    # (`water_advection.jl`), matching an exact zero updraft condensate state.
    Y_zero = copy(Y)
    fill!(parent(Y_zero.f.sgsʲs.:(1).u₃), FT(0))
    for sym in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        fill!(parent(getproperty(Y_zero.c, sym)), FT(0))
    end
    for sym in (:q_lcl, :q_icl, :q_rai, :q_sno)
        fill!(parent(getproperty(Y_zero.c.sgsʲs.:(1), sym)), FT(0))
    end
    CA.set_precomputed_quantities!(Y_zero, p, t)
    # Freeze terminal velocities across both states to isolate the flux clamp.
    w_r_up = copy(p.precomputed.ᶜwᵣʲs.:(1))
    w_l_up = copy(p.precomputed.ᶜwₗʲs.:(1))
    @. p.precomputed.ᶜwᵣʲs.:(1) = FT(4.0)
    @. p.precomputed.ᶜwₗʲs.:(1) = FT(0.02)
    Yₜ_zero = similar(Y)
    fill!(Yₜ_zero, 0)
    CA.edmfx_sgs_vertical_advection_tendency!(
        Yₜ_zero,
        Y_zero,
        p,
        t,
        p.atmos.turbconv_model,
    )

    Y_neg = copy(Y_zero)
    @. Y_neg.c.sgsʲs.:(1).q_rai = -FT(1e-4)
    @. Y_neg.c.sgsʲs.:(1).q_lcl = -FT(1e-4)
    CA.set_implicit_precomputed_quantities_core!(Y_neg, p, t)
    @. p.precomputed.ᶜwᵣʲs.:(1) = FT(4.0)
    @. p.precomputed.ᶜwₗʲs.:(1) = FT(0.02)
    Yₜ_neg = similar(Y)
    fill!(Yₜ_neg, 0)
    CA.edmfx_sgs_vertical_advection_tendency!(
        Yₜ_neg,
        Y_neg,
        p,
        t,
        p.atmos.turbconv_model,
    )
    @test Array(parent(Yₜ_neg.c.sgsʲs.:(1).q_rai)) ==
          Array(parent(Yₜ_zero.c.sgsʲs.:(1).q_rai))
    @test Array(parent(Yₜ_neg.c.sgsʲs.:(1).q_lcl)) ==
          Array(parent(Yₜ_zero.c.sgsʲs.:(1).q_lcl))

    # Negative draft `ρa` is clamped to zero in draft area and environment
    # sedimentation residual fluxes, matching `ρa = 0` when velocities are zero.
    Y_ρa0 = copy(Y)
    fill!(parent(Y_ρa0.c.sgsʲs.:(1).ρa), FT(0))
    fill!(parent(Y_ρa0.f.sgsʲs.:(1).u₃), FT(0))
    CA.set_implicit_precomputed_quantities_core!(Y_ρa0, p, t)
    @. p.precomputed.ᶜwᵣʲs.:(1) = FT(4.0)
    @. p.precomputed.ᶜwₗʲs.:(1) = FT(0.02)
    Yₜ_ρa0 = similar(Y)
    fill!(Yₜ_ρa0, 0)
    CA.edmfx_sgs_vertical_advection_tendency!(
        Yₜ_ρa0,
        Y_ρa0,
        p,
        t,
        p.atmos.turbconv_model,
    )
    Y_ρaneg = copy(Y_ρa0)
    @. Y_ρaneg.c.sgsʲs.:(1).ρa = -FT(0.05) * Y_ρa0.c.ρ
    Yₜ_ρaneg = similar(Y)
    fill!(Yₜ_ρaneg, 0)
    CA.edmfx_sgs_vertical_advection_tendency!(
        Yₜ_ρaneg,
        Y_ρaneg,
        p,
        t,
        p.atmos.turbconv_model,
    )
    @test Array(parent(Yₜ_ρaneg.c.sgsʲs.:(1).q_rai)) ==
          Array(parent(Yₜ_ρa0.c.sgsʲs.:(1).q_rai))

    @. p.precomputed.ᶜwᵣʲs.:(1) = w_r_up
    @. p.precomputed.ᶜwₗʲs.:(1) = w_l_up
end
