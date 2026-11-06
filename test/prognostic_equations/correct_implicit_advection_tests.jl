#=
Regression tests for `correct_implicit_advection_tendency!`, the first-order
upwind implicit vertical advection of sedimenting tracers, and the
stage-weighted precipitation surface-flux accumulator. At stage Courant number
`|u³| * dtγ ≤ 1`, `vertical_transport(:none) + upwinding_correction ==
vertical_transport(upwinding)` for `ρe_tot` and `ρq_tot`, and `vtt_upwind1 +
upwinding_correction == vtt_vanleer` for sedimenting tracers; at `|u³| * dtγ >
1`, the face correction scales by `1 / (|u³| * dtγ)`.
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
    # Small moist column so `ρq_tot` and 1M condensate species are present,
    # using default `vanleer_limiter` upwinding so the upwinding correction
    # is nontrivial.
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
    upwinding = p.atmos.numerics.energy_q_tot_upwinding

    # Populate nontrivial condensate profiles and sub-unity stage Courant
    # velocity (`|u³| * dtγ ≤ 0.25 < 1`) so the exact recovery identity holds.
    z_c = Fields.coordinate_field(Y.c).z
    @. Y.c.ρq_lcl = Y.c.ρ * (FT(0.001) * exp(-(z_c - FT(2000))^2 / FT(1e6)))
    @. Y.c.ρq_icl = Y.c.ρ * (FT(0.0005) * exp(-(z_c - FT(8000))^2 / FT(2e6)))
    @. Y.c.ρq_rai = Y.c.ρ * (FT(0.002) * exp(-(z_c - FT(1500))^2 / FT(1e6)))
    @. Y.c.ρq_sno = Y.c.ρ * (FT(0.001) * exp(-(z_c - FT(5000))^2 / FT(2e6)))
    u3_arr = parent(ᶠu³.components.data.:1)
    u3_arr .= (FT(0.25) / FT(dtγ)) .* sin.(range(FT(0), FT(π), length(u3_arr)))

    # Post-Newton upwinding correction: `(upwind - centered)` at C ≤ 1.
    Yₜ_corr = similar(Y)
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)
    # Only the `T_post_imp!` hook advances the precipitation stage counter, so
    # sampling the correction as a tendency diagnostic leaves it in phase.
    stage_idx = p.scratch.precip_stage_idx[]
    CA.correct_implicit_advection_tendency!(Yₜ_corr, Y, p, t)
    @test p.scratch.precip_stage_idx[] == stage_idx

    # Centered and full-upwind vertical fluxes, materialized from
    # `vertical_transport`.
    centered_ρe = similar(Y.c.ρe_tot)
    upwind_ρe = similar(Y.c.ρe_tot)
    vtt_0 = CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    vtt_up = CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, upwinding)
    @. centered_ρe = vtt_0
    @. upwind_ρe = vtt_up

    @test parent(Yₜ_corr.c.ρe_tot) .+ parent(centered_ρe) ≈ parent(upwind_ρe)

    if !(p.atmos.microphysics_model isa CA.DryModel)
        ᶜq_tot = @. CA.specific(Y.c.ρq_tot, Y.c.ρ)
        centered_ρq = similar(Y.c.ρq_tot)
        upwind_ρq = similar(Y.c.ρq_tot)
        vtt_0 = CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, Val(:none))
        vtt_up = CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, upwinding)
        @. centered_ρq = vtt_0
        @. upwind_ρq = vtt_up
        @test parent(Yₜ_corr.c.ρq_tot) .+ parent(centered_ρq) ≈
              parent(upwind_ρq)

        col_int = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
        col_mass = Fields.zeros(Spaces.horizontal_space(axes(Y.c)))
        for s_sym in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
            ᶜρχ = getproperty(Y.c, s_sym)
            ᶜχ = @. CA.specific(ᶜρχ, Y.c.ρ)
            vtt_upwind1 = Base.materialize(
                CA.vertical_transport(Y.c.ρ, ᶠu³, ᶜχ, dt, Val(:first_order)),
            )
            vtt_vl = Base.materialize(
                CA.vertical_transport(
                    Y.c.ρ,
                    ᶠu³,
                    ᶜχ,
                    dt,
                    Val(:vanleer_limiter),
                ),
            )
            @test parent(getproperty(Yₜ_corr.c, s_sym)) .+
                  parent(vtt_upwind1) ≈ parent(vtt_vl)

            ᶜρχₜ = getproperty(Yₜ_corr.c, s_sym)
            Operators.column_integral_definite!(col_int, ᶜρχₜ)
            Operators.column_integral_definite!(col_mass, ᶜρχ)
            @test abs(parent(col_int)[1]) / parent(col_mass)[1] <= FT(1e-12)
        end
    end

    # Courant limiter test: scaling `ᶠu³` to uniform stage Courant number
    # `C_stage = 4 > 1` damps the upwinding correction by `1 / C_stage = 1 / 4`
    # relative to the `C_stage ≤ 1` upwinding flux (which scales linearly
    # with `ᶠu³`).
    u3_arr .= FT(0.5) / FT(dtγ)
    Yₜ_c05 = similar(Y)
    CA.correct_implicit_advection_tendency!(Yₜ_c05, Y, p, t)
    u3_arr .= FT(4.0) / FT(dtγ)
    Yₜ_c40 = similar(Y)
    CA.correct_implicit_advection_tendency!(Yₜ_c40, Y, p, t)
    # Without Courant damping, multiplying `ᶠu³` by 8 would multiply the flux by
    # 8; with `1 / C_stage = 1 / 4` damping at `C_stage = 4`, the flux is
    # multiplied by `8 * (1 / 4) = 2`.
    @test parent(Yₜ_c40.c.ρe_tot) ≈ FT(2) .* parent(Yₜ_c05.c.ρe_tot)
    @test parent(Yₜ_c40.c.ρq_tot) ≈ FT(2) .* parent(Yₜ_c05.c.ρq_tot)
    for s_sym in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
        @test parent(getproperty(Yₜ_c40.c, s_sym)) ≈
              FT(2) .* parent(getproperty(Yₜ_c05.c, s_sym))
    end

    # Monotonicity, nonnegativity, and column closure at Courant numbers below
    # and above one, on a step profile and a Gaussian profile with nonzero
    # tails, against the implicit upwind1 baseline
    # Y* = (I - dtγ · L_upwind1)⁻¹ Y.
    matrix =
        simulation.integrator.cache.newtons_method_cache.j.cache.matrix.matrix
    s_name = CA.MatrixFields.FieldName(:c, :ρq_rai)
    Y_star = similar(Y)

    for profile_type in (:step, :gaussian)
        for C_val in (0.5, 1.0, 2.0, 4.0)
            u3_arr .= FT(C_val) / FT(dtγ)
            if profile_type == :step
                for k in 1:10
                    parent(Y.c.ρq_rai)[k] =
                        k <= 5 ? parent(Y.c.ρ)[k] * FT(0.01) : FT(0)
                end
            else
                for k in 1:10
                    z_k = (k - 0.5) * 150.0
                    gauss_val =
                        FT(0.001) +
                        FT(0.009) * exp(-((z_k - 750.0) / 300.0)^2)
                    parent(Y.c.ρq_rai)[k] = parent(Y.c.ρ)[k] * gauss_val
                end
            end

            for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
                fill!(parent(getproperty(p.precomputed, w_name)), FT(0))
            end
            CA.update_sedimentation_jacobian!(matrix, Y, p, dtγ)
            block = matrix[s_name, s_name]

            # Solve (I - dtγ · L_upwind1) Y* = Y via bidiagonal forward sweep
            @. Y_star = Y
            Y_arr = parent(Y.c.ρq_rai)
            Y_star_arr = parent(Y_star.c.ρq_rai)
            row_1 = Fields.level(block, 1)[]
            diag_1 = -row_1.entries[2]
            Y_star_arr[1] = Y_arr[1] / diag_1
            for k in 2:length(Y_arr)
                row_k = Fields.level(block, k)[]
                subdiag_k = -row_k.entries[1]
                diag_k = -row_k.entries[2]
                Y_star_arr[k] =
                    (Y_arr[k] - subdiag_k * Y_star_arr[k - 1]) / diag_k
            end
            @test Array(parent(@. -block * Y_star.c.ρq_rai)) ≈ Y_arr

            # Evaluate post-Newton antidiffusive correction at Y*
            fill!(parent(Yₜ_corr), FT(0))
            CA.correct_implicit_advection_tendency!(Yₜ_corr, Y_star, p, t)

            Y_final = Y_star_arr .+ dtγ .* parent(Yₜ_corr.c.ρq_rai)
            max_Y = maximum(Y_star_arr)
            min_final = minimum(Y_final)

            # Monotonicity and non-negativity across all Courant numbers
            @test min_final >= -eps(FT) * max_Y

            Operators.column_integral_definite!(col_int, Yₜ_corr.c.ρq_rai)
            Operators.column_integral_definite!(col_mass, Y_star.c.ρq_rai)
            @test abs(parent(col_int)[1]) / parent(col_mass)[1] <= FT(1e-12)
        end
    end

    # The post-Newton tracer correction acts on the four condensate masses;
    # `ρtke` stays on `ᶠupwind1`.
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

    # Directional finite-difference check of the self-advection Jacobian blocks
    # for implicitized grid-mean vertical tracer transport.
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
        field = getproperty(Y.c, s_sym)
        field_orig = copy(field)
        δ = FT(1e-3) * maximum(abs, arr(field_orig))
        @. field = field_orig + δ * ᶜv
        fill!(Yₜ_corr, 0)
        CA.implicit_vertical_advection_tendency!(Yₜ_corr, Y, p, t)
        t_plus = copy(arr(getproperty(Yₜ_corr.c, s_sym)))
        @. field = field_orig - δ * ᶜv
        fill!(Yₜ_corr, 0)
        CA.implicit_vertical_advection_tendency!(Yₜ_corr, Y, p, t)
        t_minus = copy(arr(getproperty(Yₜ_corr.c, s_sym)))
        @. field = field_orig
        fd = dtγ .* (t_plus .- t_minus) ./ (2 * δ)
        Jv = arr(@. block * ᶜv) .+ arr(ᶜv)
        @test norm(Jv .- fd) / norm(fd) < FT(5e-4)
    end

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
            # Exercise the 4-site non-negative sedimentation clamp (`max(q, 0)`)
            # on a negative snow cell prior to the step.
            parent(Y.c.ρq_sno)[1] = -FT(1e-6) * parent(Y.c.ρ)[1]
        elseif step == 3
            # Column whose z₁ rain is weaker than rain aloft (w_jac,1 > w₁)
            @. Y.c.ρq_rai = Y.c.ρ * FT(1e-4) * (z_c / FT(5000) + FT(0.1))
        elseif step == 4
            # Slightly negative rain cell at bottom level z₁ with rain aloft
            @. Y.c.ρq_rai = Y.c.ρ * FT(1e-4) * exp(-z_c / FT(2000))
            parent(Y.c.ρq_rai)[1] = -FT(1e-6) * parent(Y.c.ρ)[1]
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

        # The surface flux is downward on every step, including step 4, where
        # the bottom cell enters the stage negative: the Newton guess stays at
        # the stage-entering state and the Jacobian velocity is zero in the
        # negative cell, so the closure flux through the surface is downward.
        @test P_diag < FT(0)
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
end
