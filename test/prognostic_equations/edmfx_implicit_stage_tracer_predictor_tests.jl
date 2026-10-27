#=
Checks `solve_sgs_updraft_tracers_implicit_stage_analytic!`, the semi-implicit
upwind predictor for the updraft scalars in the EDMFX implicit stage, against
the recurrence its docstring states: upwind advection with the round's `u₃`,
entrainment toward the environment value, the 1M microphysics source evaluated
at the transported state, and the cell-1 surface relaxation of `mse` and
`q_tot`. A nonzero covariant `u₃` at the surface face, as over
sloped terrain, leaves cell 1 unadvected. Two limits follow: fast entrainment
pulls every scalar to the environment value, and with advection and
entrainment off only cell 1 of `mse` and `q_tot` moves, to the surface-relaxed
value, while each condensate moves by `dtγ` times its source.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, MatrixFields
import ClimaCore.MatrixFields: @name
using ClimaCore.Utilities: half

include("../test_helpers.jl")

column(f) = vec(Array(parent(f)))

@testset "EDMFX implicit-stage updraft tracer predictor (Bomex column)" begin
    config = CA.AtmosConfig(
        Dict{String, Any}(
            "initial_condition" => "Bomex",
            "FLOAT_TYPE" => "Float64",
            "config" => "column",
            "turbconv" => "prognostic_edmfx",
            "edmfx_entr_model" => "Generalized",
            "edmfx_detr_model" => "Generalized",
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => true,
            "edmfx_nh_pressure" => true,
            "implicit_diffusion" => true,
            "prognostic_tke" => true,
            "microphysics_model" => "1M",
            # Route condensation through the implicit source so the predictor
            # actually exercises `_cm_condensation_rate` (otherwise the implicit
            # path dispatches on `nothing` and the predictor is a no-op for
            # condensation).
            "cloud_liquid_formation" => "ImplicitCloudLiquidFormation",
            "implicit_microphysics" => true,
            "z_max" => 3000.0,
            "z_elem" => 30,
            "z_stretch" => false,
            "dt" => "1secs",
            "t_end" => "10secs",
            "ode_algo" => "ARS222",
            "toml" => [joinpath(pkgdir(CA), "toml", "prognostic_edmfx.toml")],
            "output_default_diagnostics" => false,
        );
        job_id = "edmfx_tracer_predictor_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)
    dtγ = FT(60)
    a_min = CAP.min_area(CAP.turbconv_params(p.params))
    sgsʲ = Y.c.sgsʲs.:(1)
    ᶠu₃ʲ = Y.f.sgsʲs.:(1).u₃.components.data.:1
    (; sfc_mass_flux_sourceʲs, sfc_mse_buoyantʲs, sfc_q_tot_buoyantʲs) =
        p.precomputed

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶠdz = Fields.Δz_field(axes(Y.f))
    z_max = FT(3000)

    # Grid-mean cloud species and a rising draft with structure in every
    # solved scalar, so that every environment value is positive.
    @. Y.c.ρtke = FT(0.5) * Y.c.ρ
    @. Y.c.ρq_lcl = Y.c.ρ * FT(2e-4) * (1 + sin(FT(2π) * ᶜz / z_max))
    @. Y.c.ρq_icl = Y.c.ρ * FT(1e-4) * (1 + sin(FT(2π) * ᶜz / z_max + 1))
    @. Y.c.ρq_rai = Y.c.ρ * FT(1e-5) * (1 + sin(FT(2π) * ᶜz / z_max + 2))
    @. Y.c.ρq_sno = Y.c.ρ * FT(5e-6) * (1 + sin(FT(2π) * ᶜz / z_max + 3))
    @. Y.c.ρq_tot =
        Y.c.ρ *
        FT(8e-3) *
        (1 + FT(0.2) * sin(FT(2π) * ᶜz / z_max + FT(0.5))) +
        Y.c.ρq_rai +
        Y.c.ρq_sno
    @. sgsʲ.ρa = Y.c.ρ * (FT(0.08) + FT(0.04) * cos(FT(2π) * ᶜz / z_max))
    @. sgsʲ.q_lcl = FT(3e-4) * (1 + FT(0.5) * cos(FT(2π) * ᶜz / z_max))
    @. sgsʲ.q_icl = FT(1e-4) * (1 + FT(0.5) * cos(FT(2π) * ᶜz / z_max + 1))
    @. sgsʲ.q_rai = FT(2e-5) * (1 + sin(FT(2π) * ᶜz / z_max + FT(1.2)))
    @. sgsʲ.q_sno = FT(1e-5) * (1 + sin(FT(2π) * ᶜz / z_max + FT(0.4)))
    @. sgsʲ.q_tot =
        FT(1e-2) * (1 + FT(0.15) * cos(FT(2π) * ᶜz / z_max)) +
        sgsʲ.q_rai +
        sgsʲ.q_sno
    # Covariant `u₃ = w Δz` with `w` [m/s] rising through the column.
    @. ᶠu₃ʲ = FT(1.5) * sin(FT(π) * ᶠz / z_max) * ᶠdz
    CA.set_precomputed_quantities!(Y, p, t)
    # The draft `mse` is the grid-mean `h_tot - K` plus a small offset.
    ᶜmse = Fields.Field(FT, axes(Y.c))
    @. ᶜmse = p.precomputed.ᶜh_tot - p.precomputed.ᶜK
    @. sgsʲ.mse = ᶜmse + FT(500) + FT(300) * sin(FT(2π) * ᶜz / z_max)
    CA.set_precomputed_quantities!(Y, p, t)

    # Prescribed surface values so the cell-1 relaxation is exercised.
    F_sfc = FT(2e-3)
    mse_sfc = column(ᶜmse)[1] + FT(1000)
    q_tot_sfc = FT(1.2e-2)
    @. $(Fields.field_values(Fields.level(sfc_mass_flux_sourceʲs.:(1), 1))) =
        F_sfc
    @. $(Fields.field_values(Fields.level(sfc_mse_buoyantʲs.:(1), 1))) =
        mse_sfc
    @. $(Fields.field_values(Fields.level(sfc_q_tot_buoyantʲs.:(1), 1))) =
        q_tot_sfc
    # A nonzero surface `u₃`, as `set_velocity_at_surface!` leaves it over
    # sloped terrain; the predictor leaves cell 1 unadvected by it.
    @. $(Fields.level(ᶠu₃ʲ, half)) = FT(0.3) * $(Fields.level(ᶠdz, half))

    Y_init = similar(Y)
    Y_init .= Y
    sgs_tracers = (
        (@name(mse), nothing),
        (@name(q_tot), nothing),
        (@name(q_lcl), @name(dq_lcl_dt)),
        (@name(q_icl), @name(dq_icl_dt)),
        (@name(q_rai), @name(dq_rai_dt)),
        (@name(q_sno), @name(dq_sno_dt)),
    )
    sfc_forced(χ_name) = χ_name === @name(mse) || χ_name === @name(q_tot)
    χ_sfc(χ_name) = χ_name === @name(mse) ? mse_sfc : q_tot_sfc

    # Reference inputs from the pre-sweep state.
    ᶠw = Fields.Field(FT, axes(Y.f))
    @. ᶠw = ᶠu₃ʲ / ᶠdz
    ᶜtmp = Fields.Field(FT, axes(Y.c))
    @. ᶜtmp = CA.ᶜbottom_bias(ᶠw) / ᶜdz
    α = column(ᶜtmp)
    @test α[1] != 0
    α[1] = 0
    @. ᶜtmp =
        CA.compute_entrainment(
            p.precomputed.ᶜentr_vel_scaleʲs.:(1),
            p.precomputed.ᶜentr_nonvel_rateʲs.:(1),
            p.precomputed.ᶜarea_bounding_entr_detrʲs.:(1),
            CA.ᶜinterp(ᶠw),
        ) + p.precomputed.ᶜturb_entrʲs.:(1)
    ε = column(ᶜtmp)
    @test all(>(0), ε)
    nz = length(α)
    ρa₁ = column(sgsʲ.ρa)[1]
    ρʲ₁ = column(p.precomputed.ᶜρʲs.:(1))[1]
    r_sfc = F_sfc / max(ρa₁, ρʲ₁ * a_min)
    @test CA.edmfx_sfc_relaxation_rate(F_sfc, ρa₁, ρʲ₁, a_min) == r_sfc
    χ_old = Dict{Any, Vector{FT}}()
    χ⁰ = Dict{Any, Vector{FT}}()
    S = Dict{Any, Vector{FT}}()
    for (χ_name, S_name) in sgs_tracers
        χ_old[χ_name] = column(MatrixFields.get_field(sgsʲ, χ_name))
        ᶜχ⁰ =
            χ_name === @name(mse) ? CA.ᶜspecific_env_mse(Y, p) :
            CA.ᶜspecific_env_value(χ_name, Y, p)
        @. ᶜtmp = ᶜχ⁰
        χ⁰[χ_name] = column(ᶜtmp)
        S[χ_name] = zeros(FT, nz)
    end
    ᶜmp_tendencyʲ = p.precomputed.ᶜmp_tendencyʲs.:(1)
    S_cached = column(MatrixFields.get_field(ᶜmp_tendencyʲ, @name(dq_lcl_dt)))

    # Implicit condensation runs between the transport sweep and the explicit
    # `ᶜmp_tendencyʲ` sources, with the stage-averaged factor
    # `τ·(1 − exp(−dtγ/τ))` on CM's instantaneous rate. The rate is evaluated
    # at the transport-state condensates with thermo at the transported draft
    # state. Both `ᶜTʲ` and `ᶜρʲ` are refreshed inside the solve just before
    # the condensation block, so reading `p.precomputed.ᶜTʲs` after the solve
    # returns gives the transport-state values.
    mp_imp = CAP.microphysics_1m_implicit_params(p.params)
    thp = CAP.thermodynamics_params(p.params)
    τ_cond = CAP.condensation_evaporation_timescale(p.params)
    cm_cond_predictor_active =
        !isnothing(mp_imp.processes.cloud_liquid_formation)
    @test cm_cond_predictor_active  # config should enable the implicit path
    τ_factor = τ_cond * (one(FT) - exp(-dtγ / τ_cond))
    function s_cond_predictor(q_lcl, q_icl, q_rai, q_sno)
        cm_cond_predictor_active || return zeros(FT, nz)
        ρʲ_col = column(p.precomputed.ᶜρʲs.:(1))
        Tʲ_col = column(p.precomputed.ᶜTʲs.:(1))
        q_tot_nonnegʲ_col = column(p.precomputed.ᶜq_tot_nonnegʲs.:(1))
        return CA._cm_condensation_rate.(
            Ref(mp_imp), Ref(thp), ρʲ_col, Tʲ_col, q_tot_nonnegʲ_col,
            q_lcl, q_icl, q_rai, q_sno,
        )
    end

    # Transport-only result of the recurrence, swept bottom to top.
    function predicted_transport(χ_name; α = α, ε = ε, r_sfc = r_sfc)
        χ_new = similar(α)
        χ_prev = zero(FT)
        for i in 1:nz
            r = i == 1 && sfc_forced(χ_name) ? r_sfc : zero(FT)
            num =
                χ_old[χ_name][i] / dtγ + ε[i] * χ⁰[χ_name][i] +
                r * χ_sfc(χ_name) + α[i] * χ_prev
            χ_new[i] = num / (1 / dtγ + α[i] + ε[i] + r)
            χ_prev = χ_new[i]
        end
        return χ_new
    end

    # Full prediction: transport + `dtγ · S` (used for `mse`, `q_tot`, and
    # the non-liquid condensates). `q_lcl` goes through its own path below
    # that interleaves the implicit condensation.
    predicted(χ_name) = predicted_transport(χ_name) .+ dtγ .* S[χ_name]

    @testset "matches the documented recurrence" begin
        CA.solve_sgs_updraft_tracers_implicit_stage_analytic!(Y, p, dtγ)
        # The condensate source the sweep evaluates at the transported state
        # stays cached.
        for (χ_name, S_name) in sgs_tracers
            isnothing(S_name) && continue
            S[χ_name] = column(MatrixFields.get_field(ᶜmp_tendencyʲ, S_name))
        end
        @test any(!=(0), S[@name(q_lcl)])
        @test S[@name(q_lcl)] != S_cached
        # Transport-only state (what the implicit condensation sees).
        q_lcl_transport = predicted_transport(@name(q_lcl))
        q_icl_transport = predicted_transport(@name(q_icl))
        q_rai_transport = predicted_transport(@name(q_rai))
        q_sno_transport = predicted_transport(@name(q_sno))
        cond_amount =
            τ_factor .* s_cond_predictor(
                q_lcl_transport, q_icl_transport, q_rai_transport,
                q_sno_transport,
            )
        @test any(!=(0), cond_amount)
        for (χ_name, _) in sgs_tracers
            # q_lcl: transport + implicit cond + explicit source.
            # others: transport + explicit source (or just transport for
            # mse/q_tot via `predicted`).
            expected =
                χ_name === @name(q_lcl) ?
                q_lcl_transport .+ cond_amount .+ dtγ .* S[@name(q_lcl)] :
                predicted(χ_name)
            @test column(MatrixFields.get_field(sgsʲ, χ_name)) ≈ expected rtol =
                1e-12
        end
        # Draft area and velocity are untouched.
        @test parent(sgsʲ.ρa) == parent(Y_init.c.sgsʲs.:(1).ρa)
        ᶠu₃ʲ_init = Y_init.f.sgsʲs.:(1).u₃.components.data.:1
        @test parent(ᶠu₃ʲ) == parent(ᶠu₃ʲ_init)
    end

    @testset "fast entrainment relaxes to the environment value" begin
        Y .= Y_init
        # ε dtγ = 6e4, so the stage and source terms leave a relative residual
        # of ~χ_old / (ε dtγ χ⁰) ≲ 1e-4.
        @. p.precomputed.ᶜturb_entrʲs.:(1) = FT(1e3)
        CA.solve_sgs_updraft_tracers_implicit_stage_analytic!(Y, p, dtγ)
        # In the fast-entrainment limit the transport recurrence is `≈ χ⁰(q_X)`,
        # which is the state the implicit condensation sees. The explicit
        # sources then stack on top for the condensates.
        cond_amount =
            τ_factor .* s_cond_predictor(
                χ⁰[@name(q_lcl)], χ⁰[@name(q_icl)],
                χ⁰[@name(q_rai)], χ⁰[@name(q_sno)],
            )
        for (χ_name, S_name) in sgs_tracers
            χ_new = column(MatrixFields.get_field(sgsʲ, χ_name))
            # Strip the explicit source off the condensates; strip the
            # implicit condensation off q_lcl too. What remains is the
            # transport result, which the fast-ε limit pulls to χ⁰.
            if !isnothing(S_name)
                S_used = column(MatrixFields.get_field(ᶜmp_tendencyʲ, S_name))
                χ_new = χ_new - dtγ * S_used
                if χ_name === @name(q_lcl)
                    χ_new = χ_new - cond_amount
                end
            end
            # Cell 1 of `mse` and `q_tot` also feels the surface relaxation.
            i₀ = sfc_forced(χ_name) ? 2 : 1
            @test χ_new[i₀:end] ≈ χ⁰[χ_name][i₀:end] rtol = 1e-3
        end
    end

    @testset "advection and entrainment off" begin
        Y .= Y_init
        parent(ᶠu₃ʲ) .= 0
        for ᶜrate in (
            p.precomputed.ᶜentr_vel_scaleʲs.:(1),
            p.precomputed.ᶜentr_nonvel_rateʲs.:(1),
            p.precomputed.ᶜarea_bounding_entr_detrʲs.:(1),
            p.precomputed.ᶜturb_entrʲs.:(1),
        )
            parent(ᶜrate) .= 0
        end
        CA.solve_sgs_updraft_tracers_implicit_stage_analytic!(Y, p, dtγ)
        # Without transport the implicit condensation sees χ_old for every
        # condensate, so the stage-averaged bump uses χ_old.
        cond_amount =
            τ_factor .* s_cond_predictor(
                χ_old[@name(q_lcl)], χ_old[@name(q_icl)],
                χ_old[@name(q_rai)], χ_old[@name(q_sno)],
            )
        for (χ_name, S_name) in sgs_tracers
            χ_new = column(MatrixFields.get_field(sgsʲ, χ_name))
            if sfc_forced(χ_name)
                @test χ_new[1] ≈
                      (χ_old[χ_name][1] / dtγ + r_sfc * χ_sfc(χ_name)) /
                      (1 / dtγ + r_sfc) rtol = 1e-12
                @test χ_new[1] != χ_old[χ_name][1]
                @test χ_new[2:end] ≈ χ_old[χ_name][2:end] rtol = 1e-14
            else
                S_used = column(MatrixFields.get_field(ᶜmp_tendencyʲ, S_name))
                expected = χ_old[χ_name] + dtγ * S_used
                if χ_name === @name(q_lcl)
                    expected = expected .+ cond_amount
                end
                @test χ_new ≈ expected rtol = 1e-12
            end
        end
    end

    @testset "Jacobian diagonal is −dtγ/τ_cond" begin
        # Build the full implicit-stage Jacobian matrix via the real cache
        # builder, zero the two condensation-affected blocks, run only the
        # condensation update, and verify the diagonals carry exactly
        # `−dtγ/τ_cond`. The sum-over-entries trick (dims=4 is the row-entry
        # dim of a MatrixRow Field) works for both `DiagonalMatrixRow` (1
        # entry) and `TridiagonalMatrixRow` (3 entries, off-diagonals left
        # at zero by this update).
        alg = CA.ManualSparseJacobian(; approximate_solve_iters = 1)
        cache = CA.jacobian_cache(alg, Y, p.atmos)
        matrix = cache.matrix.matrix
        blocks = [matrix[@name(c.ρq_lcl), @name(c.ρq_lcl)]]
        for j in 1:CA.n_mass_flux_subdomains(p.atmos.turbconv_model)
            χ_state_name = @name(c.sgsʲs.:($j).q_lcl)
            MatrixFields.has_field(Y, χ_state_name) || continue
            push!(blocks, matrix[χ_state_name, χ_state_name])
        end
        for block in blocks
            parent(block) .= 0
        end
        CA.update_lcl_condensation_jacobian!(matrix, Y, p, dtγ)
        expected_entry = -dtγ / τ_cond
        for block in blocks
            diag_per_level = vec(sum(parent(block); dims = 4))
            @test all(≈(expected_entry), diag_per_level)
        end
    end
end
