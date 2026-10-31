#=
Checks `solve_sgs_updraft_tracers_implicit_stage_analytic!`, the semi-implicit
upwind predictor for the updraft scalars in the EDMFX implicit stage, against
the recurrence its docstring states: upwind advection with the round's `u₃`,
entrainment toward the environment value, the explicit and implicit 1M
microphysics sources evaluated at the transported state (from
`ᶜmp_tendencyʲs` and `ᶜmp_tendency_implicitʲs`), and the cell-1 surface
relaxation of `mse` and `q_tot`. A nonzero covariant `u₃` at the surface face,
as over sloped terrain, leaves cell 1 unadvected. Two limits follow: fast
entrainment pulls every scalar to the environment value, and with advection
and entrainment off only cell 1 of `mse` and `q_tot` moves, to the
surface-relaxed value, while each condensate moves by `dtγ` times the sum of
its explicit and implicit cached sources.
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
            # `implicit_microphysics: true` routes cond/subl into
            # `ᶜmp_tendency_implicitʲs` so the predictor sweep adds its
            # cached contribution on top of `ᶜmp_tendencyʲs`.
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
    ᶜmp_tendency_implicitʲ = p.precomputed.ᶜmp_tendency_implicitʲs.:(1)
    S_cached = column(MatrixFields.get_field(ᶜmp_tendencyʲ, @name(dq_lcl_dt)))

    # `implicit_microphysics: true` moves `cloud_liquid_formation` and
    # `cloud_ice_formation` to `microphysics_1m_implicit_params`, so the
    # second CM BMT call `update_updraft_microphysics_tendency_cache!`
    # issues inside the predictor fills `ᶜmp_tendency_implicitʲs` with the
    # SGS-integrated cond/subl rates at the transported state.
    mp_imp = CAP.microphysics_1m_implicit_params(p.params)
    @test !isnothing(mp_imp.processes.cloud_liquid_formation)
    @test !isnothing(mp_imp.processes.cloud_ice_formation)

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

    # Full prediction: transport + `dtγ · (S_bulk + S_impl)`. `S_impl` is
    # cond/subl from `ᶜmp_tendency_implicitʲs`, nonzero only for `q_lcl`
    # and `q_icl` under `implicit_microphysics: true`.
    S_impl = Dict{Any, Vector{FT}}(
        @name(mse) => zeros(FT, nz),
        @name(q_tot) => zeros(FT, nz),
        @name(q_lcl) => zeros(FT, nz),
        @name(q_icl) => zeros(FT, nz),
        @name(q_rai) => zeros(FT, nz),
        @name(q_sno) => zeros(FT, nz),
    )
    predicted(χ_name) =
        predicted_transport(χ_name) .+ dtγ .* (S[χ_name] .+ S_impl[χ_name])

    @testset "matches the documented recurrence" begin
        CA.solve_sgs_updraft_tracers_implicit_stage_analytic!(Y, p, dtγ)
        # The explicit and implicit sources the sweep evaluates at the
        # transported state stay cached.
        for (χ_name, S_name) in sgs_tracers
            isnothing(S_name) && continue
            S[χ_name] = column(MatrixFields.get_field(ᶜmp_tendencyʲ, S_name))
            if MatrixFields.has_field(ᶜmp_tendency_implicitʲ, S_name)
                S_impl[χ_name] = column(
                    MatrixFields.get_field(ᶜmp_tendency_implicitʲ, S_name),
                )
            end
        end
        @test any(!=(0), S[@name(q_lcl)])
        @test S[@name(q_lcl)] != S_cached
        # The implicit split places cond/subl in `ᶜmp_tendency_implicitʲs`,
        # not in the bulk cache.
        @test any(!=(0), S_impl[@name(q_lcl)])
        for (χ_name, _) in sgs_tracers
            @test column(MatrixFields.get_field(sgsʲ, χ_name)) ≈
                  predicted(χ_name) rtol = 1e-12
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
        for (χ_name, S_name) in sgs_tracers
            χ_new = column(MatrixFields.get_field(sgsʲ, χ_name))
            # Strip explicit and implicit cached sources off the condensates;
            # what remains is the transport result, which the fast-ε limit
            # pulls to χ⁰.
            if !isnothing(S_name)
                S_used = column(MatrixFields.get_field(ᶜmp_tendencyʲ, S_name))
                χ_new = χ_new .- dtγ .* S_used
                if MatrixFields.has_field(ᶜmp_tendency_implicitʲ, S_name)
                    S_used_impl = column(
                        MatrixFields.get_field(ᶜmp_tendency_implicitʲ, S_name),
                    )
                    χ_new = χ_new .- dtγ .* S_used_impl
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
                expected = χ_old[χ_name] .+ dtγ .* S_used
                if MatrixFields.has_field(ᶜmp_tendency_implicitʲ, S_name)
                    S_used_impl = column(
                        MatrixFields.get_field(ᶜmp_tendency_implicitʲ, S_name),
                    )
                    expected = expected .+ dtγ .* S_used_impl
                end
                @test χ_new ≈ expected rtol = 1e-12
            end
        end
    end

    @testset "Jacobian diagonals use CM state-dependent τ for q_lcl / q_icl" begin
        # Build the full implicit-stage Jacobian via the real cache builder,
        # zero the four cond/subl-affected blocks, run just the implicit
        # microphysics update, and verify each diagonal entry matches the
        # state-dependent `-dtγ / τ_vap_to_q_lcl` / `-dtγ / τ_vap_to_q_icl`
        # evaluated at the matching (grid-mean or updraft) state.
        import CloudMicrophysics.MicrophysicsNonEq as CMNe
        alg = CA.ManualSparseJacobian(; approximate_solve_iters = 1)
        cache = CA.jacobian_cache(alg, Y, p.atmos)
        matrix = cache.matrix.matrix
        # Grid-mean block + one entry per updraft; second element is the
        # updraft index (0 = grid-mean).
        liq_blocks = Any[(matrix[@name(c.ρq_lcl), @name(c.ρq_lcl)], 0)]
        ice_blocks = Any[(matrix[@name(c.ρq_icl), @name(c.ρq_icl)], 0)]
        for j in 1:CA.n_mass_flux_subdomains(p.atmos.turbconv_model)
            χ_liq = @name(c.sgsʲs.:($j).q_lcl)
            χ_ice = @name(c.sgsʲs.:($j).q_icl)
            MatrixFields.has_field(Y, χ_liq) || continue
            push!(liq_blocks, (matrix[χ_liq, χ_liq], j))
            push!(ice_blocks, (matrix[χ_ice, χ_ice], j))
        end
        for (block, _) in vcat(liq_blocks, ice_blocks)
            parent(block) .= 0
        end
        CA.update_implicit_microphysics_jacobian!(matrix, Y, p, dtγ)

        thp = CAP.thermodynamics_params(p.params)
        mp_i = CAP.microphysics_1m_implicit_params(p.params)
        function state_cols(j)
            if j == 0
                return (
                    column(Y.c.ρ), column(p.precomputed.ᶜT),
                    column(p.precomputed.ᶜq_tot_nonneg),
                    column(Y.c.ρq_lcl) ./ column(Y.c.ρ),
                    column(Y.c.ρq_icl) ./ column(Y.c.ρ),
                    column(Y.c.ρq_rai) ./ column(Y.c.ρ),
                    column(Y.c.ρq_sno) ./ column(Y.c.ρ),
                )
            else
                sgs = Y.c.sgsʲs.:($j)
                return (
                    column(p.precomputed.ᶜρʲs.:($j)),
                    column(p.precomputed.ᶜTʲs.:($j)),
                    column(p.precomputed.ᶜq_tot_nonnegʲs.:($j)),
                    column(sgs.q_lcl), column(sgs.q_icl),
                    column(sgs.q_rai), column(sgs.q_sno),
                )
            end
        end
        _micro(q_tot, q_lcl, q_icl, q_rai, q_sno) =
            (; q_tot, q_lcl = max(0, q_lcl), q_icl = max(0, q_icl),
                q_rai = max(0, q_rai), q_sno = max(0, q_sno))
        for (block, j) in liq_blocks
            ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno = state_cols(j)
            expected = [
                -dtγ / CMNe.τ_vap_to_q_lcl(
                    mp_i.processes.cloud_liquid_formation, mp_i, thp,
                    _micro(q_tot[i], q_lcl[i], q_icl[i], q_rai[i], q_sno[i]),
                    (; ρ = ρ[i], T = T[i]),
                ) for i in 1:length(T)
            ]
            @test vec(sum(parent(block); dims = 4)) ≈ expected rtol = 1e-12
        end
        for (block, j) in ice_blocks
            ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno = state_cols(j)
            expected = [
                -dtγ / CMNe.τ_vap_to_q_icl(
                    mp_i.processes.cloud_ice_formation, mp_i, thp,
                    _micro(q_tot[i], q_lcl[i], q_icl[i], q_rai[i], q_sno[i]),
                    (; ρ = ρ[i], T = T[i]),
                ) for i in 1:length(T)
            ]
            @test vec(sum(parent(block); dims = 4)) ≈ expected rtol = 1e-12
        end
    end
end
