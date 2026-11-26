#=
julia +1.13 --project=.buildkite
using Revise; include("test/prognostic_equations/vertical_transport_scalars_tests.jl")
=#
include("vertical_transport_test_helpers.jl")

@testset "Hydrostatic moist-step ΔT (C ∈ [0.1, 5]) and column energy conservation" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()
    nz = 80
    z_max = 8000.0
    Δz = z_max / nz
    grid = CA.ColumnGrid(
        FT;
        context = ctx,
        z_elem = nz,
        z_max = z_max,
        z_stretch = false,
    )
    params = CA.ClimaAtmosParameters(FT)
    thp = CA.CAP.thermodynamics_params(params)
    cp_d = FT(CA.CAP.cp_d(params))

    # 1. Isothermal (`T_0 = 288 K`) hydrostatically balanced moist column with a
    # sharp moisture step (`1 g/kg -> 7 g/kg` at `z = 3500 m`) under uniform
    # interior vertical velocity across `C ∈ [0.1, 5]` on `ARS343` and `SSP333`.
    model_moist = CA.AtmosModel(
        grid;
        params,
        microphysics_model = CA.EquilibriumMicrophysics0M(),
        advection_test = true,
        disable_surface_flux_tendency = true,
    )
    T_0 = FT(288.0)
    dt_m = FT(10.0)

    function init_hydrostatic_moist!(
        integrator_h,
        z_step::FT,
        z_max_h::FT,
        nz_h::Int,
        C_val::FT,
    )
        Y_h = integrator_h.u
        p_h = integrator_h.p
        Δz_h = z_max_h / FT(nz_h)
        CTS.reinit!(integrator_h, Y_h)
        z_c_vec = vec(_host_arr(Fields.coordinate_field(Y_h.c).z))
        z_f_vec = vec(_host_arr(Fields.coordinate_field(Y_h.f).z))
        Φ_vec = vec(_host_arr(p_h.core.ᶜΦ))

        qt_vec = [z < z_step ? FT(7e-3) : FT(1e-3) for z in z_c_vec]
        z_zero = zeros(FT, nz_h)
        T_vec = fill(T_0, nz_h)
        p_vec = zeros(FT, nz_h)
        _solve_discrete_exner_hydrostatic_p!(
            p_vec,
            T_vec,
            qt_vec,
            z_zero,
            z_zero,
            Φ_vec,
            thp,
            cp_d,
            FT(1e5),
        )

        w_int = C_val * Δz_h / dt_m
        ramp_len = FT(1200)
        w_m_vals = [
            zf <= ramp_len ?
            w_int * FT(0.5) * (one(FT) - cos(FT(π) * zf / ramp_len)) :
            (
                zf <= z_max_h - ramp_len ? w_int :
                w_int *
                FT(0.5) *
                (one(FT) - cos(FT(π) * (z_max_h - zf) / ramp_len))
            ) for zf in z_f_vec
        ]
        _set_col_w!(Y_h.f.u₃, w_m_vals)
        fill!(Y_h.c.uₕ, Geometry.Covariant12Vector(FT(0), FT(0)))
        ᶜK_vec = vec(
            _host_arr(
                Base.materialize(CA.compute_kinetic(Y_h.c.uₕ, Y_h.f.u₃)),
            ),
        )

        ρ_prof = zeros(FT, nz_h)
        ρqt_prof = zeros(FT, nz_h)
        ρe_prof = zeros(FT, nz_h)
        for k in 1:nz_h
            R_m = TD.gas_constant_air(thp, qt_vec[k], FT(0), FT(0))
            ρ_k = p_vec[k] / (R_m * T_0)
            ρ_prof[k] = ρ_k
            ρqt_prof[k] = ρ_k * qt_vec[k]
            e_int_k = TD.internal_energy(thp, T_0, qt_vec[k], FT(0), FT(0))
            ρe_prof[k] = ρ_k * (e_int_k + ᶜK_vec[k] + Φ_vec[k])
        end
        _set_col_scalar!(Y_h.c.ρ, ρ_prof)
        _set_col_scalar!(Y_h.c.ρq_tot, ρqt_prof)
        _set_col_scalar!(Y_h.c.ρe_tot, ρe_prof)
        CA.set_precomputed_quantities!(Y_h, p_h, integrator_h.t)

        # Verify discrete hydrostatic residual vanishes before stepping.
        Yₜ_chk = zero(Y_h)
        CA.add_implicit_u₃_tendency!(
            Yₜ_chk.f,
            p_h.core.ᶠgradᵥ_ᶜΦ,
            p_h.precomputed.ᶜp,
            p_h.precomputed.ᶜT,
            p_h.precomputed.ᶜq_tot_nonneg,
            p_h.precomputed.ᶜq_liq,
            p_h.precomputed.ᶜq_ice,
            thp,
            cp_d,
        )
        @test maximum(abs, _host_arr(Yₜ_chk.f.u₃)[2:nz_h]) < FT(1e-9)

        return (
            _column_integral(Y_h.c.ρe_tot),
            _column_integral(Y_h.c.ρq_tot),
            _column_integral(Y_h.c.ρ),
        )
    end

    function check_hydrostatic_moist!(
        Y_h,
        p_h,
        nz_h::Int,
        interior_range,
        e_col_init::FT,
        qt_col_init::FT,
        ρ_col_init::FT,
        dT_tol::FT,
    )
        T_after = vec(_host_arr(p_h.precomputed.ᶜT))
        max_dT = maximum(abs, T_after[interior_range] .- T_0)
        @test max_dT <= dT_tol

        e_scale = abs(e_col_init)
        @test isapprox(
            _column_integral(Y_h.c.ρe_tot),
            e_col_init;
            atol = 16 * nz_h * eps(FT) * e_scale,
        )
        @test isapprox(
            _column_integral(Y_h.c.ρq_tot),
            qt_col_init;
            atol = 16 * nz_h * eps(FT) * abs(qt_col_init),
        )
        @test isapprox(
            _column_integral(Y_h.c.ρ),
            ρ_col_init;
            atol = 16 * nz_h * eps(FT) * abs(ρ_col_init),
        )
    end

    nz_1b = 160
    z_max_1b = FT(16000.0)
    grid_1b = CA.ColumnGrid(
        FT;
        context = ctx,
        z_elem = nz_1b,
        z_max = z_max_1b,
        z_stretch = false,
    )
    model_moist_1b = CA.AtmosModel(
        grid_1b;
        params,
        microphysics_model = CA.EquilibriumMicrophysics0M(),
        advection_test = true,
        disable_surface_flux_tendency = true,
    )

    for (alg_label, ode_config) in (
        ("ARS343", _imex_alg(CTS.ARS343())),
        ("SSP333", _imex_ssp_alg(CTS.SSP333())),
    )
        mktempdir() do tmpdir
            recorded_mask = Ref(UInt16(0))
            hook_5! =
                (Y_h, p_h, _t_h, Δt_h, cache_h) -> begin
                    recorded_mask[] = p_h.scratch.stage_entry_χ_e_mask[]
                    CA.step_vertical_scalar_transport!(Y_h, p_h, Δt_h, cache_h)
                end
            sim = CA.AtmosSimulation(
                model_moist;
                dt = dt_m,
                t_end = dt_m,
                ode_config,
                output_dir = joinpath(tmpdir, "dT_$alg_label"),
                default_callbacks = false,
                post_step_hook! = hook_5!,
            )
            integrator = sim.integrator
            Y = integrator.u
            p = integrator.p

            # 1a. Multi-stage `CTS.step!` (`C ∈ [0.1, 0.5]`), verifying
            # `stage_entry_χ_e_mask` on every `C_val`. Interior `33:48`
            # (`z ∈ [3.2, 4.8 km]`) spans the `z = 4.0 km` moisture step.
            a_imp_tab = integrator.cache.tableau.a_imp
            b_imp_tab = integrator.cache.tableau.b_imp
            expected_mask = UInt16(0)
            for i in 1:length(b_imp_tab)
                if !iszero(a_imp_tab[i, i]) || !iszero(b_imp_tab[i])
                    expected_mask |= UInt16(1) << (i - 1)
                end
            end
            for C_val in FT.((0.1, 0.25, 0.5))
                e_col_init, qt_col_init, ρ_col_init = init_hydrostatic_moist!(
                    integrator,
                    FT(4000),
                    FT(z_max),
                    nz,
                    C_val,
                )
                recorded_mask[] = UInt16(0)
                CTS.step!(integrator)
                @test float(integrator.dt) == dt_m
                @test (recorded_mask[] & expected_mask) == expected_mask
                dT_tol = C_val <= FT(0.25) ? FT(1e-4) : FT(3e-3)
                check_hydrostatic_moist!(
                    Y,
                    p,
                    nz,
                    33:48,
                    e_col_init,
                    qt_col_init,
                    ρ_col_init,
                    dT_tol,
                )
            end

            # 1b. Single-step uniform-interior-`w` hydrostatic moist step across
            # `C ∈ [0.1, 5.0]` via `step_vertical_scalar_transport!`. Placed
            # `88` cells above the `[0, 1200 m]` ramp (`nz = 160`, step at
            # `z = 11 km`, interior `102:118`) so the `(C / (1 + C))^Δk`
            # implicit M-matrix ramp tail stays below `1e-4 K` at `C = 5`.
            if alg_label == "ARS343"
                sim_1b = CA.AtmosSimulation(
                    model_moist_1b;
                    dt = dt_m,
                    t_end = dt_m,
                    ode_config,
                    output_dir = joinpath(tmpdir, "dT_1b_$alg_label"),
                    default_callbacks = false,
                    post_step_hook! = CA.step_vertical_scalar_transport!,
                )
                int_1b = sim_1b.integrator
                Y_1b = int_1b.u
                p_1b = int_1b.p
                for C_val in FT.((0.1, 0.5, 1.0, 2.5, 5.0))
                    e_col_init, qt_col_init, ρ_col_init = init_hydrostatic_moist!(
                        int_1b,
                        FT(11000),
                        z_max_1b,
                        nz_1b,
                        C_val,
                    )
                    CA.set_implicit_precomputed_quantities_core!(
                        Y_1b,
                        p_1b,
                        int_1b.t,
                    )
                    Yₜ_wave = zero(Y_1b)
                    CA.implicit_vertical_advection_tendency!(
                        Yₜ_wave,
                        Y_1b,
                        p_1b,
                        int_1b.t,
                    )
                    p_1b.scratch.stage_entry_χ_e_mask[] = UInt16(0)
                    CA.foreach_b_imp_stage(int_1b.cache) do i, _w_i, T_imp_i
                        T_imp_i .= Yₜ_wave
                        CA._write_stage_entry_χ_e!(
                            p_1b.scratch.ᶠstage_entry_χ_e[i],
                            Y_1b,
                            p_1b.core.ᶜΦ,
                        )
                        p_1b.scratch.stage_entry_χ_e_mask[] |= UInt16(1) << (i - 1)
                    end
                    @. Y_1b.c.ρ += dt_m * Yₜ_wave.c.ρ
                    @. Y_1b.c.ρe_tot += dt_m * Yₜ_wave.c.ρe_tot
                    @. Y_1b.c.ρq_tot += dt_m * Yₜ_wave.c.ρq_tot
                    CA.step_vertical_scalar_transport!(
                        Y_1b,
                        p_1b,
                        dt_m,
                        int_1b.cache,
                    )
                    CA.set_precomputed_quantities!(Y_1b, p_1b, int_1b.t)
                    check_hydrostatic_moist!(
                        Y_1b,
                        p_1b,
                        nz_1b,
                        102:118,
                        e_col_init,
                        qt_col_init,
                        ρ_col_init,
                        FT(1e-4),
                    )
                end
            end
        end
    end

    # 2. Live compressible acoustics (`advection_test = false`) on a stratified
    # moist column over 3 steps (`ARS343` `Float64` and `SSP333` `Float32`):
    # stage-entry `ê_i` evolves per stage, conserving column energy to roundoff.
    for (FT_e, ode_cfg) in (
        (Float64, _imex_alg(CTS.ARS343())),
        (Float32, _imex_ssp_alg(CTS.SSP333())),
    )
        grid_e = CA.ColumnGrid(
            FT_e;
            context = ctx,
            z_elem = 24,
            z_max = 12000.0,
            z_mesh = _stretched_z_mesh(FT_e, 24, 12000.0),
        )
        params_e = CA.ClimaAtmosParameters(FT_e)
        model_e = CA.AtmosModel(
            grid_e;
            params = params_e,
            microphysics_model = CA.EquilibriumMicrophysics0M(),
            disable_surface_flux_tendency = true,
        )
        mktempdir() do tmpdir
            sim_e = CA.AtmosSimulation(
                model_e;
                dt = FT_e(25),
                t_end = FT_e(75),
                ode_config = ode_cfg,
                output_dir = joinpath(tmpdir, "live_energy"),
                default_callbacks = false,
                post_step_hook! = CA.step_vertical_scalar_transport!,
            )
            Ye = sim_e.integrator.u
            pe = sim_e.integrator.p
            z_f_e = Fields.coordinate_field(Ye.f).z
            z_c_e = Fields.coordinate_field(Ye.c).z
            z_max_e = FT_e(Spaces.z_max(axes(Ye.f)))
            @. Ye.f.u₃ = Geometry.Covariant3Vector(
                Geometry.WVector(FT_e(3.0) * sin(FT_e(π) * z_f_e / z_max_e)),
            )
            @. Ye.c.ρq_tot =
                Ye.c.ρ * FT_e(0.004) * exp(-z_c_e / FT_e(2500))
            CA.set_precomputed_quantities!(Ye, pe, sim_e.integrator.t)

            e_init = _column_integral(Ye.c.ρe_tot)
            m_init = _column_integral(Ye.c.ρ)
            q_init = _column_integral(Ye.c.ρq_tot)

            CA.solve_atmos!(sim_e)

            atol_fac = FT_e === Float32 ? FT_e(64) : FT_e(32)
            @test isapprox(
                _column_integral(Ye.c.ρe_tot),
                e_init;
                atol = atol_fac * 24 * eps(FT_e) * abs(e_init),
            )
            @test isapprox(
                _column_integral(Ye.c.ρ),
                m_init;
                atol = atol_fac * 24 * eps(FT_e) * abs(m_init),
            )
            @test isapprox(
                _column_integral(Ye.c.ρq_tot),
                q_init;
                atol = atol_fac * 24 * eps(FT_e) * abs(q_init),
            )
        end
    end
end

@testset "Individual FCT scalars (2M/P3 numbers & rime, ρtke, PrognosticEDMFX ρaʲ) and zero allocations" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()
    nz = 24
    grid = CA.ColumnGrid(
        FT;
        context = ctx,
        z_elem = nz,
        z_max = 6000.0,
        z_mesh = _stretched_z_mesh(FT, nz, 6000.0),
    )
    params = CA.ClimaAtmosParameters(FT)

    # 1. `PrognosticEDMFX` + `1M` (`ρtke, ρq_gas_A, sgsʲs.:(1).ρa`).
    model_edmf_1m = CA.AtmosModel(
        grid;
        params,
        microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        chemistry_model = CA.GasPhaseChem(),
        turbconv_model = CA.PrognosticEDMFX(;
            n_updrafts = 1,
            prognostic_tke = true,
            area_fraction = FT(1e-5),
        ),
        edmfx_model = CA.EDMFXModel(;
            entr_model = CA.InvZEntrainment(),
            detr_model = CA.BuoyancyVelocityDetrainment(),
            scale_blending_method = CA.SmoothMinimumBlending(),
        ),
        advection_test = true,
        disable_surface_flux_tendency = true,
    )
    integrator = _make_ssprk_integrator(
        model_edmf_1m,
        CTS.ARS343(),
        FT(10),
        FT(10),
    )
    let integrator = integrator
        Y = integrator.u
        p = integrator.p
        z_f = Fields.coordinate_field(Y.f).z
        z_max = FT(Spaces.z_max(axes(Y.f)))
        @. Y.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(FT(6.0) * sin(FT(π) * z_f / z_max)),
        )

        # Pure continuity stage tendencies in `integrator.cache` test vertical
        # transport directly on `ρtke, ρq_gas_A, sgsʲs.:(1).ρa`.
        @. Y.c.ρq_tot = FT(5e-3) * Y.c.ρ
        for nm in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
            fill!(getproperty(Y.c, nm), FT(0))
        end
        ρ_host = vec(_host_arr(Y.c.ρ))
        _set_col_scalar!(
            Y.c.ρtke,
            [k == 8 ? FT(2.0) * ρ_host[8] : zero(FT) for k in 1:nz],
        )
        @. Y.c.sgsʲs.:(1).ρa = FT(0.15) * Y.c.ρ
        @. Y.c.ρq_gas_A = Y.c.ρ

        CA.set_implicit_precomputed_quantities_core!(Y, p, integrator.t)
        for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
            fill!(getproperty(p.precomputed, w_name), FT(0))
        end
        Yₜ_cont = zero(Y)
        ᶜJ = Fields.local_geometry_field(Y.c).J
        ᶠJ = Fields.local_geometry_field(Y.f).J
        ᶠρ = @. CA.ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
        @. Yₜ_cont.c.ρ = -CA.ᶜadvdivᵥ(ᶠρ * p.precomputed.ᶠu³)
        ᶜe_tot = @. CA.specific(Y.c.ρe_tot, Y.c.ρ)
        CA.add_energy_transport!(
            Yₜ_cont.c.ρe_tot,
            ᶠρ,
            p.precomputed.ᶠu³,
            ᶜe_tot,
            p.precomputed.ᶜp,
        )
        @. Yₜ_cont.c.ρq_tot = -CA.ᶜadvdivᵥ(ᶠρ * p.precomputed.ᶠu³ * FT(5e-3))

        p.scratch.stage_entry_χ_e_mask[] = UInt16(0)
        CA.foreach_b_imp_stage(integrator.cache) do i, _w_i, T_imp_i
            T_imp_i .= Yₜ_cont
            CA._write_stage_entry_χ_e!(
                p.scratch.ᶠstage_entry_χ_e[i],
                Y,
                p.core.ᶜΦ,
            )
            p.scratch.stage_entry_χ_e_mask[] |= UInt16(1) << (i - 1)
        end

        tke_init = _column_integral(Y.c.ρtke)
        ρa_init = _column_integral(Y.c.sgsʲs.:(1).ρa)
        ρa_pre_vec = copy(_host_arr(Y.c.sgsʲs.:(1).ρa))

        Δt = FT(10)
        @. Y.c.ρ += Δt * Yₜ_cont.c.ρ
        @. Y.c.ρe_tot += Δt * Yₜ_cont.c.ρe_tot
        @. Y.c.ρq_tot += Δt * Yₜ_cont.c.ρq_tot

        CA.step_vertical_scalar_transport!(Y, p, Δt, integrator.cache)

        ρ_new = vec(_host_arr(Y.c.ρ))
        tke_spec = vec(_host_arr(Y.c.ρtke)) ./ ρ_new
        gas_spec = vec(_host_arr(Y.c.ρq_gas_A)) ./ ρ_new

        @test minimum(tke_spec) >= -FT(1e-14)
        @test maximum(tke_spec) <= FT(2.0) + FT(1e-13)
        @test _host_arr(Y.c.sgsʲs.:(1).ρa) == ρa_pre_vec
        @test maximum(abs, gas_spec .- one(FT)) <= FT(1e-14)

        @test isapprox(_column_integral(Y.c.ρtke), tke_init; rtol = FT(1e-13))
        @test isapprox(_column_integral(Y.c.sgsʲs.:(1).ρa), ρa_init; rtol = FT(1e-13))

        # 2. Zero-allocation checks on CPU (`CUDADevice` wraps kernel launches).
        if ClimaComms.device(Y.c) isa ClimaComms.AbstractCPUDevice
            p.scratch.post_step.stage_imp_idx = 0
            CA.record_implicit_stage_entry_energy!(Y, p)
            p.scratch.post_step.stage_imp_idx = 0
            @test _alloc_record_imp(Y, p) == 0

            T_imp_2 = integrator.cache.T_imp[2]
            CA.maybe_record_explicit_imp_stage_energy!(T_imp_2, Y, p)
            @test _alloc_record_exp_imp(T_imp_2, Y, p) == 0

            CA.step_vertical_scalar_transport!(Y, p, Δt, integrator.cache)
            @test _alloc_vt(Y, p, Δt, integrator.cache) == 0
        end
    end

    # 2b. `2M` + `P3` individual scalars (`ρn_lcl, ρn_rai, ρn_ice, ρq_rim, ρb_rim`)
    # under both `:vanleer_limiter` (FCT) and `:none` (central bypass).
    for eh_mode in (:vanleer_limiter, :none)
        model_mp = CA.AtmosModel(
            grid;
            params,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            numerics = CA.AtmosNumerics(;
                energy_humidity_upwinding = eh_mode,
            ),
            advection_test = true,
            disable_surface_flux_tendency = true,
        )
        integrator_base = _make_ssprk_integrator(
            model_mp,
            CTS.ARS343(),
            FT(10),
            FT(10),
        )
        Y_base = integrator_base.u
        pp = integrator_base.p
        z_ft = zero(FT)
        c_mp = map(
            s -> (;
                s...,
                ρn_lcl = z_ft,
                ρn_rai = z_ft,
                ρn_ice = z_ft,
                ρq_rim = z_ft,
                ρb_rim = z_ft,
            ),
            Y_base.c,
        )
        Yp = Fields.FieldVector(; c = c_mp, f = copy(Y_base.f))
        z_f = Fields.coordinate_field(Yp.f).z
        z_max = FT(Spaces.z_max(axes(Yp.f)))
        @. Yp.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(FT(5.0) * sin(FT(π) * z_f / z_max)),
        )
        @. Yp.c.ρq_tot = FT(4e-3) * Yp.c.ρ
        for nm in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
            fill!(getproperty(Yp.c, nm), FT(0))
        end
        ρp_host = vec(_host_arr(Yp.c.ρ))
        _set_col_scalar!(
            Yp.c.ρn_lcl,
            [k == 10 ? FT(1e7) * ρp_host[10] : zero(FT) for k in 1:nz],
        )
        @. Yp.c.ρn_rai = FT(1e3) * Yp.c.ρ
        @. Yp.c.ρn_ice = FT(2e5) * Yp.c.ρ
        _set_col_scalar!(
            Yp.c.ρq_rim,
            [k == 12 ? FT(1e-4) * ρp_host[12] : zero(FT) for k in 1:nz],
        )
        @. Yp.c.ρb_rim = FT(2e-7) * Yp.c.ρ

        CA.set_implicit_precomputed_quantities_core!(Yp, pp, integrator_base.t)
        Yₜ_p3 = zero(Yp)
        ᶜJ = Fields.local_geometry_field(Yp.c).J
        ᶠJ = Fields.local_geometry_field(Yp.f).J
        ᶠρ = @. CA.ᶠinterp(Yp.c.ρ * ᶜJ) / ᶠJ
        @. Yₜ_p3.c.ρ = -CA.ᶜadvdivᵥ(ᶠρ * pp.precomputed.ᶠu³)
        CA.add_energy_transport!(
            Yₜ_p3.c.ρe_tot,
            ᶠρ,
            pp.precomputed.ᶠu³,
            (@. CA.specific(Yp.c.ρe_tot, Yp.c.ρ)),
            pp.precomputed.ᶜp,
        )
        @. Yₜ_p3.c.ρq_tot = -CA.ᶜadvdivᵥ(ᶠρ * pp.precomputed.ᶠu³ * FT(4e-3))

        cache_mp = (;
            T_imp = ntuple(
                _ -> zero(Yp),
                length(integrator_base.cache.tableau.b_imp),
            ),
            tableau = integrator_base.cache.tableau,
        )
        pp.scratch.stage_entry_χ_e_mask[] = UInt16(0)
        CA.foreach_b_imp_stage(cache_mp) do i, _w_i, T_imp_i
            T_imp_i .= Yₜ_p3
            CA._write_stage_entry_χ_e!(
                pp.scratch.ᶠstage_entry_χ_e[i],
                Yp,
                pp.core.ᶜΦ,
            )
            pp.scratch.stage_entry_χ_e_mask[] |= UInt16(1) << (i - 1)
        end

        nl_init = _column_integral(Yp.c.ρn_lcl)
        qrim_init = _column_integral(Yp.c.ρq_rim)
        Δt = FT(10)
        @. Yp.c.ρ += Δt * Yₜ_p3.c.ρ
        @. Yp.c.ρe_tot += Δt * Yₜ_p3.c.ρe_tot
        @. Yp.c.ρq_tot += Δt * Yₜ_p3.c.ρq_tot

        CA.step_vertical_scalar_transport!(Yp, pp, Δt, cache_mp)
        ρp_new = vec(_host_arr(Yp.c.ρ))
        nl_spec = vec(_host_arr(Yp.c.ρn_lcl)) ./ ρp_new
        nr_spec = vec(_host_arr(Yp.c.ρn_rai)) ./ ρp_new
        ni_spec = vec(_host_arr(Yp.c.ρn_ice)) ./ ρp_new
        qrim_spec = vec(_host_arr(Yp.c.ρq_rim)) ./ ρp_new
        brim_spec = vec(_host_arr(Yp.c.ρb_rim)) ./ ρp_new
        @test abs(nl_spec[11]) > FT(1e-4) * FT(1e7)
        @test abs(qrim_spec[13]) > FT(1e-4) * FT(1e-4)
        if eh_mode === :vanleer_limiter
            @test minimum(nl_spec) >= -FT(1e-8)
            @test maximum(nl_spec) <= FT(1e7) * (one(FT) + FT(1e-13))
            @test minimum(qrim_spec) >= -FT(1e-18)
            @test maximum(qrim_spec) <= FT(1e-4) * (one(FT) + FT(1e-13))
        end
        @test maximum(abs, nr_spec ./ FT(1e3) .- one(FT)) <= FT(1e-13)
        @test maximum(abs, ni_spec ./ FT(2e5) .- one(FT)) <= FT(1e-13)
        @test maximum(abs, brim_spec ./ FT(2e-7) .- one(FT)) <= FT(1e-13)
        @test isapprox(_column_integral(Yp.c.ρn_lcl), nl_init; rtol = FT(1e-13))
        @test isapprox(_column_integral(Yp.c.ρq_rim), qrim_init; rtol = FT(1e-13))
    end

    # 3. 3D BoxGrid zero-allocation check (SSP333 explicit-diagonal recording
    #    and vertical scalar transport).
    if ClimaComms.device(ctx) isa ClimaComms.AbstractCPUDevice
        box_grid = CA.BoxGrid(
            FT;
            context = ctx,
            x_elem = 2,
            y_elem = 2,
            x_max = 20e3,
            y_max = 20e3,
            z_elem = 8,
            z_max = 10e3,
            nh_poly = 2,
            topography = CA.CosineTopography{3, FT}(; h_max = 500.0, λ = 20e3),
        )
        box_model = CA.AtmosModel(
            box_grid;
            params,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            chemistry_model = CA.GasPhaseChem(),
            disable_surface_flux_tendency = true,
        )
        mktempdir() do tmpdir
            sim_box = CA.AtmosSimulation(
                box_model;
                dt = FT(10),
                t_end = FT(20),
                ode_config = _imex_ssp_alg(CTS.SSP333()),
                output_dir = joinpath(tmpdir, "box_alloc"),
                default_callbacks = false,
                post_step_hook! = CA.step_vertical_scalar_transport!,
            )
            CTS.step!(sim_box.integrator)
            Yb = sim_box.integrator.u
            pb = sim_box.integrator.p
            cb = sim_box.integrator.cache
            CA.maybe_record_explicit_imp_stage_energy!(cb.T_imp[1], Yb, pb)
            @test _alloc_record_exp_imp(cb.T_imp[1], Yb, pb) == 0
            CA.step_vertical_scalar_transport!(Yb, pb, FT(10), cb)
            @test _alloc_vt(Yb, pb, FT(10), cb) == 0
            @test CA.post_step_callback!(sim_box.integrator) === nothing
        end
    end

    # 4. DryModel + PrognosticEDMFX (grid-mean scalar conservation, `sgsʲs`
    #    invariance, `Float32`, and 0 CPU allocations).
    edmf_grid = CA.ColumnGrid(
        FT;
        context = ctx,
        z_elem = 32,
        z_max = 6400.0,
        z_stretch = false,
    )
    mktempdir() do tmpdir
        # 4a. DryModel regression test (no `ρq_tot` / `sgsʲ.q_tot` present).
        dry_model = CA.AtmosModel(
            edmf_grid;
            params,
            microphysics_model = CA.DryModel(),
            disable_surface_flux_tendency = true,
        )
        sim_dry = CA.AtmosSimulation(
            dry_model;
            dt = FT(10),
            t_end = FT(20),
            ode_config = _imex_ssp_alg(CTS.SSP333()),
            output_dir = joinpath(tmpdir, "dry_vt"),
            default_callbacks = false,
            post_step_hook! = CA.step_vertical_scalar_transport!,
        )
        CTS.step!(sim_dry.integrator)
        @test all(isfinite, _host_arr(sim_dry.integrator.u.c.ρe_tot))
        if ClimaComms.device(ctx) isa ClimaComms.AbstractCPUDevice
            CA.step_vertical_scalar_transport!(
                sim_dry.integrator.u,
                sim_dry.integrator.p,
                FT(10),
                sim_dry.integrator.cache,
            )
            @test _alloc_vt(
                sim_dry.integrator.u,
                sim_dry.integrator.p,
                FT(10),
                sim_dry.integrator.cache,
            ) == 0
        end

        # 4b. PrognosticEDMFX + 2M (`n_lclʲ`, `n_raiʲ` in `sgsʲ.extra`):
        #     `step_vertical_scalar_transport!` advances grid-mean scalars
        #     conservatively while leaving `sgsʲs` unchanged with zero CPU
        #     allocations.
        edmf_model = CA.AtmosModel(
            edmf_grid;
            params,
            microphysics_model = CA.NonEquilibriumMicrophysics2M(),
            turbconv = CA.AtmosTurbconv(;
                turbconv_model = CA.PrognosticEDMFX{1, true}(FT(0.05)),
                edmfx_model = CA.EDMFXModel(;
                    entr_model = CA.InvZEntrainment(),
                    detr_model = CA.BuoyancyVelocityDetrainment(),
                    sgs_mass_flux = true,
                    sgs_diffusive_flux = false,
                    nh_pressure = false,
                    vertical_diffusion = false,
                    scale_blending_method = CA.SmoothMinimumBlending(),
                ),
            ),
            disable_surface_flux_tendency = true,
        )
        sim_ed = CA.AtmosSimulation(
            edmf_model;
            dt = FT(10),
            t_end = FT(20),
            ode_config = _imex_ssp_alg(CTS.SSP333()),
            output_dir = joinpath(tmpdir, "edmf_vt"),
            default_callbacks = false,
            post_step_hook! = CA.step_vertical_scalar_transport!,
        )
        Yed = sim_ed.integrator.u
        ped = sim_ed.integrator.p
        ced = sim_ed.integrator.cache
        z_c_ed = Fields.coordinate_field(Yed.c).z
        z_f_ed = Fields.coordinate_field(Yed.f).z
        @. Yed.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(
                ifelse(
                    (z_f_ed > FT(0)) & (z_f_ed < FT(6400)),
                    FT(15.0) * sin(FT(π) * z_f_ed / FT(6400)),
                    FT(0.0),
                ),
            ),
        )
        CTS.step!(sim_ed.integrator)

        a1_prof = @. FT(0.02) + FT(0.08) * (z_c_ed / FT(6400))
        qt_ed = @. ifelse(z_c_ed < FT(3200), FT(8e-3), FT(1e-3))
        ql_ed = @. ifelse(z_c_ed < FT(3200), FT(8e-4), FT(0))
        nl_ed = @. ifelse(z_c_ed < FT(3200), FT(5e7), FT(0))
        nr_ed = @. ifelse(z_c_ed < FT(2400), FT(2e3), FT(0))
        mse_const = FT(3.15e5)
        @. Yed.c.sgsʲs.:(1).ρa = a1_prof * Yed.c.ρ
        @. Yed.c.sgsʲs.:(1).mse = mse_const
        @. Yed.c.sgsʲs.:(1).q_tot = qt_ed
        @. Yed.c.sgsʲs.:(1).q_lcl = ql_ed
        @. Yed.c.sgsʲs.:(1).q_icl = FT(0)
        @. Yed.c.sgsʲs.:(1).q_rai = FT(0)
        @. Yed.c.sgsʲs.:(1).q_sno = FT(0)
        @. Yed.c.sgsʲs.:(1).n_lcl = nl_ed
        @. Yed.c.sgsʲs.:(1).n_rai = nr_ed

        ρa_pre = copy(_host_arr(Yed.c.sgsʲs.:(1).ρa))
        mse_pre = copy(_host_arr(Yed.c.sgsʲs.:(1).mse))
        mask_ed_pre = ped.scratch.stage_entry_χ_e_mask[]
        Yed_none = deepcopy(Yed)
        qt1_pre = copy(_host_arr(Yed.c.sgsʲs.:(1).q_tot))
        ql1_pre = copy(_host_arr(Yed.c.sgsʲs.:(1).q_lcl))
        nl1_pre = copy(_host_arr(Yed.c.sgsʲs.:(1).n_lcl))
        nr1_pre = copy(_host_arr(Yed.c.sgsʲs.:(1).n_rai))
        gm_qt_pre = sum(Yed.c.ρq_tot)
        gm_ql_pre = sum(Yed.c.ρq_lcl)

        CA.step_vertical_scalar_transport!(Yed, ped, FT(10), ced)

        @test _host_arr(Yed.c.sgsʲs.:(1).ρa) == ρa_pre
        @test _host_arr(Yed.c.sgsʲs.:(1).mse) == mse_pre
        @test _host_arr(Yed.c.sgsʲs.:(1).q_tot) == qt1_pre
        @test _host_arr(Yed.c.sgsʲs.:(1).q_lcl) == ql1_pre
        @test _host_arr(Yed.c.sgsʲs.:(1).n_lcl) == nl1_pre
        @test _host_arr(Yed.c.sgsʲs.:(1).n_rai) == nr1_pre
        @test minimum(_host_arr(Yed.c.ρq_tot)) >= -FT(1e-15)
        @test minimum(_host_arr(Yed.c.ρq_lcl)) >= -FT(1e-15)
        @test isapprox(sum(Yed.c.ρq_tot), gm_qt_pre; rtol = FT(1e-13))
        @test isapprox(sum(Yed.c.ρq_lcl), gm_ql_pre; rtol = FT(1e-13))

        if ClimaComms.device(ctx) isa ClimaComms.AbstractCPUDevice
            CA.step_vertical_scalar_transport!(Yed, ped, FT(10), ced)
            @test _alloc_vt(Yed, ped, FT(10), ced) == 0
        end

        # 4c. `:none` draft upwinding leaves intensive draft scalars untouched
        #     while `Float32` PrognosticEDMFX stays finite and non-negative.
        p_none = _with_cache_atmos(
            ped,
            CA.AtmosModel(
                ped.atmos;
                numerics = CA.AtmosNumerics(;
                    edmfx_mse_humidity_upwinding = :none,
                    edmfx_tracer_upwinding = :none,
                ),
            ),
        )
        p_none.scratch.stage_entry_χ_e_mask[] = mask_ed_pre
        qt_before_none = copy(_host_arr(Yed_none.c.sgsʲs.:(1).q_tot))
        CA.step_vertical_scalar_transport!(Yed_none, p_none, FT(10), ced)
        @test _host_arr(Yed_none.c.sgsʲs.:(1).q_tot) == qt_before_none

        FT32 = Float32
        grid32 = CA.ColumnGrid(
            FT32;
            context = ctx,
            z_elem = 16,
            z_max = 6400.0,
            z_stretch = false,
        )
        model32 = CA.AtmosModel(
            grid32;
            params = CA.ClimaAtmosParameters(FT32),
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            turbconv = CA.AtmosTurbconv(;
                turbconv_model = CA.PrognosticEDMFX{1, true}(FT32(0.05)),
                edmfx_model = CA.EDMFXModel(;
                    entr_model = CA.InvZEntrainment(),
                    detr_model = CA.BuoyancyVelocityDetrainment(),
                    sgs_mass_flux = true,
                    sgs_diffusive_flux = false,
                    nh_pressure = false,
                    vertical_diffusion = false,
                    scale_blending_method = CA.SmoothMinimumBlending(),
                ),
            ),
            disable_surface_flux_tendency = true,
        )
        sim32 = CA.AtmosSimulation(
            model32;
            dt = FT32(10),
            t_end = FT32(20),
            ode_config = _imex_ssp_alg(CTS.SSP333()),
            output_dir = joinpath(tmpdir, "edmf_f32"),
            default_callbacks = false,
            post_step_hook! = CA.step_vertical_scalar_transport!,
        )
        z_f32 = Fields.coordinate_field(sim32.integrator.u.f).z
        @. sim32.integrator.u.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(
                ifelse(
                    (z_f32 > FT32(0)) & (z_f32 < FT32(6400)),
                    FT32(12) * sin(FT32(π) * z_f32 / FT32(6400)),
                    FT32(0),
                ),
            ),
        )
        CTS.step!(sim32.integrator)
        @test all(isfinite, _host_arr(sim32.integrator.u.c.sgsʲs.:(1).mse))
        @test minimum(_host_arr(sim32.integrator.u.c.sgsʲs.:(1).q_tot)) >= -FT32(1e-7)
    end
end
