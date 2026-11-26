#=
julia +1.13 --project=.buildkite
using Revise; include("test/prognostic_equations/vertical_transport_step_tests.jl")
=#
include("vertical_transport_test_helpers.jl")

@testset "Duplicate-node steep C⁰ terrain F̃ closure and continuity" begin
    ctx = ClimaComms.SingletonCommsContext()

    for (grid_label, FT, ode_config, make_grid) in (
        (
            "Box3D steep CosineTopography + LinearWarp + stretched z",
            Float64,
            _imex_alg(CTS.ARS343()),
            FT -> CA.BoxGrid(
                FT;
                context = ctx,
                x_elem = 2,
                y_elem = 2,
                x_max = 40e3,
                y_max = 40e3,
                z_elem = 12,
                z_max = 20e3,
                nh_poly = 3,
                z_mesh = _stretched_z_mesh(FT, 12, 20e3),
                topography = CA.CosineTopography{3, FT}(;
                    h_max = 1800.0,
                    λ = 20e3,
                ),
                mesh_warp_type = CA.LinearWarp(),
            ),
        ),
        (
            "Deep-atmosphere SphereGrid DCMIP200 mountain + SLEVEWarp + stretched z",
            Float32,
            _imex_ssp_alg(CTS.SSP333()),
            FT -> CA.SphereGrid(
                FT;
                context = ctx,
                h_elem = 2,
                z_elem = 10,
                z_max = 24e3,
                nh_poly = 3,
                z_mesh = _stretched_z_mesh(FT, 10, 24e3),
                deep_atmosphere = true,
                topography = CA.DCMIP200Topography(),
                mesh_warp_type = CA.SLEVEWarp{FT}(),
            ),
        ),
    )
        @testset "$grid_label ($FT)" begin
            grid = make_grid(FT)
            params = CA.ClimaAtmosParameters(FT)
            model = CA.AtmosModel(
                grid;
                params,
                microphysics_model = CA.EquilibriumMicrophysics0M(),
                chemistry_model = CA.GasPhaseChem(),
                disable_surface_flux_tendency = true,
            )
            mktempdir() do tmpdir
                pre_dss_jump_Y = Ref(FT(NaN))
                pre_dss_jump_F = Ref(FT(NaN))
                raw_col_sum_rel = Ref(FT(NaN))
                proj_div_err_rel = Ref(FT(NaN))
                direct_div_err_rel = Ref(FT(NaN))
                bc_bot_max = Ref(FT(NaN))
                bc_top_max = Ref(FT(NaN))
                f_peak = Ref(FT(0))

                hook! = function (Y, p, t, Δt, integrator_cache)
                    # Column sum `Σ_k ᶜJ_k D_raw_k` pre-projection.
                    ᶜD_raw_chk = similar(Y.c.ρ)
                    CA._accumulate_stage_tendency!(
                        ᶜD_raw_chk,
                        p,
                        integrator_cache,
                        @name(ρ),
                    )
                    ᶜJ = Fields.local_geometry_field(Y.c).J
                    J_arr = _host_arr(ᶜJ)
                    D_raw_arr = _host_arr(ᶜD_raw_chk)
                    col_sums = sum(J_arr .* D_raw_arr; dims = 1)
                    col_norms = sum(J_arr .* abs.(D_raw_arr); dims = 1)
                    raw_col_sum_rel[] =
                        maximum(abs.(col_sums)) /
                        max(maximum(col_norms), eps(FT))

                    # Run post-ARK vertical scalar transport.
                    ᶠF̃ = CA.step_vertical_scalar_transport!(
                        Y,
                        p,
                        Δt,
                        integrator_cache,
                    )
                    @test ᶠF̃ === p.scratch.ᶠF̃

                    F_arr = _host_arr(ᶠF̃)
                    f_peak[] = maximum(abs, F_arr)
                    bc_bot_max[] = maximum(abs, view(F_arr, 1:1, :, :, :, :))
                    bc_top_max[] =
                        maximum(
                            abs,
                            view(F_arr, size(F_arr, 1):size(F_arr, 1), :, :, :, :),
                        )

                    # Check `ρ^{n+1} - ρ⁰ == -Δt ᶜadvdivᵥ(ᶠF̃)`.
                    ᶜdiv_F = similar(Y.c.ρ)
                    @. ᶜdiv_F = -CA.ᶜadvdivᵥ(ᶠF̃)
                    ᶜρ0_chk = similar(Y.c.ρ)
                    Δt_ft = FT(Δt)
                    DataLayouts.foreach_column(
                        ᶜρ0_chk,
                        Y.c.ρ,
                        ᶠF̃,
                        Fields.local_geometry_field(Y.c),
                        Fields.local_geometry_field(Y.f);
                        shmem = Val(:none),
                    ) do ρ0_c, ρ_c, F_c, lg_c, flg_c
                        ρ0_fv = Fields.field_values(ρ0_c)
                        ρ_fv = Fields.field_values(ρ_c)
                        F_fv = Fields.field_values(F_c)
                        lg_fv = Fields.field_values(lg_c)
                        flg_fv = Fields.field_values(flg_c)
                        for k in 1:Spaces.nlevels(axes(ρ_c))
                            ρ0_fv[k] = CA._vt_rho0_k(
                                ρ_fv,
                                F_fv,
                                lg_fv,
                                flg_fv,
                                Δt_ft,
                                k,
                            )
                        end
                    end
                    ρ0_arr = _host_arr(ᶜρ0_chk)
                    ρ_new_arr = _host_arr(Y.c.ρ)
                    div_arr = _host_arr(ᶜdiv_F)
                    proj_div_err_rel[] =
                        maximum(
                            abs,
                            (ρ_new_arr .- ρ0_arr) .- Δt .* div_arr,
                        ) / max(maximum(abs, Δt .* div_arr), eps(FT))
                    α_proj =
                        sum(Float64.(J_arr) .* Float64.(D_raw_arr); dims = 1) ./
                        sum(Float64.(J_arr) .* Float64.(ρ_new_arr); dims = 1)
                    D_proj_arr = D_raw_arr .- FT.(α_proj) .* ρ_new_arr
                    direct_div_err_rel[] =
                        maximum(abs, div_arr .- D_proj_arr) /
                        max(maximum(abs, D_proj_arr), eps(FT))

                    # Element-boundary continuity of `Y` AND `ᶠF̃` BEFORE
                    # `post_step_callback!` calls `dss!(Y, p, t)`.
                    Y_dss = copy(Y)
                    CA.dss!(Y_dss, p, t)
                    pre_dss_jump_Y[] =
                        maximum(
                            abs,
                            _host_arr(Y.c.ρe_tot) .- _host_arr(Y_dss.c.ρe_tot),
                        ) / maximum(abs, _host_arr(Y.c.ρe_tot))

                    ᶠF_scalar = similar(p.scratch.ᶠtemp_scalar)
                    @. ᶠF_scalar = ᶠF̃.components.data.:1
                    ᶠF_dss = copy(ᶠF_scalar)
                    Spaces.weighted_dss!(ᶠF_dss)
                    pre_dss_jump_F[] =
                        maximum(
                            abs,
                            _host_arr(ᶠF_scalar) .- _host_arr(ᶠF_dss),
                        ) / max(f_peak[], eps(FT))
                end

                sim = CA.AtmosSimulation(
                    model;
                    dt = FT(20),
                    t_end = FT(20),
                    ode_config,
                    output_dir = joinpath(tmpdir, "terrain_run"),
                    default_callbacks = false,
                    post_step_hook! = hook!,
                )
                Y = sim.integrator.u
                p = sim.integrator.p
                @test sim.integrator.sol.prob.f.T_post_imp! === nothing

                # Horizontal wind, vertical wave, and moisture/tracer
                # gradients over terrain drive non-zero `T_imp` tendencies.
                z_c = Fields.coordinate_field(Y.c).z
                z_f = Fields.coordinate_field(Y.f).z
                z_max = FT(Spaces.z_max(axes(Y.f)))
                ᶜlg = Fields.local_geometry_field(Y.c)
                @. Y.c.uₕ = Geometry.Covariant12Vector(
                    Geometry.UVVector(FT(12), FT(5)),
                    ᶜlg,
                )
                @. Y.f.u₃ = Geometry.Covariant3Vector(
                    Geometry.WVector(
                        FT(1.5) * sin(FT(π) * z_f / z_max),
                    ),
                )
                @. Y.c.ρq_tot =
                    Y.c.ρ * FT(1e-4) * exp(-z_c / FT(3000))
                @. Y.c.ρq_gas_A =
                    Y.c.ρ * (FT(1) + FT(0.5) * cos(FT(2π) * z_c / z_max))
                CA.dss!(Y, p, sim.integrator.t)
                CA.set_precomputed_quantities!(Y, p, sim.integrator.t)

                mass_init = _column_integral(Y.c.ρ)
                qt_init = _column_integral(Y.c.ρq_tot)
                tr_init = _column_integral(Y.c.ρq_gas_A)
                e_init = _column_integral(Y.c.ρe_tot)

                CA.solve_atmos!(sim)

                rtol_ft = FT === Float32 ? FT(2e-5) : FT(1e-12)
                @test f_peak[] > FT(1e-5)
                @test bc_bot_max[] == zero(FT)
                @test bc_top_max[] == zero(FT)
                @test raw_col_sum_rel[] <=
                      (FT === Float32 ? FT(1e-4) : FT(1e-12))
                @test proj_div_err_rel[] <=
                      (FT === Float32 ? FT(5e-5) : FT(1e-13))
                @test direct_div_err_rel[] <=
                      (FT === Float32 ? FT(1e-6) : FT(1e-13))
                @test pre_dss_jump_Y[] <=
                      (FT === Float32 ? FT(5e-6) : FT(1e-13))
                @test pre_dss_jump_F[] <=
                      (FT === Float32 ? FT(5e-6) : FT(1e-13))

                @test isapprox(_column_integral(Y.c.ρ), mass_init; rtol = rtol_ft)
                @test isapprox(_column_integral(Y.c.ρq_tot), qt_init; rtol = rtol_ft)
                @test isapprox(_column_integral(Y.c.ρq_gas_A), tr_init; rtol = rtol_ft)
                @test isapprox(_column_integral(Y.c.ρe_tot), e_init; rtol = rtol_ft)
            end
        end
    end
end

@testset "χ ≡ 1 preservation including top cell on ARS343 and SSP333 (Float32 & Float64)" begin
    ctx = ClimaComms.SingletonCommsContext()

    for (FT, alg_label, use_ssprk) in (
        (Float64, "ARS343", false),
        (Float32, "SSP333", true),
    )
        tol = FT === Float32 ? FT(1e-6) : FT(1e-14)
        grid = CA.ColumnGrid(
            FT;
            context = ctx,
            z_elem = 32,
            z_max = 35000.0,
            z_mesh = _stretched_z_mesh(FT, 32, 35000.0),
        )
        params = CA.ClimaAtmosParameters(FT)
        numerics = CA.AtmosNumerics(;
            energy_humidity_upwinding = :vanleer_limiter,
            tracer_upwinding = :vanleer_limiter,
        )
        model = CA.AtmosModel(
            grid;
            params,
            numerics,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            chemistry_model = CA.GasPhaseChem(),
            advection_test = true,
            disable_surface_flux_tendency = true,
        )
        mktempdir() do tmpdir
            recorded_masks = UInt16[]
            Y_pre = Ref{Any}(nothing)
            mask_pre = Ref(UInt16(0))
            hook! = function (Y, p, _t, Δt, integrator_cache)
                push!(recorded_masks, p.scratch.stage_entry_χ_e_mask[])
                if FT === Float64
                    Y_pre[] = deepcopy(Y)
                    mask_pre[] = p.scratch.stage_entry_χ_e_mask[]
                end
                CA.step_vertical_scalar_transport!(
                    Y,
                    p,
                    Δt,
                    integrator_cache,
                )
            end

            integrator = if use_ssprk
                _make_ssprk_integrator(
                    model,
                    CTS.SSP333(),
                    FT(40),
                    FT(80);
                    post_step_hook! = hook!,
                )
            else
                sim = CA.AtmosSimulation(
                    model;
                    dt = FT(40),
                    t_end = FT(80),
                    ode_config = _imex_alg(CTS.ARS343()),
                    output_dir = joinpath(tmpdir, "chi1_$(alg_label)_vanleer_limiter"),
                    default_callbacks = false,
                    post_step_hook! = hook!,
                )
                sim.integrator
            end
            Y = integrator.u
            p = integrator.p
            z_f = Fields.coordinate_field(Y.f).z
            z_max = FT(Spaces.z_max(axes(Y.f)))

            # Vertical velocity spanning troposphere and stratosphere.
            @. Y.f.u₃ = Geometry.Covariant3Vector(
                Geometry.WVector(
                    FT(8.0) * sin(FT(π) * z_f / z_max),
                ),
            )
            # Subsaturated uniform `q_tot` with zero condensates keeps
            # 1M source and sedimentation tendencies identically zero.
            q0 = FT(1e-5)
            @. Y.c.ρq_tot = q0 * Y.c.ρ
            for cname in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
                fill!(getproperty(Y.c, cname), zero(FT))
            end
            @. Y.c.ρq_gas_A = Y.c.ρ
            CA.set_precomputed_quantities!(Y, p, integrator.t)

            ρ_before = copy(_host_arr(Y.c.ρ))
            CTS.solve!(integrator)
            ρ_after = _host_arr(Y.c.ρ)
            @test maximum(abs, ρ_after .- ρ_before) > FT(1e-4)

            # Stage-entry `ê_i` bitmask covers stages 2..4 (`0x000e`)
            # on `ARS343` and stages 1..3 (`0x0007`) on `SSP333`.
            expected_mask = alg_label == "ARS343" ? UInt16(0x000e) : UInt16(0x0007)
            @test all(==(expected_mask), recorded_masks)

            qt_spec = vec(_host_arr(Y.c.ρq_tot) ./ ρ_after) ./ q0
            tr_spec = vec(_host_arr(Y.c.ρq_gas_A) ./ ρ_after)

            @test maximum(abs, qt_spec .- one(FT)) <= tol
            @test abs(qt_spec[end] - one(FT)) <= tol
            @test maximum(abs, tr_spec .- one(FT)) <= tol
            @test abs(tr_spec[end] - one(FT)) <= tol

            if FT === Float64
                for tr_upw in (:third_order, :first_order, :none)
                    Y_try = deepcopy(Y_pre[])
                    p_upw = _with_cache_atmos(
                        p,
                        CA.AtmosModel(
                            p.atmos;
                            numerics = CA.AtmosNumerics(;
                                energy_humidity_upwinding = :vanleer_limiter,
                                tracer_upwinding = tr_upw,
                            ),
                        ),
                    )
                    p_upw.scratch.stage_entry_χ_e_mask[] = mask_pre[]
                    CA.step_vertical_scalar_transport!(
                        Y_try,
                        p_upw,
                        FT(40),
                        integrator.cache,
                    )
                    ρ_try = _host_arr(Y_try.c.ρ)
                    qt_try = vec(_host_arr(Y_try.c.ρq_tot) ./ ρ_try) ./ q0
                    tr_try = vec(_host_arr(Y_try.c.ρq_gas_A) ./ ρ_try)
                    @test maximum(abs, qt_try .- one(FT)) <= tol
                    @test abs(qt_try[end] - one(FT)) <= tol
                    @test maximum(abs, tr_try .- one(FT)) <= tol
                    @test abs(tr_try[end] - one(FT)) <= tol
                end
            end
        end
    end
end

@testset "Positivity, monotonicity, C ∈ [0.02, 20] spike, 3D undershoots, and counter-monotone q_vap" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()
    nz = 120
    z_max = 12000.0
    Δz = z_max / nz

    # Per-stage `ARS343` backward-Euler upwind goes `-12.8%` negative on a unit
    # spike for `C >= 5`, whereas the post-ARK M-matrix solve stays positive.
    ars_tab = CTS.tableau(CTS.ARS343())
    for (C_chk, neg_bound) in ((5.0, -0.05), (10.0, -0.075), (20.0, -0.095))
        u_per_stage = _per_stage_ark_upwind_spike(ars_tab, nz, 50, C_chk)
        @test minimum(u_per_stage) < neg_bound
    end

    grid = CA.ColumnGrid(
        FT;
        context = ctx,
        z_elem = nz,
        z_max = z_max,
        z_stretch = false,
    )
    params = CA.ClimaAtmosParameters(FT)
    model = CA.AtmosModel(
        grid;
        params,
        microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        chemistry_model = CA.GasPhaseChem(),
        advection_test = true,
        disable_surface_flux_tendency = true,
    )

    dt_step = FT(10)
    for (alg_label, ode_config) in (
        ("ARS343", _imex_alg(CTS.ARS343())),
        ("SSP333", _imex_ssp_alg(CTS.SSP333())),
    )
        mktempdir() do tmpdir
            sim = CA.AtmosSimulation(
                model;
                dt = dt_step,
                t_end = dt_step,
                ode_config,
                output_dir = joinpath(tmpdir, "pos_$alg_label"),
                default_callbacks = false,
                post_step_hook! = CA.step_vertical_scalar_transport!,
            )
            integrator = sim.integrator
            Y = integrator.u
            p = integrator.p
            z_f_vec = vec(_host_arr(Fields.coordinate_field(Y.f).z))

            # Velocity profile with 40-cell boundary ramps (`|Δt ∂_z w| <= 0.5`)
            # and uniform interior `w = C * Δz / dt` on cells `41..80`, keeping
            # `ρ⁰ > 0` and `ρ^{n+1} > 0` for all `C ∈ [0.02, 20]`.
            function set_courant_velocity!(C_val)
                w_int = C_val * FT(Δz) / dt_step
                w_vals = [
                    zf <= 40 * Δz ? w_int * (zf / FT(40 * Δz)) :
                    (
                        zf <= 80 * Δz ? w_int :
                        w_int * ((FT(z_max) - zf) / FT(40 * Δz))
                    ) for zf in z_f_vec
                ]
                _set_col_w!(Y.f.u₃, w_vals)
                return nothing
            end

            function init_spike_square_state!(C_val)
                CTS.reinit!(integrator, Y)
                fill!(Y.c.ρ, FT(1))
                @. Y.c.ρe_tot = Y.c.ρ * (FT(2.5e5) + p.core.ᶜΦ)
                for cname in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
                    fill!(getproperty(Y.c, cname), FT(0))
                end
                # Unit spike on `q_tot` (at k = 48, subsaturated) and square
                # step on `q_gas_A` (on k = 45:55).
                q_scale = FT(1e-4)
                _set_col_scalar!(
                    Y.c.ρq_tot,
                    [k == 48 ? q_scale : zero(FT) for k in 1:nz],
                )
                _set_col_scalar!(
                    Y.c.ρq_gas_A,
                    [(45 <= k <= 55) ? one(FT) : zero(FT) for k in 1:nz],
                )
                set_courant_velocity!(C_val)
                return q_scale
            end

            function check_spike_square_result!(q_scale, qt_col_before, tr_col_before)
                ρ_vec = vec(_host_arr(Y.c.ρ))
                qt_end = vec(_host_arr(Y.c.ρq_tot)) ./ (ρ_vec .* q_scale)
                tr_end = vec(_host_arr(Y.c.ρq_gas_A)) ./ ρ_vec

                @test minimum(ρ_vec) > FT(0.4)
                @test minimum(qt_end) >= -FT(1e-14)
                @test maximum(qt_end) <= one(FT) + FT(1e-13)
                @test minimum(tr_end) >= -FT(1e-14)
                @test maximum(tr_end) <= one(FT) + FT(1e-13)

                # Spike unimodality on both flanks inside the uniform-`w`
                # interior (`42:77`, clear of the `k = 80..81` convergence kink).
                peak_idx = argmax(qt_end)
                diffs_left = diff(qt_end[42:peak_idx])
                diffs_right = diff(qt_end[peak_idx:77])
                @test all(>=(-FT(1e-13)), diffs_left)
                @test all(<=(FT(1e-13)), diffs_right)

                @test isapprox(
                    _column_integral(Y.c.ρq_tot),
                    qt_col_before;
                    rtol = FT(1e-13),
                )
                @test isapprox(
                    _column_integral(Y.c.ρq_gas_A),
                    tr_col_before;
                    rtol = FT(1e-13),
                )
            end

            function run_direct_wave_step!()
                CA.set_implicit_precomputed_quantities_core!(Y, p, integrator.t)
                for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
                    fill!(getproperty(p.precomputed, w_name), FT(0))
                end
                Yₜ_wave = zero(Y)
                CA.implicit_vertical_advection_tendency!(
                    Yₜ_wave,
                    Y,
                    p,
                    integrator.t,
                )
                p.scratch.stage_entry_χ_e_mask[] = UInt16(0)
                CA.foreach_b_imp_stage(integrator.cache) do i, _w_i, T_imp_i
                    T_imp_i .= Yₜ_wave
                    CA._write_stage_entry_χ_e!(
                        p.scratch.ᶠstage_entry_χ_e[i],
                        Y,
                        p.core.ᶜΦ,
                    )
                    p.scratch.stage_entry_χ_e_mask[] |= UInt16(1) << (i - 1)
                end
                @. Y.c.ρ += dt_step * Yₜ_wave.c.ρ
                @. Y.c.ρe_tot += dt_step * Yₜ_wave.c.ρe_tot
                @. Y.c.ρq_tot += dt_step * Yₜ_wave.c.ρq_tot
                @. Y.c.ρq_gas_A += dt_step * Yₜ_wave.c.ρq_gas_A
                CA.step_vertical_scalar_transport!(
                    Y,
                    p,
                    dt_step,
                    integrator.cache,
                )
            end

            # 1a. Full `CTS.step!` over explicit-continuity-safe `C <= 0.8`.
            for C_val in FT.((0.02, 0.25, 0.8))
                q_scale = init_spike_square_state!(C_val)
                CA.set_precomputed_quantities!(Y, p, integrator.t)
                qt_col_before = _column_integral(Y.c.ρq_tot)
                tr_col_before = _column_integral(Y.c.ρq_gas_A)
                CTS.step!(integrator)
                @test float(integrator.dt) == dt_step
                check_spike_square_result!(q_scale, qt_col_before, tr_col_before)
            end

            # 1b. Direct `step_vertical_scalar_transport!` across `C ∈ [0.02, 20]`.
            for C_val in FT.((0.02, 0.25, 0.9, 1.5, 5.0, 20.0))
                q_scale = init_spike_square_state!(C_val)
                qt_col_before = _column_integral(Y.c.ρq_tot)
                tr_col_before = _column_integral(Y.c.ρq_gas_A)
                run_direct_wave_step!()
                check_spike_square_result!(q_scale, qt_col_before, tr_col_before)
            end

            # 2. Incoming 3D horizontal undershoot (`min χ⁰ < 0`): never clipped
            # non-conservatively, and `min χ^{n+1} >= min(0, min χ⁰)`.
            for (use_cts, C_vals) in (
                (true, FT.((0.3, 0.8))),
                (false, FT.((0.3, 1.0, 8.0, 20.0))),
            )
                for C_val in C_vals
                    CTS.reinit!(integrator, Y)
                    fill!(Y.c.ρ, FT(1))
                    @. Y.c.ρe_tot = Y.c.ρ * (FT(2.5e5) + p.core.ᶜΦ)
                    fill!(Y.c.ρq_tot, FT(0))
                    for cname in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
                        fill!(getproperty(Y.c, cname), FT(0))
                    end
                    _set_col_scalar!(
                        Y.c.ρq_gas_A,
                        [
                            k == 50 ? FT(-0.025) :
                            (51 <= k <= 56 ? FT(0.5) : zero(FT)) for k in 1:nz
                        ],
                    )
                    set_courant_velocity!(C_val)
                    tr_col_before = _column_integral(Y.c.ρq_gas_A)
                    if use_cts
                        CA.set_precomputed_quantities!(Y, p, integrator.t)
                        CTS.step!(integrator)
                        @test float(integrator.dt) == dt_step
                    else
                        run_direct_wave_step!()
                    end

                    ρ_vec = vec(_host_arr(Y.c.ρ))
                    tr_end = vec(_host_arr(Y.c.ρq_gas_A)) ./ ρ_vec
                    @test minimum(tr_end) >= FT(-0.025) - FT(1e-14)
                    @test maximum(tr_end) <= FT(0.5) + FT(1e-13)
                    @test isapprox(
                        _column_integral(Y.c.ρq_gas_A),
                        tr_col_before;
                        atol = FT(1e-12),
                    )
                end
            end

            # 3. Counter-monotone `q_tot` and `q_lcl` profiles where `q_vap`
            # drops to `0` while `q_tot` is linear.
            for C_val in FT.((0.25, 0.6, 1.5, 5.0))
                fill!(Y.c.ρ, FT(1))
                @. Y.c.ρe_tot = Y.c.ρ * (FT(2.5e5) + p.core.ᶜΦ)
                for cname in (:ρq_icl, :ρq_rai, :ρq_sno)
                    fill!(getproperty(Y.c, cname), FT(0))
                end
                qt_prof = zeros(FT, nz)
                ql_prof = zeros(FT, nz)
                for k in 1:nz
                    s = clamp(k - 44, 0, 16)
                    qt_k = FT(2e-3) - FT(1e-4) * FT(s)
                    ql_k = k <= 50 ? qt_k : FT(0)
                    qt_prof[k] = qt_k
                    ql_prof[k] = ql_k
                end
                _set_col_scalar!(Y.c.ρq_tot, qt_prof)
                _set_col_scalar!(Y.c.ρq_lcl, ql_prof)
                # Pure continuity stage in `integrator.cache` isolates transport
                # of `q_lcl` and `q_vap` from 1M phase sources.
                set_courant_velocity!(C_val)
                CA.set_implicit_precomputed_quantities_core!(Y, p, integrator.t)
                for w_name in (:ᶜwₗ, :ᶜwᵢ, :ᶜwᵣ, :ᶜwₛ)
                    fill!(getproperty(p.precomputed, w_name), FT(0))
                end
                Yₜ_wave = zero(Y)
                CA.implicit_vertical_advection_tendency!(
                    Yₜ_wave,
                    Y,
                    p,
                    integrator.t,
                )
                # Advance `(ρ, ρe_tot, ρq_tot, ρq_lcl)` by central wave-core
                # tendency and record stage-entry `ê`.
                p.scratch.stage_entry_χ_e_mask[] = UInt16(0)
                CA.foreach_b_imp_stage(integrator.cache) do i, _w_i, T_imp_i
                    T_imp_i .= Yₜ_wave
                    CA._write_stage_entry_χ_e!(
                        p.scratch.ᶠstage_entry_χ_e[i],
                        Y,
                        p.core.ᶜΦ,
                    )
                    p.scratch.stage_entry_χ_e_mask[] |= UInt16(1) << (i - 1)
                end
                qt_init_int = _column_integral(Y.c.ρq_tot)
                ql_init_int = _column_integral(Y.c.ρq_lcl)
                @. Y.c.ρ += dt_step * Yₜ_wave.c.ρ
                @. Y.c.ρe_tot += dt_step * Yₜ_wave.c.ρe_tot
                @. Y.c.ρq_tot += dt_step * Yₜ_wave.c.ρq_tot
                @. Y.c.ρq_lcl += dt_step * Yₜ_wave.c.ρq_lcl

                # Unlimited third-order drives `q_vap < 0` on this profile at
                # `C_val < 1`, showing the shared `q_vap` ratio is active.
                if C_val == FT(0.25)
                    Y_unlim = copy(Y)
                    num_3rd = CA.AtmosNumerics(;
                        energy_humidity_upwinding = :third_order,
                    )
                    model_3rd = CA.AtmosModel(
                        grid;
                        params,
                        numerics = num_3rd,
                        microphysics_model = CA.NonEquilibriumMicrophysics1M(),
                        advection_test = true,
                        disable_surface_flux_tendency = true,
                    )
                    p_3rd = CA.build_cache(
                        Y_unlim,
                        model_3rd,
                        params,
                        dt_step,
                        DateTime(2010, 1, 1),
                        nothing;
                        ode_algo = ode_config,
                    )
                    p_3rd.scratch.post_step.cache_box.rc = integrator.cache
                    CA.step_vertical_scalar_transport!(
                        Y_unlim,
                        p_3rd,
                        dt_step,
                        integrator.cache,
                    )
                    q_vap_3rd =
                        vec(
                            _host_arr(Y_unlim.c.ρq_tot) .-
                            _host_arr(Y_unlim.c.ρq_lcl),
                        ) ./ vec(_host_arr(Y_unlim.c.ρ))
                    @test minimum(q_vap_3rd[44:64]) < -FT(1e-5)
                end

                CA.step_vertical_scalar_transport!(
                    Y,
                    p,
                    dt_step,
                    integrator.cache,
                )
                ρ_vec = vec(_host_arr(Y.c.ρ))
                qt_vec = vec(_host_arr(Y.c.ρq_tot)) ./ ρ_vec
                ql_vec = vec(_host_arr(Y.c.ρq_lcl)) ./ ρ_vec
                q_vap_vec = qt_vec .- ql_vec
                @test minimum(ql_vec) >= -FT(1e-16)
                @test minimum(q_vap_vec) >= -FT(1e-15)
                @test isapprox(
                    _column_integral(Y.c.ρq_tot),
                    qt_init_int;
                    rtol = FT(1e-13),
                )
                @test isapprox(
                    _column_integral(Y.c.ρq_lcl),
                    ql_init_int;
                    rtol = FT(1e-13),
                )
            end
        end
    end
end

@testset "Clear-air _fct_ratio(P <= 0) = 1 third-order convergence (uniform & stretched)" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()
    params = CA.ClimaAtmosParameters(FT)
    z_max = 4000.0
    w0 = 2.0

    for stretch in (false, true)
        errs = Float64[]
        nz_list = (32, 64, 128)
        for nz in nz_list
            grid =
                stretch ?
                CA.ColumnGrid(
                    FT;
                    context = ctx,
                    z_elem = nz,
                    z_max = z_max,
                    z_mesh = _stretched_z_mesh(FT, nz, z_max),
                ) :
                CA.ColumnGrid(
                    FT;
                    context = ctx,
                    z_elem = nz,
                    z_max = z_max,
                    z_stretch = false,
                )
            # 1M moist model in clear air (`ρq_lcl = ... = ρq_sno = 0`)
            # exercises `_vt_fct_ratio(0, Q) == 1` in the shared limiter.
            model = CA.AtmosModel(
                grid;
                params,
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
                advection_test = true,
                disable_surface_flux_tendency = true,
            )
            # Keep `w * dt` tiny (`dt ∝ nz^-3`) so `O(dt)` backward-Euler time
            # error is negligible and spatial `O(Δz³)` truncation is isolated.
            dt_c = FT(1e-3 * (32 / nz)^3)
            ode_config = _imex_alg(CTS.ARS343())
            mktempdir() do tmpdir
                sim = CA.AtmosSimulation(
                    model;
                    dt = dt_c,
                    t_end = dt_c,
                    ode_config,
                    output_dir = joinpath(tmpdir, "conv_$(stretch)_$nz"),
                    default_callbacks = false,
                    post_step_hook! = CA.step_vertical_scalar_transport!,
                )
                integrator = sim.integrator
                Y = integrator.u
                p = integrator.p

                z_f_vec = vec(_host_arr(Fields.coordinate_field(Y.f).z))
                z_c_field = Fields.coordinate_field(Y.c).z
                fill!(Y.c.ρ, FT(1))
                @. Y.c.ρe_tot =
                    Y.c.ρ * (FT(2.5e5) + FT(15.0) * z_c_field + p.core.ᶜΦ)
                for cname in (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
                    fill!(getproperty(Y.c, cname), FT(0))
                end

                # Cell-averaged compactly-supported pulse in `[800, 3200]` m
                # compared against the analytic flux difference.
                q_exact_fn(z) =
                    (800.0 <= z <= 3200.0) ?
                    1e-4 * (1.0 + sin(π * (z - 800.0) / 2400.0)^4) : 1e-4
                q0_vec = zeros(FT, nz)
                dqdt_exact = zeros(FT, nz)
                for k in 1:nz
                    za = z_f_vec[k]
                    zb = z_f_vec[k + 1]
                    hk = zb - za
                    # 4-point Gauss-Legendre quadrature for cell average.
                    nodes = (
                        -0.8611363115940526,
                        -0.33998104358485626,
                        0.33998104358485626,
                        0.8611363115940526,
                    )
                    weights = (
                        0.34785484513745385,
                        0.6521451548625461,
                        0.6521451548625461,
                        0.34785484513745385,
                    )
                    zm = 0.5 * (za + zb)
                    avg = 0.0
                    for (ξ, wt) in zip(nodes, weights)
                        avg += 0.5 * wt * q_exact_fn(zm + 0.5 * hk * ξ)
                    end
                    q0_vec[k] = avg
                    dqdt_exact[k] = -w0 * (q_exact_fn(zb) - q_exact_fn(za)) / hk
                end
                _set_col_scalar!(Y.c.ρq_tot, q0_vec)

                w_f_vals = [
                    zf <= 400.0 ? w0 * (zf / 400.0) :
                    (zf <= 3600.0 ? w0 : w0 * ((z_max - zf) / 400.0)) for
                    zf in z_f_vec
                ]
                _set_col_w!(Y.f.u₃, w_f_vals)
                CA.set_precomputed_quantities!(Y, p, integrator.t)
                CTS.step!(integrator)

                ρ_after = vec(_host_arr(Y.c.ρ))
                q_after = vec(_host_arr(Y.c.ρq_tot)) ./ ρ_after
                dqdt_num = (q_after .- q0_vec) ./ dt_c

                # Evaluate L∞ error over the monotone rising flank `[1000, 1750]` m
                # where `w == w0` and neither `q_tot` nor `χ_e` has an extremum.
                mask = [
                    (z_f_vec[k] >= 1000.0) && (z_f_vec[k + 1] <= 1750.0) for
                    k in 1:nz
                ]
                push!(errs, maximum(abs, dqdt_num[mask] .- dqdt_exact[mask]))
            end
        end
        rate_1 = log2(errs[1] / errs[2])
        rate_2 = log2(errs[2] / errs[3])
        @test rate_1 >= 2.85
        @test rate_2 >= 2.92
    end
end
