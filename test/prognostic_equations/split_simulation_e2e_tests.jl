#=
julia --project=.buildkite
using Revise; include("test/prognostic_equations/split_simulation_e2e_tests.jl")
=#
include("split_simulation_e2e_test_helpers.jl")

@testset "Post-ARK stage-cache trimming and pre-column thermo refresh" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()
    mktempdir() do tmpdir
        cfg = CA.AtmosConfig(
            Dict{String, Any}(
                "config" => "column",
                "FLOAT_TYPE" => "Float64",
                "z_elem" => 12,
                "z_max" => 6000.0,
                "z_stretch" => false,
                "dt" => "20secs",
                "t_end" => "40secs",
                "output_default_diagnostics" => false,
                "output_dir" => joinpath(tmpdir, "cfg_default"),
            );
            job_id = "split_e2e_default_cfg",
            comms_ctx = ctx,
        )
        sim_cfg = CA.get_simulation(cfg)
        @test sim_cfg.integrator.p.scratch.post_step.in_stage_loop === true
        @test CA.is_stage_cache_update(
            sim_cfg.integrator.cache.U,
            sim_cfg.integrator.p,
        )
        @test !CA.is_stage_cache_update(
            sim_cfg.integrator.u,
            sim_cfg.integrator.p,
        )

        # Stage-cache trimming skips surface conditions, vertical EDMF closures,
        # and microphysics caches, while updating core/EDMF thermo and
        # horizontal Smagorinsky-Lilly diffusivities.
        box_grid = CA.BoxGrid(
            FT;
            context = ctx,
            x_elem = 2,
            x_max = 2e4,
            y_elem = 2,
            y_max = 2e4,
            z_elem = 8,
            z_max = 6e3,
            z_stretch = false,
            nh_poly = 1,
        )
        params = CA.ClimaAtmosParameters(
            FT;
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        )
        tc = CA.PrognosticEDMFX{1, true}(FT(0.05))
        edmfx = CA.EDMFXModel(;
            entr_model = CA.InvZEntrainment(),
            detr_model = CA.BuoyancyVelocityDetrainment(),
            sgs_mass_flux = true,
            sgs_diffusive_flux = true,
            nh_pressure = true,
            vertical_diffusion = true,
            scale_blending_method = CA.SmoothMinimumBlending(),
        )
        mo = CA.SurfaceConditions.MoninObukhov(; z0m = FT(1e-3), z0b = FT(1e-3))
        sfc = CA.AtmosSurface(;
            flux_scheme = mo,
            temperature = CA.SurfaceConditions.SlabOceanTemperature{FT}(;
                q_flux = false,
            ),
        )
        model = CA.AtmosModel(
            box_grid;
            params,
            turbconv_model = tc,
            edmfx_model = edmfx,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            cloud_model = CA.GridScaleCloud(),
            smagorinsky_lilly = CA.SmagorinskyLilly(; axes = :UV),
            surface = sfc,
            disable_surface_flux_tendency = false,
        )
        sim = CA.AtmosSimulation(
            model;
            dt = 20,
            t_end = 40,
            output_dir = joinpath(tmpdir, "trim"),
            default_callbacks = false,
        )
        Y, p, U = sim.integrator.u, sim.integrator.p, sim.integrator.cache.U
        _seed_moist_column!(Y, p, FT)
        copyto!(U, Y)
        @. U.c.ρe_tot += FT(5000) * U.c.ρ

        # Poison fields that must not be overwritten during intermediate ARK stages.
        poison = FT(-9876.5)
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(poison)
        fill!(p.precomputed.ᶠK_h, poison)
        fill!(p.precomputed.ᶜl_mix, poison)
        fill!(p.precomputed.ᶜentr_vel_scaleʲs.:(1), poison)
        fill!(p.precomputed.surface_rain_flux, poison)
        fill!(p.precomputed.ᶜwᵣ, poison)
        fill!(p.precomputed.ᶜD_h, poison)
        T_before = Array(parent(p.precomputed.ᶜT))

        CA.set_precomputed_quantities!(U, p, FT(10))

        @test maximum(abs.(Array(parent(p.precomputed.ᶜT)) .- T_before)) > FT(1)
        @test all(>=(zero(FT)), Array(parent(p.precomputed.ᶜD_h)))
        @test all(
            ==(poison),
            Array(parent(p.precomputed.sfc_conditions.ρ_flux_h_tot.components.data.:1)),
        )
        @test all(==(poison), Array(parent(p.precomputed.ᶠK_h)))
        @test all(==(poison), Array(parent(p.precomputed.ᶜl_mix)))
        @test all(==(poison), Array(parent(p.precomputed.ᶜentr_vel_scaleʲs.:(1))))
        @test all(==(poison), Array(parent(p.precomputed.surface_rain_flux)))
        @test all(==(poison), Array(parent(p.precomputed.ᶜwᵣ)))
        if _is_cpu(U.c)
            @test _allocs_stage_cache(U, p, FT(10)) ==
                  _allocs_smag(U, p, model.smagorinsky_lilly)
        end

        # Pre-column-physics thermo refresh updates core, EDMF thermo, and
        # `GridScaleCloud` `ᶜcloud_fraction` while preserving surface conditions
        # and mixing length.
        p.scratch.post_step.in_stage_loop = false
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(poison)
        fill!(p.precomputed.ᶜl_mix, poison)
        fill!(p.precomputed.ᶜcloud_fraction, poison)
        @. Y.c.ρe_tot += FT(5000) * Y.c.ρ
        T_pre_mp = Array(parent(p.precomputed.ᶜT))
        CA.refresh_pre_microphysics_thermo!(Y, p, FT(0))
        @test maximum(abs.(Array(parent(p.precomputed.ᶜT)) .- T_pre_mp)) > FT(1)
        @test all(
            c -> c == zero(FT) || c == one(FT),
            Array(parent(p.precomputed.ᶜcloud_fraction)),
        )
        @test all(
            ==(poison),
            Array(parent(p.precomputed.sfc_conditions.ρ_flux_h_tot.components.data.:1)),
        )
        @test all(==(poison), Array(parent(p.precomputed.ᶜl_mix)))
        if _is_cpu(Y.c)
            @test _allocs_pre_mp(Y, p, FT(0)) == 0
        end
    end
end

@testset "End-to-end solve_atmos! across 0M and 1M+EDMFX+Slab" begin
    ctx = ClimaComms.SingletonCommsContext()

    mktempdir() do tmpdir
        # 1. EquilibriumMicrophysics0M: ARS343 Float64 (with GaussianSGS) & SSP333 Float32.
        for (FT, alg_tag, alg_name, constraint) in (
            (Float64, "ARS343", CTS.ARS343(), CTS.Unconstrained()),
            (Float32, "SSP333", CTS.SSP333(), CTS.Unconstrained()),
        )

            grid = CA.ColumnGrid(
                FT;
                context = ctx,
                z_elem = 16,
                z_max = FT(8000),
                z_stretch = false,
            )
            params = CA.ClimaAtmosParameters(FT)
            vd = CA.VerticalDiffusion{FT}(;
                disable_momentum_vertical_diffusion = false,
                C_E = FT(1e-3),
            )
            sgs_quad =
                (FT === Float64 && alg_tag == "ARS343") ?
                CA.SGSQuadrature(
                    FT;
                    distribution = CA.GaussianSGS(),
                    quadrature_order = 3,
                ) :
                CA.SGSQuadrature(FT; distribution = CA.GridMeanSGS())
            slab = CA.SurfaceConditions.SlabOceanTemperature{FT}(; q_flux = false)
            mo = CA.SurfaceConditions.MoninObukhov(; z0m = FT(1e-3), z0b = FT(1e-3))
            sfc = CA.AtmosSurface(; flux_scheme = mo, temperature = slab)
            model = CA.AtmosModel(
                grid;
                params,
                microphysics_model = CA.EquilibriumMicrophysics0M(),
                vertical_diffusion = vd,
                sgs_quadrature = sgs_quad,
                surface = sfc,
                disable_surface_flux_tendency = false,
            )
            sim = CA.AtmosSimulation(
                model;
                dt = FT(20),
                t_end = FT(100),
                ode_config = _make_imex_algo(alg_name, constraint),
                output_dir = joinpath(tmpdir, "0m_$(FT)_$(alg_tag)"),
                default_callbacks = false,
            )
            Y, p = sim.integrator.u, sim.integrator.p
            _seed_moist_column!(Y, p, FT; with_condensates = false)

            h_space = Spaces.horizontal_space(axes(Y.c))
            nz = Spaces.nlevels(axes(Y.c))
            hcap = slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
            ρd_0 = _column_integral(Y.c.ρ .- Y.c.ρq_tot)
            w_0 = _column_integral(Y.c.ρq_tot) + _surface_integral(Y.sfc.water, h_space)
            e_atmos_0 = _column_integral(Y.c.ρe_tot)
            e_atmos_scale = _column_integral(abs.(Y.c.ρe_tot))
            T_sfc_0 = copy(Y.sfc.T)
            e_slab_scale = hcap * _surface_integral(T_sfc_0, h_space)

            CA.solve_atmos!(sim)

            rtol_ft = FT === Float32 ? FT(5e-5) : FT(1e-12)
            ΔE_atmos = _column_integral(Y.c.ρe_tot) - e_atmos_0
            ΔE_slab = hcap * _surface_integral(Y.sfc.T .- T_sfc_0, h_space)
            e_tol = FT(32) * (nz * e_atmos_scale + e_slab_scale) * eps(FT)
            @test _all_finite(Y)
            @test minimum(Array(parent(Y.c.ρ))) > zero(FT)
            @test minimum(Array(parent(Y.c.ρq_tot))) >= zero(FT)
            @test _column_integral(Y.c.ρ .- Y.c.ρq_tot) ≈ ρd_0 rtol = rtol_ft
            @test _column_integral(Y.c.ρq_tot) + _surface_integral(Y.sfc.water, h_space) ≈
                  w_0 rtol = rtol_ft
            @test abs(ΔE_atmos + ΔE_slab) <= e_tol

            if _is_cpu(Y.c) && FT === Float64 && alg_tag == "ARS343"
                Yₜ = zero(Y)
                @test _allocs_imp_tendency(Yₜ, Y, p, FT(100)) == 0
                @test _allocs_vt(Y, p, FT(20), sim.integrator.cache) == 0
                @test _allocs_stage_cache(sim.integrator.cache.U, p, FT(100)) == 0
            end
        end

        # 2. 1M + PrognosticEDMFX + SlabOceanTemperature:
        #    ARS343 Float64 & SSP333 Float32, multi-step atmos+slab
        #    conservation, positivity, realized surface/precip flux preservation
        #    in p.precomputed and diagnostics, and zero-alloc split steps.
        for (FT, alg_tag, alg_name, constraint) in (
            (Float64, "ARS343", CTS.ARS343(), CTS.Unconstrained()),
            (Float32, "SSP333", CTS.SSP333(), CTS.Unconstrained()),
        )

            grid = CA.ColumnGrid(
                FT;
                context = ctx,
                z_elem = 14,
                z_max = FT(8000),
                z_stretch = true,
                dz_bottom = FT(250),
                dz_top = FT(1000),
            )
            params = CA.ClimaAtmosParameters(
                FT;
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            )
            tc = CA.PrognosticEDMFX{1, true}(FT(0.05))
            edmfx = CA.EDMFXModel(;
                entr_model = CA.InvZEntrainment(),
                detr_model = CA.BuoyancyVelocityDetrainment(),
                sgs_mass_flux = true,
                sgs_diffusive_flux = true,
                nh_pressure = true,
                vertical_diffusion = true,
                scale_blending_method = CA.SmoothMinimumBlending(),
            )
            slab = CA.SurfaceConditions.SlabOceanTemperature{FT}(; q_flux = false)
            mo = CA.SurfaceConditions.MoninObukhov(; z0m = FT(1e-3), z0b = FT(1e-3))
            sfc = CA.AtmosSurface(; flux_scheme = mo, temperature = slab)
            model = CA.AtmosModel(
                grid;
                params,
                turbconv_model = tc,
                edmfx_model = edmfx,
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
                sgs_quadrature = CA.SGSQuadrature(FT; distribution = CA.GridMeanSGS()),
                surface = sfc,
                disable_surface_flux_tendency = false,
            )

            diag_specs = [
                Dict{String, Any}(
                    "short_name" => sn,
                    "period" => "1steps",
                    "writer" => "dict",
                )
                for sn in ("evspsbl", "hfls", "pr", "prra", "prsn")
            ]
            sim = CA.AtmosSimulation(
                model;
                dt = FT(20),
                t_end = FT(100),
                ode_config = _make_imex_algo(alg_name, constraint),
                output_dir = joinpath(tmpdir, "1m_edmf_$(FT)_$(alg_tag)"),
                default_callbacks = false,
                diagnostics = CA.DiagnosticsConfig(;
                    default = false,
                    additional = diag_specs,
                ),
            )
            raw_int =
                hasproperty(sim.integrator, :integrator) ?
                sim.integrator.integrator : sim.integrator
            Y, p = raw_int.u, raw_int.p
            _seed_moist_column!(Y, p, FT)

            h_space = Spaces.horizontal_space(axes(Y.c))
            nz = Spaces.nlevels(axes(Y.c))
            hcap = slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
            a_max = FT(CAP.max_area(CAP.turbconv_params(p.params)))
            ρd_0 = _column_integral(Y.c.ρ .- Y.c.ρq_tot)
            w_0 = _column_integral(Y.c.ρq_tot) + _surface_integral(Y.sfc.water, h_space)
            e_atmos_0 = _column_integral(Y.c.ρe_tot)
            e_atmos_scale = _column_integral(abs.(Y.c.ρe_tot))
            T_sfc_0 = copy(Y.sfc.T)
            e_slab_scale = hcap * _surface_integral(T_sfc_0, h_space)

            CA.solve_atmos!(sim)

            rtol_ft = FT === Float32 ? FT(5e-5) : FT(1e-12)
            ΔE_atmos = _column_integral(Y.c.ρe_tot) - e_atmos_0
            ΔE_slab = hcap * _surface_integral(Y.sfc.T .- T_sfc_0, h_space)
            e_tol = FT(32) * (nz * e_atmos_scale + e_slab_scale) * eps(FT)
            @test _all_finite(Y)
            @test minimum(Array(parent(Y.c.ρ))) > zero(FT)
            @test minimum(Array(parent(Y.c.ρq_tot))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρq_lcl))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρq_icl))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρq_rai))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρq_sno))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρtke))) >= zero(FT)
            @test minimum(Array(parent(Y.c.sgsʲs.:(1).ρa))) >= zero(FT)
            @test all(
                Array(parent(Y.c.sgsʲs.:(1).ρa ./ Y.c.ρ)) .<=
                a_max + FT(100) * eps(FT),
            )
            @test minimum(Array(parent(Y.c.sgsʲs.:(1).q_tot))) >= zero(FT)
            @test minimum(
                Array(parent(Y.f.sgsʲs.:(1).u₃.components.data.:1)),
            ) >= zero(FT)

            @test _column_integral(Y.c.ρ .- Y.c.ρq_tot) ≈ ρd_0 rtol = rtol_ft
            @test _column_integral(Y.c.ρq_tot) + _surface_integral(Y.sfc.water, h_space) ≈
                  w_0 rtol = rtol_ft
            @test abs(ΔE_atmos + ΔE_slab) <= e_tol
            @test p.scratch.post_step.in_stage_loop === true

            # Realized fluxes from the final step must be preserved in p.precomputed
            # and recorded identically by the end-of-step diagnostics, while
            # T_sfc tracks the updated SlabOcean temperature.
            @test minimum(Array(parent(p.precomputed.surface_rain_flux))) < FT(-1e-6)
            @test Array(parent(p.precomputed.sfc_conditions.T_sfc)) ==
                  Array(parent(Y.sfc.T))

            realized_h_flux = Array(
                parent(p.precomputed.sfc_conditions.ρ_flux_h_tot.components.data.:1),
            )
            realized_q_flux = Array(
                parent(p.precomputed.sfc_conditions.ρ_flux_q_tot.components.data.:1),
            )
            realized_prra = Array(parent(p.precomputed.surface_rain_flux))
            realized_prsn = Array(parent(p.precomputed.surface_snow_flux))
            @test maximum(abs, realized_q_flux) > zero(FT)

            dict_writer = sim.output_writers[1]
            for (sn, expected) in (
                ("prra", realized_prra),
                ("prsn", realized_prsn),
                ("pr", realized_prra .+ realized_prsn),
            )
                @test _last_diag(dict_writer.dict, sn) ≈ expected rtol = rtol_ft
            end
            diag_evspsbl = _last_diag(dict_writer.dict, "evspsbl")
            @test diag_evspsbl ≈
                  Array(
                parent(
                    @. dot(
                        p.precomputed.sfc_conditions.ρ_flux_q_tot,
                        p.core.surface_ct3_unit,
                    )
                ),
            ) rtol = rtol_ft
            diag_hfls = _last_diag(dict_writer.dict, "hfls")
            LH_v0 = FT(TD.Parameters.LH_v0(CAP.thermodynamics_params(p.params)))
            @test diag_hfls ≈ diag_evspsbl .* LH_v0 rtol = rtol_ft

            # Calling update_surface_conditions! on the post-step state changes
            # the surface fluxes relative to the realized vertical diffusion fluxes.
            CA.SurfaceConditions.update_surface_conditions!(Y, p, FT(raw_int.t))
            post_step_q_flux = Array(
                parent(p.precomputed.sfc_conditions.ρ_flux_q_tot.components.data.:1),
            )
            @test !isapprox(post_step_q_flux, realized_q_flux; rtol = eps(FT))

            if _is_cpu(Y.c) && FT === Float64
                p.scratch.post_step.in_stage_loop = false
                ᶠF̃ = CA.step_vertical_scalar_transport!(
                    Y,
                    p,
                    FT(20),
                    raw_int.cache,
                )
                @test _allocs_vt(Y, p, FT(20), raw_int.cache) == 0
                @test _allocs_cphys(Y, p, FT(100), FT(20), ᶠF̃) == 0
                @test _allocs_pre_mp(Y, p, FT(100)) == 0
                p.scratch.post_step.in_stage_loop = true
                @test _allocs_stage_cache(raw_int.cache.U, p, FT(100)) == 0
                Yₜ = zero(Y)
                @test _allocs_imp_tendency(Yₜ, Y, p, FT(100)) == 0
            end
        end
    end
end
