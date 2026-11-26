#=
julia --project=.buildkite
using Revise; include("test/prognostic_equations/split_simulation_e2e_tests.jl")
=#
using Test
using Dates
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Geometry, Spaces, Topologies
import ClimaDiagnostics as CD
import ClimaTimeSteppers as CTS
const TD = CA.TD

_is_cpu(field) = ClimaComms.device(field) isa ClimaComms.AbstractCPUDevice

function _all_finite(Y)
    ok = all(isfinite, Array(parent(Y.c))) && all(isfinite, Array(parent(Y.f)))
    if hasproperty(Y, :sfc)
        ok &= all(isfinite, Array(parent(Y.sfc)))
    end
    return ok
end

function _column_integral(field)
    return sum(field)
end

function _surface_integral(sfc_field, h_space)
    if h_space isa Spaces.PointSpace
        return only(Array(parent(sfc_field)))
    end
    return sum(Fields.Field(Fields.field_values(sfc_field), h_space))
end

function _last_diag(dict, sn)
    k = only(filter(k -> k == sn || startswith(k, "$(sn)_"), collect(keys(dict))))
    d = dict[k]
    return Array(parent(d[maximum(keys(d))]))
end

@noinline function _allocs_vt(Y, p, Δt, ic)
    CA.step_vertical_scalar_transport!(Y, p, Δt, ic)
    return @allocated CA.step_vertical_scalar_transport!(Y, p, Δt, ic)
end

@noinline function _allocs_cphys(Y, p, t, Δt, ᶠF̃)
    CA.step_column_physics!(Y, p, t, Δt; ᶠF̃)
    return @allocated CA.step_column_physics!(Y, p, t, Δt; ᶠF̃)
end

@noinline function _allocs_pre_mp(Y, p, t)
    CA.refresh_pre_microphysics_thermo!(Y, p, t)
    return @allocated CA.refresh_pre_microphysics_thermo!(Y, p, t)
end

@noinline function _allocs_stage_cache(U, p, t)
    CA.set_precomputed_quantities!(U, p, t)
    return @allocated CA.set_precomputed_quantities!(U, p, t)
end

@noinline function _allocs_smag(U, p, sl)
    CA.set_smagorinsky_lilly_precomputed_quantities!(U, p, sl)
    return @allocated CA.set_smagorinsky_lilly_precomputed_quantities!(U, p, sl)
end

@noinline function _allocs_imp_tendency(Yₜ, Y, p, t)
    CA.implicit_tendency!(Yₜ, Y, p, t)
    return @allocated CA.implicit_tendency!(Yₜ, Y, p, t)
end

function _make_imex_algo(alg_name, constraint = CTS.Unconstrained())
    nm = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    return CTS.IMEXAlgorithm(alg_name, nm, constraint)
end

function _seed_moist_column!(Y, p, FT; with_condensates::Bool = true)
    tps = CAP.thermodynamics_params(p.params)
    grav = FT(CAP.grav(p.params))
    R_d = FT(CAP.R_d(p.params))
    zc = Fields.coordinate_field(Y.c).z
    zf = Fields.coordinate_field(Y.f).z
    zmax = FT(Spaces.z_max(axes(zf)))
    lgc = Fields.local_geometry_field(Y.c)
    lgf = Fields.local_geometry_field(Y.f)

    @. Y.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(FT(8), FT(-3)), lgc)
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(FT(0.4) * sin(FT(π) * zf / zmax)),
        lgf,
    )
    @. Y.c.ρ = FT(1.15) * exp(-grav * zc / (R_d * FT(275)))
    T_prof = @. FT(296) - FT(6.5e-3) * zc
    q_prof = @. FT(0.015) * exp(-zc / FT(2500)) + FT(5e-4)
    if with_condensates && hasproperty(Y.c, :ρq_lcl)
        ql_prof = @. FT(6e-4) * exp(-((zc - FT(1800)) / FT(600))^2)
        qi_prof = @. FT(2e-4) * exp(-((zc - FT(4200)) / FT(700))^2)
        qr_prof = @. FT(8e-4) * exp(-(zc / FT(1500))^2)
        qs_prof = @. FT(2e-4) * exp(-((zc - FT(3000)) / FT(1000))^2)
        @. Y.c.ρq_tot = Y.c.ρ * (q_prof + ql_prof + qi_prof + qr_prof + qs_prof)
        @. Y.c.ρq_lcl = Y.c.ρ * ql_prof
        @. Y.c.ρq_icl = Y.c.ρ * qi_prof
        @. Y.c.ρq_rai = Y.c.ρ * qr_prof
        @. Y.c.ρq_sno = Y.c.ρ * qs_prof
        e_prof = @. TD.internal_energy(
            tps,
            T_prof,
            q_prof + ql_prof + qi_prof + qr_prof + qs_prof,
            ql_prof + qr_prof,
            qi_prof + qs_prof,
        )
    else
        @. Y.c.ρq_tot = Y.c.ρ * q_prof
        e_prof = @. TD.internal_energy(tps, T_prof, q_prof, FT(0), FT(0))
    end
    if hasproperty(Y.c, :ρn_lcl)
        @. Y.c.ρn_lcl = Y.c.ρ * FT(1e8) * exp(-((zc - FT(1800)) / FT(600))^2)
        @. Y.c.ρn_rai = Y.c.ρ * FT(1e5) * exp(-(zc / FT(1500))^2)
    end
    if hasproperty(Y.c, :ρn_ice)
        @. Y.c.ρn_ice = Y.c.ρ * FT(1e5) * exp(-((zc - FT(4200)) / FT(700))^2)
        @. Y.c.ρq_rim = FT(0.25) * Y.c.ρq_icl
        @. Y.c.ρb_rim = Y.c.ρq_rim / FT(400)
    end
    CA.set_implicit_precomputed_quantities_core!(Y, p, FT(0))
    @. Y.c.ρe_tot = Y.c.ρ * (e_prof + p.precomputed.ᶜK + p.core.ᶜΦ)
    if hasproperty(Y.c, :ρtke)
        @. Y.c.ρtke = Y.c.ρ * FT(0.6)
    end
    if hasproperty(Y.c, :sgsʲs)
        @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.06)
        @. Y.f.sgsʲs.:(1).u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(FT(3.0) * sin(FT(π) * zf / zmax)),
            lgf,
        )
        @. Y.c.sgsʲs.:(1).q_tot = Y.c.ρq_tot / Y.c.ρ + FT(1e-3)
        if hasproperty(Y.c.sgsʲs.:(1), :q_lcl)
            @. Y.c.sgsʲs.:(1).q_lcl = Y.c.ρq_lcl / Y.c.ρ
            @. Y.c.sgsʲs.:(1).q_icl = Y.c.ρq_icl / Y.c.ρ
            @. Y.c.sgsʲs.:(1).q_rai = Y.c.ρq_rai / Y.c.ρ
            @. Y.c.sgsʲs.:(1).q_sno = Y.c.ρq_sno / Y.c.ρ
        end
        CA.set_implicit_precomputed_quantities_core!(Y, p, FT(0))
        @. Y.c.sgsʲs.:(1).mse =
            p.precomputed.ᶜh_tot - p.precomputed.ᶜK + FT(800)
    end
    if hasproperty(Y, :sfc) && hasproperty(Y.sfc, :T)
        @. Y.sfc.T = FT(301)
        @. Y.sfc.water = FT(0)
    end
    CA.dss!(Y, p, FT(0))
    CA.set_precomputed_quantities!(Y, p, FT(0))
    return nothing
end

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

@testset "End-to-end solve_atmos! across 0M, 1M+EDMFX+Slab, 2M, 3D terrain, and ShipwayHill2012" begin
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

        # 3. 2M: ARS343 Float64 & SSP333 Float32
        for (mp_tag, mp_model) in (("2M", CA.NonEquilibriumMicrophysics2M()),),
            (FT, alg_tag, alg_name, constraint) in (
                (Float64, "ARS343", CTS.ARS343(), CTS.Unconstrained()),
                (Float32, "SSP333", CTS.SSP333(), CTS.Unconstrained()),
            )

            grid = CA.ColumnGrid(
                FT;
                context = ctx,
                z_elem = 14,
                z_max = FT(8000),
                z_stretch = false,
            )
            params = CA.ClimaAtmosParameters(FT; microphysics_model = mp_model)
            vd = CA.VerticalDiffusion{FT}(;
                disable_momentum_vertical_diffusion = false,
                C_E = FT(1e-3),
            )
            slab = CA.SurfaceConditions.SlabOceanTemperature{FT}(; q_flux = false)
            mo = CA.SurfaceConditions.MoninObukhov(; z0m = FT(1e-3), z0b = FT(1e-3))
            sfc = CA.AtmosSurface(; flux_scheme = mo, temperature = slab)
            model = CA.AtmosModel(
                grid;
                params,
                microphysics_model = mp_model,
                vertical_diffusion = vd,
                cloud_model = CA.GridScaleCloud(),
                sgs_quadrature = CA.SGSQuadrature(FT; distribution = CA.GridMeanSGS()),
                surface = sfc,
                disable_surface_flux_tendency = false,
            )
            sim = CA.AtmosSimulation(
                model;
                dt = FT(20),
                t_end = FT(80),
                ode_config = _make_imex_algo(alg_name, constraint),
                output_dir = joinpath(tmpdir, "$(mp_tag)_$(FT)_$(alg_tag)"),
                default_callbacks = false,
            )
            Y, p = sim.integrator.u, sim.integrator.p
            _seed_moist_column!(Y, p, FT)

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
            @test minimum(Array(parent(Y.c.ρq_tot))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρq_lcl))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρq_rai))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρn_lcl))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρn_rai))) >= zero(FT)
            @test minimum(Array(parent(p.precomputed.surface_rain_flux))) < FT(-1e-6)
            @test _column_integral(Y.c.ρ .- Y.c.ρq_tot) ≈ ρd_0 rtol = rtol_ft
            @test _column_integral(Y.c.ρq_tot) + _surface_integral(Y.sfc.water, h_space) ≈
                  w_0 rtol = rtol_ft
            @test abs(ΔE_atmos + ΔE_slab) <= e_tol

            if _is_cpu(Y.c) && FT === Float64 && alg_tag == "ARS343"
                p.scratch.post_step.in_stage_loop = false
                ᶠF̃ = CA.step_vertical_scalar_transport!(
                    Y,
                    p,
                    FT(20),
                    sim.integrator.cache,
                )
                @test _allocs_vt(Y, p, FT(20), sim.integrator.cache) == 0
                @test _allocs_cphys(Y, p, FT(80), FT(20), ᶠF̃) == 0
                @test _allocs_pre_mp(Y, p, FT(80)) == 0
                p.scratch.post_step.in_stage_loop = true
                @test _allocs_stage_cache(sim.integrator.cache.U, p, FT(80)) == 0
            end
        end

        # 4. 3D BoxGrid steep terrain: 1M + PrognosticEDMFX + SlabOceanTemperature
        for FT in (Float64,)
            topo = CA.CosineTopography{3, FT}(; h_max = FT(1000), λ = FT(20e3))
            box_grid = CA.BoxGrid(
                FT;
                context = ctx,
                x_elem = 2,
                x_max = FT(40e3),
                y_elem = 2,
                y_max = FT(40e3),
                z_elem = 10,
                z_max = FT(8000),
                z_stretch = true,
                dz_bottom = FT(300),
                dz_top = FT(1500),
                nh_poly = 1,
                periodic_x = true,
                periodic_y = true,
                topography = topo,
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
                box_grid;
                params,
                turbconv_model = tc,
                edmfx_model = edmfx,
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
                sgs_quadrature = CA.SGSQuadrature(FT; distribution = CA.GridMeanSGS()),
                surface = sfc,
                disable_surface_flux_tendency = false,
            )
            sim = CA.AtmosSimulation(
                model;
                dt = FT(10),
                t_end = FT(40),
                ode_config = _make_imex_algo(CTS.ARS343()),
                output_dir = joinpath(tmpdir, "box3d_$(FT)"),
                default_callbacks = false,
            )
            Y, p = sim.integrator.u, sim.integrator.p
            _seed_moist_column!(Y, p, FT)

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

            rtol_ft = FT === Float32 ? FT(1e-4) : FT(1e-11)
            ΔE_atmos = _column_integral(Y.c.ρe_tot) - e_atmos_0
            ΔE_slab = hcap * _surface_integral(Y.sfc.T .- T_sfc_0, h_space)
            e_tol = FT(32) * (nz * e_atmos_scale + e_slab_scale) * eps(FT)
            @test _all_finite(Y)
            @test minimum(Array(parent(Y.c.ρq_tot))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρtke))) >= zero(FT)
            @test minimum(Array(parent(Y.c.sgsʲs.:(1).ρa))) >= zero(FT)
            @test _column_integral(Y.c.ρ .- Y.c.ρq_tot) ≈ ρd_0 rtol = rtol_ft
            @test _column_integral(Y.c.ρq_tot) + _surface_integral(Y.sfc.water, h_space) ≈
                  w_0 rtol = rtol_ft
            @test abs(ΔE_atmos + ΔE_slab) <= e_tol

            # Final state must already be continuous across horizontal elements.
            Y_dss_check = copy(Y)
            CA.dss!(Y_dss_check, p, FT(40))
            @test Array(parent(Y_dss_check.c.ρ)) == Array(parent(Y.c.ρ))
            @test Array(parent(Y_dss_check.c.ρq_tot)) == Array(parent(Y.c.ρq_tot))

            if _is_cpu(Y.c) && FT === Float64
                p.scratch.post_step.in_stage_loop = false
                ᶠF̃ = CA.step_vertical_scalar_transport!(
                    Y,
                    p,
                    FT(10),
                    sim.integrator.cache,
                )
                @test _allocs_vt(Y, p, FT(10), sim.integrator.cache) == 0
                @test _allocs_cphys(Y, p, FT(40), FT(10), ᶠF̃) == 0
                @test _allocs_pre_mp(Y, p, FT(40)) == 0
                p.scratch.post_step.in_stage_loop = true
                @test _allocs_stage_cache(sim.integrator.cache.U, p, FT(40)) == 0
            end
        end

        # 5. PrescribedFlow (ShipwayHill2012) full solve_atmos! without custom hook
        for FT in (Float64,)
            sh_grid = CA.ColumnGrid(
                FT;
                context = ctx,
                z_elem = 32,
                z_max = FT(3000),
                z_stretch = false,
            )
            params = CA.ClimaAtmosParameters(
                FT;
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            )
            sh_setup = CA.Setups.ShipwayHill2012(;
                thermo_params = CAP.thermodynamics_params(params),
            )
            sh_model = CA.AtmosModel(
                sh_grid;
                params,
                setup = sh_setup,
                microphysics_model = CA.NonEquilibriumMicrophysics1M(),
                sgs_quadrature = CA.SGSQuadrature(FT; distribution = CA.GridMeanSGS()),
                hyperdiff = nothing,
            )
            sim_sh = CA.AtmosSimulation(
                sh_model;
                dt = FT(20),
                t_end = FT(360),
                output_dir = joinpath(tmpdir, "shipway_$(FT)"),
                default_callbacks = false,
            )
            Y, p = sim_sh.integrator.u, sim_sh.integrator.p
            @test p.atmos.prescribed_flow isa CA.PrescribedFlow

            CA.solve_atmos!(sim_sh)

            @test _all_finite(Y)
            @test minimum(Array(parent(Y.c.ρ))) > zero(FT)
            @test minimum(Array(parent(Y.c.ρq_tot))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρq_lcl))) >= zero(FT)
            @test minimum(Array(parent(Y.c.ρq_rai))) >= zero(FT)
            @test maximum(Array(parent(Y.c.ρq_lcl))) > zero(FT)
            @test all(<=(zero(FT)), Array(parent(p.precomputed.surface_rain_flux)))
            @test p.scratch.post_step.in_stage_loop === true

            if _is_cpu(Y.c) && FT === Float64
                p.scratch.post_step.in_stage_loop = false
                @test _allocs_cphys(Y, p, FT(360), FT(20), nothing) == 0
                @test _allocs_pre_mp(Y, p, FT(360)) == 0
                p.scratch.post_step.in_stage_loop = true
                @test _allocs_stage_cache(sim_sh.integrator.cache.U, p, FT(360)) == 0
            end
        end
    end
end

@testset "2MP3 is rejected by the post-ARK column step" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()
    grid = CA.ColumnGrid(
        FT;
        context = ctx,
        z_elem = 14,
        z_max = FT(8000),
        z_stretch = false,
    )
    mp_model = CA.NonEquilibriumMicrophysics2MP3()
    params = CA.ClimaAtmosParameters(FT; microphysics_model = mp_model)
    model = CA.AtmosModel(
        grid;
        params,
        microphysics_model = mp_model,
        cloud_model = CA.GridScaleCloud(),
        sgs_quadrature = CA.SGSQuadrature(FT; distribution = CA.GridMeanSGS()),
    )
    mktempdir() do tmpdir
        sim = CA.AtmosSimulation(
            model;
            dt = FT(20),
            t_end = FT(80),
            output_dir = tmpdir,
            default_callbacks = false,
        )
        @test_throws ArgumentError CTS.step!(sim.integrator)
    end
    @test_throws ArgumentError CA.get_microphysics_model(
        Dict("microphysics_model" => "2MP3"),
    )
end
