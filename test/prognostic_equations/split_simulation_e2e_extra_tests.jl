#=
julia --project=.buildkite
using Revise; include("test/prognostic_equations/split_simulation_e2e_extra_tests.jl")
=#
include("split_simulation_e2e_test_helpers.jl")

@testset "End-to-end solve_atmos! across 2M, 3D terrain, and ShipwayHill2012" begin
    ctx = ClimaComms.SingletonCommsContext()

    mktempdir() do tmpdir
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
