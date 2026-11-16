#=
Verification tests for the wave-only implicit tendency and exact
`BlockArrowheadSolve` Jacobian: solver selection across all microphysics and
turbulence configurations, finite-difference Jacobian agreement and zero heap
allocations on `1M + PrognosticEDMFX` and `0M`, round-off column continuity
closure, slab ocean turbulent-flux exclusion, and `PrescribedFlow`
(`ShipwayHill2012`) wiring.
=#
using Test
using Dates
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Geometry, MatrixFields, Spaces
import ClimaCore.MatrixFields: @name
import ClimaTimeSteppers as CTS
const TD = CA.TD

include("../test_helpers.jl")

const u₁_name = @name(c.uₕ.components.data.:1)
const u₂_name = @name(c.uₕ.components.data.:2)
const u₃_name = @name(f.u₃.components.data.:1)

@noinline function _allocs_implicit_tendency(Yₜ, Y, p, t)
    CA.implicit_tendency!(Yₜ, Y, p, t)
    return @allocated CA.implicit_tendency!(Yₜ, Y, p, t)
end

@noinline function _allocs_update_jacobian(jac, Y, p, dtγ, t)
    CA.update_jacobian!(jac, Y, p, dtγ, t)
    return @allocated CA.update_jacobian!(jac, Y, p, dtγ, t)
end

@noinline function _allocs_ldiv(ΔY, jac, R)
    ldiv!(ΔY, jac, R)
    return @allocated ldiv!(ΔY, jac, R)
end

@noinline function _allocs_prescribed_flow(Yₜ, Y, p, t)
    CA.prescribed_flow_vertical_advection_tendency!(Yₜ, Y, p, t)
    return @allocated CA.prescribed_flow_vertical_advection_tendency!(Yₜ, Y, p, t)
end

@testset "BlockArrowheadSolve across all microphysics and turbulence configs" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()
    col_grid =
        CA.ColumnGrid(FT; context = ctx, z_elem = 10, z_max = 6e3, z_stretch = false)
    topo_grid = CA.BoxGrid(
        FT;
        context = ctx,
        x_elem = 2,
        x_max = 1e4,
        y_elem = 2,
        y_max = 1e4,
        z_elem = 8,
        z_max = 4e3,
        z_stretch = false,
        nh_poly = 1,
        periodic_x = true,
        periodic_y = true,
        topography = CA.CosineTopography{3, FT}(),
    )

    edmfx_opts = CA.EDMFXModel(;
        entr_model = CA.InvZEntrainment(),
        detr_model = CA.BuoyancyVelocityDetrainment(),
        sgs_mass_flux = true,
        sgs_diffusive_flux = true,
        nh_pressure = true,
        vertical_diffusion = true,
        scale_blending_method = CA.SmoothMinimumBlending(),
    )
    prog_edmf = CA.PrognosticEDMFX(;
        n_updrafts = 1,
        prognostic_tke = true,
        area_fraction = FT(0.1),
    )
    edonly_edmf = CA.EDOnlyEDMFX()

    cases = (
        ("Dry", col_grid, CA.DryModel(), nothing, nothing),
        ("0M", col_grid, CA.EquilibriumMicrophysics0M(), nothing, nothing),
        ("1M", col_grid, CA.NonEquilibriumMicrophysics1M(), nothing, nothing),
        ("2M", col_grid, CA.NonEquilibriumMicrophysics2M(), nothing, nothing),
        ("2MP3", col_grid, CA.NonEquilibriumMicrophysics2MP3(), nothing, nothing),
        (
            "EDOnlyEDMFX",
            col_grid,
            CA.NonEquilibriumMicrophysics1M(),
            edonly_edmf,
            edmfx_opts,
        ),
        (
            "PrognosticEDMFX",
            col_grid,
            CA.NonEquilibriumMicrophysics1M(),
            prog_edmf,
            edmfx_opts,
        ),
        (
            "Topo3D_PrognosticEDMFX",
            topo_grid,
            CA.NonEquilibriumMicrophysics1M(),
            prog_edmf,
            edmfx_opts,
        ),
        ("Topo3D_2MP3", topo_grid, CA.NonEquilibriumMicrophysics2MP3(), nothing, nothing),
    )

    for (label, grid, mm, tc, edmfx) in cases
        params = CA.ClimaAtmosParameters(FT; microphysics_model = mm)
        sgs_quad =
            mm isa CA.EquilibriumMicrophysics0M ? CA.SGSQuadrature(FT) : nothing
        vert_diff =
            isnothing(tc) ?
            CA.DecayWithHeightDiffusion{false, FT}(; H = 1000.0, D₀ = 10.0) :
            nothing
        model = CA.AtmosModel(
            grid;
            params,
            microphysics_model = mm,
            sgs_quadrature = sgs_quad,
            turbconv_model = tc,
            edmfx_model = edmfx,
            vertical_diffusion = vert_diff,
            chemistry_model = CA.GasPhaseChem(),
            temperature = CA.SurfaceConditions.SlabOceanTemperature{FT}(),
            rayleigh_sponge = CA.RayleighSponge{FT}(;
                zd = FT(3e3),
                α_uₕ = FT(0.1),
                α_w = FT(0.1),
            ),
        )
        Y = CA.initial_state(model)
        atmos = CA.physics_only(model)
        jcache = CA.jacobian_cache(CA.ManualSparseJacobian(), Y, atmos)

        @test jcache.matrix.solver.alg isa MatrixFields.BlockArrowheadSolve
        @test !haskey(jcache.derivative_flags, :diffusion_flag)

        mat = jcache.matrix.matrix
        for name in CA.jacobian_diagonal_names(Y)
            name == @name(f.u₃) && continue
            @test (name, name) in keys(mat)
            @test !(mat[name, name] isa Fields.Field)
        end

        # Verify absence of diffusion, sedimentation, ρtke, number/rime, and SGS blocks.
        @test !((@name(c.ρe_tot), @name(c.ρq_tot)) in keys(mat))
        @test !((@name(c.ρe_tot), @name(c.ρ)) in keys(mat))
        if MatrixFields.has_field(Y, @name(c.ρq_lcl))
            @test !((@name(c.ρ), @name(c.ρq_lcl)) in keys(mat))
            @test !((@name(c.ρq_tot), @name(c.ρq_lcl)) in keys(mat))
            @test !((@name(c.ρe_tot), @name(c.ρq_lcl)) in keys(mat))
            @test (@name(c.ρq_lcl), @name(f.u₃)) in keys(mat)
            @test (@name(f.u₃), @name(c.ρq_lcl)) in keys(mat)
        end
        if MatrixFields.has_field(Y, @name(c.ρn_lcl))
            @test !((@name(c.ρn_lcl), @name(f.u₃)) in keys(mat))
        end
        if MatrixFields.has_field(Y, @name(c.ρq_rim))
            @test !((@name(c.ρq_rim), @name(f.u₃)) in keys(mat))
        end
        if MatrixFields.has_field(Y, @name(c.sgsʲs.:(1).mse))
            @test !((@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)) in keys(mat))
        end

        # Verify P3 uses frozen condensate loading across all hydrometeors (`!use_moist_wave_jacobian`).
        if mm isa CA.NonEquilibriumMicrophysics2MP3
            ᶜz = Fields.coordinate_field(Y.c).z
            ᶜp = @. FT(9e4) * exp(-(ᶜz) / FT(8e3))
            ᶜT = @. FT(265) - FT(5e-3) * ᶜz
            @. Y.c.ρq_tot = Y.c.ρ * FT(0.01)
            @. Y.c.ρq_lcl = Y.c.ρ * FT(5e-4)
            @. Y.c.ρq_rai = Y.c.ρ * FT(5e-4)
            @. Y.c.ρq_icl = Y.c.ρ * FT(5e-4)
            @. Y.c.ρq_sno = Y.c.ρ * FT(5e-4)
            ᶜq_tot_nonneg = @. Y.c.ρq_tot / Y.c.ρ
            ᶜq_liq = @. (Y.c.ρq_lcl + Y.c.ρq_rai) / Y.c.ρ
            ᶜq_ice = @. (Y.c.ρq_icl + Y.c.ρq_sno) / Y.c.ρ
            ᶜK = zeros(axes(Y.c))
            ᶜcloud_fraction = similar(Y.c, FT)
            fill!(parent(ᶜcloud_fraction), FT(0.45))
            p_p3 = (;
                atmos,
                params,
                core = (; ᶜΦ = (@. FT(CAP.grav(params)) * ᶜz)),
                precomputed = (;
                    ᶜK,
                    ᶜp,
                    ᶜT,
                    ᶜq_tot_nonneg,
                    ᶜq_liq,
                    ᶜq_ice,
                    ᶜcloud_fraction,
                ),
                scratch = (;
                    ᶜtemp_scalar = similar(Y.c, FT),
                    ᶜtemp_scalar_2 = similar(Y.c, FT),
                    ᶜtemp_scalar_3 = similar(Y.c, FT),
                ),
            )
            @test !CA.use_moist_wave_jacobian(p_p3)
            CA.update_jacobian!(CA.ManualSparseJacobian(), jcache, Y, p_p3, FT(5), FT(0))
            col_mat = CA.first_column_view(jcache.matrix.matrix)
            u3_comp(name) =
                map(
                    x -> x.components.data.:1,
                    MatrixFields.column_field2array(col_mat[@name(f.u₃), name]),
                )
            b_lcl = u3_comp(@name(c.ρq_lcl))
            b_rai = u3_comp(@name(c.ρq_rai))
            b_icl = u3_comp(@name(c.ρq_icl))
            b_sno = u3_comp(@name(c.ρq_sno))
            n_int = 2:Spaces.nlevels(axes(Y.c))
            @test isapprox(b_lcl[n_int, :], b_rai[n_int, :]; rtol = 1e-12)
            @test isapprox(b_icl[n_int, :], b_sno[n_int, :]; rtol = 1e-12)
        end
    end
end

@testset "Pure wave-core FD Jacobian agreement, exact solve, continuity, and 0 allocs" begin
    function fd_jacobian_blocks(Y, p, dtγ, t, row_names, col_names)
        Y_work = copy(Y)
        Yₜ_plus = similar(Y)
        Yₜ_minus = similar(Y)
        R_plus = similar(Y)
        R_minus = similar(Y)

        eval_R! = function (R_out, Yₜ_buf, Y_in)
            CA.set_implicit_precomputed_quantities!(Y_in, p, t)
            CA.implicit_tendency!(Yₜ_buf, Y_in, p, t)
            @. R_out = dtγ * Yₜ_buf - Y_in
            return nothing
        end

        blocks = Dict{Tuple{Any, Any}, Matrix{Float64}}()
        for col_name in col_names
            col_vec = vec(parent(MatrixFields.get_field(Y, col_name)))
            work_vec = vec(parent(MatrixFields.get_field(Y_work, col_name)))
            n_col = length(col_vec)
            for rname in row_names
                n_row = length(vec(parent(MatrixFields.get_field(Y, rname))))
                blocks[(rname, col_name)] = zeros(Float64, n_row, n_col)
            end
            col_scale = max(maximum(abs, col_vec), 1e-3)
            rel_step = col_name == u₃_name ? 1e-5 : 1e-6
            for j in 1:n_col
                base = col_vec[j]
                h = rel_step * max(abs(base), col_scale)
                work_vec[j] = base + h
                eval_R!(R_plus, Yₜ_plus, Y_work)
                work_vec[j] = base - h
                eval_R!(R_minus, Yₜ_minus, Y_work)
                work_vec[j] = base
                inv_2h = 1.0 / (2.0 * h)
                for rname in row_names
                    rp = vec(parent(MatrixFields.get_field(R_plus, rname)))
                    rm = vec(parent(MatrixFields.get_field(R_minus, rname)))
                    @views blocks[(rname, col_name)][:, j] .= (rp .- rm) .* inv_2h
                end
            end
        end
        CA.set_implicit_precomputed_quantities!(Y, p, t)
        return blocks
    end

    # 1. `0M` column: subsaturated + supersaturated regimes, FD vs ManualSparseJacobian,
    #    exact `ldiv!`, pure continuity, and 0 allocations (CPU for scalar-indexed FD).
    cpu_ctx = ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    z_elem = 24
    cfg_0m = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "column",
            "FLOAT_TYPE" => "Float64",
            "dt" => "20secs",
            "z_max" => 12000.0,
            "z_elem" => z_elem,
            "z_stretch" => false,
            "microphysics_model" => "0M",
            "cloud_model" => "grid_scale",
            "use_sgs_quadrature" => true,
            "sgs_distribution" => "mean",
            "implicit_diffusion" => true,
            "vert_diff" => "DecayWithHeightDiffusion",
            "output_default_diagnostics" => false,
        );
        job_id = "wave_core_fd_0m",
        comms_ctx = cpu_ctx,
    )
    nt_0m = generate_test_simulation(cfg_0m)
    Y0, p0, sim0 = nt_0m.Y, nt_0m.p, nt_0m.simulation
    thermo_params = CAP.thermodynamics_params(p0.params)
    grav = Float64(CAP.grav(p0.params))
    R_d = Float64(CAP.R_d(p0.params))

    zc0 = Fields.coordinate_field(Y0.c).z
    zf0 = Fields.coordinate_field(Y0.f).z
    zmax0 = Spaces.z_max(axes(zf0))
    lgc0 = Fields.local_geometry_field(Y0.c)
    lgf0 = Fields.local_geometry_field(Y0.f)
    @. Y0.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(6.0, -3.0), lgc0)
    @. Y0.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(1.2 * sin(π * zf0 / zmax0)),
        lgf0,
    )
    CA.set_implicit_precomputed_quantities!(Y0, p0, 0.0)
    zc_vec = vec(parent(zc0))
    K0_vec = vec(parent(p0.precomputed.ᶜK))
    for k in 1:z_elem
        zk = zc_vec[k]
        Tk = 295.0 - 5.5e-3 * zk
        ρk = 1.15 * exp(-grav * zk / (R_d * 270.0))
        q_sat_k = TD.q_vap_saturation(thermo_params, Tk, ρk)
        # Lower half supersaturated (`q_tot > q_sat`), upper half subsaturated.
        qk = k <= z_elem ÷ 2 ? (1.45 * q_sat_k + 1.5e-3) : 0.4 * q_sat_k
        ek =
            qk > q_sat_k ?
            TD.internal_energy_sat(thermo_params, Tk, ρk, qk) :
            TD.internal_energy(thermo_params, Tk, qk, 0.0, 0.0)
        parent(Y0.c.ρ)[k] = ρk
        parent(Y0.c.ρq_tot)[k] = ρk * qk
        parent(Y0.c.ρe_tot)[k] = ρk * (ek + K0_vec[k] + grav * zk)
    end
    CA.set_precomputed_quantities!(Y0, p0, 0.0)

    jac0 = sim0.integrator.cache.newtons_method_cache.j
    @test jac0.cache.matrix.solver.alg isa MatrixFields.BlockArrowheadSolve
    dtγ0 = Float64(p0.dtγ)

    Yₜ0 = similar(Y0)
    R0 = similar(Y0)
    ΔY0 = similar(Y0)
    CA.implicit_tendency!(Yₜ0, Y0, p0, 0.0)
    @. R0 = dtγ0 * Yₜ0 - Y0

    # Pure continuity: `Σ_k ᶜJ_k T_imp.c.ρ_k == 0` to round-off (even with 0M precip).
    ᶜJ0 = vec(parent(Fields.local_geometry_field(Y0.c).J))
    col_mass_rate_0 = sum(ᶜJ0 .* vec(parent(Yₜ0.c.ρ)))
    @test abs(col_mass_rate_0) < 20 * eps(Float64) * sum(ᶜJ0 .* abs.(vec(parent(Yₜ0.c.ρ))))

    @test _allocs_implicit_tendency(Yₜ0, Y0, p0, 0.0) == 0
    @test _allocs_update_jacobian(jac0, Y0, p0, dtγ0, 0.0) == 0
    @test _allocs_ldiv(ΔY0, jac0, R0) == 0

    # Exact linear solve residual `(∂R/∂Y) * ΔY == R`.
    res0 = similar(Y0)
    @. res0 = jac0.cache.matrix.matrix * ΔY0 - R0
    @test norm(parent(res0)) / norm(parent(R0)) < 1e-12

    manual0 = CA.first_column_block_arrays(jac0.alg, Y0, p0, dtγ0, 0.0)
    cols0 = (@name(c.ρ), @name(c.ρe_tot), @name(c.ρq_tot), u₁_name, u₂_name, u₃_name)
    rows0 = (@name(c.ρ), @name(c.ρe_tot), @name(c.ρq_tot), u₃_name)
    fd0 = fd_jacobian_blocks(Y0, p0, dtγ0, 0.0, rows0, cols0)
    int0 = 2:z_elem

    for sname in (@name(c.ρ), @name(c.ρq_tot))
        # `ManualSparseJacobian` omits `O(CFL_w)` scalar vertical advection on the diagonal.
        @test norm(fd0[(sname, sname)] + I) < 0.1
        @test norm(manual0[(sname, u₃_name)][:, int0] - fd0[(sname, u₃_name)][:, int0]) /
              norm(fd0[(sname, u₃_name)][:, int0]) < 1e-9
    end
    @test norm(fd0[(@name(c.ρe_tot), @name(c.ρe_tot))] + I) < 0.1
    @test norm(
        manual0[(@name(c.ρe_tot), u₃_name)][:, int0] -
        fd0[(@name(c.ρe_tot), u₃_name)][:, int0],
    ) / norm(fd0[(@name(c.ρe_tot), u₃_name)][:, int0]) < 1e-4

    for cname in (@name(c.ρ), @name(c.ρe_tot), @name(c.ρq_tot), u₁_name, u₂_name)
        @test norm(
            manual0[(u₃_name, cname)][int0, :] - fd0[(u₃_name, cname)][int0, :],
        ) / norm(fd0[(u₃_name, cname)][int0, :]) < 0.1
    end
    @test norm(
        (manual0[(u₃_name, u₃_name)][int0, int0] + I) -
        (fd0[(u₃_name, u₃_name)][int0, int0] + I),
    ) / norm(fd0[(u₃_name, u₃_name)][int0, int0] + I) < 0.1

    # 2. `1M + PrognosticEDMFX` column: FD vs ManualSparseJacobian, stage-hook
    #    bypass, zero allocations, and pure continuity with active condensates,
    #    terminal velocities, TKE, and updrafts.
    cfg_1m = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "column",
            "FLOAT_TYPE" => "Float64",
            "dt" => "20secs",
            "z_max" => 10000.0,
            "z_elem" => z_elem,
            "z_stretch" => false,
            "microphysics_model" => "1M",
            "implicit_microphysics" => true,
            "turbconv" => "prognostic_edmfx",
            "prognostic_tke" => true,
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => true,
            "implicit_diffusion" => true,
            "output_default_diagnostics" => false,
        );
        job_id = "wave_core_fd_1m_edmf",
        comms_ctx = cpu_ctx,
    )
    nt_1m = generate_test_simulation(cfg_1m)
    Y1, p1, sim1 = nt_1m.Y, nt_1m.p, nt_1m.simulation
    @test hasproperty(p1.scratch, :post_step)
    @test !CA.use_moist_wave_jacobian(p1)

    zc1 = Fields.coordinate_field(Y1.c).z
    zf1 = Fields.coordinate_field(Y1.f).z
    zmax1 = Spaces.z_max(axes(zf1))
    lgc1 = Fields.local_geometry_field(Y1.c)
    lgf1 = Fields.local_geometry_field(Y1.f)
    @. Y1.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(5.0, -2.5), lgc1)
    @. Y1.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(1.5 * sin(π * zf1 / zmax1)),
        lgf1,
    )
    @. Y1.f.sgsʲs.:(1).u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(4.0 * sin(π * zf1 / zmax1)),
        lgf1,
    )
    @. Y1.c.ρtke = Y1.c.ρ * (0.4 + 0.2 * cos(π * zc1 / zmax1))
    @. Y1.c.sgsʲs.:(1).ρa = Y1.c.ρ * 0.08
    @. Y1.c.ρq_tot = Y1.c.ρ * (2e-4 * exp(-zc1 / 3000.0) + 1e-4)
    @. Y1.c.ρq_lcl = Y1.c.ρ * 1e-5 * exp(-zc1 / 4000.0)
    @. Y1.c.ρq_icl = Y1.c.ρ * 5e-6 * exp(-zc1 / 5000.0)
    @. Y1.c.ρq_rai = Y1.c.ρ * 1.5e-5 * exp(-zc1 / 3000.0)
    @. Y1.c.ρq_sno = Y1.c.ρ * 7e-6 * exp(-zc1 / 4000.0)
    @. Y1.c.sgsʲs.:(1).q_tot = 2e-3
    @. Y1.c.sgsʲs.:(1).q_lcl = 4e-4
    @. Y1.c.sgsʲs.:(1).q_rai = 3e-4
    CA.set_precomputed_quantities!(Y1, p1, 0.0)
    # Set subsaturated cloud fraction to 0 for exact FD comparison of the
    # pure wave-core residual (`q_tot < q_sat` everywhere in this profile).
    fill!(parent(p1.precomputed.ᶜcloud_fraction), 0.0)
    @test all(
        vec(parent(p1.precomputed.ᶜq_tot_nonneg)) .<
        TD.q_vap_saturation.(
            thermo_params,
            vec(parent(p1.precomputed.ᶜT)),
            vec(parent(Y1.c.ρ)),
        ),
    )

    jac1 = sim1.integrator.cache.newtons_method_cache.j
    @test jac1.cache.matrix.solver.alg isa MatrixFields.BlockArrowheadSolve
    dtγ1 = Float64(p1.dtγ)

    # Verify `initialize_implicit_stage_problem!` and
    # `set_implicit_precomputed_quantities!` leave `Y1` unchanged, and
    # `set_implicit_precomputed_quantities!` preserves EDMF draft/env thermo
    # while `set_implicit_precomputed_quantities_core!` updates them.
    Y1_snap = copy(Y1)
    CA.initialize_implicit_stage_problem!(Y1, p1, dtγ1)
    @test parent(Y1) == parent(Y1_snap)
    mse1_orig = copy(Y1.c.sgsʲs.:(1).mse)
    T1_edmf_before = Array(parent(p1.precomputed.ᶜTʲs.:(1)))
    ρ1_edmf_before = Array(parent(p1.precomputed.ᶜρʲs.:(1)))
    T0_edmf_before = Array(parent(p1.precomputed.ᶜT⁰))
    @. Y1.c.sgsʲs.:(1).mse += 2500.0
    CA.set_implicit_precomputed_quantities!(Y1, p1, 0.0)
    @test Array(parent(p1.precomputed.ᶜTʲs.:(1))) == T1_edmf_before
    @test Array(parent(p1.precomputed.ᶜρʲs.:(1))) == ρ1_edmf_before
    @test Array(parent(p1.precomputed.ᶜT⁰)) == T0_edmf_before
    CA.set_implicit_precomputed_quantities_core!(Y1, p1, 0.0)
    @test maximum(abs.(Array(parent(p1.precomputed.ᶜTʲs.:(1))) .- T1_edmf_before)) > 1.0
    @. Y1.c.sgsʲs.:(1).mse = mse1_orig
    CA.set_implicit_precomputed_quantities_core!(Y1, p1, 0.0)
    @test parent(Y1) == parent(Y1_snap)

    Yₜ1 = similar(Y1)
    R1 = similar(Y1)
    ΔY1 = similar(Y1)
    CA.implicit_tendency!(Yₜ1, Y1, p1, 0.0)
    @. R1 = dtγ1 * Yₜ1 - Y1

    # Pure continuity and zero non-wave-core tendencies (`ρtke`, `sgsʲs`).
    ᶜJ1 = vec(parent(Fields.local_geometry_field(Y1.c).J))
    @test abs(sum(ᶜJ1 .* vec(parent(Yₜ1.c.ρ)))) <
          20 * eps(Float64) * sum(ᶜJ1 .* abs.(vec(parent(Yₜ1.c.ρ))))
    @test all(iszero, parent(Yₜ1.c.ρtke))
    @test all(iszero, parent(Yₜ1.c.sgsʲs))
    @test all(iszero, parent(Yₜ1.f.sgsʲs))

    @test _allocs_implicit_tendency(Yₜ1, Y1, p1, 0.0) == 0
    @test _allocs_update_jacobian(jac1, Y1, p1, dtγ1, 0.0) == 0
    @test _allocs_ldiv(ΔY1, jac1, R1) == 0

    res1 = similar(Y1)
    @. res1 = jac1.cache.matrix.matrix * ΔY1 - R1
    @test norm(parent(res1)) / norm(parent(R1)) < 1e-12

    manual1 = CA.first_column_block_arrays(jac1.alg, Y1, p1, dtγ1, 0.0)
    cond_syms = (@name(c.ρq_lcl), @name(c.ρq_icl), @name(c.ρq_rai), @name(c.ρq_sno))
    cols1 = (
        @name(c.ρ),
        @name(c.ρe_tot),
        @name(c.ρq_tot),
        cond_syms...,
        @name(c.ρtke),
        @name(c.sgsʲs.:(1).ρa),
        @name(c.sgsʲs.:(1).mse),
        @name(c.sgsʲs.:(1).q_tot),
        u₁_name,
        u₂_name,
        u₃_name,
    )
    rows1 = (
        @name(c.ρ),
        @name(c.ρe_tot),
        @name(c.ρq_tot),
        cond_syms...,
        @name(c.ρtke),
        @name(c.sgsʲs.:(1).ρa),
        @name(c.sgsʲs.:(1).mse),
        u₃_name,
    )
    fd1 = fd_jacobian_blocks(Y1, p1, dtγ1, 0.0, rows1, cols1)

    for sname in (@name(c.ρ), @name(c.ρq_tot), cond_syms...)
        @test norm(fd1[(sname, sname)] + I) < 0.1
        @test norm(manual1[(sname, u₃_name)][:, int0] - fd1[(sname, u₃_name)][:, int0]) /
              norm(fd1[(sname, u₃_name)][:, int0]) < 1e-9
    end
    for inactive in (@name(c.ρtke), @name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse))
        @test norm(fd1[(inactive, inactive)] + I) < 1e-9
        @test norm(fd1[(inactive, u₃_name)]) < 1e-12
        @test norm(fd1[(u₃_name, inactive)]) < 1e-12
    end
    @test norm(
        manual1[(@name(c.ρe_tot), u₃_name)][:, int0] -
        fd1[(@name(c.ρe_tot), u₃_name)][:, int0],
    ) / norm(fd1[(@name(c.ρe_tot), u₃_name)][:, int0]) < 1e-4

    for cname in
        (@name(c.ρ), @name(c.ρe_tot), @name(c.ρq_tot), cond_syms..., u₁_name, u₂_name)
        @test norm(
            manual1[(u₃_name, cname)][int0, :] - fd1[(u₃_name, cname)][int0, :],
        ) / norm(fd1[(u₃_name, cname)][int0, :]) < 0.1
    end
    @test norm(
        (manual1[(u₃_name, u₃_name)][int0, int0] + I) -
        (fd1[(u₃_name, u₃_name)][int0, int0] + I),
    ) / norm(fd1[(u₃_name, u₃_name)][int0, int0] + I) < 0.1

    # 3. 3D topography + stretched grid continuity check (`Σ_k ᶜJ_k T_imp.c.ρ_k == 0`
    #    in every column).
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()
    box_grid = CA.BoxGrid(
        FT;
        context = ctx,
        x_elem = 2,
        x_max = 2e4,
        y_elem = 2,
        y_max = 2e4,
        z_elem = 10,
        z_max = 12e3,
        z_stretch = true,
        dz_bottom = 300.0,
        dz_top = 2500.0,
        nh_poly = 2,
        periodic_x = true,
        periodic_y = true,
        topography = CA.CosineTopography{3, FT}(),
    )
    params_3d = CA.ClimaAtmosParameters(FT)
    model_3d = CA.AtmosModel(
        box_grid;
        params = params_3d,
        microphysics_model = CA.NonEquilibriumMicrophysics1M(),
    )
    Y_3d = CA.initial_state(model_3d)
    zf_3d = Fields.coordinate_field(Y_3d.f).z
    zmax_3d = Spaces.z_max(axes(zf_3d))
    lgc_3d = Fields.local_geometry_field(Y_3d.c)
    lgf_3d = Fields.local_geometry_field(Y_3d.f)
    @. Y_3d.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(8.0, 4.0), lgc_3d)
    @. Y_3d.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(2.0 * sin(π * zf_3d / zmax_3d)),
        lgf_3d,
    )
    @. Y_3d.c.ρq_tot = Y_3d.c.ρ * 0.01
    @. Y_3d.c.ρq_rai = Y_3d.c.ρ * 1e-3
    p_3d = CA.build_cache(
        Y_3d,
        model_3d,
        params_3d,
        FT(20),
        DateTime(2010, 1, 1),
        nothing,
    )
    CA.set_implicit_precomputed_quantities!(Y_3d, p_3d, FT(0))
    Yₜ_3d = similar(Y_3d)
    CA.implicit_tendency!(Yₜ_3d, Y_3d, p_3d, FT(0))
    ᶜJ_arr = reshape(Array(parent(Fields.local_geometry_field(Y_3d.c).J)), 10, :)
    ρₜ_arr = reshape(Array(parent(Yₜ_3d.c.ρ)), 10, :)
    for col in axes(ρₜ_arr, 2)
        scale = sum(@views ᶜJ_arr[:, col] .* abs.(ρₜ_arr[:, col]))
        @test abs(sum(@views ᶜJ_arr[:, col] .* ρₜ_arr[:, col])) <= 30 * eps(FT) * scale
    end
end

@testset "Slab ocean turbulent split and PrescribedFlow (ShipwayHill2012) wiring" begin
    FT = Float64
    ctx = ClimaComms.SingletonCommsContext()

    # 1. SlabOceanTemperature: turbulent fluxes omitted from `surface_temp_tendency!`
    #    while Q-flux is retained.
    sph_grid = CA.SphereGrid(FT; context = ctx, h_elem = 2, z_elem = 6, z_stretch = false)
    params = CA.ClimaAtmosParameters(FT)
    slab = CA.SurfaceConditions.SlabOceanTemperature{FT}(; q_flux = true, Q₀ = FT(35))
    model_split = CA.AtmosModel(
        sph_grid;
        params,
        microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        temperature = slab,
    )
    model_nofx = CA.AtmosModel(
        sph_grid;
        params,
        microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        temperature = slab,
        disable_surface_flux_tendency = true,
    )

    Y_s = CA.initial_state(model_split)
    lgc_s = Fields.local_geometry_field(Y_s.c)
    @. Y_s.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(7.0, 3.0), lgc_s)
    @. Y_s.c.ρq_tot = Y_s.c.ρ * 0.005
    p_split =
        CA.build_cache(Y_s, model_split, params, FT(30), DateTime(2010, 1, 1), nothing)
    p_nofx = CA.build_cache(Y_s, model_nofx, params, FT(30), DateTime(2010, 1, 1), nothing)

    @test maximum(abs, Array(parent(p_split.precomputed.sfc_conditions.ρ_flux_h_tot))) > 0
    @test maximum(abs, Array(parent(p_split.precomputed.sfc_conditions.ρ_flux_q_tot))) > 0

    Yₜ_split = zero(Y_s)
    Yₜ_nofx = zero(Y_s)
    CA.surface_temp_tendency!(Yₜ_split, Y_s, p_split, FT(0), slab)
    CA.surface_temp_tendency!(Yₜ_nofx, Y_s, p_nofx, FT(0), slab)

    @test all(iszero, Array(parent(Yₜ_split.sfc.water)))
    @test maximum(abs, Array(parent(Yₜ_split.sfc.T))) > 0 # Q-flux retained
    @test Array(parent(Yₜ_split.sfc.T)) == Array(parent(Yₜ_nofx.sfc.T))

    # 2. `PrescribedFlow` (`ShipwayHill2012`): `fully_explicit_tendency!` runs
    #    with zero allocations, matches wave-core vertical transport across
    #    upwinding schemes (including passive tracers), and steps cleanly through
    #    `AtmosSimulation`.
    sh_grid = CA.ColumnGrid(FT; context = ctx, z_elem = 32, z_max = 8e3, z_stretch = false)
    sh_setup =
        CA.Setups.ShipwayHill2012(; thermo_params = CAP.thermodynamics_params(params))
    for upw in (:none, :first_order, :third_order, :vanleer_limiter)
        sh_model = CA.AtmosModel(
            sh_grid;
            params,
            setup = sh_setup,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            chemistry_model = CA.GasPhaseChem(),
            energy_humidity_upwinding = upw,
            tracer_upwinding = upw,
            hyperdiff = nothing,
        )
        Y_sh = CA.initial_state(sh_model)
        zc_sh = Fields.coordinate_field(Y_sh.c).z
        zmax_sh = Spaces.z_max(axes(Y_sh.f))
        @. Y_sh.c.ρq_lcl = Y_sh.c.ρ * 2e-4
        @. Y_sh.c.ρq_rai = Y_sh.c.ρ * 1e-4
        @. Y_sh.c.ρq_gas_A = Y_sh.c.ρ * (1e-3 + 5e-4 * sin(π * zc_sh / zmax_sh))
        p_sh = CA.build_cache(Y_sh, sh_model, params, FT(1), DateTime(2010, 1, 1), nothing)
        CA.constrain_state!(Y_sh, p_sh, FT(60))
        CA.set_precomputed_quantities!(Y_sh, p_sh, FT(60))

        Yₜ_sh = similar(Y_sh)
        Yₜ_lim_sh = similar(Y_sh)
        if !(ClimaComms.device(ctx) isa ClimaComms.CUDADevice)
            Yₜ_pf = zero(Y_sh)
            @test _allocs_prescribed_flow(Yₜ_pf, Y_sh, p_sh, FT(60)) == 0
        end
        CA.fully_explicit_tendency!(Yₜ_sh, Yₜ_lim_sh, Y_sh, p_sh, FT(60))
        @test maximum(abs, Array(parent(Yₜ_sh.c.ρq_gas_A))) > 0

        # Compare against `remaining_tendency!` + wave-core `implicit_tendency!`
        # (for `ρ` and `f.u₃`, and full `Yₜ` at `upw === :none`), and explicit
        # `vertical_transport` at `Val(upw)` on tracers and `e_tot - Φ`.
        Yₜ_ref = similar(Y_sh)
        Yₜ_lim_ref = similar(Y_sh)
        Yₜ_imp_ref = similar(Y_sh)
        CA.remaining_tendency!(Yₜ_ref, Yₜ_lim_ref, Y_sh, p_sh, FT(60))
        CA.implicit_tendency!(Yₜ_imp_ref, Y_sh, p_sh, FT(60))
        ᶜJ = Fields.local_geometry_field(axes(Y_sh.c)).J
        ᶠJ = Fields.local_geometry_field(axes(Y_sh.f)).J
        ᶠu³ = p_sh.precomputed.ᶠu³
        ᶠρ = @. CA.ᶠinterp(Y_sh.c.ρ * ᶜJ) / ᶠJ
        ᶜq_gas_A = @. Y_sh.c.ρq_gas_A / Y_sh.c.ρ
        if upw === :none
            Yₜ_ref .+= Yₜ_imp_ref
            CA.add_vertical_transport!(
                Yₜ_ref.c.ρq_gas_A,
                ᶠρ,
                ᶠu³,
                ᶜq_gas_A,
                zero(FT),
                Val(upw),
            )
        else
            ᶜΦ = p_sh.core.ᶜΦ
            @. Yₜ_ref.c.ρ += Yₜ_imp_ref.c.ρ
            @. Yₜ_ref.f += Yₜ_imp_ref.f
            @. Yₜ_ref.c.ρe_tot -=
                CA.ᶜadvdivᵥ(
                    ᶠρ * ᶠu³ * CA.ᶠinterp(ᶜΦ) + ᶠu³ * CA.ᶠinterp(p_sh.precomputed.ᶜp),
                )
            ᶜe_rel = @. Y_sh.c.ρe_tot / Y_sh.c.ρ - ᶜΦ
            ᶜq_tot = @. Y_sh.c.ρq_tot / Y_sh.c.ρ
            ᶜq_lcl = @. Y_sh.c.ρq_lcl / Y_sh.c.ρ
            ᶜq_rai = @. Y_sh.c.ρq_rai / Y_sh.c.ρ
            CA.add_vertical_transport!(Yₜ_ref.c.ρe_tot, ᶠρ, ᶠu³, ᶜe_rel, zero(FT), Val(upw))
            CA.add_vertical_transport!(Yₜ_ref.c.ρq_tot, ᶠρ, ᶠu³, ᶜq_tot, zero(FT), Val(upw))
            CA.add_vertical_transport!(Yₜ_ref.c.ρq_lcl, ᶠρ, ᶠu³, ᶜq_lcl, zero(FT), Val(upw))
            CA.add_vertical_transport!(Yₜ_ref.c.ρq_rai, ᶠρ, ᶠu³, ᶜq_rai, zero(FT), Val(upw))
            CA.add_vertical_transport!(
                Yₜ_ref.c.ρq_gas_A,
                ᶠρ,
                ᶠu³,
                ᶜq_gas_A,
                zero(FT),
                Val(upw),
            )
        end
        @test isapprox(
            Array(parent(Yₜ_sh.c)),
            Array(parent(Yₜ_ref.c));
            rtol = 1e-12,
            atol = 1e-13,
        )
        @test isapprox(
            Array(parent(Yₜ_sh.f)),
            Array(parent(Yₜ_ref.f));
            rtol = 1e-12,
            atol = 1e-13,
        )
    end

    mktempdir() do tmpdir
        sh_model = CA.AtmosModel(
            sh_grid;
            params,
            setup = sh_setup,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            energy_humidity_upwinding = :first_order,
            tracer_upwinding = :first_order,
            hyperdiff = nothing,
        )
        hook_calls = Ref(0)
        sim_sh = CA.AtmosSimulation(
            sh_model;
            dt = 1,
            t_end = 3,
            output_dir = joinpath(tmpdir, "shipway_split"),
            default_callbacks = false,
            post_step_hook! = (Y, p, t, Δt, cache) -> (hook_calls[] += 1; nothing),
        )
        CA.solve_atmos!(sim_sh)
        @test hook_calls[] == 3
        @test all(isfinite, Array(parent(sim_sh.integrator.u.c.ρ)))
        @test all(isfinite, Array(parent(sim_sh.integrator.u.c.ρq_tot)))
    end
end
