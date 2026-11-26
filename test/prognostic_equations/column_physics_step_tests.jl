#=
julia --project=.buildkite
using Revise; include("test/prognostic_equations/column_physics_step_tests.jl")
=#
using Test
using Dates
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaAtmos: TD, SA
import ClimaCore: Fields, Geometry, Spaces, Operators, DataLayouts
import ClimaCore.Geometry: ⊗
import ClimaCore.MatrixFields

# ============================================================================
# Test helpers
# ============================================================================

function _build_sim(
    FT;
    grid = nothing,
    z_elem = 10,
    z_max = FT(3000),
    z_stretch = false,
    vertical_diffusion = nothing,
    smagorinsky_lilly = nothing,
    turbconv_model = nothing,
    edmfx_model = CA.EDMFXModel(;
        scale_blending_method = CA.SmoothMinimumBlending(),
    ),
    microphysics_model = CA.EquilibriumMicrophysics0M(),
    surface_setup = CA.AtmosSurface(),
    disable_surface_flux_tendency = true,
    advection_test = false,
    dt = FT(60),
    sgs_quadrature = CA.SGSQuadrature(FT; distribution = CA.GridMeanSGS()),
)
    ctx = ClimaComms.SingletonCommsContext()
    g = isnothing(grid) ? CA.ColumnGrid(FT; context = ctx, z_elem, z_max, z_stretch) : grid
    params = CA.ClimaAtmosParameters(FT; microphysics_model)
    model = CA.AtmosModel(
        g;
        params,
        vertical_diffusion,
        smagorinsky_lilly,
        turbconv_model,
        edmfx_model,
        microphysics_model,
        surface = surface_setup,
        disable_surface_flux_tendency,
        sgs_quadrature,
        advection_test,
    )
    Y = CA.initial_state(model)
    p = CA.build_cache(
        Y,
        model,
        model.params,
        dt,
        DateTime(2010, 1, 1),
        nothing,
    )
    return (; Y, p)
end

@noinline function _alloc_check(f::F, args::Vararg{Any, N}) where {F, N}
    f(args...)  # warmup
    return @allocated f(args...)
end

_is_cpu(field) = ClimaComms.device(field) isa ClimaComms.AbstractCPUDevice

function _col_sum(field, ᶜJ)
    return sum(Array(parent(field .* ᶜJ)))
end

function _with_cache(
    p::CA.AtmosCache;
    atmos = p.atmos,
    dt = p.dt,
    params = p.params,
)
    vals = map(fieldnames(CA.AtmosCache)) do f
        f === :atmos ? atmos :
        (f === :dt ? dt : (f === :params ? params : getfield(p, f)))
    end
    return CA.AtmosCache(vals...)
end

# ============================================================================
# Tests
# ============================================================================

# Horizontal surface area `ᶠJ ∂ξ³/∂x³` of a face.
_surface_area(ᶠlg) = ᶠlg.J * parent(ᶠlg.∂ξ∂x)[end]

@testset "Column Physics Step" begin

    @testset "Thomas solvers" begin
        FT = Float64
        nz = 20

        # Scalar tridiagonal
        d = fill(FT(4.0), nz)
        u = fill(FT(-1.0), nz - 1)
        l = fill(FT(-1.0), nz - 1)
        b = rand(FT, nz)
        x = zeros(FT, nz)
        A = diagm(0 => d, 1 => u, -1 => l)
        x_ref = A \ b
        CA.column_thomas_solve!(x, copy(b), d, copy(u), l, nz)
        @test x ≈ x_ref rtol = 1e-12

        # Shared-matrix 2-RHS Thomas
        b1 = rand(FT, nz)
        b2 = rand(FT, nz)
        x1 = zeros(FT, nz)
        x2 = zeros(FT, nz)
        CA.column_thomas_solve_2!(x1, x2, copy(b1), copy(b2), d, copy(u), l, nz)
        @test x1 ≈ A \ b1 rtol = 1e-12
        @test x2 ≈ A \ b2 rtol = 1e-12

        # nz == 1 boundary guard
        d1 = FT[4.0];
        b1_1 = FT[8.0];
        x1_1 = FT[0.0]
        b2_1 = FT[12.0];
        x2_1 = FT[0.0]
        u_empty = FT[];
        l_empty = FT[]
        CA.column_thomas_solve!(x1_1, copy(b1_1), d1, u_empty, l_empty, 1)
        @test x1_1[1] ≈ FT(2.0) rtol = 1e-14
        CA.column_thomas_solve_2!(
            x1_1, x2_1, copy(b1_1), copy(b2_1), d1, u_empty, l_empty, 1,
        )
        @test x1_1[1] ≈ FT(2.0) rtol = 1e-14
        @test x2_1[1] ≈ FT(3.0) rtol = 1e-14
    end

    @testset "Face conductance, area ratio, and metric upwind3 on flat, stretched, terrain columns" begin
        FT = Float64
        (; Y) = _build_sim(FT; z_elem = 5, z_max = FT(1000))
        ᶠlg = Fields.local_geometry_field(Y.f)
        G_host = vec(Array(parent(@. CA.face_conductance(ᶠlg, FT(1.0)))))
        A_host = vec(Array(parent(@. _surface_area(ᶠlg))))
        G = G_host[2]
        @test G > 0
        for f in 2:5
            @test G_host[f] ≈ G rtol = 1e-14
        end
        @test A_host[1] ≈ FT(1.0) rtol = 1e-14

        # Metric-aware 3rd-order finite-volume reconstruction exact on quadratics over
        # non-uniform stretched vertical faces.
        nz_s = 10
        ᶠz_str = cumsum(FT[0; [100 * FT(1.2)^k for k in 0:(nz_s - 1)]])
        p_quad(z) = FT(1.5) + FT(2.0) * (z / FT(1000)) - FT(0.7) * (z / FT(1000))^2
        P_quad(z) =
            FT(1.5) * z + FT(1.0) * z^2 / FT(1000) - FT(0.7) / 3 * z^3 / FT(1000)^2
        χ_cell = [
            (P_quad(ᶠz_str[k + 1]) - P_quad(ᶠz_str[k])) / (ᶠz_str[k + 1] - ᶠz_str[k])
            for k in 1:nz_s
        ]
        for f in 3:(nz_s - 1)
            @test CA._face_ho_upwind3(χ_cell, f, FT(1.0), ᶠz_str, nz_s) ≈ p_quad(ᶠz_str[f]) rtol =
                1e-12
            @test CA._face_ho_upwind3(χ_cell, f, FT(-1.0), ᶠz_str, nz_s) ≈ p_quad(ᶠz_str[f]) rtol =
                1e-12
        end

        # Non-unit horizontal area and terrain-sloped g³³ on 3D CosineTopography BoxGrid
        ctx = ClimaComms.SingletonCommsContext()
        topo = CA.CosineTopography{3, FT}(; h_max = FT(400), λ = FT(50e3))
        g_topo = CA.BoxGrid(
            FT;
            context = ctx,
            x_elem = 2,
            y_elem = 2,
            z_elem = 6,
            z_max = FT(3000),
            x_max = FT(50e3),
            y_max = FT(50e3),
            nh_poly = 2,
            topography = topo,
        )
        fspace_topo = Spaces.FaceExtrudedFiniteDifferenceSpace(g_topo)
        ᶠlg_topo = Fields.local_geometry_field(fspace_topo)
        A_topo = Array(parent(@. _surface_area(ᶠlg_topo)))
        G_topo = Array(parent(@. CA.face_conductance(ᶠlg_topo, FT(1.0))))
        flat_fac = Array(parent(@. _surface_area(ᶠlg_topo)^2 / ᶠlg_topo.J))
        @test minimum(A_topo) > FT(1e6)  # J_h >> 1 on 50km horizontal elements
        @test maximum(G_topo ./ flat_fac) > FT(1.001)  # |∇h|² increases g³³ on slopes
    end

    @testset "Grid-mean vertical diffusion conservation, equilibrium, Robin vs Neumann" begin
        FT = Float64
        params_ref = CA.ClimaAtmosParameters(FT)
        vdp = CAP.vert_diff_params(params_ref)
        vd = CA.VerticalDiffusion{FT}(;
            disable_momentum_vertical_diffusion = false,
            C_E = vdp.C_E,
        )
        # Standalone surface (MoninObukhov)
        mo = CA.SurfaceConditions.MoninObukhov(; z0m = FT(1e-3), z0b = FT(1e-3))
        sfc_sa = CA.AtmosSurface(;
            flux_scheme = mo,
            temperature = CA.SurfaceConditions.AnalyticTemperature(Returns(FT(300))),
        )
        (; Y, p) = _build_sim(
            FT;
            z_elem = 12,
            z_max = FT(3000),
            vertical_diffusion = vd,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_sa,
            disable_surface_flux_tendency = false,
            dt = FT(60),
        )
        Y0 = deepcopy(Y)
        ᶜJ = Fields.local_geometry_field(Y.c).J
        ᶠlg = Fields.local_geometry_field(Y.f)
        sfc_factor = vec(Array(parent(@. CA.surface_c3_flux_factor(ᶠlg))))[1]
        tps = CAP.thermodynamics_params(p.params)

        # 1. Closed column conservation (dry air cell-wise, water & energy column-wise)
        copyto!(Y, Y0)
        z = Fields.coordinate_field(Y.c).z
        @. Y.c.uₕ = CA.C12(FT(10), FT(-5))
        @. Y.c.ρq_tot = Y.c.ρ * (FT(0.005) + FT(0.008) * exp(-z / FT(1000)))
        @. Y.c.ρq_lcl = Y.c.ρ * FT(0.0005) * exp(-((z - FT(1200)) / FT(400))^2)
        @. Y.c.ρq_icl = Y.c.ρ * FT(0.0002) * exp(-((z - FT(2200)) / FT(400))^2)
        @. Y.c.ρq_rai = Y.c.ρ * FT(1e-4)
        @. Y.c.ρq_sno = Y.c.ρ * FT(5e-5)

        CA.set_precomputed_quantities!(Y, p, FT(0))
        # Zero surface fluxes for closed-column budget check
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_uₕ =
            CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))

        ρ_d_before = Array(parent(Y.c.ρ .- Y.c.ρq_tot))
        qt_int_before = _col_sum(Y.c.ρq_tot, ᶜJ)
        ql_int_before = _col_sum(Y.c.ρq_lcl, ᶜJ)
        qi_int_before = _col_sum(Y.c.ρq_icl, ᶜJ)
        qr_before = Array(parent(Y.c.ρq_rai))
        qs_before = Array(parent(Y.c.ρq_sno))
        e_int_before = _col_sum(Y.c.ρe_tot, ᶜJ)

        Δt = FT(300)
        CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), Δt)

        ρ_d_after = Array(parent(Y.c.ρ .- Y.c.ρq_tot))
        @test maximum(abs.(ρ_d_after .- ρ_d_before)) < 100 * eps(FT)
        @test abs(_col_sum(Y.c.ρq_tot, ᶜJ) - qt_int_before) <
              100 * eps(FT) * abs(qt_int_before)
        @test abs(_col_sum(Y.c.ρq_lcl, ᶜJ) - ql_int_before) <
              100 * eps(FT) * abs(ql_int_before)
        @test abs(_col_sum(Y.c.ρq_icl, ᶜJ) - qi_int_before) <
              100 * eps(FT) * abs(qi_int_before)
        @test abs(_col_sum(Y.c.ρe_tot, ᶜJ) - e_int_before) <
              100 * eps(FT) * abs(e_int_before)
        # Precipitation species do not diffuse under VerticalDiffusion
        @test Array(parent(Y.c.ρq_rai)) == qr_before
        @test Array(parent(Y.c.ρq_sno)) == qs_before

        # 2. Adiabatic equilibrium (c_p ∇T + ∇Φ = 0, ∇q = 0) fixed point
        copyto!(Y, Y0)
        q0 = FT(0.008)
        cp0 = TD.cp_m(tps, q0, FT(0), FT(0))
        T_sfc = FT(300)
        @. Y.c.uₕ = CA.C12(FT(8), FT(0))
        @. Y.c.ρq_tot = Y.c.ρ * q0
        @. Y.c.ρq_lcl = FT(0)
        @. Y.c.ρq_icl = FT(0)
        @. Y.c.ρq_rai = FT(0)
        @. Y.c.ρq_sno = FT(0)
        ᶜT_eq = @. T_sfc - p.core.ᶜΦ / cp0
        ᶜK = @. 0.5 * (64)
        @. Y.c.ρe_tot = Y.c.ρ * TD.total_energy(tps, ᶜK, p.core.ᶜΦ, ᶜT_eq, q0)
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_uₕ =
            CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))
        e_eq_before = Array(parent(Y.c.ρe_tot))
        CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), FT(120))
        @test maximum(abs.(Array(parent(Y.c.ρe_tot)) .- e_eq_before)) <
              1e-8 * maximum(abs, e_eq_before)

        # 2b. Saturated EquilibriumMicrophysics0M() fixed point well-mixed in (mse, q_tot)
        sim_0m = _build_sim(
            FT;
            z_elem = 10,
            z_max = FT(3000),
            vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
                disable_momentum_vertical_diffusion = false,
                H = FT(2000),
                D₀ = FT(50),
            ),
            microphysics_model = CA.EquilibriumMicrophysics0M(),
            disable_surface_flux_tendency = true,
        )
        Y_0m, p_0m = sim_0m.Y, sim_0m.p
        tps_0m = CAP.thermodynamics_params(p_0m.params)
        q_sat_mix = FT(0.02)
        mse_mix =
            TD.enthalpy(tps_0m, FT(280), q_sat_mix, FT(0.01), FT(0)) + FT(1.5e4)
        @. Y_0m.c.uₕ = CA.C12(FT(0), FT(0))
        @. Y_0m.f.u₃ = CA.C3(FT(0))
        ᶜp_0m = p_0m.precomputed.ᶜp
        ᶜΦ_0m = p_0m.core.ᶜΦ
        sa_eq_0m = @. TD.saturation_adjustment(
            tps_0m, TD.ph(), ᶜp_0m, mse_mix - ᶜΦ_0m, q_sat_mix,
        )
        @. Y_0m.c.ρ = TD.air_density(
            tps_0m, sa_eq_0m.T, ᶜp_0m, q_sat_mix, sa_eq_0m.q_liq, sa_eq_0m.q_ice,
        )
        @. Y_0m.c.ρq_tot = Y_0m.c.ρ * q_sat_mix
        @. Y_0m.c.ρe_tot =
            Y_0m.c.ρ * (
                TD.internal_energy(
                    tps_0m, sa_eq_0m.T, q_sat_mix, sa_eq_0m.q_liq, sa_eq_0m.q_ice,
                ) + ᶜΦ_0m
            )
        CA.set_precomputed_quantities!(Y_0m, p_0m, FT(0))
        @test minimum(Array(parent(p_0m.precomputed.ᶜq_liq .+ p_0m.precomputed.ᶜq_ice))) >
              FT(1e-4)
        T_0m_before = Array(parent(p_0m.precomputed.ᶜT))
        CA.step_vertical_diffusion_and_surface!(Y_0m, p_0m, FT(0), FT(300))
        CA.set_precomputed_quantities!(Y_0m, p_0m, FT(0))
        @test maximum(abs.(Array(parent(p_0m.precomputed.ᶜT)) .- T_0m_before)) < FT(0.05)

        # 3. Standalone implicit Robin drag over large Δt: uₕ stays bounded,
        # does not reverse sign, realized flux written back, and momentum budget holds.
        copyto!(Y, Y0)
        @. Y.c.uₕ = CA.C12(FT(10), FT(0))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        u_vec_before = Array(parent(Y.c.uₕ.components.data.:1))
        m_robin_before = Array(parent(Y.c.ρ .* ᶜJ))
        u1_before = vec(u_vec_before)[1]
        τ_old = vec(
            Array(parent(p.precomputed.sfc_conditions.ρ_flux_uₕ.components.data.:1)),
        )[1]
        @test τ_old < 0  # downward momentum flux (drag)
        Δt_robin = FT(3600)
        CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), Δt_robin)
        u_vec_after = Array(parent(Y.c.uₕ.components.data.:1))
        u1_after = vec(u_vec_after)[1]
        τ_new = vec(
            Array(parent(p.precomputed.sfc_conditions.ρ_flux_uₕ.components.data.:1)),
        )[1]
        @test 0 <= u1_after < u1_before
        @test τ_new < 0 && abs(τ_new) < abs(τ_old)
        Δmom_col = sum((u_vec_after .- u_vec_before) .* m_robin_before)
        @test Δmom_col ≈ Δt_robin * sfc_factor * τ_new rtol = 1e-12

        # 3b. Stiff condensate diffusion (K Δt / Δz² ≈ 1000): bounded physical T
        p_stiff = _with_cache(
            p;
            atmos = CA.AtmosModel(
                p.atmos;
                vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
                    disable_momentum_vertical_diffusion = false,
                    H = FT(1e6),
                    D₀ = FT(1e5),
                ),
            ),
        )
        copyto!(Y, Y0)
        @. Y.c.ρq_tot = Y.c.ρ * FT(0.015)
        @. Y.c.ρq_lcl = Y.c.ρ * FT(3e-3) * exp(-((z - FT(1500)) / FT(400))^2)
        @. Y.c.ρq_icl = Y.c.ρ * FT(1e-3) * exp(-((z - FT(2200)) / FT(400))^2)
        CA.set_precomputed_quantities!(Y, p_stiff, FT(0))
        @. p_stiff.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p_stiff.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p_stiff.precomputed.sfc_conditions.ρ_flux_uₕ =
            CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))
        T_stiff_before = Array(parent(p_stiff.precomputed.ᶜT))
        e_stiff_before = _col_sum(Y.c.ρe_tot, ᶜJ)
        ql_stiff_before = _col_sum(Y.c.ρq_lcl, ᶜJ)
        CA.step_vertical_diffusion_and_surface!(Y, p_stiff, FT(0), FT(625))
        CA.set_precomputed_quantities!(Y, p_stiff, FT(0))
        T_stiff_after = Array(parent(p_stiff.precomputed.ᶜT))
        @test minimum(T_stiff_after) > FT(200) && maximum(T_stiff_after) < FT(350)
        @test maximum(abs.(T_stiff_after .- T_stiff_before)) < FT(10)
        @test abs(_col_sum(Y.c.ρq_lcl, ᶜJ) - ql_stiff_before) <
              100 * eps(FT) * ql_stiff_before
        @test abs(_col_sum(Y.c.ρe_tot, ᶜJ) - e_stiff_before) <
              100 * eps(FT) * abs(e_stiff_before)

        # 4. Zero heap allocations (CPU)
        if _is_cpu(Y.c)
            @test _alloc_check(
                CA.step_vertical_diffusion_and_surface!,
                Y,
                p,
                FT(0),
                FT(60),
            ) == 0
        end

        # 5. DecayWithHeightDiffusion & vertical SmagorinskyLilly conservation + zero alloc
        p_dwh = _with_cache(
            p;
            atmos = CA.AtmosModel(
                p.atmos;
                vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
                    disable_momentum_vertical_diffusion = false,
                    H = FT(1200),
                    D₀ = FT(5),
                ),
            ),
        )
        copyto!(Y, Y0)
        @. Y.c.ρq_tot = Y.c.ρ * (FT(0.004) + FT(0.006) * exp(-z / FT(1000)))
        CA.set_precomputed_quantities!(Y, p_dwh, FT(0))
        @. p_dwh.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p_dwh.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p_dwh.precomputed.sfc_conditions.ρ_flux_uₕ =
            CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))
        qt_dwh_before = _col_sum(Y.c.ρq_tot, ᶜJ)
        e_dwh_before = _col_sum(Y.c.ρe_tot, ᶜJ)
        CA.step_vertical_diffusion_and_surface!(Y, p_dwh, FT(0), FT(120))
        @test abs(_col_sum(Y.c.ρq_tot, ᶜJ) - qt_dwh_before) < 100 * eps(FT) * qt_dwh_before
        @test abs(_col_sum(Y.c.ρe_tot, ᶜJ) - e_dwh_before) <
              100 * eps(FT) * abs(e_dwh_before)

        sim_smag = _build_sim(
            FT;
            z_elem = 8,
            z_max = FT(2400),
            smagorinsky_lilly = CA.SmagorinskyLilly(; axes = :UV_W),
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            disable_surface_flux_tendency = true,
        )
        Y_smag = sim_smag.Y
        p_smag = sim_smag.p
        ᶜJ_smag = Fields.local_geometry_field(Y_smag.c).J
        z_smag = Fields.coordinate_field(Y_smag.c).z
        zf_smag = Fields.coordinate_field(Y_smag.f).z
        @. Y_smag.c.uₕ = CA.C12(FT(100) * (z_smag / FT(2400)), FT(0))
        @. Y_smag.f.u₃ = CA.C3(FT(30) * sin(FT(π) * zf_smag / FT(2400)) * FT(300))
        @. Y_smag.c.ρq_tot = Y_smag.c.ρ * (FT(0.005) + FT(0.005) * exp(-z_smag / FT(1000)))
        @. Y_smag.c.ρq_rai = Y_smag.c.ρ * FT(1e-4) * exp(-((z_smag - FT(1200)) / FT(400))^2)
        CA.set_precomputed_quantities!(Y_smag, p_smag, FT(0))
        qt_smag_before = _col_sum(Y_smag.c.ρq_tot, ᶜJ_smag)
        qr_smag_before = _col_sum(Y_smag.c.ρq_rai, ᶜJ_smag)
        e_smag_before = _col_sum(Y_smag.c.ρe_tot, ᶜJ_smag)
        qr_smag_vec = Array(parent(Y_smag.c.ρq_rai))
        CA.step_vertical_diffusion_and_surface!(Y_smag, p_smag, FT(0), FT(60))
        @test abs(_col_sum(Y_smag.c.ρq_tot, ᶜJ_smag) - qt_smag_before) <
              100 * eps(FT) * qt_smag_before
        @test abs(_col_sum(Y_smag.c.ρq_rai, ᶜJ_smag) - qr_smag_before) <
              100 * eps(FT) * qr_smag_before
        @test abs(_col_sum(Y_smag.c.ρe_tot, ᶜJ_smag) - e_smag_before) <
              100 * eps(FT) * abs(e_smag_before)
        @test maximum(abs.(Array(parent(Y_smag.c.ρq_rai)) .- qr_smag_vec)) > FT(0)
        if _is_cpu(Y_smag.c)
            @test _alloc_check(
                CA.step_vertical_diffusion_and_surface!,
                Y_smag,
                p_smag,
                FT(0),
                FT(60),
            ) == 0
        end

        # 6. Surface-only path (no vertical diffusion / Smagorinsky / turbconv closure)
        p_sfconly = _with_cache(
            p;
            atmos = CA.AtmosModel(p.atmos; vertical_diffusion = nothing),
        )
        copyto!(Y, Y0)
        @. Y.c.uₕ = CA.C12(FT(10), FT(0))
        @. Y.c.ρq_tot = Y.c.ρ * (FT(0.005) + FT(0.005) * exp(-z / FT(1000)))
        CA.set_precomputed_quantities!(Y, p_sfconly, FT(0))
        qt_so_before = _col_sum(Y.c.ρq_tot, ᶜJ)
        e_so_before = _col_sum(Y.c.ρe_tot, ᶜJ)
        u_so_before = Array(parent(Y.c.uₕ.components.data.:1))
        m_so_before = Array(parent(Y.c.ρ .* ᶜJ))
        qt_so_cells_before = Array(parent(Y.c.ρq_tot))
        Δt_so = FT(120)
        CA.step_vertical_diffusion_and_surface!(Y, p_sfconly, FT(0), Δt_so)
        evap_so = vec(
            Array(
                parent(
                    p_sfconly.precomputed.sfc_conditions.ρ_flux_q_tot.components.data.:1,
                ),
            ),
        )[1]
        shf_so = vec(
            Array(
                parent(
                    p_sfconly.precomputed.sfc_conditions.ρ_flux_h_tot.components.data.:1,
                ),
            ),
        )[1]
        τ_so = vec(
            Array(
                parent(p_sfconly.precomputed.sfc_conditions.ρ_flux_uₕ.components.data.:1),
            ),
        )[1]
        @test _col_sum(Y.c.ρq_tot, ᶜJ) ≈ qt_so_before + Δt_so * sfc_factor * evap_so rtol =
            1e-12
        @test _col_sum(Y.c.ρe_tot, ᶜJ) ≈ e_so_before + Δt_so * sfc_factor * shf_so rtol =
            1e-12
        u_so_after = Array(parent(Y.c.uₕ.components.data.:1))
        @test sum((u_so_after .- u_so_before) .* m_so_before) ≈
              Δt_so * sfc_factor * τ_so rtol = 1e-12
        # Cells above level 1 see zero turbulent transport when all closures are off
        @test vec(Array(parent(Y.c.ρq_tot)))[2:end] == vec(qt_so_cells_before)[2:end]

        # Surface-only path with NonEquilibriumMicrophysics1M and dirty strain scratch:
        # interior ρq_tot and ρe_tot (k >= 2) stay unchanged when G == 0.
        sim_so_ne = _build_sim(
            FT;
            z_elem = 10,
            z_max = FT(3000),
            vertical_diffusion = nothing,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_sa,
            disable_surface_flux_tendency = false,
        )
        Y_sone, p_sone = sim_so_ne.Y, sim_so_ne.p
        CA.set_precomputed_quantities!(Y_sone, p_sone, FT(0))
        @. p_sone.scratch.ᶠtemp_UVWxUVW.components.data.:2 = FT(1e-3)
        @. p_sone.scratch.ᶠtemp_UVWxUVW.components.data.:3 = FT(-1e-3)
        ᶜJ_sone = Fields.local_geometry_field(Y_sone.c).J
        ᶠlg_sone = Fields.local_geometry_field(Y_sone.f)
        sfc_fac_sone = vec(Array(parent(@. CA.surface_c3_flux_factor(ᶠlg_sone))))[1]
        e_sone_total_before = _col_sum(Y_sone.c.ρe_tot, ᶜJ_sone)
        e_sone_cells_before = Array(parent(Y_sone.c.ρe_tot))
        qt_sone_cells_before = Array(parent(Y_sone.c.ρq_tot))
        CA.step_vertical_diffusion_and_surface!(Y_sone, p_sone, FT(0), Δt_so)
        shf_sone = vec(
            Array(
                parent(p_sone.precomputed.sfc_conditions.ρ_flux_h_tot.components.data.:1),
            ),
        )[1]
        @test _col_sum(Y_sone.c.ρe_tot, ᶜJ_sone) ≈
              e_sone_total_before + Δt_so * sfc_fac_sone * shf_sone rtol = 1e-12
        @test vec(Array(parent(Y_sone.c.ρq_tot)))[2:end] ==
              vec(qt_sone_cells_before)[2:end]
        @test vec(Array(parent(Y_sone.c.ρe_tot)))[2:end] ==
              vec(e_sone_cells_before)[2:end]

        # 7. Float32 vertical diffusion conservation, Robin drag, and zero alloc
        FT32 = Float32
        vdp32 = CAP.vert_diff_params(CA.ClimaAtmosParameters(FT32))
        vd32 = CA.VerticalDiffusion{FT32}(;
            disable_momentum_vertical_diffusion = false,
            C_E = vdp32.C_E,
        )
        mo32 = CA.SurfaceConditions.MoninObukhov(; z0m = FT32(1e-3), z0b = FT32(1e-3))
        sfc32 = CA.AtmosSurface(;
            flux_scheme = mo32,
            temperature = CA.SurfaceConditions.AnalyticTemperature(Returns(FT32(300))),
        )
        sim32 = _build_sim(
            FT32;
            z_elem = 10,
            z_max = FT32(3000),
            vertical_diffusion = vd32,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc32,
            disable_surface_flux_tendency = false,
            dt = FT32(60),
        )
        Y32, p32 = sim32.Y, sim32.p
        ᶜJ32 = Fields.local_geometry_field(Y32.c).J
        ᶠlg32 = Fields.local_geometry_field(Y32.f)
        sfc_fac32 = vec(Array(parent(@. CA.surface_c3_flux_factor(ᶠlg32))))[1]
        z32 = Fields.coordinate_field(Y32.c).z
        @. Y32.c.uₕ = CA.C12(FT32(10), FT32(-4))
        @. Y32.c.ρq_tot = Y32.c.ρ * (FT32(0.005) + FT32(0.006) * exp(-z32 / FT32(1000)))
        @. Y32.c.ρq_lcl = Y32.c.ρ * FT32(4e-4) * exp(-((z32 - FT32(1200)) / FT32(400))^2)
        @. Y32.c.ρq_icl = Y32.c.ρ * FT32(2e-4) * exp(-((z32 - FT32(2200)) / FT32(400))^2)
        CA.set_precomputed_quantities!(Y32, p32, FT32(0))
        @. p32.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT32(0))
        @. p32.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT32(0))
        u32_before = Array(parent(Y32.c.uₕ.components.data.:1))
        m32_before = Array(parent(Y32.c.ρ .* ᶜJ32))
        ρd32_before = Array(parent(Y32.c.ρ .- Y32.c.ρq_tot))
        qt32_before = _col_sum(Y32.c.ρq_tot, ᶜJ32)
        ql32_before = _col_sum(Y32.c.ρq_lcl, ᶜJ32)
        qi32_before = _col_sum(Y32.c.ρq_icl, ᶜJ32)
        e32_before = _col_sum(Y32.c.ρe_tot, ᶜJ32)
        Δt32 = FT32(120)
        CA.step_vertical_diffusion_and_surface!(Y32, p32, FT32(0), Δt32)
        τ32_new = vec(
            Array(parent(p32.precomputed.sfc_conditions.ρ_flux_uₕ.components.data.:1)),
        )[1]
        u32_after = Array(parent(Y32.c.uₕ.components.data.:1))
        nz32 = Spaces.nlevels(axes(Y32.c))
        tol32 = FT32(50) * nz32 * eps(FT32)
        @test maximum(abs.(Array(parent(Y32.c.ρ .- Y32.c.ρq_tot)) .- ρd32_before)) < tol32
        @test abs(_col_sum(Y32.c.ρq_tot, ᶜJ32) - qt32_before) < tol32 * qt32_before
        @test abs(_col_sum(Y32.c.ρq_lcl, ᶜJ32) - ql32_before) < tol32 * ql32_before
        @test abs(_col_sum(Y32.c.ρq_icl, ᶜJ32) - qi32_before) < tol32 * qi32_before
        @test abs(_col_sum(Y32.c.ρe_tot, ᶜJ32) - e32_before) < tol32 * abs(e32_before)
        @test sum((u32_after .- u32_before) .* m32_before) ≈
              Δt32 * sfc_fac32 * τ32_new rtol = tol32
        if _is_cpu(Y32.c)
            @test _alloc_check(
                CA.step_vertical_diffusion_and_surface!, Y32, p32, FT32(0), Δt32,
            ) == 0
        end
    end

    @testset "ClimaCoupler Neumann BC, blended cloud positivity, and EDOnlyEDMFX TKE" begin
        FT = Float64
        # ClimaCoupler mode: flux_scheme === nothing, EDOnlyEDMFX + 1M + ρtke
        # triggers blended_cloud_species_flux = true.
        coupler_sfc = CA.AtmosSurface(;
            flux_scheme = nothing,
            temperature = CA.SurfaceConditions.AnalyticTemperature(Returns(FT(300))),
        )
        edmfx = CA.EDMFXModel(;
            sgs_diffusive_flux = true,
            scale_blending_method = CA.SmoothMinimumBlending(),
        )
        tc = CA.EDOnlyEDMFX()
        quad = CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), quadrature_order = 3)
        (; Y, p) = _build_sim(
            FT;
            z_elem = 10,
            z_max = FT(3000),
            turbconv_model = tc,
            edmfx_model = edmfx,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = coupler_sfc,
            disable_surface_flux_tendency = false,
            sgs_quadrature = quad,
        )
        Y0_ed = deepcopy(Y)
        ᶜJ = Fields.local_geometry_field(Y.c).J
        ᶠlg = Fields.local_geometry_field(Y.f)
        sfc_factor = vec(Array(parent(@. CA.surface_c3_flux_factor(ᶠlg))))[1]
        z_ed = Fields.coordinate_field(Y.c).z

        # Non-uniform cloud fields + strong surface fluxes -> unconditional positivity
        # and tight relative conservation.
        @. Y.c.ρtke = Y.c.ρ * FT(1.5)
        @. Y.c.ρq_tot = Y.c.ρ * (FT(0.012) + FT(0.004) * exp(-z_ed / FT(1200)))
        @. Y.c.ρq_lcl =
            Y.c.ρ * (FT(1e-6) + FT(5e-5) * exp(-((z_ed - FT(1200)) / FT(400))^2))
        @. Y.c.ρq_icl =
            Y.c.ρ * (FT(1e-6) + FT(3e-5) * exp(-((z_ed - FT(2100)) / FT(400))^2))
        CA.set_precomputed_quantities!(Y, p, FT(0))

        # Apply strong surface evaporation + heat flux (ClimaCoupler prescribed)
        evap = FT(1e-4)
        shf = FT(200.0)
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(evap)
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(shf)
        @. p.precomputed.sfc_conditions.ρ_flux_uₕ =
            CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))

        qt_before = _col_sum(Y.c.ρq_tot, ᶜJ)
        e_before = _col_sum(Y.c.ρe_tot, ᶜJ)
        ql_before = _col_sum(Y.c.ρq_lcl, ᶜJ)
        qi_before = _col_sum(Y.c.ρq_icl, ᶜJ)

        Δt = FT(600)
        CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), Δt)

        @test minimum(Array(parent(Y.c.ρq_lcl))) >= 0
        @test minimum(Array(parent(Y.c.ρq_icl))) >= 0
        @test _col_sum(Y.c.ρq_tot, ᶜJ) ≈ qt_before + Δt * sfc_factor * evap rtol = 1e-13
        @test _col_sum(Y.c.ρe_tot, ᶜJ) ≈ e_before + Δt * sfc_factor * shf rtol = 1e-13
        @test abs(_col_sum(Y.c.ρq_lcl, ᶜJ) - ql_before) < 100 * eps(FT) * ql_before
        @test abs(_col_sum(Y.c.ρq_icl, ᶜJ) - qi_before) < 100 * eps(FT) * qi_before

        # Flux-form conservation even when initial blended cloud field has tiny negative values
        copyto!(Y, Y0_ed)
        @. Y.c.ρtke = Y.c.ρ * FT(1.5)
        @. Y.c.ρq_tot = Y.c.ρ * FT(0.015)
        @. Y.c.ρq_lcl = Y.c.ρ * FT(1e-4) * exp(-((z_ed - FT(1500)) / FT(400))^2)
        ql_neg_host = Array(parent(Y.c.ρq_lcl))
        ql_neg_host[3] = FT(-1e-6)
        copyto!(parent(Y.c.ρq_lcl), ql_neg_host)
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_uₕ = CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))
        ql_neg_before = _col_sum(Y.c.ρq_lcl, ᶜJ)
        CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), FT(120))
        @test abs(_col_sum(Y.c.ρq_lcl, ᶜJ) - ql_neg_before) <
              100 * eps(FT) * abs(ql_neg_before)

        # EDOnlyEDMFX under step_column_physics!: steps ρtke, cell-1 buoyancy active,
        # shear production uses updated uₕ (bounded by KE loss), and refreshes K_h.
        copyto!(Y, Y0_ed)
        @. Y.c.ρtke = Y.c.ρ * FT(0.5)
        @. Y.c.uₕ = CA.C12(ifelse(z_ed < FT(600), FT(20), FT(0)), FT(0))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_uₕ = CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))
        tke_ed_before = Array(parent(Y.c.ρtke))
        Kh_ed_before = Array(parent(p.precomputed.ᶠK_h))
        CA.step_column_physics!(Y, p, FT(0), FT(120))
        @test maximum(abs.(Array(parent(Y.c.ρtke)) .- tke_ed_before)) > FT(1e-4)
        @test minimum(Array(parent(Y.c.ρtke))) >= 0
        @test maximum(abs.(Array(parent(p.precomputed.ᶠK_h)) .- Kh_ed_before)) > FT(1e-5)

        # Cell-1 buoyancy production check: unstable surface layer (N² < 0 at face 2)
        # gives strictly larger ρtke[1] than zero-buoyancy baseline with zero shear.
        p_nodiff_ed = _with_cache(
            p;
            atmos = CA.AtmosModel(p.atmos; disable_surface_flux_tendency = true),
        )
        copyto!(Y, Y0_ed)
        @. Y.c.uₕ = CA.C12(FT(0), FT(0))
        @. Y.f.u₃ = CA.C3(FT(0))
        @. Y.c.ρtke = Y.c.ρ * FT(2.0)
        CA.set_precomputed_quantities!(Y, p_nodiff_ed, FT(0))
        p_nodiff_ed.precomputed.ᶜK .= FT(0)
        Y_nb = deepcopy(Y)
        p_nb = deepcopy(p_nodiff_ed)
        # Inject unstable buoygrad = -1e-2 at face 2 with nonzero K_h vs neutral buoygrad = 0
        @. p_nodiff_ed.precomputed.ᶠK_h = FT(1.0)
        @. p_nb.precomputed.ᶠK_h = FT(1.0)
        bg_host = Array(parent(p_nodiff_ed.precomputed.ᶠbuoygrad))
        bg_host .= FT(0)
        bg_host[2] = FT(-1e-2)
        copyto!(parent(p_nodiff_ed.precomputed.ᶠbuoygrad), bg_host)
        fill!(p_nb.precomputed.ᶠbuoygrad, FT(0))
        CA.step_prognostic_tke!(Y, p_nodiff_ed, FT(60))
        CA.step_prognostic_tke!(Y_nb, p_nb, FT(60))
        @test vec(Array(parent(Y.c.ρtke)))[1] >
              vec(Array(parent(Y_nb.c.ρtke)))[1] + FT(1e-2)

        # Stiff shear production bounded by mean KE loss (small initial tke and
        # strong shear ensure shear production dominates dissipation so Δtke > 0).
        copyto!(Y, Y0_ed)
        @. Y.c.uₕ = CA.C12(ifelse(z_ed < FT(600), FT(25), FT(0)), FT(0))
        @. Y.c.ρtke = Y.c.ρ * FT(0.01)
        CA.set_precomputed_quantities!(Y, p_nodiff_ed, FT(0))
        fill!(p_nodiff_ed.precomputed.ᶠbuoygrad, FT(0))
        @. p_nodiff_ed.precomputed.ᶠK_u = FT(10.0)
        ke_before = _col_sum(Y.c.ρ .* p_nodiff_ed.precomputed.ᶜK, ᶜJ)
        tke_sum_before = _col_sum(Y.c.ρtke, ᶜJ)
        CA.step_vertical_diffusion_and_surface!(Y, p_nodiff_ed, FT(0), FT(60))
        CA.step_prognostic_tke!(Y, p_nodiff_ed, FT(60))
        ke_after = _col_sum(Y.c.ρ .* CA.compute_kinetic(Y.c.uₕ, Y.f.u₃), ᶜJ)
        tke_sum_after = _col_sum(Y.c.ρtke, ᶜJ)
        @test ke_before > ke_after
        @test tke_sum_after > tke_sum_before
        @test tke_sum_after - tke_sum_before <= (ke_before - ke_after) + FT(1e-10)
        if _is_cpu(Y.c)
            @test _alloc_check(
                CA.step_vertical_diffusion_and_surface!,
                Y,
                p,
                FT(0),
                FT(60),
            ) == 0
            @test _alloc_check(CA.step_prognostic_tke!, Y, p, FT(60)) == 0
            @test _alloc_check(CA.repair_column_water_conservatively!, Y, p) == 0
            @test _alloc_check(CA.step_coupled_microphysics_sedimentation!, Y, p, FT(60)) ==
                  0
            @test _alloc_check(CA.step_slab_ocean_turbulent_and_precip!, Y, p, FT(60)) == 0
        end
    end

    @testset "PrognosticEDMFX two-stream, extreme updrafts, 3D F̃, moisture step" begin
        FT = Float64
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
        sfc_sa = CA.AtmosSurface(;
            flux_scheme = mo,
            temperature = CA.SurfaceConditions.AnalyticTemperature(Returns(FT(302))),
        )
        (; Y, p) = _build_sim(
            FT;
            z_elem = 16,
            z_max = FT(4000),
            turbconv_model = tc,
            edmfx_model = edmfx,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_sa,
            disable_surface_flux_tendency = false,
            dt = FT(60),
        )
        Y0 = deepcopy(Y)
        ᶜJ = Fields.local_geometry_field(Y.c).J
        ᶠlg = Fields.local_geometry_field(Y.f)
        a_max = CAP.max_area(CAP.turbconv_params(p.params))
        tps = CAP.thermodynamics_params(p.params)

        # 1. Extreme updrafts: w¹ = 25 m/s, Δt = 900 s (Δt E > ρa⁰)
        copyto!(Y, Y0)
        z = Fields.coordinate_field(Y.c).z
        zf = Fields.coordinate_field(Y.f).z
        @. Y.c.ρtke = Y.c.ρ * FT(0.8)
        @. Y.c.ρq_tot = Y.c.ρ * (FT(0.002) + FT(0.012) * exp(-z / FT(1500)))
        @. Y.c.ρq_lcl = Y.c.ρ * FT(5e-4) * exp(-((z - FT(1500)) / FT(500))^2)
        @. Y.c.ρq_icl = Y.c.ρ * FT(1e-4) * exp(-((z - FT(3000)) / FT(500))^2)
        @. Y.c.ρq_rai = Y.c.ρ * FT(1e-5)
        @. Y.c.ρq_sno = Y.c.ρ * FT(5e-6)
        @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.05)
        @. Y.c.sgsʲs.:(1).q_tot = FT(0.014)
        @. Y.c.sgsʲs.:(1).q_lcl = FT(1e-3)
        @. Y.c.sgsʲs.:(1).q_icl = FT(2e-4)
        @. Y.c.sgsʲs.:(1).q_rai = FT(2e-5)
        @. Y.c.sgsʲs.:(1).q_sno = FT(1e-5)
        @. Y.c.sgsʲs.:(1).mse = FT(3.1e5)
        @. Y.f.sgsʲs.:(1).u₃ =
            CA.C3(ifelse((zf > 0) & (zf < FT(4000)), FT(25.0) * FT(250), FT(0.0)))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        # Zero surface flux for exact closed-column conservation check
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_uₕ =
            CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))

        ρd_int_before = _col_sum(Y.c.ρ .- Y.c.ρq_tot, ᶜJ)
        qt_int_before = _col_sum(Y.c.ρq_tot, ᶜJ)
        ql_int_before = _col_sum(Y.c.ρq_lcl, ᶜJ)
        qi_int_before = _col_sum(Y.c.ρq_icl, ᶜJ)
        qr_int_before = _col_sum(Y.c.ρq_rai, ᶜJ)
        qs_int_before = _col_sum(Y.c.ρq_sno, ᶜJ)
        e_int_before = _col_sum(Y.c.ρe_tot, ᶜJ)

        Δt_ext = FT(900)
        CA.step_edmf_and_env_turbulence!(Y, p, FT(0), Δt_ext)
        CA.step_prognostic_tke!(Y, p, Δt_ext)

        # Boundedness and positivity of raw step (updraft scalar bound enforced
        # by `enforce_edmf_updraft_constraints!` below, matching
        # `step_column_physics!`).
        @test minimum(Array(parent(Y.c.sgsʲs.:(1).ρa))) >= 0
        @test all(Array(parent(Y.c.sgsʲs.:(1).ρa ./ Y.c.ρ)) .<= a_max + 100 * eps(FT))
        @test minimum(Array(parent(Y.c.ρq_tot))) >= 0
        @test minimum(
            Array(
                parent(
                    Y.c.ρq_tot .- Y.c.ρq_lcl .- Y.c.ρq_icl .- Y.c.ρq_rai .- Y.c.ρq_sno,
                ),
            ),
        ) >= -100 * eps(FT)
        @test minimum(Array(parent(Y.c.ρq_lcl))) >= -100 * eps(FT)
        @test minimum(Array(parent(Y.c.ρq_icl))) >= -100 * eps(FT)
        @test minimum(Array(parent(Y.c.ρq_rai))) >= -100 * eps(FT)
        @test minimum(Array(parent(Y.c.ρq_sno))) >= -100 * eps(FT)
        @test minimum(Array(parent(Y.c.sgsʲs.:(1).q_tot))) >= 0
        @test minimum(Array(parent(Y.c.sgsʲs.:(1).q_lcl))) >= 0
        @test minimum(Array(parent(Y.c.sgsʲs.:(1).q_icl))) >= 0
        @test minimum(Array(parent(Y.c.sgsʲs.:(1).q_rai))) >= 0
        @test minimum(Array(parent(Y.c.sgsʲs.:(1).q_sno))) >= 0
        @test minimum(Array(parent(Y.c.ρtke))) >= 0
        @test all(>=(0), Array(parent(Y.f.sgsʲs.:(1).u₃.components.data.:1)))

        # Updraft filter enforces subdomain mass bound ρa¹ q_tot¹ ≤ ρ q_tot
        # without altering grid-mean conservation.
        CA.enforce_edmf_updraft_constraints!(Y, p, FT(0), tc)
        @test minimum(
            Array(parent(Y.c.ρq_tot .- Y.c.sgsʲs.:(1).ρa .* Y.c.sgsʲs.:(1).q_tot)),
        ) >= -100 * eps(FT)

        # Conservation to round-off (all water species including qi and qs)
        @test abs(_col_sum(Y.c.ρ .- Y.c.ρq_tot, ᶜJ) - ρd_int_before) < 1e-12 * ρd_int_before
        @test abs(_col_sum(Y.c.ρq_tot, ᶜJ) - qt_int_before) < 1e-12 * qt_int_before
        @test abs(_col_sum(Y.c.ρq_lcl, ᶜJ) - ql_int_before) < 1e-12 * (ql_int_before + 1)
        @test abs(_col_sum(Y.c.ρq_icl, ᶜJ) - qi_int_before) < 1e-12 * (qi_int_before + 1)
        @test abs(_col_sum(Y.c.ρq_rai, ᶜJ) - qr_int_before) < 1e-12 * (qr_int_before + 1)
        @test abs(_col_sum(Y.c.ρq_sno, ᶜJ) - qs_int_before) < 1e-12 * (qs_int_before + 1)
        @test abs(_col_sum(Y.c.ρe_tot, ᶜJ) - e_int_before) < 1e-12 * abs(e_int_before)

        # 1b. Nonzero Monin-Obukhov surface fluxes under PrognosticEDMFX + cell-wise ρ_d
        sfc_factor = vec(Array(parent(@. CA.surface_c3_flux_factor(ᶠlg))))[1]
        copyto!(Y, Y0)
        @. Y.c.uₕ = CA.C12(FT(8), FT(0))
        @. Y.c.ρtke = Y.c.ρ * FT(0.8)
        @. Y.c.ρq_tot = Y.c.ρ * (FT(0.004) + FT(0.008) * exp(-z / FT(1500)))
        @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.05)
        @. Y.c.sgsʲs.:(1).q_tot = FT(0.012)
        @. Y.c.sgsʲs.:(1).mse = FT(3.05e5)
        @. Y.f.sgsʲs.:(1).u₃ =
            CA.C3(ifelse((zf > 0) & (zf < FT(4000)), FT(10.0) * FT(250), FT(0.0)))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        evap_ed = vec(
            Array(parent(p.precomputed.sfc_conditions.ρ_flux_q_tot.components.data.:1)),
        )[1]
        shf_ed = vec(
            Array(parent(p.precomputed.sfc_conditions.ρ_flux_h_tot.components.data.:1)),
        )[1]
        @test abs(evap_ed) > FT(0) && abs(shf_ed) > FT(0)
        ρd_ed_cells = Array(parent(Y.c.ρ .- Y.c.ρq_tot))
        qt_ed_before = _col_sum(Y.c.ρq_tot, ᶜJ)
        e_ed_before = _col_sum(Y.c.ρe_tot, ᶜJ)
        Δt_ed_sfc = FT(180)
        CA.step_edmf_and_env_turbulence!(Y, p, FT(0), Δt_ed_sfc)
        @test maximum(abs.(Array(parent(Y.c.ρ .- Y.c.ρq_tot)) .- ρd_ed_cells)) <
              100 * eps(FT)
        @test _col_sum(Y.c.ρq_tot, ᶜJ) ≈ qt_ed_before + Δt_ed_sfc * sfc_factor * evap_ed rtol =
            1e-12
        @test _col_sum(Y.c.ρe_tot, ᶜJ) ≈ e_ed_before + Δt_ed_sfc * sfc_factor * shf_ed rtol =
            1e-12

        # 2. Constant scalar χ ≡ const and uniform η preserved with ᶠF̃ ≠ 0 (sfc flux disabled)
        # Environment-mean entrainment (`edmf_tail_entr_fraction = 1`) preserves
        # uniform `η`; tail entrainment (`f = 0.5`) draws warmer air from the
        # environment's subgrid distribution while conserving column energy.
        tc_notail =
            CA.TurbulenceConvectionParameters(FT, (; edmf_tail_entr_fraction = FT(1)))
        params_notail = CAP.ClimaAtmosParameters(
            map(
                f -> f === :turbconv_params ? tc_notail : getfield(p.params, f),
                fieldnames(CAP.ClimaAtmosParameters),
            )...,
        )
        p_nosfc = _with_cache(
            p;
            params = params_notail,
            atmos = CA.AtmosModel(
                p.atmos;
                disable_surface_flux_tendency = true,
                turbconv = CA.AtmosTurbconv(;
                    turbconv_model = p.atmos.turbconv_model,
                    edmfx_model = CA.EDMFXModel(;
                        entr_model = CA.InvZEntrainment(),
                        detr_model = CA.BuoyancyVelocityDetrainment(),
                        sgs_mass_flux = true,
                        sgs_diffusive_flux = true,
                        nh_pressure = true,
                        vertical_diffusion = false,
                        scale_blending_method = CA.SmoothMinimumBlending(),
                    ),
                ),
            ),
        )
        copyto!(Y, Y0)
        @. Y.c.uₕ = CA.C12(FT(0), FT(0))
        @. Y.f.u₃ = CA.C3(FT(0))
        @. Y.c.ρq_lcl = FT(0);
        @. Y.c.ρq_icl = FT(0);
        @. Y.c.ρq_rai = FT(0);
        @. Y.c.ρq_sno = FT(0)
        @. Y.c.sgsʲs.:(1).q_lcl = FT(0);
        @. Y.c.sgsʲs.:(1).q_icl = FT(0)
        @. Y.c.sgsʲs.:(1).q_rai = FT(0);
        @. Y.c.sgsʲs.:(1).q_sno = FT(0)
        CA.set_precomputed_quantities!(Y, p_nosfc, FT(0))
        q_const_ed = FT(0.002)
        η_const = TD.enthalpy(tps, FT(310), q_const_ed)
        sa_const = @. TD.saturation_adjustment(
            tps, TD.ph(), p_nosfc.precomputed.ᶜp, η_const - p.core.ᶜΦ, q_const_ed,
        )
        @. Y.c.ρ = TD.air_density(
            tps, sa_const.T, p_nosfc.precomputed.ᶜp, q_const_ed, sa_const.q_liq,
            sa_const.q_ice,
        )
        @. Y.c.ρq_tot = Y.c.ρ * q_const_ed
        @. Y.c.ρe_tot =
            Y.c.ρ * (
                TD.internal_energy(
                    tps, sa_const.T, q_const_ed, sa_const.q_liq, sa_const.q_ice,
                ) + p.core.ᶜΦ
            )
        @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.04)
        @. Y.c.sgsʲs.:(1).q_tot = q_const_ed
        @. Y.f.sgsʲs.:(1).u₃ =
            CA.C3(ifelse((zf > 0) & (zf < FT(4000)), FT(6.0) * FT(250), FT(0.0)))
        ᶠF̃ = similar(Y.f, FT)
        @. ᶠF̃ = ifelse((zf > 0) & (zf < FT(4000)), FT(0.4), FT(0.0))
        CA.set_precomputed_quantities!(Y, p_nosfc, FT(0))
        @. Y.c.sgsʲs.:(1).mse = η_const
        CA.set_precomputed_quantities!(Y, p_nosfc, FT(0))
        K1_const_before = Array(parent(p_nosfc.precomputed.ᶜKʲs.:(1)))
        e_col_const_before = _col_sum(Y.c.ρe_tot, ᶜJ)
        Y_const_init = copy(Y)
        CA.step_edmf_and_env_turbulence!(Y, p_nosfc, FT(0), FT(120); ᶠF̃)
        @test maximum(abs.(Array(parent(Y.c.ρq_tot ./ Y.c.ρ)) .- q_const_ed)) < 1e-12
        @test maximum(abs.(Array(parent(Y.c.sgsʲs.:(1).q_tot)) .- q_const_ed)) < 1e-12
        @test maximum(
            abs.(Array(parent(Y.c.sgsʲs.:(1).mse)) .- η_const),
        ) < 1e-11 * abs(η_const)
        @test maximum(
            abs.(Array(parent(p_nosfc.precomputed.ᶜKʲs.:(1))) .- K1_const_before),
        ) > FT(1)
        @test abs(_col_sum(Y.c.ρe_tot, ᶜJ) - e_col_const_before) <
              1e-12 * abs(e_col_const_before)

        # Default tail entrainment (`f_tail = 0.5`) from the same initial state
        copyto!(Y, Y_const_init)
        p_nosfc_tail = _with_cache(p_nosfc; params = p.params)
        CA.set_precomputed_quantities!(Y, p_nosfc_tail, FT(0))
        e_col_tail_before = _col_sum(Y.c.ρe_tot, ᶜJ)
        CA.step_edmf_and_env_turbulence!(Y, p_nosfc_tail, FT(0), FT(120); ᶠF̃)
        @test maximum(Array(parent(Y.c.sgsʲs.:(1).mse))) > η_const + FT(1)
        @test abs(_col_sum(Y.c.ρe_tot, ᶜJ) - e_col_tail_before) <
              1e-12 * abs(e_col_tail_before)

        # 3. 3D EDMF (ᶠF̃ ≠ 0) conservation and updraft area/velocity bounds
        p_nodiff = _with_cache(
            p;
            atmos = CA.AtmosModel(
                p.atmos;
                turbconv = CA.AtmosTurbconv(;
                    turbconv_model = p.atmos.turbconv_model,
                    edmfx_model = CA.EDMFXModel(;
                        entr_model = CA.InvZEntrainment(),
                        detr_model = CA.BuoyancyVelocityDetrainment(),
                        sgs_mass_flux = true,
                        sgs_diffusive_flux = false,
                        nh_pressure = true,
                        vertical_diffusion = false,
                        scale_blending_method = CA.SmoothMinimumBlending(),
                    ),
                ),
            ),
        )
        copyto!(Y, Y0)
        @. Y.c.ρ = FT(1.0)
        @. Y.c.sgsʲs.:(1).ρa = FT(0.1)
        @. Y.c.sgsʲs.:(1).q_tot = ifelse(z < FT(2000), FT(0.02), FT(0.01))
        @. Y.c.ρq_tot =
            FT(0.1) * Y.c.sgsʲs.:(1).q_tot +
            FT(0.9) * ifelse(z < FT(2000), FT(0.008), FT(0.003))
        @. Y.c.ρq_lcl = FT(0);
        @. Y.c.ρq_icl = FT(0);
        @. Y.c.ρq_rai = FT(0);
        @. Y.c.ρq_sno = FT(0)
        @. Y.c.sgsʲs.:(1).q_lcl = FT(0);
        @. Y.c.sgsʲs.:(1).q_icl = FT(0)
        @. Y.c.sgsʲs.:(1).q_rai = FT(0);
        @. Y.c.sgsʲs.:(1).q_sno = FT(0)
        @. Y.f.sgsʲs.:(1).u₃ =
            CA.C3(ifelse((zf > 0) & (zf < FT(4000)), FT(8.0) * FT(250), FT(0.0)))
        CA.set_precomputed_quantities!(Y, p_nodiff, FT(0))
        @. p_nodiff.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p_nodiff.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p_nodiff.precomputed.sfc_conditions.ρ_flux_uₕ =
            CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))
        # Set resolved face mass flux F̃ on interior faces
        ᶠF̃ = similar(Y.f, FT)
        @. ᶠF̃ = ifelse((zf > 0) & (zf < FT(4000)), FT(0.5), FT(0.0))
        qt_before_3d = _col_sum(Y.c.ρq_tot, ᶜJ)
        Δt_3d = FT(60)
        CA.step_edmf_and_env_turbulence!(Y, p_nodiff, FT(0), Δt_3d; ᶠF̃)
        @test abs(_col_sum(Y.c.ρq_tot, ᶜJ) - qt_before_3d) < 1e-13 * qt_before_3d
        a_max_3d = FT(CAP.max_area(CAP.turbconv_params(p_nodiff.params)))
        ρa1_new_arr = vec(Array(parent(Y.c.sgsʲs.:(1).ρa)))
        ρ1_new_arr = vec(Array(parent(p_nodiff.precomputed.ᶜρʲs.:(1))))
        u31_new_arr = vec(Array(parent(Y.f.sgsʲs.:(1).u₃)))
        @test all(>=(zero(FT)), ρa1_new_arr)
        @test all(ρa1_new_arr .<= a_max_3d .* ρ1_new_arr .+ FT(1e-12))
        @test all(>=(zero(FT)), u31_new_arr)
        @test minimum(Array(parent(Y.c.ρq_tot))) >= zero(FT)
        @test minimum(Array(parent(Y.c.sgsʲs.:(1).q_tot))) >= zero(FT)

        # 3b. Relative-flux cancellation check:
        # With zero buoyancy, zero entr/detr/nh-drag, and F¹ == s¹ F̃ (F¹_rel == 0),
        # the relative two-stream increment vanishes identically even for
        # non-uniform q¹(z) == q⁰(z) and nonzero resolved flux F̃.
        p_cancel = _with_cache(
            p;
            atmos = CA.AtmosModel(
                p.atmos;
                disable_surface_flux_tendency = true,
                turbconv = CA.AtmosTurbconv(;
                    turbconv_model = p.atmos.turbconv_model,
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
            ),
        )
        copyto!(Y, Y0)
        @. Y.c.uₕ = CA.C12(FT(0), FT(0))
        @. Y.c.ρ = FT(1.0)
        @. Y.c.ρtke = FT(0)
        @. Y.c.sgsʲs.:(1).ρa = FT(0.1)
        q_prof = @. FT(0.005) + FT(0.008) * exp(-z / FT(1500))
        @. Y.c.sgsʲs.:(1).q_tot = q_prof
        @. Y.c.ρq_tot = Y.c.ρ * q_prof
        @. Y.c.ρe_tot = Y.c.ρ * TD.total_energy(tps, FT(0), p.core.ᶜΦ, FT(300), q_prof)
        @. Y.c.ρq_lcl = FT(0);
        @. Y.c.ρq_icl = FT(0)
        @. Y.c.ρq_rai = FT(0);
        @. Y.c.ρq_sno = FT(0)
        @. Y.c.sgsʲs.:(1).q_lcl = FT(0);
        @. Y.c.sgsʲs.:(1).q_icl = FT(0)
        @. Y.c.sgsʲs.:(1).q_rai = FT(0);
        @. Y.c.sgsʲs.:(1).q_sno = FT(0)
        Δt_cancel = FT(60)
        w_cancel = FT(5.0)
        dz_cancel = vec(Array(parent(Fields.Δz_field(axes(Y.f)))))[2]
        @. Y.f.sgsʲs.:(1).u₃ = ifelse(
            abs(zf - dz_cancel) < FT(1),
            CA.C3(w_cancel * dz_cancel + Δt_cancel * w_cancel^2 / 2),
            ifelse(
                (zf > 0) & (zf < FT(4000)),
                Geometry.Covariant3Vector(Geometry.WVector(w_cancel), ᶠlg),
                CA.C3(FT(0.0)),
            ),
        )
        @. ᶠF̃ = ifelse((zf > 0) & (zf < FT(4000)), w_cancel, FT(0.0))
        CA.set_precomputed_quantities!(Y, p_cancel, FT(0))
        @. Y.c.sgsʲs.:(1).mse = p_cancel.precomputed.ᶜh_tot - p_cancel.precomputed.ᶜK
        CA.set_precomputed_quantities!(Y, p_cancel, FT(0))
        fill!(p_cancel.precomputed.ᶜentr_vel_scaleʲs.:(1), FT(0))
        fill!(p_cancel.precomputed.ᶜentr_nonvel_rateʲs.:(1), FT(0))
        fill!(p_cancel.precomputed.ᶜρ_diffʲs.:(1), FT(0))
        fill!(p_cancel.precomputed.sfc_mass_flux_sourceʲs.:(1), FT(0))
        ρqt_cancel_before = Array(parent(Y.c.ρq_tot))
        CA.step_edmf_and_env_turbulence!(Y, p_cancel, FT(0), Δt_cancel; ᶠF̃)
        @test maximum(abs.(Array(parent(Y.c.ρq_tot)) .- ρqt_cancel_before)) < 1e-13
        @test maximum(abs, Array(parent(p_cancel.scratch.ᶠtemp_scalar_2))) < 1e-13

        # 4. EDMF moisture-step test (a¹ > 0): no ΔT spike across sharp q_tot step
        copyto!(Y, Y0)
        T_iso = FT(290)
        @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.05)
        @. Y.c.ρq_lcl = FT(0);
        @. Y.c.ρq_icl = FT(0);
        @. Y.c.ρq_rai = FT(0);
        @. Y.c.ρq_sno = FT(0)
        @. Y.c.sgsʲs.:(1).q_lcl = FT(0);
        @. Y.c.sgsʲs.:(1).q_icl = FT(0)
        @. Y.c.sgsʲs.:(1).q_rai = FT(0);
        @. Y.c.sgsʲs.:(1).q_sno = FT(0)
        qt_step = @. ifelse(z < FT(2000), FT(0.015), FT(0.002))
        @. Y.c.ρq_tot = Y.c.ρ * qt_step
        @. Y.c.sgsʲs.:(1).q_tot = qt_step
        @. Y.c.ρe_tot = Y.c.ρ * TD.total_energy(tps, FT(0), p.core.ᶜΦ, T_iso, qt_step)
        @. Y.c.sgsʲs.:(1).mse =
            TD.enthalpy(tps, T_iso, qt_step) + p.core.ᶜΦ
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_uₕ =
            CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))
        @. ᶠF̃ = ifelse((zf > 0) & (zf < FT(4000)), FT(0.3), FT(0.0))
        CA.step_edmf_and_env_turbulence!(Y, p, FT(0), FT(60); ᶠF̃)
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @test maximum(abs.(Array(parent(p.precomputed.ᶜT)) .- T_iso)) < FT(0.05)

        # 5. step_prognostic_tke! uses realized Δt (never p.dt) & zero allocs (CPU)
        Y_tke1 = deepcopy(Y)
        Y_tke2 = deepcopy(Y)
        Y_tke3 = deepcopy(Y)
        p_bigdt = _with_cache(p; dt = FT(1e9))
        CA.set_precomputed_quantities!(Y_tke1, p, FT(0))
        CA.step_prognostic_tke!(Y_tke1, p, FT(10))
        CA.set_precomputed_quantities!(Y_tke2, p, FT(0))
        CA.step_prognostic_tke!(Y_tke2, p, FT(600))
        CA.set_precomputed_quantities!(Y_tke3, p_bigdt, FT(0))
        CA.step_prognostic_tke!(Y_tke3, p_bigdt, FT(10))
        @test maximum(abs.(Array(parent(Y_tke1.c.ρtke)) .- Array(parent(Y_tke2.c.ρtke)))) >
              FT(1e-3)
        @test Array(parent(Y_tke1.c.ρtke)) == Array(parent(Y_tke3.c.ρtke))
        if _is_cpu(Y.c)
            @test _alloc_check(CA.step_edmf_and_env_turbulence!, Y, p, FT(0), FT(60)) == 0
            @test _alloc_check(CA.step_prognostic_tke!, Y, p, FT(60)) == 0
        end

        # 6. PrognosticEDMFX + 1M + ρtke + GaussianSGS (blended cloud species flux)
        quad_ed =
            CA.SGSQuadrature(FT; distribution = CA.GaussianSGS(), quadrature_order = 3)
        sim_bl = _build_sim(
            FT;
            z_elem = 10,
            z_max = FT(3000),
            turbconv_model = tc,
            edmfx_model = edmfx,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_sa,
            disable_surface_flux_tendency = true,
            sgs_quadrature = quad_ed,
        )
        Y_bl, p_bl = sim_bl.Y, sim_bl.p
        @test CA.blended_cloud_species_flux(Y_bl, p_bl)
        ᶜJ_bl = Fields.local_geometry_field(Y_bl.c).J
        z_bl = Fields.coordinate_field(Y_bl.c).z
        zf_bl = Fields.coordinate_field(Y_bl.f).z
        @. Y_bl.c.ρtke = Y_bl.c.ρ * FT(1.2)
        @. Y_bl.c.ρq_tot = Y_bl.c.ρ * (FT(0.012) + FT(0.004) * exp(-z_bl / FT(1200)))
        @. Y_bl.c.ρq_lcl =
            Y_bl.c.ρ * (FT(1e-6) + FT(4e-5) * exp(-((z_bl - FT(1200)) / FT(400))^2))
        @. Y_bl.c.ρq_icl =
            Y_bl.c.ρ * (FT(1e-6) + FT(2e-5) * exp(-((z_bl - FT(2100)) / FT(400))^2))
        @. Y_bl.c.sgsʲs.:(1).ρa = Y_bl.c.ρ * FT(0.05)
        @. Y_bl.c.sgsʲs.:(1).q_tot = FT(0.015)
        @. Y_bl.c.sgsʲs.:(1).q_lcl = FT(2e-4)
        @. Y_bl.c.sgsʲs.:(1).q_icl = FT(5e-5)
        @. Y_bl.f.sgsʲs.:(1).u₃ =
            CA.C3(ifelse((zf_bl > 0) & (zf_bl < FT(3000)), FT(8.0) * FT(300), FT(0.0)))
        CA.set_precomputed_quantities!(Y_bl, p_bl, FT(0))
        ql_bl_before = _col_sum(Y_bl.c.ρq_lcl, ᶜJ_bl)
        qi_bl_before = _col_sum(Y_bl.c.ρq_icl, ᶜJ_bl)
        e_bl_before = _col_sum(Y_bl.c.ρe_tot, ᶜJ_bl)
        CA.step_edmf_and_env_turbulence!(Y_bl, p_bl, FT(0), FT(120))
        @test minimum(Array(parent(Y_bl.c.ρq_lcl))) >= 0
        @test minimum(Array(parent(Y_bl.c.ρq_icl))) >= 0
        @test minimum(Array(parent(Y_bl.c.sgsʲs.:(1).q_lcl))) >= 0
        @test minimum(Array(parent(Y_bl.c.sgsʲs.:(1).q_icl))) >= 0
        @test abs(_col_sum(Y_bl.c.ρq_lcl, ᶜJ_bl) - ql_bl_before) < 1e-12 * ql_bl_before
        @test abs(_col_sum(Y_bl.c.ρq_icl, ᶜJ_bl) - qi_bl_before) < 1e-12 * qi_bl_before
        @test abs(_col_sum(Y_bl.c.ρe_tot, ᶜJ_bl) - e_bl_before) < 1e-12 * abs(e_bl_before)
        if _is_cpu(Y_bl.c)
            @test _alloc_check(
                CA.step_edmf_and_env_turbulence!,
                Y_bl,
                p_bl,
                FT(0),
                FT(60),
            ) == 0
        end

        # 7. EDMFExtraScalarTwoStreamSweep for 2M/P3 updraft number/rime and passive tracers
        # (re-seed scratch column masses and fluxes explicitly before sweeping).
        ᶜn1_ex = similar(Y_bl.c.sgsʲs.:(1).q_lcl)
        ᶜρn_ex = similar(Y_bl.c.ρq_lcl)
        @. ᶜn1_ex = FT(2e7)
        @. ᶜρn_ex = Y_bl.c.ρ * FT(1e7) * exp(-((z_bl - FT(1200)) / FT(400))^2)
        n1_ex_before = Array(parent(ᶜn1_ex))
        rn_ex_before = _col_sum(ᶜρn_ex, ᶜJ_bl)
        @. p_bl.scratch.ᶜtemp_scalar = Y_bl.c.sgsʲs.:(1).ρa
        @. p_bl.scratch.ᶜtemp_scalar_2 = max(zero(FT), Y_bl.c.ρ - Y_bl.c.sgsʲs.:(1).ρa)
        fill!(p_bl.scratch.ᶜtemp_scalar_3, FT(1e-3))
        @. p_bl.scratch.ᶜtemp_scalar_4 = max(zero(FT), Y_bl.c.ρ - Y_bl.c.sgsʲs.:(1).ρa)
        fill!(p_bl.scratch.ᶜtemp_scalar_5, zero(FT))
        fill!(p_bl.scratch.ᶜtemp_scalar_6, zero(FT))
        fill!(p_bl.scratch.ᶜtemp_scalar_7, zero(FT))
        @. p_bl.scratch.ᶠtemp_scalar =
            ifelse((zf_bl > 0) & (zf_bl < FT(3000)), FT(1.0), FT(0))
        @. p_bl.scratch.ᶠtemp_scalar_2 =
            ifelse((zf_bl > 0) & (zf_bl < FT(3000)), FT(0.05), FT(0))
        @. p_bl.scratch.ᶠtemp_UVW.components.data.:1 = -p_bl.scratch.ᶠtemp_scalar_2
        @. p_bl.scratch.ᶠtemp_UVWxUVW.components.data.:1 =
            ifelse((zf_bl > 0) & (zf_bl < FT(3000)), FT(0.5), FT(0))

        # Uniform updraft mse_eff¹ stays invariant under vertical_diffusion despite a
        # steep updraft q_tot¹ gradient (dry-enthalpy cross-term goes to row 0).
        copyto!(Y, Y0)
        @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.05)
        @. Y.c.sgsʲs.:(1).mse = FT(3.1e5)
        @. Y.c.sgsʲs.:(1).q_tot = FT(0.002) + FT(0.015) * exp(-z / FT(1000))
        @. Y.c.sgsʲs.:(1).q_rai = FT(0);
        @. Y.c.sgsʲs.:(1).q_sno = FT(0)
        @. Y.c.ρq_rai = FT(0);
        @. Y.c.ρq_sno = FT(0)
        @. Y.f.sgsʲs.:(1).u₃ = CA.C3(FT(0))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @. p.precomputed.ᶠK_h = FT(50.0)
        fill!(p.precomputed.ᶜentr_vel_scaleʲs.:(1), FT(0))
        fill!(p.precomputed.ᶜentr_nonvel_rateʲs.:(1), FT(0))
        fill!(p.precomputed.ᶜarea_bounding_entr_detrʲs.:(1), FT(0))
        fill!(p.precomputed.ᶜturb_entrʲs.:(1), FT(0))
        fill!(p.precomputed.sfc_mass_flux_sourceʲs.:(1), FT(0))
        fill!(p.precomputed.ᶜρ_diffʲs.:(1), FT(0))
        p_mse_iso = _with_cache(
            p;
            atmos = CA.AtmosModel(p.atmos; disable_surface_flux_tendency = true),
        )
        CA.step_edmf_and_env_turbulence!(Y, p_mse_iso, FT(0), FT(60))
        mse1_iso = vec(Array(parent(Y.c.sgsʲs.:(1).mse)))
        @test maximum(abs.(mse1_iso .- FT(3.1e5))) < FT(1e-7)
        extra_geom_ex =
            (; ᶜJ = ᶜJ_bl, ᶠlg = Fields.local_geometry_field(Y_bl.f), ᶠz = zf_bl)
        extra_pre_ex = (; p_bl.precomputed.ᶠK_h)
        scratch_s_ex = (;
            ρa1_in = p_bl.scratch.ᶜtemp_scalar,
            ρa0_in = p_bl.scratch.ᶜtemp_scalar_2,
            ρa1_prev = p_bl.scratch.ᶜtemp_scalar_3,
            ρa0_new = p_bl.scratch.ᶜtemp_scalar_4,
            E_col = p_bl.scratch.ᶜtemp_scalar_5,
            D_col = p_bl.scratch.ᶜtemp_scalar_6,
            empty_reg = p_bl.scratch.ᶜtemp_scalar_7,
            JF1 = p_bl.scratch.ᶠtemp_scalar,
            JF1_rel = p_bl.scratch.ᶠtemp_scalar_2,
            JF0_rel = p_bl.scratch.ᶠtemp_UVW.components.data.:1,
        )
        scratch_v_ex = (;
            uv_c = p_bl.scratch.ᶜtemp_UVWxUVW,
            st_c = p_bl.scratch.ᶜtemp_strain,
            c12_c = p_bl.scratch.ᶜtemp_C12,
            c3_c = p_bl.scratch.ᶜtemp_C3,
            ct123_c = p_bl.scratch.ᶜtemp_CT123,
            uvw_c = p_bl.scratch.ᶜtemp_UVW,
            uv_f = p_bl.scratch.ᶠtemp_UVWxUVW,
            st_f = p_bl.scratch.ᶠtemp_strain,
            c123_f = p_bl.scratch.ᶠtemp_C123,
        )
        kh_fac_ex =
            FT(CAP.edmfx_updraft_kh_diffusion_factor(CAP.turbconv_params(p_bl.params)))
        sw_ex_diff = CA.EDMFExtraScalarTwoStreamSweep{true, FT}(
            kh_fac_ex,
            FT(60),
            true,
            true,
            Int8(2),
            Int8(1),
        )
        DataLayouts.foreach_column(
            sw_ex_diff, ᶜn1_ex, ᶜρn_ex, Y_bl.c, extra_geom_ex, extra_pre_ex, scratch_s_ex,
            scratch_v_ex;
            shmem = Val(:none),
        )
        @test maximum(abs.(Array(parent(ᶜn1_ex)) .- n1_ex_before)) > FT(1)
        @test minimum(Array(parent(ᶜn1_ex))) >= 0
        @test minimum(Array(parent(ᶜρn_ex))) >= 0
        @test abs(_col_sum(ᶜρn_ex, ᶜJ_bl) - rn_ex_before) < 1e-12 * rn_ex_before
        if _is_cpu(Y_bl.c)
            _run_ex_alloc(sw::S, f1, f2, f3, f4, f5, f6, f7) where {S} =
                DataLayouts.foreach_column(
                    sw,
                    f1,
                    f2,
                    f3,
                    f4,
                    f5,
                    f6,
                    f7;
                    shmem = Val(:none),
                )
            @test _alloc_check(
                _run_ex_alloc,
                sw_ex_diff, ᶜn1_ex, ᶜρn_ex, Y_bl.c, extra_geom_ex, extra_pre_ex,
                scratch_s_ex, scratch_v_ex,
            ) == 0
        end

        # 8. Float32 PrognosticEDMFX on stretched vertical grid
        FT32 = Float32
        tc32 = CA.PrognosticEDMFX{1, true}(FT32(0.05))
        mo32 = CA.SurfaceConditions.MoninObukhov(; z0m = FT32(1e-3), z0b = FT32(1e-3))
        sfc32 = CA.AtmosSurface(;
            flux_scheme = mo32,
            temperature = CA.SurfaceConditions.AnalyticTemperature(Returns(FT32(302))),
        )
        sim32_ed = _build_sim(
            FT32;
            z_elem = 8,
            z_max = FT32(12000),
            z_stretch = true,
            turbconv_model = tc32,
            edmfx_model = edmfx,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc32,
            disable_surface_flux_tendency = false,
        )
        Y32, p32 = sim32_ed.Y, sim32_ed.p
        tps32 = CAP.thermodynamics_params(p32.params)
        ᶜJ32 = Fields.local_geometry_field(Y32.c).J
        ᶠlg32 = Fields.local_geometry_field(Y32.f)
        sfc_fac32 = vec(Array(parent(@. CA.surface_c3_flux_factor(ᶠlg32))))[1]
        z32 = Fields.coordinate_field(Y32.c).z
        zf32 = Fields.coordinate_field(Y32.f).z
        @. Y32.c.uₕ = CA.C12(FT32(8), FT32(0))
        @. Y32.c.ρtke = Y32.c.ρ * FT32(0.8)
        @. Y32.c.ρq_tot = Y32.c.ρ * (FT32(0.004) + FT32(0.01) * exp(-z32 / FT32(1500)))
        @. Y32.c.ρq_lcl = Y32.c.ρ * FT32(4e-4) * exp(-((z32 - FT32(1500)) / FT32(500))^2)
        @. Y32.c.ρq_icl = Y32.c.ρ * FT32(1e-4) * exp(-((z32 - FT32(3000)) / FT32(500))^2)
        @. Y32.c.ρq_rai = Y32.c.ρ * FT32(2e-5)
        @. Y32.c.ρq_sno = Y32.c.ρ * FT32(1e-5)
        @. Y32.c.sgsʲs.:(1).ρa = Y32.c.ρ * FT32(0.05)
        @. Y32.c.sgsʲs.:(1).q_tot = FT32(0.014)
        @. Y32.c.sgsʲs.:(1).q_lcl = FT32(8e-4)
        @. Y32.c.sgsʲs.:(1).q_icl = FT32(2e-4)
        @. Y32.c.sgsʲs.:(1).q_rai = FT32(3e-5)
        @. Y32.c.sgsʲs.:(1).q_sno = FT32(1.5e-5)
        @. Y32.c.sgsʲs.:(1).mse =
            TD.enthalpy(tps32, FT32(290), FT32(0.014)) + FT32(5000)
        @. Y32.f.sgsʲs.:(1).u₃ = ifelse(
            (zf32 > 0) & (zf32 < FT32(12000)),
            Geometry.Covariant3Vector(Geometry.WVector(FT32(12.0)), ᶠlg32),
            CA.C3(FT32(0)),
        )
        ᶠF̃32 = similar(Y32.f, FT32)
        @. ᶠF̃32 = ifelse((zf32 > 0) & (zf32 < FT32(12000)), FT32(0.3), FT32(0))
        CA.set_precomputed_quantities!(Y32, p32, FT32(0))
        evap32 = vec(
            Array(parent(p32.precomputed.sfc_conditions.ρ_flux_q_tot.components.data.:1)),
        )[1]
        shf32 = vec(
            Array(parent(p32.precomputed.sfc_conditions.ρ_flux_h_tot.components.data.:1)),
        )[1]
        ρd32_before = Array(parent(Y32.c.ρ .- Y32.c.ρq_tot))
        qt32_before = _col_sum(Y32.c.ρq_tot, ᶜJ32)
        ql32_before = _col_sum(Y32.c.ρq_lcl, ᶜJ32)
        qi32_before = _col_sum(Y32.c.ρq_icl, ᶜJ32)
        qr32_before = _col_sum(Y32.c.ρq_rai, ᶜJ32)
        qs32_before = _col_sum(Y32.c.ρq_sno, ᶜJ32)
        e32_before = _col_sum(Y32.c.ρe_tot, ᶜJ32)
        Δt32_ed = FT32(120)
        CA.step_edmf_and_env_turbulence!(Y32, p32, FT32(0), Δt32_ed; ᶠF̃ = ᶠF̃32)
        nz32_ed = Spaces.nlevels(axes(Y32.c))
        tol32_cell = FT32(10) * eps(FT32)
        tol32_ed = FT32(10) * nz32_ed * eps(FT32)
        @test maximum(abs.(Array(parent(Y32.c.ρ .- Y32.c.ρq_tot)) .- ρd32_before)) <
              tol32_cell
        @test _col_sum(Y32.c.ρq_tot, ᶜJ32) ≈ qt32_before + Δt32_ed * sfc_fac32 * evap32 rtol =
            tol32_ed
        @test _col_sum(Y32.c.ρe_tot, ᶜJ32) ≈ e32_before + Δt32_ed * sfc_fac32 * shf32 rtol =
            tol32_ed
        @test abs(_col_sum(Y32.c.ρq_lcl, ᶜJ32) - ql32_before) < tol32_ed * ql32_before
        @test abs(_col_sum(Y32.c.ρq_icl, ᶜJ32) - qi32_before) < tol32_ed * qi32_before
        @test abs(_col_sum(Y32.c.ρq_rai, ᶜJ32) - qr32_before) < tol32_ed * qr32_before
        @test abs(_col_sum(Y32.c.ρq_sno, ᶜJ32) - qs32_before) < tol32_ed * qs32_before
        @test minimum(Array(parent(Y32.c.ρq_tot))) >= 0
        @test minimum(Array(parent(Y32.c.sgsʲs.:(1).ρa))) >= 0
        if _is_cpu(Y32.c)
            @test _alloc_check(
                CA.step_edmf_and_env_turbulence!, Y32, p32, FT32(0), Δt32_ed,
            ) == 0
        end
    end

    @testset "PrognosticEDMFX terrain fluxes, area bounds, and SGS flux target" begin
        FT = Float64
        ctx = ClimaComms.SingletonCommsContext()

        # Draft fluxes on a sloped column: with `u₃¹ = u₃` the full draft
        # velocity equals the grid-mean `u³`, so the relative mass flux and the
        # SGS flux weight vanish although `uₕ³ ≠ 0`.
        topo = CA.CosineTopography{3, FT}(; h_max = FT(400), λ = FT(50e3))
        g_topo = CA.BoxGrid(
            FT;
            context = ctx,
            x_elem = 2,
            y_elem = 2,
            z_elem = 8,
            z_max = FT(3000),
            x_max = FT(50e3),
            y_max = FT(50e3),
            nh_poly = 2,
            topography = topo,
        )
        cspace = Spaces.CenterExtrudedFiniteDifferenceSpace(g_topo)
        fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(g_topo)
        ᶜlg = Fields.local_geometry_field(cspace)
        ᶠlg = Fields.local_geometry_field(fspace)
        ᶜz = Fields.coordinate_field(cspace).z
        ᶠz = Fields.coordinate_field(fspace).z
        ᶜρ = @. FT(1.2) * exp(-1 * ᶜz / FT(8000))
        ᶜuₕ = @. CA.C12(Geometry.UVVector(FT(12), FT(-7)), ᶜlg)
        ᶠuₕ³ = Base.materialize(CA.compute_ᶠuₕ³(ᶜuₕ, ᶜρ))
        ᶠu₃ = @. CA.C3(Geometry.WVector(FT(0.3) * sin(FT(π) * ᶠz / FT(3000))), ᶠlg)
        ᶠu³ = @. ᶠuₕ³ + CA.CT3(ᶠu₃)
        ᶜρa = @. ᶜρ * (FT(0.05) + FT(0.03) * sin(FT(π) * ᶜz / FT(3000)))
        uh3_max = maximum(abs, Array(parent(ᶠuₕ³)))
        @test uh3_max > FT(1e-4) * maximum(abs, Array(parent(ᶠu³)))
        nz = Spaces.nlevels(cspace)
        F_rel_max = zero(FT)
        W1_max = zero(FT)
        leak_max = zero(FT)
        for (i, j, h) in ((1, 1, 1), (2, 3, 2), (3, 2, 4))
            col(field) = Fields.field_values(Fields.column(field, i, j, h))
            ᶜlg_c, ᶠlg_c = col(ᶜlg), col(ᶠlg)
            ᶜρ_c, ᶜρa_c = col(ᶜρ), col(ᶜρa)
            ᶠuₕ³_c, ᶠu₃_c, ᶠu³_c = col(ᶠuₕ³), col(ᶠu₃), col(ᶠu³)
            for f in 2:nz
                u3_cont_1 = CA._draft_face_u3_contravariant(
                    ᶠuₕ³_c[f], ᶠlg_c[f], ᶠu₃_c[f].u₃,
                )
                @test u3_cont_1 ≈ ᶠu³_c[f].u³ rtol = 1e-13
                c1 = CA._draft_face_conductance(
                    u3_cont_1, ᶜρ_c[f - 1], ᶜρ_c[f], ᶜlg_c[f - 1].J, ᶜlg_c[f].J,
                )
                F_rel, V1, W1 = CA._draft_relative_face_fluxes(
                    c1, u3_cont_1, ᶠu³_c[f].u³, ᶜρa_c[f - 1], ᶜρa_c[f],
                    ᶜρ_c[f - 1], ᶜρ_c[f], ᶜlg_c[f - 1].J, ᶜlg_c[f].J, ᶠlg_c[f].J,
                )
                JF_scale = abs(V1) * ᶜρa_c[f]
                F_rel_max = max(F_rel_max, abs(F_rel) / JF_scale)
                W1_max = max(W1_max, abs(W1))
                # Flux of `ρa¹` without the `uₕ³` projection.
                c1_vert = CA.face_conductance(ᶠlg_c[f], ᶠu₃_c[f].u₃)
                leak_max = max(
                    leak_max,
                    abs(c1_vert * ᶜρa_c[f - 1] - c1 * ᶜρa_c[f - 1]) / JF_scale,
                )
            end
            # Draft mass-flux divergence of the TKE detrainment matches the
            # donor-cell divergence of `ρa¹ u³¹` with the full `u³¹`.
            ρa_don(f) = ᶠu³_c[f].u³ >= 0 ? ᶜρa_c[f - 1] : ᶜρa_c[f]
            for k in 1:nz
                div_ref =
                    (
                        (k < nz ? ᶠlg_c[k + 1].J * ᶠu³_c[k + 1].u³ * ρa_don(k + 1) : 0) -
                        (k > 1 ? ᶠlg_c[k].J * ᶠu³_c[k].u³ * ρa_don(k) : 0)
                    ) / ᶜlg_c[k].J
                div_mf = CA._draft_mass_flux_divergence(
                    ᶠu₃_c, ᶠuₕ³_c, ᶠlg_c, ᶜρa_c, ᶜlg_c[k].J, k, nz,
                )
                @test div_mf ≈ div_ref rtol = 1e-12 atol = 1e-15
            end
        end
        @test F_rel_max < 1e-12
        @test W1_max == 0
        @test leak_max > 1e-3

        # The SGS flux target equals the low-order flux for subdomain values
        # that are uniform in height, also when `ρ¹ ≠ ρ`; a weight with the
        # draft face density misses it by the factor `ρ¹ / ρ`.
        nz_l = 8
        ᶠz_l = collect(FT, 0:250:(250 * nz_l))
        J_l = diff(ᶠz_l)
        ρ_l = fill(FT(1.1), nz_l)
        ρ1_l = fill(FT(1.3), nz_l)
        ρa1_l = fill(FT(0.12), nz_l)
        ρa0_l = ρ_l .- ρa1_l
        x1_l = fill(FT(3), nz_l)
        x0_l = fill(FT(1), nz_l)
        JF1_rel_l = zeros(FT, nz_l + 1)
        W1_l = zeros(FT, nz_l + 1)
        W1_draft_l = zeros(FT, nz_l + 1)
        for f in 2:nz_l
            u3_cont_1 = FT(2.5) / J_l[f]
            u3_cont_gm = FT(0.4) / J_l[f]
            c1 = CA._draft_face_conductance(
                u3_cont_1, ρ1_l[f - 1], ρ1_l[f], J_l[f - 1], J_l[f],
            )
            JF1_rel_l[f], _, W1_l[f] = CA._draft_relative_face_fluxes(
                c1, u3_cont_1, u3_cont_gm, ρa1_l[f - 1], ρa1_l[f],
                ρ_l[f - 1], ρ_l[f], J_l[f - 1], J_l[f], J_l[f],
            )
            W1_draft_l[f] = W1_l[f] * ρ1_l[f] / ρ_l[f]
        end
        @test all(>(0), JF1_rel_l[2:nz_l])
        A_out_l = zeros(FT, nz_l + 1)
        bufs = (zeros(FT, nz_l + 1), zeros(FT, nz_l + 1), zeros(FT, nz_l + 1))
        F_scale = maximum(JF1_rel_l) * (x1_l[1] - x0_l[1])
        for mode in (Int8(0), Int8(1), Int8(2))
            CA._compute_edmf_sgs_flux_corr!(
                A_out_l, fill(FT(1), nz_l), x1_l, x0_l, JF1_rel_l, W1_l,
                ρa1_l, ρa0_l, J_l, ᶠz_l, bufs..., FT(60), nz_l, false, mode,
            )
            @test maximum(abs, A_out_l) < 1e-13 * F_scale
            CA._compute_edmf_sgs_flux_corr!(
                A_out_l, fill(FT(1), nz_l), x1_l, x0_l, JF1_rel_l, W1_draft_l,
                ρa1_l, ρa0_l, J_l, ᶠz_l, bufs..., FT(60), nz_l, false, mode,
            )
            @test maximum(abs, A_out_l) > FT(0.1) * F_scale
        end

        # Strong surface flux over sloped terrain: the draft stays below the
        # grid-mean density when `ρ¹ > ρ`, and the column stays finite under
        # the centered SGS flux.
        tc = CA.PrognosticEDMFX{1, true}(FT(0.1))
        edmfx = CA.EDMFXModel(;
            entr_model = CA.InvZEntrainment(),
            detr_model = CA.BuoyancyVelocityDetrainment(),
            sgs_mass_flux = true,
            sgs_diffusive_flux = true,
            nh_pressure = true,
            vertical_diffusion = true,
            scale_blending_method = CA.SmoothMinimumBlending(),
            updraft_picard_rounds = 2,
        )
        g_box = CA.BoxGrid(
            FT;
            context = ctx,
            x_elem = 1,
            y_elem = 1,
            z_elem = 12,
            z_max = FT(3000),
            x_max = FT(50e3),
            y_max = FT(50e3),
            nh_poly = 2,
            topography = topo,
        )
        (; Y, p) = _build_sim(
            FT;
            grid = g_box,
            turbconv_model = tc,
            edmfx_model = edmfx,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            disable_surface_flux_tendency = false,
            dt = FT(120),
        )
        tps = CAP.thermodynamics_params(p.params)
        a_max = FT(CAP.max_area(CAP.turbconv_params(p.params)))
        z = Fields.coordinate_field(Y.c).z
        ᶜlg_b = Fields.local_geometry_field(Y.c)
        Y0 = deepcopy(Y)
        @. Y.c.uₕ = CA.C12(Geometry.UVVector(FT(15), FT(5)), ᶜlg_b)
        @. Y.c.ρtke = Y.c.ρ * FT(1)
        @. Y.c.ρq_tot = Y.c.ρ * FT(0.004) * exp(-z / FT(2000))
        @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.69)
        @. Y.c.sgsʲs.:(1).q_tot = Y.c.ρq_tot / Y.c.ρ
        @. Y.c.sgsʲs.:(1).mse =
            TD.enthalpy(tps, FT(185), Y.c.sgsʲs.:(1).q_tot) +
            CAP.grav(p.params) * z
        CA.set_precomputed_quantities!(Y, p, FT(0))
        ρ1_over_ρ = Array(parent(p.precomputed.ᶜρʲs.:(1) ./ Y.c.ρ))
        @test maximum(ρ1_over_ρ) > 1 / a_max
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(1000))
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(2e-4))
        for _ in 1:3
            CA.step_edmf_and_env_turbulence!(Y, p, FT(0), FT(120))
            @test minimum(Array(parent(Y.c.ρ .- Y.c.sgsʲs.:(1).ρa))) > 0
            @test all(isfinite, Array(parent(Y.c.ρe_tot)))
            @test all(isfinite, Array(parent(Y.c.sgsʲs.:(1).mse)))
            @test all(isfinite, Array(parent(Y.f.sgsʲs.:(1).u₃)))
            @test minimum(Array(parent(Y.c.ρq_tot))) >= 0
        end

        # An empty draft without a surface mass-flux source keeps `ρa¹ = 0`,
        # and its scalars do not enter the grid-mean solves.
        copyto!(Y, Y0)
        @. Y.c.uₕ = CA.C12(Geometry.UVVector(FT(15), FT(5)), ᶜlg_b)
        @. Y.c.ρq_tot = Y.c.ρ * FT(0.004) * exp(-z / FT(2000))
        @. Y.c.sgsʲs.:(1).ρa = FT(0)
        @. Y.c.sgsʲs.:(1).q_tot = Y.c.ρq_tot / Y.c.ρ
        CA.set_precomputed_quantities!(Y, p, FT(0))
        fill!(p.precomputed.sfc_mass_flux_sourceʲs.:(1), FT(0))
        Y_a = deepcopy(Y)
        Y_b = deepcopy(Y)
        @. Y_b.c.sgsʲs.:(1).mse += FT(3000)
        @. Y_b.c.sgsʲs.:(1).q_tot *= FT(1.5)
        @. Y_b.f.sgsʲs.:(1).u₃ += CA.C3(FT(50))
        p_a = deepcopy(p)
        p_b = deepcopy(p)
        CA.step_edmf_and_env_turbulence!(Y_a, p_a, FT(0), FT(120))
        CA.step_edmf_and_env_turbulence!(Y_b, p_b, FT(0), FT(120))
        @test all(iszero, Array(parent(Y_a.c.sgsʲs.:(1).ρa)))
        @test all(iszero, Array(parent(Y_b.c.sgsʲs.:(1).ρa)))
        for name in (:ρ, :ρe_tot, :ρq_tot, :ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
            @test Array(parent(getproperty(Y_a.c, name))) ==
                  Array(parent(getproperty(Y_b.c, name)))
        end
        if _is_cpu(Y.c)
            @test _alloc_check(CA.step_edmf_and_env_turbulence!, Y_a, p_a, FT(0), FT(60)) ==
                  0
            @test _alloc_check(CA.step_prognostic_tke!, Y_a, p_a, FT(60)) == 0
        end
    end

    @testset "Blended cloud-species energy flux and 1M updraft relaxation parity" begin
        FT = Float64
        # A layer well mixed in `(s_l, q_t)` keeps its energy for every blend
        # weight: the energy flux is `F_sl + (h_v + Φ) F_qt`, whatever the
        # cloud-species flux.
        coupler_sfc = CA.AtmosSurface(;
            flux_scheme = nothing,
            temperature = CA.SurfaceConditions.AnalyticTemperature(Returns(FT(300))),
        )
        (; Y, p) = _build_sim(
            FT;
            z_elem = 10,
            z_max = FT(3000),
            turbconv_model = CA.EDOnlyEDMFX(),
            edmfx_model = CA.EDMFXModel(;
                sgs_diffusive_flux = true,
                scale_blending_method = CA.SmoothMinimumBlending(),
            ),
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = coupler_sfc,
            disable_surface_flux_tendency = false,
            sgs_quadrature = CA.SGSQuadrature(
                FT;
                distribution = CA.GaussianSGS(),
                quadrature_order = 3,
            ),
        )
        @test CA.blended_cloud_species_flux(Y, p)
        tps = CAP.thermodynamics_params(p.params)
        grav = FT(CAP.grav(p.params))
        z_c = vec(Array(parent(Fields.coordinate_field(Y.c).z)))
        nz = length(z_c)
        q_t = FT(0.012)
        q_l = @. FT(2e-4) * exp(-((z_c - FT(1500)) / FT(500))^2)
        cp_c = [TD.cp_m(tps, q_t, q_l[k], FT(0)) for k in 1:nz]
        h_lv(T) = TD.enthalpy_liquid(tps, T) - TD.enthalpy_vapor(tps, T)
        T_c = zeros(FT, nz)
        T_c[1] = FT(292)
        for k in 2:nz
            T_k = T_c[k - 1]
            for _ in 1:20
                hlv_f = (h_lv(T_c[k - 1]) + h_lv(T_k)) / 2
                T_k =
                    T_c[k - 1] -
                    (grav * (z_c[k] - z_c[k - 1]) + hlv_f * (q_l[k] - q_l[k - 1])) /
                    ((cp_c[k - 1] + cp_c[k]) / 2)
            end
            T_c[k] = T_k
        end
        ρ_c = vec(Array(parent(Y.c.ρ)))
        e_c = [
            ρ_c[k] * (TD.internal_energy(tps, T_c[k], q_t, q_l[k], FT(0)) + grav * z_c[k])
            for k in 1:nz
        ]
        @. Y.c.uₕ = CA.C12(FT(0), FT(0))
        @. Y.f.u₃ = CA.C3(FT(0))
        @. Y.c.ρtke = Y.c.ρ * FT(1)
        @. Y.c.ρq_tot = Y.c.ρ * q_t
        copyto!(parent(Y.c.ρq_lcl), ρ_c .* q_l)
        copyto!(parent(Y.c.ρe_tot), e_c)
        @. Y.c.ρq_icl = FT(0)
        @. Y.c.ρq_rai = FT(0)
        @. Y.c.ρq_sno = FT(0)
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(FT(0))
        @. p.precomputed.sfc_conditions.ρ_flux_uₕ = CA.C3(FT(0)) ⊗ CA.C12(FT(0), FT(0))
        fill!(p.precomputed.ᶠK_h, FT(30))
        fill!(p.precomputed.ᶠK_u, FT(30))
        Y_mixed = deepcopy(Y)
        ql_after = map((FT(1e-3), FT(1e5))) do l_mix
            copyto!(Y, Y_mixed)
            fill!(p.precomputed.ᶜl_mix, l_mix)
            CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), FT(600))
            Δe = Array(parent(Y.c.ρe_tot)) .- e_c
            @test maximum(abs, Δe) < 1e-11 * maximum(abs, e_c)
            Array(parent(Y.c.ρq_lcl))
        end
        @test maximum(abs, ql_after[1] .- ql_after[2]) > FT(1e-3) * maximum(ρ_c .* q_l)

        # Updraft cloud relaxation reproduces the CloudMicrophysics rate,
        # including the evaporation bound `S ≥ -q/τ`.
        params = CA.ClimaAtmosParameters(
            FT;
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        )
        tps1 = CAP.thermodynamics_params(params)
        mp = CAP.cloud_formation_1m_params(params)
        T_min_sgs = FT(CAP.T_min_sgs(params))
        p_air = FT(9e4)
        cases = (
            # (phase is liquid, T, q_vap / q_sat, q_lcl, q_icl)
            (true, FT(285), FT(1.03), FT(5e-4), FT(0)),
            (true, FT(285), FT(0.97), FT(5e-4), FT(0)),
            (true, FT(285), FT(0.6), FT(1e-6), FT(0)),
            (false, FT(245), FT(1.05), FT(0), FT(1e-5)),
            (false, FT(245), FT(0.5), FT(0), FT(1e-7)),
        )
        for (is_liq, T, s, q_lcl, q_icl) in cases
            ρ_dry = p_air / (FT(TD.Parameters.R_d(tps1)) * T)
            phase = is_liq ? TD.Liquid() : TD.Ice()
            q_sat = TD.q_vap_saturation(tps1, T, ρ_dry, phase)
            q_tot = s * q_sat + q_lcl + q_icl
            ρ = TD.air_density(tps1, T, p_air, q_tot, q_lcl, q_icl)
            for _ in 1:5
                q_sat = TD.q_vap_saturation(tps1, T, ρ, phase)
                q_tot = s * q_sat + q_lcl + q_icl
                ρ = TD.air_density(tps1, T, p_air, q_tot, q_lcl, q_icl)
            end
            @test T > T_min_sgs
            mse = [TD.enthalpy(tps1, T, q_tot, q_lcl, q_icl)]
            r_relax = zeros(FT, 1)
            q_eq = zeros(FT, 1)
            CA._compute_updraft_1m_relax!(
                Val(is_liq), r_relax, q_eq, mp, tps1, T_min_sgs, [FT(0.1)], mse,
                [q_tot], [q_lcl], [q_icl], [FT(0)], [FT(0)], [FT(0)], [p_air],
                false, 1,
            )
            micro = (; q_tot, q_lcl, q_icl, q_rai = FT(0), q_sno = FT(0))
            thermo = (; ρ, T, w = FT(0))
            procs = mp.processes
            S_cm, τ_cm = if is_liq
                (
                    CA.CMNe.conv_q_vap_to_q_lcl(
                        procs.cloud_liquid_formation, mp, tps1, micro, thermo,
                    ),
                    CA.CMNe.τ_vap_to_q_lcl(
                        procs.cloud_liquid_formation, mp, tps1, micro, thermo,
                    ),
                )
            else
                (
                    CA.CMNe.conv_q_vap_to_q_icl(
                        procs.cloud_ice_formation, mp, tps1, micro, thermo,
                    ),
                    CA.CMNe.τ_vap_to_q_icl(
                        procs.cloud_ice_formation, mp, tps1, micro, thermo,
                    ),
                )
            end
            q_self = is_liq ? q_lcl : q_icl
            @test r_relax[1] * (q_eq[1] - q_self) ≈ S_cm rtol = 1e-8 atol = 1e-18
            if S_cm != 0
                @test r_relax[1] ≈ 1 / τ_cm rtol = 1e-12
                @test S_cm >= -q_self / τ_cm * (1 + 1e-12)
            end
        end
    end

    @testset "Column-conservative water repair & SlabOcean conservation" begin
        FT = Float64
        slab = CA.SurfaceConditions.SlabOceanTemperature{FT}(;
            depth_ocean = FT(50),
            ρ_ocean = FT(1025),
            cp_ocean = FT(3985),
            q_flux = false,
        )
        mo = CA.SurfaceConditions.MoninObukhov(; z0m = FT(1e-3), z0b = FT(1e-3))
        sfc_slab = CA.AtmosSurface(; flux_scheme = mo, temperature = slab)
        (; Y, p) = _build_sim(
            FT;
            z_elem = 8,
            z_max = FT(2000),
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_slab,
            disable_surface_flux_tendency = false,
        )
        ᶜJ = Fields.local_geometry_field(Y.c).J
        ᶠlg = Fields.local_geometry_field(Y.f)
        sfc_factor = vec(Array(parent(@. CA.surface_c3_flux_factor(ᶠlg))))[1]
        A_sfc = vec(Array(parent(@. _surface_area(ᶠlg))))[1]
        hcap = slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean

        # Inject negative condensate and negative vapor in some levels
        @. Y.c.ρq_tot = Y.c.ρ * FT(0.01)
        qt_host = Array(parent(Y.c.ρq_tot))
        qt_host[2] = FT(-1e-4)
        copyto!(parent(Y.c.ρq_tot), qt_host)
        ql_host = Array(parent(Y.c.ρq_lcl))
        ql_host[3] = FT(-5e-5)
        copyto!(parent(Y.c.ρq_lcl), ql_host)
        qi_host = Array(parent(Y.c.ρq_icl))
        qi_host[4] = FT(1e-4)
        copyto!(parent(Y.c.ρq_icl), qi_host)
        CA.set_precomputed_quantities!(Y, p, FT(0))

        ρd_before = Array(parent(Y.c.ρ .- Y.c.ρq_tot))
        qt_before = _col_sum(Y.c.ρq_tot, ᶜJ)
        e_before = _col_sum(Y.c.ρe_tot, ᶜJ)

        CA.repair_column_water_conservatively!(Y, p)

        @test minimum(Array(parent(Y.c.ρq_lcl))) >= 0
        @test minimum(Array(parent(Y.c.ρq_icl))) >= 0
        @test minimum(
            Array(
                parent(
                    Y.c.ρq_tot .- Y.c.ρq_lcl .- Y.c.ρq_icl .- Y.c.ρq_rai .- Y.c.ρq_sno,
                ),
            ),
        ) >= -100 * eps(FT)
        @test maximum(abs.(Array(parent(Y.c.ρ .- Y.c.ρq_tot)) .- ρd_before)) < 100 * eps(FT)
        @test abs(_col_sum(Y.c.ρq_tot, ᶜJ) - qt_before) < 100 * eps(FT) * qt_before
        @test abs(_col_sum(Y.c.ρe_tot, ᶜJ) - e_before) < 100 * eps(FT) * abs(e_before)

        # Slab ocean water and energy conservation with vertical diffusion
        p_slab = _with_cache(
            p;
            atmos = CA.AtmosModel(
                p.atmos;
                vertical_diffusion = CA.DecayWithHeightDiffusion{FT}(;
                    disable_momentum_vertical_diffusion = false,
                    H = FT(1000),
                    D₀ = FT(10),
                ),
            ),
        )
        @. Y.c.ρq_tot = Y.c.ρ * FT(0.01)
        @. Y.c.ρq_lcl = FT(0);
        @. Y.c.ρq_icl = FT(0);
        @. Y.c.ρq_rai = FT(0);
        @. Y.c.ρq_sno = FT(0)
        CA.set_precomputed_quantities!(Y, p_slab, FT(0))
        evap = FT(2e-4)
        shf = FT(150.0)
        @. p_slab.precomputed.sfc_conditions.ρ_flux_q_tot = CA.C3(evap)
        @. p_slab.precomputed.sfc_conditions.ρ_flux_h_tot = CA.C3(shf)
        @. p_slab.precomputed.surface_rain_flux = FT(-1e-4)
        @. p_slab.precomputed.surface_snow_flux = FT(0)
        @. p_slab.precomputed.col_integrated_precip_energy_tendency = FT(-50.0)

        w_tot_before =
            _col_sum(Y.c.ρq_tot, ᶜJ) + vec(Array(parent(Y.sfc.water)))[1] * A_sfc
        e_tot_before =
            _col_sum(Y.c.ρe_tot, ᶜJ) + hcap * vec(Array(parent(Y.sfc.T)))[1] * A_sfc
        Δt_s = FT(120)
        CA.step_vertical_diffusion_and_surface!(Y, p_slab, FT(0), Δt_s)
        CA.step_slab_ocean_turbulent_and_precip!(Y, p_slab, Δt_s)
        # Turbulent water & heat exchange between atmos and slab cancels; slab also
        # receives prescribed P_liq and pet.
        w_tot_after =
            _col_sum(Y.c.ρq_tot, ᶜJ) + vec(Array(parent(Y.sfc.water)))[1] * A_sfc
        @test w_tot_after ≈ w_tot_before - Δt_s * FT(-1e-4) * A_sfc rtol = 1e-13
        e_tot_after =
            _col_sum(Y.c.ρe_tot, ᶜJ) + hcap * vec(Array(parent(Y.sfc.T)))[1] * A_sfc
        @test e_tot_after ≈ e_tot_before - Δt_s * FT(-50.0) * A_sfc rtol = 1e-13

        if _is_cpu(Y.c)
            @test _alloc_check(CA.repair_column_water_conservatively!, Y, p_slab) == 0
            @test _alloc_check(CA.step_slab_ocean_turbulent_and_precip!, Y, p_slab, Δt_s) ==
                  0
        end

        # End-to-end step_column_physics! on 1M + SlabOcean over non-unit
        # horizontal area and stretched vertical grid with active precip.
        ctx_e2e = ClimaComms.SingletonCommsContext()
        grid_e2e = CA.BoxGrid(
            FT;
            context = ctx_e2e,
            x_elem = 2,
            y_elem = 2,
            z_elem = 8,
            z_max = FT(12000),
            x_max = FT(100e3),
            y_max = FT(100e3),
            nh_poly = 2,
            z_stretch = true,
        )
        vdp_e2e = CAP.vert_diff_params(CA.ClimaAtmosParameters(FT))
        vd_e2e = CA.VerticalDiffusion{FT}(;
            disable_momentum_vertical_diffusion = false,
            C_E = vdp_e2e.C_E,
        )
        sim_e2e = _build_sim(
            FT;
            grid = grid_e2e,
            vertical_diffusion = vd_e2e,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_slab,
            disable_surface_flux_tendency = false,
        )
        Y_e2e, p_e2e = sim_e2e.Y, sim_e2e.p
        z_e2e = Fields.coordinate_field(Y_e2e.c).z
        @. Y_e2e.c.uₕ = CA.C12(FT(10), FT(0))
        @. Y_e2e.c.ρq_tot = Y_e2e.c.ρ * FT(0.012)
        @. Y_e2e.c.ρq_lcl = Y_e2e.c.ρ * FT(5e-4) * exp(-((z_e2e - FT(1200)) / FT(400))^2)
        @. Y_e2e.c.ρq_rai = Y_e2e.c.ρ * FT(1e-3) * exp(-(z_e2e / FT(1500))^2)
        @. Y_e2e.c.ρq_sno = Y_e2e.c.ρ * FT(5e-4) * exp(-((z_e2e - FT(500)) / FT(800))^2)
        @. Y_e2e.sfc.T = FT(302)
        CA.set_precomputed_quantities!(Y_e2e, p_e2e, FT(0))

        hs_e2e = Spaces.horizontal_space(axes(Y_e2e.c))
        _sfc_sum_e2e(f) = sum(Fields.Field(Fields.field_values(f), hs_e2e))
        w_e2e_before = sum(Y_e2e.c.ρq_tot) + _sfc_sum_e2e(Y_e2e.sfc.water)
        e_e2e_before = sum(Y_e2e.c.ρe_tot) + hcap * _sfc_sum_e2e(Y_e2e.sfc.T)
        ρd_e2e_before = sum(Y_e2e.c.ρ .- Y_e2e.c.ρq_tot)

        CA.step_column_physics!(Y_e2e, p_e2e, FT(0), FT(120))

        @test minimum(Array(parent(p_e2e.precomputed.surface_rain_flux))) < FT(-1e-6)
        @test sum(Y_e2e.c.ρ .- Y_e2e.c.ρq_tot) ≈ ρd_e2e_before rtol = 1e-13
        @test sum(Y_e2e.c.ρq_tot) + _sfc_sum_e2e(Y_e2e.sfc.water) ≈ w_e2e_before rtol =
            1e-13
        @test sum(Y_e2e.c.ρe_tot) + hcap * _sfc_sum_e2e(Y_e2e.sfc.T) ≈ e_e2e_before rtol =
            1e-13
    end

    @testset "3D terrain duplicate-node + dss! check" begin
        FT = Float64
        ctx = ClimaComms.SingletonCommsContext()
        topo = CA.CosineTopography{3, FT}(; h_max = FT(200), λ = FT(100e3))
        grid = CA.BoxGrid(
            FT;
            context = ctx,
            x_elem = 2,
            y_elem = 2,
            z_elem = 8,
            z_max = FT(3000),
            x_max = FT(100e3),
            y_max = FT(100e3),
            nh_poly = 2,
            topography = topo,
        )
        vdp = CAP.vert_diff_params(CA.ClimaAtmosParameters(FT))
        vd = CA.VerticalDiffusion{FT}(;
            disable_momentum_vertical_diffusion = false,
            C_E = vdp.C_E,
        )
        (; Y, p) = _build_sim(
            FT;
            grid,
            vertical_diffusion = vd,
            microphysics_model = CA.EquilibriumMicrophysics0M(),
            disable_surface_flux_tendency = true,
        )
        z = Fields.coordinate_field(Y.c).z
        @. Y.c.ρq_tot = Y.c.ρ * max(FT(0), FT(0.002) * (one(FT) - z / FT(3000)))
        @. Y.c.uₕ = CA.C12(FT(12), FT(4))
        CA.set_precomputed_quantities!(Y, p, FT(0))

        ᶜJ = Fields.local_geometry_field(Y.c).J
        # Global weighted integrals via ClimaCore sum(field)
        qt_before = sum(Y.c.ρq_tot)
        e_before = sum(Y.c.ρe_tot)
        ρd_before = sum(Y.c.ρ .- Y.c.ρq_tot)

        CA.step_column_physics!(Y, p, FT(0), FT(60))
        CA.dss!(Y, p, FT(0))

        @test maximum(abs, Array(parent(p.precomputed.surface_rain_flux))) == FT(0)
        @test sum(Y.c.ρq_tot) ≈ qt_before rtol = 1e-12
        @test sum(Y.c.ρe_tot) ≈ e_before rtol = 1e-12
        @test sum(Y.c.ρ .- Y.c.ρq_tot) ≈ ρd_before rtol = 1e-12
        # Verify element-boundary continuity: DSS after step_column_physics! is idempotent
        Y_copy = copy(Y)
        CA.dss!(Y_copy, p, FT(0))
        @test Array(parent(Y_copy.c.ρq_tot)) == Array(parent(Y.c.ρq_tot))
        @test Array(parent(Y_copy.c.ρe_tot)) == Array(parent(Y.c.ρe_tot))
    end

    @testset "PrognosticEDMFX CT3 flux complement, FCT isolation, and subcycling" begin
        FT = Float64
        tc_sub = CA.PrognosticEDMFX{1, true}(FT(0.05))
        edmf_ho = CA.EDMFXModel(;
            entr_model = CA.InvZEntrainment(),
            detr_model = CA.BuoyancyVelocityDetrainment(),
            sgs_mass_flux = true,
            sgs_diffusive_flux = true,
            nh_pressure = false,
            vertical_diffusion = true,
            scale_blending_method = CA.SmoothMinimumBlending(),
        )
        slab_sub = CA.SurfaceConditions.SlabOceanTemperature{FT}(;
            depth_ocean = FT(1),
            q_flux = false,
        )
        hcap_sub = slab_sub.ρ_ocean * slab_sub.cp_ocean * slab_sub.depth_ocean
        sfc_sub = CA.AtmosSurface(; temperature = slab_sub)
        sim_sub = _build_sim(
            FT;
            z_elem = 16,
            z_max = FT(4000),
            turbconv_model = tc_sub,
            edmfx_model = edmf_ho,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_sub,
            disable_surface_flux_tendency = false,
            dt = FT(150),
        )
        Y_s, p_s = sim_sub.Y, sim_sub.p
        z_c = Fields.coordinate_field(Y_s.c).z
        z_f = Fields.coordinate_field(Y_s.f).z
        ᶠlg_s = Fields.local_geometry_field(Y_s.f)
        tps_s = CAP.thermodynamics_params(p_s.params)

        p_pure = _with_cache(
            p_s;
            atmos = CA.AtmosModel(
                p_s.atmos;
                disable_surface_flux_tendency = true,
                turbconv = CA.AtmosTurbconv(;
                    turbconv_model = p_s.atmos.turbconv_model,
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
            ),
        )

        # 1. `CT3` `p.scratch.ᶠF̃` complement (including `u_rel < 0`)
        Y_ct3 = deepcopy(Y_s)
        w_gm = FT(4.0)
        dz_s = vec(Array(parent(Fields.Δz_field(axes(Y_ct3.f)))))[2]
        Δt_ct3 = FT(20.0)
        q_prof_ct3 = @. FT(0.005) + FT(0.008) * exp(-z_c / FT(1500))
        @. Y_ct3.c.uₕ = CA.C12(FT(0), FT(0))
        @. Y_ct3.c.ρ = FT(1.0)
        @. Y_ct3.c.ρtke = FT(0)
        @. Y_ct3.c.sgsʲs.:(1).ρa = FT(0.1)
        @. Y_ct3.c.ρq_tot = Y_ct3.c.ρ * q_prof_ct3
        @. Y_ct3.c.sgsʲs.:(1).q_tot = q_prof_ct3
        @. Y_ct3.c.ρe_tot =
            Y_ct3.c.ρ * TD.total_energy(tps_s, FT(0), p_s.core.ᶜΦ, FT(300), q_prof_ct3)
        @. Y_ct3.c.ρq_lcl = FT(0)
        @. Y_ct3.c.ρq_icl = FT(0)
        @. Y_ct3.c.ρq_rai = FT(0)
        @. Y_ct3.c.ρq_sno = FT(0)
        @. Y_ct3.c.sgsʲs.:(1).q_lcl = FT(0)
        @. Y_ct3.c.sgsʲs.:(1).q_icl = FT(0)
        @. Y_ct3.c.sgsʲs.:(1).q_rai = FT(0)
        @. Y_ct3.c.sgsʲs.:(1).q_sno = FT(0)
        @. Y_ct3.f.sgsʲs.:(1).u₃ = ifelse(
            abs(z_f - dz_s) < FT(1),
            CA.C3(w_gm * dz_s + Δt_ct3 * w_gm^2 / 2),
            ifelse(
                (z_f > FT(0)) & (z_f < FT(4000)),
                Geometry.Covariant3Vector(Geometry.WVector(w_gm), ᶠlg_s),
                CA.C3(FT(0)),
            ),
        )
        @. p_pure.scratch.ᶠF̃ = ifelse(
            (z_f > FT(0)) & (z_f < FT(4000)),
            Geometry.Contravariant3Vector(Geometry.WVector(w_gm), ᶠlg_s),
            Geometry.Contravariant3Vector(Geometry.WVector(FT(0)), ᶠlg_s),
        )
        CA.set_precomputed_quantities!(Y_ct3, p_pure, FT(0))
        @. Y_ct3.c.sgsʲs.:(1).mse = p_pure.precomputed.ᶜh_tot - p_pure.precomputed.ᶜK
        CA.set_precomputed_quantities!(Y_ct3, p_pure, FT(0))
        fill!(p_pure.precomputed.ᶜentr_vel_scaleʲs.:(1), FT(0))
        fill!(p_pure.precomputed.ᶜentr_nonvel_rateʲs.:(1), FT(0))
        fill!(p_pure.precomputed.ᶜρ_diffʲs.:(1), FT(0))
        fill!(p_pure.precomputed.sfc_mass_flux_sourceʲs.:(1), FT(0))
        Y_neg = deepcopy(Y_ct3)

        CA.step_edmf_and_env_turbulence!(
            Y_ct3,
            p_pure,
            FT(0),
            Δt_ct3;
            ᶠF̃ = p_pure.scratch.ᶠF̃,
        )
        # `CT3` `ᶠF̃` cancels `JF1_full` so `JF1_rel` (`ᶠtemp_scalar_2`) is 0.
        @test maximum(abs.(Array(parent(Y_ct3.c.ρq_tot)) .- Array(parent(q_prof_ct3)))) <
              FT(1e-13)
        @test maximum(abs, Array(parent(p_pure.scratch.ᶠtemp_scalar_2))) < FT(1e-13)

        # Supply larger `ᶠF̃` so `JF1_rel < 0` (downward relative updraft flux)
        # and verify conservation.
        @. p_pure.scratch.ᶠF̃ = ifelse(
            (z_f > FT(0)) & (z_f < FT(4000)),
            Geometry.Contravariant3Vector(Geometry.WVector(FT(2.0) * w_gm), ᶠlg_s),
            Geometry.Contravariant3Vector(Geometry.WVector(FT(0)), ᶠlg_s),
        )
        ρqt_neg_pre = sum(Y_neg.c.ρq_tot)
        CA.step_edmf_and_env_turbulence!(
            Y_neg,
            p_pure,
            FT(0),
            Δt_ct3;
            ᶠF̃ = p_pure.scratch.ᶠF̃,
        )
        jf1_rel_arr = vec(Array(parent(p_pure.scratch.ᶠtemp_scalar_2)))
        @test minimum(jf1_rel_arr[3:(end - 2)]) < -FT(0.1)
        @test sum(Y_neg.c.ρq_tot) ≈ ρqt_neg_pre rtol = 1e-13

        # 2. Environment FCT isolation (`α_face_fv[2] == 0` on `q_tot` does not disable
        #    environment FCT on `η` at upper faces) + `:first_order` vs `:third_order`
        num_3rd = CA.AtmosNumerics(; edmfx_sgsflux_upwinding = :third_order)
        num_1st = CA.AtmosNumerics(; edmfx_sgsflux_upwinding = :first_order)
        p_3rd = _with_cache(
            p_pure;
            atmos = CA.AtmosModel(p_pure.atmos; numerics = num_3rd),
        )
        p_1st = _with_cache(
            p_pure;
            atmos = CA.AtmosModel(p_pure.atmos; numerics = num_1st),
        )
        Y_fct3 = deepcopy(Y_s)
        # Put a sharp surface-adjacent extremum in `q_tot` (forcing `α_face[2] = 0`)
        # and an elevated smooth θ perturbation aloft where `:third_order` differs from `:first_order`
        qt_ext =
            @. ifelse(z_c < FT(300), FT(0.002), ifelse(z_c < FT(600), FT(0.012), FT(0.004)))
        T_pert = @. FT(290) + FT(4.0) * exp(-((z_c - FT(2200)) / FT(450))^2)
        @. Y_fct3.c.ρq_tot = Y_fct3.c.ρ * qt_ext
        @. Y_fct3.c.ρq_lcl = FT(0)
        @. Y_fct3.c.ρq_icl = FT(0)
        @. Y_fct3.c.ρq_rai = FT(0)
        @. Y_fct3.c.ρq_sno = FT(0)
        @. Y_fct3.c.ρe_tot =
            Y_fct3.c.ρ * TD.total_energy(tps_s, FT(0), p_s.core.ᶜΦ, T_pert, qt_ext)
        @. Y_fct3.c.sgsʲs.:(1).ρa = FT(0.08) * Y_fct3.c.ρ
        @. Y_fct3.c.sgsʲs.:(1).q_tot = qt_ext
        @. Y_fct3.c.sgsʲs.:(1).q_lcl = FT(0)
        @. Y_fct3.c.sgsʲs.:(1).q_icl = FT(0)
        @. Y_fct3.c.sgsʲs.:(1).q_rai = FT(0)
        @. Y_fct3.c.sgsʲs.:(1).q_sno = FT(0)
        @. Y_fct3.c.sgsʲs.:(1).mse = TD.enthalpy(tps_s, T_pert, qt_ext) + p_s.core.ᶜΦ
        @. Y_fct3.f.sgsʲs.:(1).u₃ = ifelse(
            (z_f > FT(0)) & (z_f < FT(4000)),
            Geometry.Covariant3Vector(Geometry.WVector(FT(6.0)), ᶠlg_s),
            CA.C3(FT(0)),
        )
        fill!(parent(p_3rd.scratch.ᶠF̃), FT(0))
        Y_fct1 = deepcopy(Y_fct3)
        CA.set_precomputed_quantities!(Y_fct3, p_3rd, FT(0))
        fill!(p_3rd.precomputed.ᶜentr_vel_scaleʲs.:(1), FT(0))
        fill!(p_3rd.precomputed.ᶜentr_nonvel_rateʲs.:(1), FT(0))
        fill!(p_3rd.precomputed.ᶜρ_diffʲs.:(1), FT(0))
        fill!(p_3rd.precomputed.sfc_mass_flux_sourceʲs.:(1), FT(0))
        CA.set_precomputed_quantities!(Y_fct1, p_1st, FT(0))
        fill!(p_1st.precomputed.ᶜentr_vel_scaleʲs.:(1), FT(0))
        fill!(p_1st.precomputed.ᶜentr_nonvel_rateʲs.:(1), FT(0))
        fill!(p_1st.precomputed.ᶜρ_diffʲs.:(1), FT(0))
        fill!(p_1st.precomputed.sfc_mass_flux_sourceʲs.:(1), FT(0))
        CA.step_edmf_and_env_turbulence!(Y_fct3, p_3rd, FT(0), FT(25))
        CA.step_edmf_and_env_turbulence!(Y_fct1, p_1st, FT(0), FT(25))
        ρe_diff_upper =
            maximum(
                abs.(
                    vec(Array(parent(Y_fct3.c.ρe_tot)))[6:14] .-
                    vec(Array(parent(Y_fct1.c.ρe_tot)))[6:14],
                ),
            )
        @test ρe_diff_upper > FT(1.0)
        @test minimum(Array(parent(Y_fct3.c.ρq_tot))) >= FT(0)
        @test minimum(Array(parent(Y_fct1.c.ρq_tot))) >= FT(0)

        # 3. Subcycled `step_column_physics!` at `Δt = 150 s` (`n_sub = 3`):
        #    exact dry-air, water, and energy + `SlabOcean` conservation and
        #    0 CPU allocations.
        @. Y_s.c.uₕ = CA.C12(FT(8), FT(2))
        @. Y_s.c.ρq_tot = Y_s.c.ρ * FT(0.013)
        @. Y_s.c.ρq_lcl =
            Y_s.c.ρ * FT(6e-4) * exp(-((z_c - FT(1200)) / FT(400))^2)
        @. Y_s.c.ρq_rai = Y_s.c.ρ * FT(1e-3) * exp(-(z_c / FT(1500))^2)
        @. Y_s.c.sgsʲs.:(1).ρa = FT(0.05) * Y_s.c.ρ
        @. Y_s.c.sgsʲs.:(1).q_tot = FT(0.014)
        @. Y_s.c.sgsʲs.:(1).q_lcl =
            FT(8e-4) * exp(-((z_c - FT(1200)) / FT(400))^2)
        @. Y_s.c.sgsʲs.:(1).q_rai = FT(5e-4) * exp(-(z_c / FT(1500))^2)
        @. Y_s.f.sgsʲs.:(1).u₃ = ifelse(
            (z_f > FT(0)) & (z_f < FT(4000)),
            Geometry.Covariant3Vector(Geometry.WVector(FT(2.5)), ᶠlg_s),
            CA.C3(FT(0)),
        )
        @. Y_s.sfc.T = FT(302)
        CA.set_precomputed_quantities!(Y_s, p_s, FT(0))
        @. Y_s.c.sgsʲs.:(1).mse = p_s.precomputed.ᶜh_tot - p_s.precomputed.ᶜK + FT(500)
        CA.set_precomputed_quantities!(Y_s, p_s, FT(0))
        fill!(parent(p_s.scratch.ᶠF̃), FT(0))

        A_sfc_s = vec(Array(parent(@. _surface_area(ᶠlg_s))))[1]
        _sfc_sum_s(f) = sum(Array(parent(f))) * A_sfc_s
        ρd_s_pre = sum(Y_s.c.ρ .- Y_s.c.ρq_tot)
        w_s_pre = sum(Y_s.c.ρq_tot) + _sfc_sum_s(Y_s.sfc.water)
        e_s_pre = sum(Y_s.c.ρe_tot) + hcap_sub * _sfc_sum_s(Y_s.sfc.T)

        CA.step_column_physics!(Y_s, p_s, FT(0), FT(150); ᶠF̃ = p_s.scratch.ᶠF̃)

        @test minimum(Array(parent(p_s.precomputed.surface_rain_flux))) < FT(-1e-6)
        @test sum(Y_s.c.ρ .- Y_s.c.ρq_tot) ≈ ρd_s_pre rtol = 1e-13
        @test sum(Y_s.c.ρq_tot) + _sfc_sum_s(Y_s.sfc.water) ≈ w_s_pre rtol = 1e-13
        @test sum(Y_s.c.ρe_tot) + hcap_sub * _sfc_sum_s(Y_s.sfc.T) ≈ e_s_pre rtol =
            1e-13
        if _is_cpu(Y_s.c.ρ)
            Y_alloc_s = deepcopy(Y_s)
            CA.set_precomputed_quantities!(Y_alloc_s, p_s, FT(0))
            CA.step_column_physics!(
                Y_alloc_s,
                p_s,
                FT(0),
                FT(150);
                ᶠF̃ = p_s.scratch.ᶠF̃,
            )
            CA.set_precomputed_quantities!(Y_alloc_s, p_s, FT(0))
            _alloc_step_sub(Y_, p_) = @allocated CA.step_column_physics!(
                Y_,
                p_,
                FT(0),
                FT(150);
                ᶠF̃ = p_.scratch.ᶠF̃,
            )
            b_sub = _alloc_step_sub(Y_alloc_s, p_s)
            @test b_sub == 0
        end
    end

    @testset "Physical-component momentum diffusion and vertical Smagorinsky stress" begin
        FT = Float64
        ctx = ClimaComms.SingletonCommsContext()
        vdp = CAP.vert_diff_params(CA.ClimaAtmosParameters(FT))
        vd = CA.VerticalDiffusion{FT}(;
            disable_momentum_vertical_diffusion = false,
            C_E = vdp.C_E,
        )
        # Physical horizontal velocity at centers, as in `ᶜu`.
        _physical_uv(Y) = @. Geometry.UVVector(Y.c.uₕ) +
           Geometry.UVVector(CA.ᶜinterp(Y.f.u₃))

        # 1. Flat column: the physical-component solve equals the scalar
        # backward-Euler solve of the covariant components.
        (; Y, p) = _build_sim(
            FT;
            z_elem = 12,
            z_max = FT(3000),
            vertical_diffusion = vd,
            microphysics_model = CA.DryModel(),
        )
        zc = Fields.coordinate_field(Y.c).z
        @. Y.c.uₕ = CA.C12(FT(10) * exp(-zc / FT(800)), FT(-3) + zc / FT(1000))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        Δt = FT(600)
        ρ_col = vec(Array(parent(Y.c.ρ)))
        J_col = vec(Array(parent(Fields.local_geometry_field(Y.c).J)))
        ᶜK = CA.ᶜcompute_eddy_diffusivity_coefficient(
            Y.c.uₕ, p.precomputed.ᶜp, vd,
        )
        ᶠρK = @. CA.ᶠinterp(Y.c.ρ) / CA.ᶠinterp(1 / max(ᶜK, eps(FT)))
        ᶠlg = Fields.local_geometry_field(Y.f)
        G = vec(Array(parent(@. CA.face_conductance(ᶠlg, ᶠρK))))
        _uc(Y, c) = vec(Array(parent(getproperty(Y.c.uₕ.components.data, c))))
        u_old = (_uc(Y, 1), _uc(Y, 2))
        CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), Δt)
        nz = length(ρ_col)
        A = zeros(FT, nz, nz)
        for k in 1:nz
            A[k, k] = J_col[k] * ρ_col[k]
            if k < nz
                A[k, k] += Δt * G[k + 1]
                A[k, k + 1] = -Δt * G[k + 1]
            end
            if k > 1
                A[k, k] += Δt * G[k]
                A[k, k - 1] = -Δt * G[k]
            end
        end
        for c in 1:2
            ref = A \ (J_col .* ρ_col .* u_old[c])
            @test _uc(Y, c) ≈ ref rtol = 1e-12
        end
        @test maximum(abs, _uc(Y, 1) .- u_old[1]) > FT(1e-3)

        # 2. Terrain: a height-independent physical wind with a sloped `w`
        # profile is a fixed point of momentum diffusion, although `u₁`
        # varies with height through `w ∂z/∂ξ¹`.
        topo = CA.CosineTopography{3, FT}(; h_max = FT(600), λ = FT(20e3))
        grid_topo = CA.BoxGrid(
            FT;
            context = ctx,
            x_elem = 2,
            y_elem = 2,
            z_elem = 10,
            z_max = FT(3000),
            x_max = FT(20e3),
            y_max = FT(20e3),
            nh_poly = 2,
            topography = topo,
        )
        (; Y, p) = _build_sim(
            FT;
            grid = grid_topo,
            vertical_diffusion = vd,
            microphysics_model = CA.DryModel(),
        )
        # The state is set after the cache update, which would reset the
        # bottom-face `u₃` to the kinematic boundary condition.
        CA.set_precomputed_quantities!(Y, p, FT(0))
        zf = Fields.coordinate_field(Y.f).z
        @. Y.f.u₃ = CA.C3(Geometry.WVector(FT(2) * sin(FT(π) * zf / FT(3000))))
        uv_0 = Geometry.UVVector(FT(10), FT(-4))
        ᶜuv_w = @. Geometry.UVVector(CA.ᶜinterp(Y.f.u₃))
        ᶜuv_h = ᶜuv_w .* (-1) .+ Ref(uv_0)
        @. Y.c.uₕ = CA.C12(ᶜuv_h)
        u₁_0 = Array(parent(Y.c.uₕ.components.data.:1))
        u₃_0 = Array(parent(Y.f.u₃))
        @test maximum(u₁_0) - minimum(u₁_0) > FT(0.1)
        CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), FT(600))
        uv = _physical_uv(Y)
        @test maximum(abs, Array(parent(uv.components.data.:1)) .- FT(10)) <
              FT(1e-10)
        @test maximum(abs, Array(parent(uv.components.data.:2)) .+ FT(4)) <
              FT(1e-10)
        @test Array(parent(Y.f.u₃)) == u₃_0

        # Column momentum of the physical wind is conserved on slopes.
        zc = Fields.coordinate_field(Y.c).z
        @. Y.c.uₕ = CA.C12(
            Geometry.UVVector(FT(8) * exp(-zc / FT(700)), FT(2)) - ᶜuv_w,
        )
        ᶜJ = Fields.local_geometry_field(Y.c).J
        ρJ = @. Y.c.ρ * ᶜJ
        mom_0 = sum(Array(parent(ρJ .* _physical_uv(Y).components.data.:1)))
        uv_before = Array(parent(_physical_uv(Y).components.data.:1))
        CA.step_vertical_diffusion_and_surface!(Y, p, FT(0), FT(600))
        uv_after = Array(parent(_physical_uv(Y).components.data.:1))
        @test maximum(abs, uv_after .- uv_before) > FT(1e-2)
        @test sum(Array(parent(ρJ)) .* uv_after) ≈ mom_0 rtol = 1e-12
        if _is_cpu(Y.c)
            @test _alloc_check(
                CA.step_vertical_diffusion_and_surface!, Y, p, FT(0), FT(60),
            ) == 0
        end

        # 3. Vertical Smagorinsky-Lilly on a flat box: the implicit column
        # solve plus the explicit remainder equals the full stress operator.
        L = FT(4e3)
        H = FT(1e3)
        grid_box = CA.BoxGrid(
            FT;
            context = ctx,
            x_elem = 2,
            y_elem = 2,
            z_elem = 16,
            z_max = H,
            x_max = L,
            y_max = L,
            nh_poly = 3,
        )
        smag = CA.SmagorinskyLilly(; axes = :UVW)
        (; Y, p) = _build_sim(
            FT;
            grid = grid_box,
            smagorinsky_lilly = smag,
            microphysics_model = CA.DryModel(),
        )
        cc = Fields.coordinate_field(Y.c)
        cf = Fields.coordinate_field(Y.f)
        kx = 2 * FT(π) / L
        @. Y.c.uₕ = CA.C12(
            Geometry.UVVector(
                FT(5) + FT(20) * sin(FT(π) * cc.z / H) * cos(kx * cc.y),
                FT(-10) * cos(FT(π) * cc.z / H) * sin(kx * cc.x),
            ),
        )
        @. Y.f.u₃ = CA.C3(
            Geometry.WVector(
                FT(3) * sin(FT(π) * cf.z / H) * cos(kx * cf.x) *
                (FT(1) + FT(0.5) * sin(kx * cf.y)),
            ),
        )
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @test maximum(Array(parent(p.precomputed.ᶜνₜ_v))) > 0

        Yₜ_full = zero(Y)
        CA.vertical_smagorinsky_lilly_tendency!(Yₜ_full, Y, p, FT(0), smag)
        Yₜ_expl = zero(Y)
        CA.vertical_smagorinsky_lilly_explicit_tendency!(
            Yₜ_expl, Y, p, FT(0), smag,
        )
        # Backward Euler differs from the operator by O(Δt ‖L‖) ≈ 2e-5 here.
        Δt_s = FT(1e-4)
        Y_impl = copy(Y)
        CA.step_vertical_diffusion_and_surface!(Y_impl, p, FT(0), Δt_s)

        for (full, expl, new, old) in (
            (Yₜ_full.c.uₕ, Yₜ_expl.c.uₕ, Y_impl.c.uₕ, Y.c.uₕ),
            (Yₜ_full.f.u₃, Yₜ_expl.f.u₃, Y_impl.f.u₃, Y.f.u₃),
        )
            ref = Array(parent(full))
            split =
                Array(parent(expl)) .+
                (Array(parent(new)) .- Array(parent(old))) ./ Δt_s
            scale = maximum(abs, ref)
            @test scale > 0
            @test maximum(abs, split .- ref) < FT(1e-4) * scale
        end
        # The explicit cross term `∂_z(ρνₜ∇ₕw)` matters in `uₕ`, and the
        # explicit remainder vanishes in `u₃` on flat coordinates.
        @test maximum(abs, Array(parent(Yₜ_expl.c.uₕ))) >
              FT(1e-2) * maximum(abs, Array(parent(Yₜ_full.c.uₕ)))
        @test maximum(abs, Array(parent(Yₜ_expl.f.u₃))) <
              FT(1e-12) * maximum(abs, Array(parent(Yₜ_full.f.u₃)))
        if _is_cpu(Y.c)
            @test _alloc_check(
                CA.step_vertical_diffusion_and_surface!, Y, p, FT(0), Δt_s,
            ) == 0
        end
    end

    @testset "DryModel column physics step" begin
        FT = Float64
        mo = CA.SurfaceConditions.MoninObukhov(; z0m = FT(1e-3), z0b = FT(1e-3))
        sfc_dry = CA.AtmosSurface(;
            flux_scheme = mo,
            temperature = CA.SurfaceConditions.AnalyticTemperature(Returns(FT(300))),
        )
        vd_dry = CA.VerticalDiffusion{FT}(;
            disable_momentum_vertical_diffusion = false,
            C_E = CAP.vert_diff_params(CA.ClimaAtmosParameters(FT)).C_E,
        )
        (; Y, p) = _build_sim(
            FT;
            z_elem = 12,
            z_max = FT(3000),
            vertical_diffusion = vd_dry,
            microphysics_model = CA.DryModel(),
            surface_setup = sfc_dry,
            disable_surface_flux_tendency = false,
            dt = FT(60),
        )
        ᶜJ = Fields.local_geometry_field(Y.c).J
        @. Y.c.uₕ = CA.C12(FT(10), FT(-5))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        mass_0 = _col_sum(Y.c.ρ, ᶜJ)
        uₕ_0 = Array(parent(Y.c.uₕ))

        CA.step_column_physics!(Y, p, FT(0), FT(60))

        @test all(isfinite, Array(parent(Y.c.ρ)))
        @test all(isfinite, Array(parent(Y.c.ρe_tot)))
        @test all(isfinite, Array(parent(Y.c.uₕ)))
        @test _col_sum(Y.c.ρ, ᶜJ) ≈ mass_0 rtol = 10 * eps(FT)
        # Surface drag changes the horizontal wind.
        @test Array(parent(Y.c.uₕ)) != uₕ_0
        @test all(iszero, Array(parent(p.precomputed.surface_rain_flux)))
        @test all(iszero, Array(parent(p.precomputed.surface_snow_flux)))
    end

    @testset "advection_test freezes all velocities" begin
        FT = Float64
        params_ref = CA.ClimaAtmosParameters(FT)
        mo = CA.SurfaceConditions.MoninObukhov(; z0m = FT(1e-3), z0b = FT(1e-3))
        sfc_adv = CA.AtmosSurface(;
            flux_scheme = mo,
            temperature = CA.SurfaceConditions.AnalyticTemperature(Returns(FT(300))),
        )
        vd = CA.VerticalDiffusion{FT}(;
            disable_momentum_vertical_diffusion = false,
            C_E = CAP.vert_diff_params(params_ref).C_E,
        )
        (; Y, p) = _build_sim(
            FT;
            z_elem = 12,
            z_max = FT(3000),
            vertical_diffusion = vd,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_adv,
            disable_surface_flux_tendency = false,
            advection_test = true,
            dt = FT(60),
        )
        zc = Fields.coordinate_field(Y.c).z
        zf = Fields.coordinate_field(Y.f).z
        @. Y.c.uₕ = CA.C12(FT(10), FT(-5))
        @. Y.f.u₃ = CA.C3(FT(0.2) * sin(FT(π) * zf / FT(3000)))
        @. Y.c.ρq_tot = Y.c.ρ * (FT(0.005) + FT(0.006) * exp(-zc / FT(1000)))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        uₕ_0 = copy(Array(parent(Y.c.uₕ)))
        u3_0 = copy(Array(parent(Y.f.u₃)))
        qt_0 = copy(Array(parent(Y.c.ρq_tot)))

        CA.step_column_physics!(Y, p, FT(0), FT(60))

        @test Array(parent(Y.c.uₕ)) == uₕ_0
        @test Array(parent(Y.f.u₃)) == u3_0
        @test Array(parent(Y.c.ρq_tot)) != qt_0

        tc = CA.PrognosticEDMFX{1, true}(FT(0.05))
        edmfx = CA.EDMFXModel(;
            entr_model = CA.InvZEntrainment(),
            detr_model = CA.BuoyancyVelocityDetrainment(),
            sgs_mass_flux = true,
            sgs_diffusive_flux = true,
            nh_pressure = true,
            vertical_diffusion = true,
            filter = true,
            scale_blending_method = CA.SmoothMinimumBlending(),
        )
        (; Y, p) = _build_sim(
            FT;
            z_elem = 12,
            z_max = FT(3000),
            turbconv_model = tc,
            edmfx_model = edmfx,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            surface_setup = sfc_adv,
            disable_surface_flux_tendency = false,
            advection_test = true,
            dt = FT(60),
        )
        zc_ed = Fields.coordinate_field(Y.c).z
        zf_ed = Fields.coordinate_field(Y.f).z
        @. Y.c.uₕ = CA.C12(FT(8), FT(-4))
        @. Y.f.u₃ = CA.C3(FT(0.15) * sin(FT(π) * zf_ed / FT(3000)))
        @. Y.f.sgsʲs.:(1).u₃ = CA.C3(FT(1.5) * sin(FT(π) * zf_ed / FT(3000)))
        @. Y.c.sgsʲs.:(1).ρa = FT(0.04) * Y.c.ρ
        @. Y.c.sgsʲs.:(1).q_tot = FT(0.012) * exp(-zc_ed / FT(2000))
        @. Y.c.ρq_tot = Y.c.ρ * FT(0.008) * exp(-zc_ed / FT(2000))
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @. Y.c.sgsʲs.:(1).mse = p.precomputed.ᶜh_tot + FT(400)
        CA.set_precomputed_quantities!(Y, p, FT(0))
        uₕ_ed0 = copy(Array(parent(Y.c.uₕ)))
        u3_ed0 = copy(Array(parent(Y.f.u₃)))
        u3_up0 = copy(Array(parent(Y.f.sgsʲs.:(1).u₃)))
        ρa_ed0 = copy(Array(parent(Y.c.sgsʲs.:(1).ρa)))

        CA.step_column_physics!(Y, p, FT(0), FT(60))

        @test Array(parent(Y.c.uₕ)) == uₕ_ed0
        @test Array(parent(Y.f.u₃)) == u3_ed0
        @test Array(parent(Y.f.sgsʲs.:(1).u₃)) == u3_up0
        @test Array(parent(Y.c.sgsʲs.:(1).ρa)) != ρa_ed0
        @test all(isfinite, Array(parent(Y.c.ρ)))
        @test all(isfinite, Array(parent(Y.c.ρe_tot)))
    end

end  # @testset "Column Physics Step"
