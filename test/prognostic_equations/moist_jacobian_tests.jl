#=
Unit and physics verification tests for saturation-equilibrium pressure
derivatives in the wave-core Jacobian (`moist_pressure_derivatives.jl` and
`update_advection_jacobian!`): ForwardDiff exactness across warm, mixed-phase,
and ice regimes, column `Wfact!` finite-difference agreement and zero
allocations, single-iteration W-method stability on `ARS343` and `ARS222`, and
2D (`k_h > 0`) dry versus saturated `0M` gravity-wave eigenvalues.
=#
using Test
using LinearAlgebra
import ForwardDiff
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Geometry, MatrixFields, Operators, Spaces
import ClimaCore.MatrixFields: @name
import ClimaTimeSteppers as CTS
import Thermodynamics as TD

include("../test_helpers.jl")

const u₁_name = @name(c.uₕ.components.data.:1)
const u₂_name = @name(c.uₕ.components.data.:2)
const u₃_name = @name(f.u₃.components.data.:1)

@noinline function _measure_update_jacobian_allocs(jac, Y, p, dtγ, t)
    CA.update_jacobian!(jac, Y, p, dtγ, t)
    return @allocated CA.update_jacobian!(jac, Y, p, dtγ, t)
end

@testset "AD exactness of saturated pressure derivatives and subsaturated C_v,sat" begin
    for (FT, rtol) in ((Float64, 1e-11), (Float32, 5e-5))
        params = CA.ClimaAtmosParameters(FT)
        thermo_params = CAP.thermodynamics_params(params)
        K = FT(42)
        Φ = FT(12500)

        @test TD.∂λ_∂T_ramp(thermo_params, FT(260)) > zero(FT)
        @test iszero(TD.∂λ_∂T_ramp(thermo_params, FT(295)))
        @test iszero(TD.∂λ_∂T_ramp(thermo_params, FT(225)))

        for (regime, T_eq, ρ) in (
            ("warm", FT(295), FT(1.12)),
            ("mixed_phase", FT(260), FT(0.75)),
            ("ice", FT(225), FT(0.42)),
        )
            q_sat = TD.q_vap_saturation(thermo_params, T_eq, ρ)
            q_tot = FT(1.6) * q_sat + FT(2e-3)
            @test q_tot > q_sat

            e_int = TD.internal_energy_sat(thermo_params, T_eq, ρ, q_tot)
            ρe_tot = ρ * (e_int + K + Φ)
            ρq_tot = ρ * q_tot

            closed = CA.saturated_pressure_derivatives(
                thermo_params,
                T_eq,
                ρ,
                q_tot,
                K,
                Φ,
            )

            p_from_conserved = function (x)
                ρ_x, ρe_x, ρq_x = x[1], x[2], x[3]
                q_x = ρq_x / ρ_x
                e_int_x = ρe_x / ρ_x - K - Φ
                sa = TD.saturation_adjustment(
                    thermo_params,
                    TD.ρe(),
                    ρ_x,
                    e_int_x,
                    q_x;
                    maxiter = 25,
                )
                return TD.air_pressure(
                    thermo_params,
                    sa.T,
                    ρ_x,
                    q_x,
                    sa.q_liq,
                    sa.q_ice,
                )
            end

            grad_ad = ForwardDiff.gradient(
                p_from_conserved,
                [ρ, ρe_tot, ρq_tot],
            )
            @test isapprox(closed.∂p_∂ρ_sat, grad_ad[1]; rtol)
            @test isapprox(closed.κ_sat, grad_ad[2]; rtol)
            @test isapprox(closed.∂p_∂ρq_tot_sat, grad_ad[3]; rtol)

            κ_helper = CA.saturated_kappa(
                thermo_params,
                T_eq,
                ρ,
                q_tot,
                one(FT),
            )
            @test isapprox(κ_helper, closed.κ_sat; rtol = eps(FT) * 10)
        end

        # Subsaturated cells with positive cloud fraction `f > 0`: evaluating
        # at `q_tot_cld = max(q_tot, q_sat) >= q_sat` keeps `q_c >= 0` in
        # `L_f q_c dλ/dT`, so `C_v,sat > cv_m > 0` across all temperatures.
        for (T_sub, ρ_sub) in (
            (FT(295), FT(1.1)),
            (FT(260), FT(0.75)),
            (FT(225), FT(0.42)),
        )
            q_sat = TD.q_vap_saturation(thermo_params, T_sub, ρ_sub)
            q_tot_sub = FT(0.5) * q_sat
            q_tot_cld = CA.in_cloud_q_tot(
                thermo_params,
                T_sub,
                ρ_sub,
                q_tot_sub,
            )
            @test q_tot_cld >= q_sat
            derivs = CA.saturated_pressure_derivatives(
                thermo_params,
                T_sub,
                ρ_sub,
                q_tot_cld,
            )
            cv_m = TD.cv_m(thermo_params, q_tot_sub, zero(FT), zero(FT))
            @test derivs.C_v_sat > cv_m > zero(FT)
            @test derivs.κ_sat > zero(FT)
        end
    end
end

@testset "Column Wfact! FD agreement, zero allocations, and 1M gating" begin
    # Helper to build a 0M column at arbitrary vertical resolution with a
    # warm-to-ice supersaturated profile and nonzero `uₕ` and `u₃`.
    function setup_saturated_0m_column(z_elem::Int)
        cfg = CA.AtmosConfig(
            Dict{String, Any}(
                "config" => "column",
                "FLOAT_TYPE" => "Float64",
                "dt" => "10secs",
                "z_max" => 9000.0,
                "z_elem" => z_elem,
                "z_stretch" => false,
                "microphysics_model" => "0M",
                "cloud_model" => "grid_scale",
                "sgs_distribution" => "mean",
                "vert_diff" => nothing,
                "surface_setup" => "DefaultExchangeCoefficients",
                "disable_surface_flux_tendency" => true,
                "rayleigh_sponge" => false,
                "output_default_diagnostics" => false,
            );
            job_id = "moist_jac_0m_col_$(z_elem)",
        )
        (; Y, p, simulation) = generate_test_simulation(cfg)
        FT = eltype(Y)
        thermo_params = CAP.thermodynamics_params(p.params)
        grav = FT(CAP.grav(p.params))
        z_f_field = Fields.coordinate_field(Y.f).z
        z_c = vec(parent(Fields.coordinate_field(Y.c).z))
        z_max = FT(9000)

        # Warm (295 K) near surface, mixed-phase (~260 K) mid-column, ice
        # (~225 K) aloft, all supersaturated (`q_tot = q_sat + 3e-4` so 0M's
        # 2-iteration `saturation_adjustment` converges cleanly).
        ρ_arr = parent(Y.c.ρ)
        ρq_arr = parent(Y.c.ρq_tot)
        ρe_arr = parent(Y.c.ρe_tot)
        uₕ_u1 = parent(Y.c.uₕ.components.data.:1)
        uₕ_u2 = parent(Y.c.uₕ.components.data.:2)
        u₃_arr = parent(Y.f.u₃.components.data.:1)

        for k in 1:z_elem
            zk = z_c[k]
            uₕ_u1[k, 1, 1, 1] = FT(7) + FT(4) * (zk / z_max)
            uₕ_u2[k, 1, 1, 1] = FT(-3) + FT(2) * (zk / z_max)
        end
        @. Y.f.u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(FT(1.5) * sin(FT(π) * z_f_field / z_max)),
        )
        u₃_arr[1, 1, 1, 1] = zero(FT)
        u₃_arr[end, 1, 1, 1] = zero(FT)

        # Set velocity-consistent kinetic energy first, then assign saturated
        # `(ρ, ρq_tot, ρe_tot)`.
        CA.set_implicit_precomputed_quantities!(Y, p, FT(0))
        K_vec = vec(parent(p.precomputed.ᶜK))
        Φ_vec = vec(parent(p.core.ᶜΦ))
        for k in 1:z_elem
            zk = z_c[k]
            Tk = FT(295) - FT(70) * (zk / z_max)
            ρk = FT(1.15) * exp(-zk / FT(8000))
            q_sat_k = TD.q_vap_saturation(thermo_params, Tk, ρk)
            qk = q_sat_k + FT(3e-4)
            e_int_k = TD.internal_energy_sat(thermo_params, Tk, ρk, qk)
            ρ_arr[k, 1, 1, 1] = ρk
            ρq_arr[k, 1, 1, 1] = ρk * qk
            ρe_arr[k, 1, 1, 1] = ρk * (e_int_k + K_vec[k] + Φ_vec[k])
        end

        t = simulation.integrator.t
        CA.set_precomputed_quantities!(Y, p, t)
        CA.set_implicit_precomputed_quantities!(Y, p, t)
        return (; Y, p, simulation, t)
    end

    # Compute centered finite-difference blocks of `R(Y) = dtγ Yₜ(Y) - Y`.
    function fd_column_blocks(Y, p, dtγ, t)
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

        col_names = (
            @name(c.ρ),
            @name(c.ρe_tot),
            @name(c.ρq_tot),
            u₁_name,
            u₂_name,
            u₃_name,
        )
        row_names = (
            @name(c.ρ),
            @name(c.ρe_tot),
            @name(c.ρq_tot),
            u₃_name,
        )
        fd_blocks = Dict{Tuple{Any, Any}, Matrix{Float64}}()

        for col_name in col_names
            col_vec = vec(parent(MatrixFields.get_field(Y, col_name)))
            work_vec = vec(parent(MatrixFields.get_field(Y_work, col_name)))
            n_col = length(col_vec)
            for rname in row_names
                n_row = length(vec(parent(MatrixFields.get_field(Y, rname))))
                fd_blocks[(rname, col_name)] = zeros(Float64, n_row, n_col)
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
                    @views fd_blocks[(rname, col_name)][:, j] .=
                        (rp .- rm) .* inv_2h
                end
            end
        end
        CA.set_implicit_precomputed_quantities!(Y, p, t)
        return fd_blocks
    end

    st24 = setup_saturated_0m_column(24)
    @test CA.use_moist_wave_jacobian(st24.p)
    T_col = vec(parent(st24.p.precomputed.ᶜT))
    @test maximum(T_col) > 285.0
    @test minimum(T_col) < 235.0
    @test any(T -> 245.0 < T < 270.0, T_col)

    jac24 = st24.simulation.integrator.cache.newtons_method_cache.j
    dtγ = Float64(st24.p.dtγ)
    allocs =
        _measure_update_jacobian_allocs(jac24, st24.Y, st24.p, dtγ, st24.t)
    @test allocs == 0

    manual24 = CA.first_column_block_arrays(
        jac24.alg,
        st24.Y,
        st24.p,
        dtγ,
        st24.t,
    )
    fd24 = fd_column_blocks(st24.Y, st24.p, dtγ, st24.t)

    # Interior face indices `2:z_elem` carry the PGF tendency (`u₃` at boundary
    # faces 1 and `z_elem + 1` is fixed at zero).
    int24 = 2:24
    @test norm(
        manual24[(@name(c.ρ), u₃_name)][:, int24] -
        fd24[(@name(c.ρ), u₃_name)][:, int24],
    ) / norm(fd24[(@name(c.ρ), u₃_name)][:, int24]) < 1e-9
    @test norm(
        manual24[(@name(c.ρq_tot), u₃_name)][:, int24] -
        fd24[(@name(c.ρq_tot), u₃_name)][:, int24],
    ) / norm(fd24[(@name(c.ρq_tot), u₃_name)][:, int24]) < 1e-9
    # `(c.ρe_tot, f.u₃)` freezes `ᶜp` and `ᶜK`, omitting `O(w² / c_s²) ~ 2e-5`.
    @test norm(
        manual24[(@name(c.ρe_tot), u₃_name)][:, int24] -
        fd24[(@name(c.ρe_tot), u₃_name)][:, int24],
    ) / norm(fd24[(@name(c.ρe_tot), u₃_name)][:, int24]) < 1e-4

    # On the stratified grid, discrete Exner `cp_d ᶠinterp(θ_v) ᶠgradᵥ(Π)` and
    # `-ᶠgradᵥ(p) / ᶠinterp(ρ)` agree to `O(Δz / H) ~ 1–2%` and converge as
    # `z_elem` doubles (a frozen Jacobian would err by ~35–55% here).
    st48 = setup_saturated_0m_column(48)
    jac48 = st48.simulation.integrator.cache.newtons_method_cache.j
    manual48 = CA.first_column_block_arrays(
        jac48.alg,
        st48.Y,
        st48.p,
        Float64(st48.p.dtγ),
        st48.t,
    )
    fd48 = fd_column_blocks(st48.Y, st48.p, Float64(st48.p.dtγ), st48.t)
    int48 = 2:48

    for col_name in (
        @name(c.ρ),
        @name(c.ρe_tot),
        @name(c.ρq_tot),
        u₁_name,
        u₂_name,
    )
        err24 =
            norm(
                manual24[(u₃_name, col_name)][int24, :] -
                fd24[(u₃_name, col_name)][int24, :],
            ) / norm(fd24[(u₃_name, col_name)][int24, :])
        err48 =
            norm(
                manual48[(u₃_name, col_name)][int48, :] -
                fd48[(u₃_name, col_name)][int48, :],
            ) / norm(fd48[(u₃_name, col_name)][int48, :])
        @test err24 < 2.5e-2
        @test err48 < 0.65 * err24
    end

    # Check the kinetic-energy contribution `(f.u₃, f.u₃) + I` directly on
    # interior faces (since `-I` otherwise dominates the norm of `W_{u₃, u₃}`).
    du3_m24 = manual24[(u₃_name, u₃_name)][int24, int24] + I
    du3_fd24 = fd24[(u₃_name, u₃_name)][int24, int24] + I
    du3_m48 = manual48[(u₃_name, u₃_name)][int48, int48] + I
    du3_fd48 = fd48[(u₃_name, u₃_name)][int48, int48] + I
    err_u3_24 = norm(du3_m24 - du3_fd24) / norm(du3_fd24)
    err_u3_48 = norm(du3_m48 - du3_fd48) / norm(du3_fd48)
    @test err_u3_24 < 2.5e-2
    @test err_u3_48 < 0.65 * err_u3_24

    # Verify full frozen-condensate loading weights (`b_lcl ≈ b_rai`, `b_icl ≈ b_sno`),
    # zero allocations, and `use_moist_wave_jacobian == false` on 1M.
    let
        cfg_1m = CA.AtmosConfig(
            Dict{String, Any}(
                "config" => "column",
                "FLOAT_TYPE" => "Float64",
                "dt" => "10secs",
                "z_max" => 6000.0,
                "z_elem" => 10,
                "z_stretch" => false,
                "microphysics_model" => "1M",
                "vert_diff" => nothing,
                "surface_setup" => "DefaultExchangeCoefficients",
                "disable_surface_flux_tendency" => true,
                "output_default_diagnostics" => false,
            );
            job_id = "moist_jac_1m",
        )
        nt_1m = generate_test_simulation(cfg_1m)
        Y1, p1, sim1 = nt_1m.Y, nt_1m.p, nt_1m.simulation
        @test !CA.use_moist_wave_jacobian(p1)

        @. Y1.c.ρq_tot = Y1.c.ρ * 0.012
        @. Y1.c.ρq_lcl = Y1.c.ρ * 0.001
        @. Y1.c.ρq_rai = Y1.c.ρ * 0.001
        @. Y1.c.ρq_icl = Y1.c.ρ * 0.0005
        @. Y1.c.ρq_sno = Y1.c.ρ * 0.0005
        CA.set_precomputed_quantities!(Y1, p1, 0.0)
        parent(p1.precomputed.ᶜcloud_fraction) .= 0.4

        jac1 = sim1.integrator.cache.newtons_method_cache.j
        @test _measure_update_jacobian_allocs(
            jac1,
            Y1,
            p1,
            Float64(p1.dtγ),
            0.0,
        ) == 0

        blocks1 = CA.first_column_block_arrays(
            jac1.alg,
            Y1,
            p1,
            Float64(p1.dtγ),
            0.0,
        )
        b_lcl = blocks1[(u₃_name, @name(c.ρq_lcl))][2:10, :]
        b_rai = blocks1[(u₃_name, @name(c.ρq_rai))][2:10, :]
        b_icl = blocks1[(u₃_name, @name(c.ρq_icl))][2:10, :]
        b_sno = blocks1[(u₃_name, @name(c.ρq_sno))][2:10, :]
        @test isapprox(b_lcl, b_rai; rtol = 1e-12)
        @test isapprox(b_icl, b_sno; rtol = 1e-12)
    end
end

@testset "Single-iteration W-method amplification for r in [0.80, 0.95] (ARS343 & ARS222)" begin
    # 2x2 wave mode `Y = [ζ, w]` over one Lie-split step: column adjustment leaves
    # `q_c^n` in equilibrium with `ζ^n` at step entry (`p' ~ -r ω_f^2 ζ^n`); during
    # the ARK stages condensate is frozen (`p' - p'^n ~ -ω_f^2 (ζ - ζ^n)`), while
    # the moist Jacobian linearizes with `-r ω_f^2`, and post-step adjustment restores
    # equilibrium at `ζ^{n+1}` at step end.
    function w_method_step_matrix(alg_name, z::Float64, r::Float64)
        tab = CTS.IMEXTableau(alg_name)
        A = tab.a_imp
        b = tab.b_imp
        s = length(b)
        G = zeros(Float64, 2, 2)
        T_imp = [zeros(Float64, 2) for _ in 1:s]
        for col in 1:2
            y0 = col == 1 ? [1.0, 0.0] : [0.0, 1.0]
            ζ0 = y0[1]
            for i in 1:s
                temp = copy(y0)
                for j in 1:(i - 1)
                    aij = A[i, j]
                    !iszero(aij) && (@. temp += aij * T_imp[j])
                end
                γ = A[i, i]
                res = [z * temp[2], -r * z * ζ0 - z * (temp[1] - ζ0)]
                if iszero(γ)
                    T_imp[i] .= res
                else
                    W = [1.0 (-γ * z); (γ * r * z) 1.0]
                    T_imp[i] .= W \ res
                end
            end
            yn1 = copy(y0)
            for i in 1:s
                bi = b[i]
                !iszero(bi) && (@. yn1 += bi * T_imp[i])
            end
            G[:, col] .= yn1
        end
        return G
    end

    spectral_radius(M) = maximum(abs, eigvals(M))

    z_grid = [10.0^e for e in range(-2.0, 4.0; length = 241)]
    for r in (0.80, 0.85, 0.90, 0.95), alg in (CTS.ARS343(), CTS.ARS222())
        radii = [spectral_radius(w_method_step_matrix(alg, z, r)) for z in z_grid]
        @test all(<(1.0), radii)
        if r == 0.85
            ρ_inf = spectral_radius(w_method_step_matrix(alg, 1e6, r))
            @test isapprox(ρ_inf, 0.43; atol = 0.01)
        end
    end
end

@testset "NonEquilibriumMicrophysics1M cloudy column SSP333 stability" begin
    mktempdir() do tmpdir
        FT = Float64
        ctx = ClimaComms.SingletonCommsContext()
        grid =
            CA.ColumnGrid(FT; context = ctx, z_elem = 15, z_max = 6000.0, z_stretch = false)
        params = CA.ClimaAtmosParameters(
            FT;
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
        )
        thermo_params = CAP.thermodynamics_params(params)
        model = CA.AtmosModel(
            grid;
            params,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            disable_surface_flux_tendency = true,
        )
        ode_config = CTS.IMEXAlgorithm(
            CTS.SSP333(),
            CTS.NewtonsMethod(; max_iters = 1),
            CTS.Unconstrained(),
        )
        sim = CA.AtmosSimulation(
            model;
            dt = FT(20),
            t_end = FT(60),
            ode_config,
            output_dir = joinpath(tmpdir, "ssp333_1m"),
            default_callbacks = false,
        )
        Y, p = sim.integrator.u, sim.integrator.p
        ᶜz = Fields.coordinate_field(Y.c).z
        ᶜT = @. FT(288) - FT(6.5e-3) * ᶜz
        ᶜq_lcl = @. FT(6e-4) * exp(-((ᶜz - FT(1800)) / FT(600))^2)
        ᶜq_rai = @. FT(8e-4) * exp(-(ᶜz / FT(1500))^2)
        ᶜq_v = @. TD.q_vap_saturation(thermo_params, ᶜT, Y.c.ρ)
        @. Y.c.ρq_lcl = Y.c.ρ * ᶜq_lcl
        @. Y.c.ρq_rai = Y.c.ρ * ᶜq_rai
        @. Y.c.ρq_tot = Y.c.ρ * (ᶜq_v + ᶜq_lcl + ᶜq_rai)
        @. Y.c.ρe_tot =
            Y.c.ρ * (
                TD.internal_energy(
                    thermo_params,
                    ᶜT,
                    Y.c.ρq_tot / Y.c.ρ,
                    ᶜq_lcl + ᶜq_rai,
                    FT(0),
                ) +
                p.core.ᶜΦ
            )
        CA.set_precomputed_quantities!(Y, p, FT(0))
        @test !CA.use_moist_wave_jacobian(p)
        CA.solve_atmos!(sim)
        Y_end = sim.integrator.u
        @test all(isfinite, Array(parent(Y_end.c.ρ)))
        @test all(isfinite, Array(parent(Y_end.c.ρe_tot)))
        @test all(>=(0.0), Array(parent(Y_end.c.ρq_tot)))
        @test all(>=(0.0), Array(parent(Y_end.c.ρq_lcl)))
        @test all(>=(0.0), Array(parent(Y_end.c.ρq_rai)))
    end
end

@testset "2D (k_h > 0) dry vs saturated 0M gravity-wave eigenvalues" begin
    L_x = 10_000.0
    z_max = 1_000.0
    z_elem = 16
    k_h = 2π / L_x
    m_v = π / z_max

    # Build a resting, horizontally uniform, isothermal (`T_0 = 290 K`)
    # hydrostatic 2D `(x, z)` slice (`DryModel` or saturated `0M` with uniform
    # `q_tot > q_sat`) and assemble the linearized wave-core operator on the
    # `(cos(k_h x), sin(k_h x))` Fourier subspace from `ManualSparseJacobian`
    # (`Wfact!`) for the vertical terms plus `horizontal_dynamics_tendency!` and
    # `horizontal_tracer_advection_tendency!` (with DSS) for the horizontal
    # terms.
    function build_2d_wave_operator(mp_model::String; use_frozen_jac::Bool = false)
        dict = Dict{String, Any}(
            "config" => "plane",
            "FLOAT_TYPE" => "Float64",
            "x_max" => L_x,
            "x_elem" => 6,
            "z_max" => z_max,
            "z_elem" => z_elem,
            "z_stretch" => false,
            "dt" => "1secs",
            "microphysics_model" => mp_model,
            "vert_diff" => nothing,
            "hyperdiff" => nothing,
            "surface_setup" => "DefaultExchangeCoefficients",
            "disable_surface_flux_tendency" => true,
            "rayleigh_sponge" => false,
            "viscous_sponge" => false,
            "output_default_diagnostics" => false,
        )
        if mp_model == "0M"
            dict["cloud_model"] = "grid_scale"
            dict["sgs_distribution"] = "mean"
        end
        cfg = CA.AtmosConfig(dict; job_id = "gw_2d_$(mp_model)_$(use_frozen_jac)")
        (; Y, p, simulation) = generate_test_simulation(cfg)
        thermo_params = CAP.thermodynamics_params(p.params)
        grav = Float64(CAP.grav(p.params))
        R_d = Float64(CAP.R_d(p.params))

        T0 = 290.0
        q0 = mp_model == "0M" ? 0.014 : 0.0
        ρ_sfc = 1.20

        # Exact hydrostatic density profile at isothermal `T0` and uniform `q0`:
        # Dry: `p = ρ R_d T0` => `H_ρ = R_d T0 / g`.
        # Saturated 0M: `p = (1 - q0) ρ R_d T0 + p_v,sat(T0)` =>
        # `dp/dz = (1 - q0) R_d T0 dρ/dz = -ρ g` => `H_ρ = (1 - q0) R_d T0 / g`.
        H_ρ = (1.0 - q0) * R_d * T0 / grav

        z_c = Fields.coordinate_field(Y.c).z
        @. Y.c.uₕ = Geometry.Covariant12Vector(0.0, 0.0)
        @. Y.f.u₃ = Geometry.Covariant3Vector(0.0)
        @. Y.c.ρ = ρ_sfc * exp(-z_c / H_ρ)
        if mp_model == "0M"
            @. Y.c.ρq_tot = Y.c.ρ * q0
            @. Y.c.ρe_tot =
                Y.c.ρ * (
                    TD.internal_energy_sat(thermo_params, T0, Y.c.ρ, q0) +
                    p.core.ᶜΦ
                )
        else
            @. Y.c.ρe_tot =
                Y.c.ρ * (TD.internal_energy_dry(thermo_params, T0) + p.core.ᶜΦ)
        end

        t = simulation.integrator.t
        CA.set_precomputed_quantities!(Y, p, t)
        if mp_model == "0M"
            q_sat_col =
                TD.q_vap_saturation.(
                    Ref(thermo_params),
                    T0,
                    vec(reshape(parent(Y.c.ρ), z_elem, :)[:, 1]),
                )
            @test all(q0 .> q_sat_col)
        end
        CA.set_implicit_precomputed_quantities!(Y, p, t)

        # Theoretical `c_s^2(z)` and `N^2(z) = g (1 / H_ρ - g / c_s^2)` at
        # mid-height.
        ρ_mid = ρ_sfc * exp(-0.5 * z_max / H_ρ)
        Φ_mid = grav * 0.5 * z_max
        c_s2_theory = if mp_model == "0M" && !use_frozen_jac
            d_mid = CA.saturated_pressure_derivatives(
                thermo_params,
                T0,
                ρ_mid,
                q0,
                0.0,
                Φ_mid,
            )
            e_mid = TD.internal_energy_sat(thermo_params, T0, ρ_mid, q0) + Φ_mid
            q_liq_m, q_ice_m =
                TD.condensate_partition(thermo_params, T0, ρ_mid, q0)
            p_mid = TD.air_pressure(
                thermo_params,
                T0,
                ρ_mid,
                q0,
                q_liq_m,
                q_ice_m,
            )
            d_mid.∂p_∂ρ_sat +
            d_mid.κ_sat * (e_mid + p_mid / ρ_mid) +
            q0 * d_mid.∂p_∂ρq_tot_sat
        else
            Float64(CAP.cp_d(p.params) / CAP.cv_d(p.params)) * R_d * T0
        end
        N2_theory = grav * (1.0 / H_ρ - grav / c_s2_theory)
        B_disp = c_s2_theory * (k_h^2 + m_v^2 + 1.0 / (4.0 * H_ρ^2))
        ω_gw_theory =
            sqrt(0.5 * (B_disp - sqrt(B_disp^2 - 4.0 * c_s2_theory * N2_theory * k_h^2)))

        # Vertical wave operator `(W + I) / dtγ` from `ManualSparseJacobian`.
        dtγ = Float64(p.dtγ)
        jac = simulation.integrator.cache.newtons_method_cache.j
        if use_frozen_jac
            column_Y = CA.first_column_view(Y)
            column_p = CA.first_column_view(p)
            column_cache =
                CA.jacobian_cache(jac.alg, column_Y, p.atmos; verbose = false)
            CA.update_jacobian!(jac.alg, column_cache, column_Y, column_p, dtγ, t)
            CA._update_u₃_advection_jacobian!(
                column_cache.matrix,
                column_Y,
                column_p,
                dtγ,
                column_cache.derivative_flags.topography_flag,
                CA.IgnoreDerivative(),
            )
            CA.zero_velocity_jacobian!(column_cache.matrix, column_Y, column_p, t)
            vblocks = Dict()
            scalar_names = CA.scalar_field_names(Y)
            for k in Iterators.product(scalar_names, scalar_names)
                k in keys(column_cache.matrix) || continue
                bv = Base.materialize(column_cache.matrix[k])
                vblocks[k] =
                    bv isa Fields.Field ? MatrixFields.column_field2array(bv) : bv
            end
        else
            vblocks = CA.first_column_block_arrays(jac.alg, Y, p, dtγ, t)
        end

        has_q = mp_model == "0M"
        scalar_syms =
            has_q ? (@name(c.ρ), @name(c.ρe_tot), @name(c.ρq_tot)) :
            (@name(c.ρ), @name(c.ρe_tot))
        n_s = length(scalar_syms)
        n_w = z_elem - 1 # interior `u₃` faces `2:z_elem`
        # State ordering on Fourier mode `k_h`:
        # `[ρ(1:nz) cos, ρe(1:nz) cos, (ρq cos), u₁(1:nz) sin, u₃(2:nz) cos]`.
        n_tot = n_s * z_elem + z_elem + n_w
        L_vert = zeros(Float64, n_tot, n_tot)
        L_horz = zeros(Float64, n_tot, n_tot)

        idx_s(si) = ((si - 1) * z_elem + 1):(si * z_elem)
        idx_u1 = (n_s * z_elem + 1):((n_s + 1) * z_elem)
        idx_w = ((n_s + 1) * z_elem + 1):n_tot
        int_w = 2:z_elem

        for (si, sname) in enumerate(scalar_syms)
            L_vert[idx_s(si), idx_w] .=
                vblocks[(sname, u₃_name)][:, int_w] ./ dtγ
            L_vert[idx_w, idx_s(si)] .=
                vblocks[(u₃_name, sname)][int_w, :] ./ dtγ
        end

        # Horizontal explicit wave operator on `(cos(k_h x), sin(k_h x))` via
        # centered finite differences of `horizontal_dynamics_tendency!`,
        # `horizontal_tracer_advection_tendency!`, and DSS. On the uniform
        # periodic mesh, `uₕ` component 1 (`u₁_name`) and `u₃` (`u₃_name`) are
        # constant multiples of physical `(u, w)`, so `L_vert + L_horz` is
        # similarity-conjugate to the physical-velocity wave operator.
        x_c = Fields.coordinate_field(Y.c).x
        cos_f = Base.materialize(@. cos(k_h * x_c))
        sin_f = Base.materialize(@. sin(k_h * x_c))
        cos_1d = vec(reshape(parent(cos_f), z_elem, :)[1, :])
        sin_1d = vec(reshape(parent(sin_f), z_elem, :)[1, :])
        wj_1d =
            vec(reshape(parent(Fields.local_geometry_field(Y.c).WJ), z_elem, :)[1, :])
        w_cos = wj_1d .* cos_1d
        w_sin = wj_1d .* sin_1d
        norm_cos = dot(w_cos, cos_1d)
        norm_sin = dot(w_sin, sin_1d)

        Y_pert = copy(Y)
        Yₜ_p = similar(Y)
        Yₜ_m = similar(Y)

        eval_horz! = function (Yₜ_out, Y_in)
            CA.set_precomputed_quantities!(Y_in, p, t)
            fill!(Yₜ_out, 0.0)
            CA.horizontal_tracer_advection_tendency!(Yₜ_out, Y_in, p, t)
            CA.horizontal_dynamics_tendency!(Yₜ_out, Y_in, p, t)
            Spaces.weighted_dss!(Yₜ_out.c)
            return nothing
        end

        project_mode! = function (col_out, Yₜ_p, Yₜ_m, inv_2h)
            for (si, sname) in enumerate(scalar_syms)
                arr_p = reshape(parent(MatrixFields.get_field(Yₜ_p, sname)), z_elem, :)
                arr_m = reshape(parent(MatrixFields.get_field(Yₜ_m, sname)), z_elem, :)
                col_out[idx_s(si)] .=
                    ((arr_p .- arr_m) * w_cos) .* (inv_2h / norm_cos)
            end
            arr_up = reshape(parent(MatrixFields.get_field(Yₜ_p, u₁_name)), z_elem, :)
            arr_um = reshape(parent(MatrixFields.get_field(Yₜ_m, u₁_name)), z_elem, :)
            col_out[idx_u1] .=
                ((arr_up .- arr_um) * w_sin) .* (inv_2h / norm_sin)
            return nothing
        end

        # Perturb scalar modes (`cos(k_h x)`) and `u₁_name` (`sin(k_h x)`).
        for (si, sname) in enumerate(scalar_syms)
            f_base = MatrixFields.get_field(Y, sname)
            f_pert = MatrixFields.get_field(Y_pert, sname)
            for k in 1:z_elem
                scale_k = abs(reshape(parent(f_base), z_elem, :)[k, 1])
                h = 1e-6 * max(scale_k, 1.0)
                @. f_pert = f_base
                @views reshape(parent(f_pert), z_elem, :)[k, :] .+= h .* cos_1d
                eval_horz!(Yₜ_p, Y_pert)
                @. f_pert = f_base
                @views reshape(parent(f_pert), z_elem, :)[k, :] .-= h .* cos_1d
                eval_horz!(Yₜ_m, Y_pert)
                @. f_pert = f_base
                col_idx = idx_s(si)[k]
                @views project_mode!(L_horz[:, col_idx], Yₜ_p, Yₜ_m, 1.0 / (2.0 * h))
            end
        end

        u1_base = MatrixFields.get_field(Y, u₁_name)
        u1_pert = MatrixFields.get_field(Y_pert, u₁_name)
        for k in 1:z_elem
            h = 1e-3
            @. Y_pert.c.uₕ = Y.c.uₕ
            @views reshape(parent(u1_pert), z_elem, :)[k, :] .+= h .* sin_1d
            eval_horz!(Yₜ_p, Y_pert)
            @. Y_pert.c.uₕ = Y.c.uₕ
            @views reshape(parent(u1_pert), z_elem, :)[k, :] .-= h .* sin_1d
            eval_horz!(Yₜ_m, Y_pert)
            @. Y_pert.c.uₕ = Y.c.uₕ
            col_idx = idx_u1[k]
            @views project_mode!(L_horz[:, col_idx], Yₜ_p, Yₜ_m, 1.0 / (2.0 * h))
        end

        L_2d = L_vert .+ L_horz
        return (;
            L_2d,
            L_vert,
            N_theory = sqrt(N2_theory),
            c_s_theory = sqrt(c_s2_theory),
            ω_gw_theory,
            idx_w,
        )
    end

    # Extract the `m_v = π / z_max` fundamental gravity-wave frequency by
    # selecting the non-acoustic imaginary eigenmode with largest projection on
    # `sin(π z_f / z_max)` in `u₃`.
    function fundamental_gw_frequency(op)
        F = eigen(op.L_2d)
        z_f_int = [(k / z_elem) * z_max for k in 1:(z_elem - 1)]
        sin_prof = sin.(m_v .* z_f_int)
        normalize!(sin_prof)

        best_proj = -1.0
        best_ω = 0.0
        for idx in eachindex(F.values)
            λ = F.values[idx]
            ω = imag(λ)
            # Filter positive-frequency gravity-wave band (`0 < ω < 0.05 s^-1`,
            # well below acoustic modes `ω_ac >= π c_s / z_max ~ 1.0 s^-1`).
            if ω > 1e-4 && ω < 0.1 && abs(real(λ)) < 1e-6
                w_vec = F.vectors[op.idx_w, idx]
                nrm = norm(w_vec)
                nrm > 0 || continue
                proj = abs(dot(sin_prof, w_vec)) / nrm
                if proj > best_proj
                    best_proj = proj
                    best_ω = ω
                end
            end
        end
        return (; ω = best_ω, proj = best_proj)
    end

    dry_op = build_2d_wave_operator("dry")
    moist_op = build_2d_wave_operator("0M")
    frozen_0m_op = build_2d_wave_operator("0M"; use_frozen_jac = true)

    @test moist_op.N_theory < dry_op.N_theory
    @test moist_op.c_s_theory < dry_op.c_s_theory

    # At `k_h = 0` (`L_vert` alone), gravity-wave frequencies vanish identically
    # and only vertical acoustic modes (`ω >= m_v c_s ~ 1 s^-1`) are nonzero.
    for op in (dry_op, moist_op)
        ωs_1d = abs.(imag.(eigvals(op.L_vert)))
        @test all(ω -> ω < 1e-10 || ω > 0.5, ωs_1d)
    end

    gw_dry = fundamental_gw_frequency(dry_op)
    gw_moist = fundamental_gw_frequency(moist_op)
    gw_frozen = fundamental_gw_frequency(frozen_0m_op)

    @test gw_dry.proj > 0.98
    @test gw_moist.proj > 0.98
    @test isapprox(gw_dry.ω, dry_op.ω_gw_theory; rtol = 2e-2)
    @test isapprox(
        gw_moist.ω / gw_dry.ω,
        moist_op.ω_gw_theory / dry_op.ω_gw_theory;
        rtol = 3e-2,
    )
    @test gw_moist.ω < gw_dry.ω
    # Pairing frozen `κ_m` in `Wfact!` (`L_vert`) with saturated `0M` pressure
    # in `L_horz` mismatches vertical and horizontal compressibility, distorting
    # the gravity-wave frequency by >10× the moist-Jacobian WKB discrepancy.
    @test abs(gw_frozen.ω - moist_op.ω_gw_theory) >
          10 * abs(gw_moist.ω - moist_op.ω_gw_theory)
end
