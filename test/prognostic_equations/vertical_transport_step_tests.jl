#=
julia +1.13 --project=.buildkite
using Revise; include("test/prognostic_equations/vertical_transport_step_tests.jl")
=#
using Test
using Dates
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: DataLayouts, Domains, Fields, Geometry, Meshes, Operators, Spaces
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name
import ClimaTimeSteppers as CTS
const TD = CA.TD

@noinline _alloc_vt(Y, p, Δt, cache) =
    @allocated CA.step_vertical_scalar_transport!(Y, p, Δt, cache)

@noinline _alloc_record_imp(Y, p) =
    @allocated CA.record_implicit_stage_entry_energy!(Y, p)

@noinline _alloc_record_exp_imp(Yₜ, Y, p) =
    @allocated CA.maybe_record_explicit_imp_stage_energy!(Yₜ, Y, p)

@noinline _alloc_post_step_cb(integrator) =
    @allocated CA.post_step_callback!(integrator)

function _stretched_z_mesh(::Type{FT}, z_elem::Int, z_max::Real) where {FT}
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(z_max));
        boundary_names = (:bottom, :top),
    )
    return Meshes.IntervalMesh(
        domain,
        Meshes.ExponentialStretching(FT(z_max) / FT(2));
        nelems = z_elem,
    )
end

function _host_arr(f)
    f_contig = similar(f, eltype(f))
    f_contig .= f
    return Array(parent(f_contig))
end

function _set_col_scalar!(ᶜdest, vals::AbstractVector)
    f_contig = similar(ᶜdest, eltype(ᶜdest))
    copyto!(parent(f_contig), reshape(eltype(ᶜdest).(vals), size(parent(f_contig))))
    ᶜdest .= f_contig
    return ᶜdest
end

function _set_col_w!(ᶠu₃_dest, w_vals::AbstractVector)
    FT = Spaces.undertype(axes(ᶠu₃_dest))
    ᶠw = Fields.Field(FT, axes(ᶠu₃_dest))
    copyto!(parent(ᶠw), reshape(FT.(w_vals), size(parent(ᶠw))))
    @. ᶠu₃_dest = Geometry.Covariant3Vector(Geometry.WVector(ᶠw))
    return ᶠu₃_dest
end

function _imex_alg(alg_name)
    nm = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    return CTS.IMEXAlgorithm(alg_name, nm, CTS.Unconstrained())
end

function _imex_ssp_alg(alg_name)
    nm = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    return CTS.IMEXAlgorithm(alg_name, nm, CTS.Unconstrained())
end

function _make_ssprk_integrator(
    model,
    alg_name,
    dt::FT,
    t_end::FT;
    post_step_hook! = CA.step_vertical_scalar_transport!,
) where {FT}
    alg = CTS.IMEXAlgorithm(
        alg_name,
        CTS.NewtonsMethod(;
            max_iters = 1,
            update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
        ),
    )
    Y = CA.initial_state(model)
    p = CA.build_cache(
        Y,
        model,
        model.params,
        dt,
        DateTime(2010, 1, 1),
        nothing;
        ode_algo = alg,
        update_cache_every = "stage",
        post_step_hook!,
    )
    cb = CTS.CallbackSet(CA.post_step_discrete_callback())
    int_args, int_kwargs = CA.args_integrator(
        Y,
        p,
        (zero(FT), t_end),
        alg,
        cb,
        CA.ManualSparseJacobian(),
        false,
        model.prescribed_flow,
        dt,
        "stage",
        "stage",
    )
    return CTS.init(int_args...; int_kwargs...)
end

function _column_integral(ᶜf)
    return sum(ᶜf)
end

# Discrete Exner-hydrostatic pressure profile where `add_implicit_u₃_tendency!`
# vanishes to machine precision at `t = 0` for any `(T, q_tot, q_liq, q_ice)`.
function _solve_discrete_exner_hydrostatic_p!(
    p_vec::Vector{FT},
    T_vec::Vector{FT},
    qt_vec::Vector{FT},
    ql_vec::Vector{FT},
    qi_vec::Vector{FT},
    Φ_vec::Vector{FT},
    thp,
    cp_d::FT,
    p_bot::FT,
) where {FT}
    nz = length(p_vec)
    p_vec[1] = p_bot
    for k in 1:(nz - 1)
        pk = p_vec[k]
        Φk = Φ_vec[k]
        Φkp = Φ_vec[k + 1]
        dΦ = Φkp - Φk
        θvk = CA.theta_v(thp, T_vec[k], pk, qt_vec[k], ql_vec[k], qi_vec[k])
        θvrk = CA.theta_vr(thp, pk)
        Φrk = CA.phi_r(thp, pk)
        Πk = TD.exner_given_pressure(thp, pk)

        R_mk = TD.gas_constant_air(thp, qt_vec[k], ql_vec[k], qi_vec[k])
        p_guess = pk * exp(-dΦ / (R_mk * T_vec[k]))
        for _ in 1:6
            f_val = begin
                θvkp = CA.theta_v(
                    thp,
                    T_vec[k + 1],
                    p_guess,
                    qt_vec[k + 1],
                    ql_vec[k + 1],
                    qi_vec[k + 1],
                )
                θvrkp = CA.theta_vr(thp, p_guess)
                Φrkp = CA.phi_r(thp, p_guess)
                Πkp = TD.exner_given_pressure(thp, p_guess)
                (dΦ - (Φrkp - Φrk)) +
                cp_d * FT(0.5) * ((θvk - θvrk) + (θvkp - θvrkp)) * (Πkp - Πk)
            end
            dp = p_guess * FT(1e-6)
            p_hi = p_guess + dp
            θvkp_h = CA.theta_v(
                thp,
                T_vec[k + 1],
                p_hi,
                qt_vec[k + 1],
                ql_vec[k + 1],
                qi_vec[k + 1],
            )
            f_hi =
                (dΦ - (CA.phi_r(thp, p_hi) - Φrk)) +
                cp_d *
                FT(0.5) *
                ((θvk - θvrk) + (θvkp_h - CA.theta_vr(thp, p_hi))) *
                (TD.exner_given_pressure(thp, p_hi) - Πk)
            df_dp = (f_hi - f_val) / dp
            p_guess -= f_val / df_dp
        end
        p_vec[k + 1] = p_guess
    end
    return p_vec
end

# Exact 1-step linear response of per-stage backward-Euler upwind applied inside
# an S-stage ARK tableau `b_imp` (with diagonal `γ = a_imp[i, i]`):
# on `ARS343` (`b_3 < 0`), a unit spike goes `-12.8%` negative for `C >= 5`.
function _per_stage_ark_upwind_spike(tab, nz::Int, spike_k::Int, C::Float64)
    a = Matrix{Float64}(tab.a_imp.coeffs)
    b = Vector{Float64}(tab.b_imp.coeffs)
    s = length(b)
    B = zeros(Float64, nz, nz)
    for k in 1:nz
        if k < nz
            B[k, k] += 1.0
        end
        if k > 1
            B[k, k - 1] -= 1.0
        end
    end
    u0 = zeros(Float64, nz)
    u0[spike_k] = 1.0
    T_stages = [zeros(Float64, nz) for _ in 1:s]
    for i in 1:s
        rhs = copy(u0)
        for j in 1:(i - 1)
            rhs .+= a[i, j] .* T_stages[j]
        end
        γ_i = a[i, i]
        U_i = iszero(γ_i) ? rhs : (I + (γ_i * C) .* B) \ rhs
        T_stages[i] .= iszero(γ_i) ? (-C .* (B * U_i)) : ((U_i .- rhs) ./ γ_i)
    end
    u_end = copy(u0)
    for i in 1:s
        u_end .+= b[i] .* T_stages[i]
    end
    return u_end
end

@testset "Duplicate-node steep C⁰ terrain F̃ closure and continuity" begin
    ctx = ClimaComms.SingletonCommsContext()

    for (grid_label, make_grid) in (
        (
            "Box3D steep CosineTopography + LinearWarp + stretched z",
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
        @testset "$grid_label ($FT)" for FT in (Float64, Float32)
            grid = make_grid(FT)
            params = CA.ClimaAtmosParameters(FT)
            model = CA.AtmosModel(
                grid;
                params,
                microphysics_model = CA.EquilibriumMicrophysics0M(),
                chemistry_model = CA.GasPhaseChem(),
                disable_surface_flux_tendency = true,
            )

            for ode_config in (_imex_alg(CTS.ARS343()), _imex_ssp_alg(CTS.SSP333()))
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
                        return nothing
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
end

@testset "χ ≡ 1 preservation including top cell on ARS343 and SSP333 (Float32 & Float64)" begin
    ctx = ClimaComms.SingletonCommsContext()

    for FT in (Float64, Float32)
        tol = FT === Float32 ? FT(1e-6) : FT(1e-14)
        grid = CA.ColumnGrid(
            FT;
            context = ctx,
            z_elem = 32,
            z_max = 35000.0,
            z_mesh = _stretched_z_mesh(FT, 32, 35000.0),
        )
        params = CA.ClimaAtmosParameters(FT)

        for tr_upw in (:vanleer_limiter, :third_order, :first_order, :none)
            numerics = CA.AtmosNumerics(;
                energy_humidity_upwinding = :vanleer_limiter,
                tracer_upwinding = tr_upw,
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

            for (alg_label, use_ssprk) in (
                ("ARS343", false),
                ("SSP333", true),
            )
                (tr_upw === :vanleer_limiter || (FT === Float64 && !use_ssprk)) ||
                    continue
                mktempdir() do tmpdir
                    recorded_masks = UInt16[]
                    hook! = function (Y, p, _t, Δt, integrator_cache)
                        push!(recorded_masks, p.scratch.stage_entry_χ_e_mask[])
                        CA.step_vertical_scalar_transport!(
                            Y,
                            p,
                            Δt,
                            integrator_cache,
                        )
                        return nothing
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
                            output_dir = joinpath(tmpdir, "chi1_$(alg_label)_$(tr_upw)"),
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
                    atol = FT(1e-14),
                )
                @test isapprox(
                    _column_integral(Y.c.ρq_lcl),
                    ql_init_int;
                    atol = FT(1e-14),
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

    # 2. Live compressible acoustics (`advection_test = false`) on a stratified
    # moist column over 3 steps (`ARS343` and `SSP333`, `Float64` and `Float32`):
    # stage-entry `ê_i` evolves per stage, conserving column energy to roundoff.
    for FT_e in (Float64, Float32)
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
        for ode_cfg in (_imex_alg(CTS.ARS343()), _imex_ssp_alg(CTS.SSP333()))
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
        edmf_none = CA.AtmosModel(
            edmf_grid;
            params,
            microphysics_model = CA.NonEquilibriumMicrophysics1M(),
            numerics = CA.AtmosNumerics(;
                edmfx_mse_humidity_upwinding = :none,
                edmfx_tracer_upwinding = :none,
            ),
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
        sim_none = CA.AtmosSimulation(
            edmf_none;
            dt = FT(10),
            t_end = FT(20),
            ode_config = _imex_ssp_alg(CTS.SSP333()),
            output_dir = joinpath(tmpdir, "edmf_none"),
            default_callbacks = false,
            post_step_hook! = CA.step_vertical_scalar_transport!,
        )
        Yn = sim_none.integrator.u
        pn = sim_none.integrator.p
        cn = sim_none.integrator.cache
        @. Yn.f.u₃ = Yed.f.u₃
        CTS.step!(sim_none.integrator)
        @. Yn.c.sgsʲs.:(1).q_tot = qt_ed
        qt_before_none = copy(_host_arr(Yn.c.sgsʲs.:(1).q_tot))
        CA.step_vertical_scalar_transport!(Yn, pn, FT(10), cn)
        @test _host_arr(Yn.c.sgsʲs.:(1).q_tot) == qt_before_none

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
