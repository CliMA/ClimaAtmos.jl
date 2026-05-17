# Water-filling GH quadrature diagnostic: one 3×3 figure in (T, qₜ).
#
# ## Use a persistent REPL (not `julia -e`)
# `julia -e '...'` starts a **new process** every time → full load + compile (see
# https://julialang.org/blog/2021/01/precompile_tutorial/). Keep one session open:
#
#   julia --project=/tmp/climaatmos_wf_quad_plot
#   include("scripts/experiments/plot_water_filling_quadrature.jl")
#   plot_water_filling_quadrature("/tmp/water_filling_quadrature.png")  # fast after first call
#
# Root `Project.toml` has no CairoMakie; this script uses a **fixed** env at `/tmp/climaatmos_wf_quad_plot`
# (not a new hashed folder per machine path).
#
# CLI cold start (slow once): julia scripts/experiments/plot_water_filling_quadrature.jl [out.png]

import Pkg

"""Fixed path — always the same env on this machine (avoids hash(~/repo) ≠ hash(/net/.../repo))."""
wf_quad_env_dir() = joinpath(tempdir(), "climaatmos_wf_quad_plot")

wf_quad_repo_dir() = realpath(normpath(joinpath(@__DIR__, "..", "..")))

"""`Pkg.activate` the plot env; first time only, `develop` ClimaAtmos + `add` CairoMakie."""
function wf_quad_activate_env!()
    env = wf_quad_env_dir()
    if !ispath(joinpath(env, "Project.toml"))
        mkpath(env)
        Pkg.activate(env)
        Pkg.develop(Pkg.PackageSpec(path = wf_quad_repo_dir()))
        Pkg.add("CairoMakie")
        println("WF quadrature plot: initialized ", env)
    else
        Pkg.activate(env)
    end
    return env
end

wf_quad_activate_env!()
using ClimaAtmos
using CairoMakie
using LinearAlgebra
using Printf

function wf_quad_thermo_params(FT = Float64)
    return ClimaAtmos.ClimaAtmosParameters(FT).thermodynamics_params
end

function wf_quad_targets(thp, ρ, q_tot_mean, T_mean, μ, q′q′, T′T′, corr; lf = 0.5)
    FT = typeof(T_mean)
    σ_S = ClimaAtmos.compute_sigma_S(
        thp, FT(ρ), FT(q_tot_mean), FT(T_mean), FT(q′q′), FT(T′T′), FT(corr); lf = FT(lf),
    )
    qcμ = ClimaAtmos.qc_from_lambda_1d(FT(μ), σ_S)
    return (; σ_S, qcμ)
end

"""Riemann reference on χ ∈ [-4,4]² (same transform / `qcond_hat` as production GH)."""
function wf_quad_dense_ref_bulk(
    thp, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr, μ, λ, α, lf, scale_pos;
    n::Int = 64,
)
    FT = eltype(T_mean)
    σ_q, σ_T, corr_c = ClimaAtmos.sgs_stddevs_and_correlation(q′q′, T′T′, corr)
    dist = ClimaAtmos.GaussianSGS()
    transform = ClimaAtmos.PhysicalPointTransform(
        dist, T_mean, q_tot_mean, σ_T, σ_q, corr_c, FT(150), FT(0.1),
    )
    χ = range(-FT(4), FT(4); length = n)
    dχ = χ[2] - χ[1]
    acc = zero(FT)
    invπ = one(FT) / FT(π)
    for χ1 in χ, χ2 in χ
        T_hat, q_hat = transform(χ1, χ2)
        qc = ClimaAtmos.qcond_hat_water_filling(
            thp, ρ, q_hat, T_hat, μ, λ, α; scale_pos = scale_pos,
        )
        acc += qc * exp(-χ1 * χ1 - χ2 * χ2) * dχ * dχ * invπ
    end
    return acc
end

const WF_QUAD_ALPHAS = (0.0, 0.5, 1.0)

"""
    wf_quad_default_cases(FT, thp; kwargs...)

Nine regimes covering **shift** (λ > μ), **sub-capacity** (`scale_pos < 1`, λ = μ), and
**μ < 0** trace / wide / anti-correlated states. Every panel has T′T′ > 0 so GH is 2D in (T, qₜ).
"""
function wf_quad_default_cases(
    FT,
    thp;
    ρ = 1.0,
    T_mean = 280.0,
    lf = 0.5,
    μ_pos = 3e-4,
    μ_neg = -3e-3,
    σ_q_narrow = 1e-5,
    σ_q_wide = 4e-5,
    T′T′_narrow = 0.25,
    T′T′_wide = 1.0,
    T′T′_mp = 0.5,
    corr_mp = 0.6,
)
    ρ, T_mean, lf = FT(ρ), FT(T_mean), FT(lf)
    μ_pos, μ_neg = FT(μ_pos), FT(μ_neg)
    σ_n, σ_w = FT(σ_q_narrow), FT(σ_q_wide)
    q′q′_n, q′q′_w = σ_n^2, σ_w^2
    Tt_n, Tt_w, Tt_mp = FT(T′T′_narrow), FT(T′T′_wide), FT(T′T′_mp)
    μ0 = zero(FT)

    base = (; ρ, T_mean, lf)
    q_tot_from_μ(μ) = ClimaAtmos.TD.q_vap_saturation(thp, T_mean, ρ) + μ
    t(μ, q′q′, T′T′, corr) = wf_quad_targets(thp, ρ, q_tot_from_μ(μ), T_mean, μ, q′q′, T′T′, corr; lf)

    t1 = t(μ_pos, q′q′_n, Tt_n, FT(0.5))
    t2 = t(μ_pos, q′q′_w, Tt_w, zero(FT))
    t3 = t(μ_pos, q′q′_n, Tt_n, zero(FT))
    t4 = t(μ0, q′q′_n, Tt_n, FT(0.5))
    t5 = t(μ0, q′q′_w, Tt_mp, FT(corr_mp))
    t7 = t(μ_pos, q′q′_w, Tt_w, FT(0.3))
    t8 = t(μ_neg, q′q′_w, Tt_w, zero(FT))
    t9 = t(μ_neg, q′q′_n, Tt_n, FT(-0.7))

    return [
        (; base..., title = "μ>0 shift +0.2σ (narrow)", μ = μ_pos, q′q′ = q′q′_n, T′T′ = Tt_n, corr = FT(0.5),
            q_cond = t1.qcμ + FT(0.2) * t1.σ_S),
        (; base..., title = "μ>0 shift +0.35σ (wide)", μ = μ_pos, q′q′ = q′q′_w, T′T′ = Tt_w, corr = zero(FT),
            q_cond = t2.qcμ + FT(0.35) * t2.σ_S),
        (; base..., title = "μ>0 sub-cap scale (40% qcμ)", μ = μ_pos, q′q′ = q′q′_n, T′T′ = Tt_n, corr = zero(FT),
            q_cond = FT(0.4) * t3.qcμ),
        (; base..., title = "μ>0 far shift +0.55σ", μ = μ_pos, q′q′ = q′q′_w, T′T′ = Tt_w, corr = FT(0.3),
            q_cond = t7.qcμ + FT(0.55) * t7.σ_S),
        (; base..., title = "μ≈0 shift +0.1σ", μ = μ0, q′q′ = q′q′_n, T′T′ = Tt_n, corr = FT(0.5),
            q_cond = t4.qcμ + FT(0.1) * t4.σ_S),
        (; base..., title = "μ≈0 wide ρ=0.6 shift", μ = μ0, q′q′ = q′q′_w, T′T′ = Tt_mp, corr = FT(corr_mp),
            q_cond = t5.qcμ + FT(0.18) * t5.σ_S),
        (; base..., title = "μ<0 trace 1e-6 (subsaturated mean)", μ = μ_neg, q′q′ = q′q′_n, T′T′ = Tt_n, corr = zero(FT),
            q_cond = FT(1e-6)),
        (; base..., title = "μ<0 wide shift", μ = μ_neg, q′q′ = q′q′_w, T′T′ = Tt_w, corr = zero(FT),
            q_cond = max(FT(8e-5), t8.qcμ + FT(0.2) * t8.σ_S)),
        (; base..., title = "μ<0 anti-corr shift", μ = μ_neg, q′q′ = q′q′_n, T′T′ = Tt_n, corr = FT(-0.7),
            q_cond = max(FT(5e-5), t9.qcμ + FT(0.15) * t9.σ_S)),
    ]
end

"""
Build [`WaterFillingSGSEvaluator`](@ref) with the same branch logic as
[`compute_sgs_condensate_water_filling`](@ref) (plot-only helper; not exported from ClimaAtmos).
"""
function wf_quad_water_filling_setup(
    thp, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr, q_liq, q_ice, α; iters = 1,
)
    FT = typeof(T_mean)
    q_cond_mean = q_liq + q_ice
    μ = q_tot_mean - ClimaAtmos.TD.q_vap_saturation(thp, T_mean, ρ)
    if !(q_cond_mean > FT(0))
        λ, lf, A, scale_pos = μ, zero(FT), zero(FT), one(FT)
    else
        lf = q_liq / max(q_cond_mean, FT(1e-30))
        σ_S = ClimaAtmos.compute_sigma_S(thp, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr; lf = lf)
        A = α * σ_S
        λ, scale_pos = ClimaAtmos.water_filling_λ_scale_pos(
            q_cond_mean, μ, A, α, thp, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr;
            iters, lf,
        )
    end
    eval = ClimaAtmos.WaterFillingSGSEvaluator(thp, ρ, μ, λ, α, lf, scale_pos)
    return (; eval, μ, λ, A, α, lf, scale_pos, q_cond_mean)
end

"""
Sample GH nodes and bulk condensate for each ``α ∈ {0, 0.5, 1}``.

Also integrates the same partition on a dense χ-grid ([`wf_quad_dense_ref_bulk`](@ref)) so recovery
can be judged independently of GH order.
"""
function wf_quad_sample_case(
    thp, case; quadrature_order = 5, iters = 1, dense_grid::Int = 64,
)
    FT = typeof(case.μ)
    N = quadrature_order
    ρ, T_mean, lf = case.ρ, case.T_mean, case.lf
    q′q′, T′T′, corr = case.q′q′, case.T′T′, case.corr
    TD = ClimaAtmos.TD
    q_sat_mean = TD.q_vap_saturation(thp, T_mean, ρ)
    q_tot_mean = q_sat_mean + case.μ
    q_liq = lf * case.q_cond
    q_ice = (one(lf) - lf) * case.q_cond
    q_cond_tgt = case.q_cond

    SG = ClimaAtmos.SGSQuadrature(FT; quadrature_order = N)
    by_α = Dict{FT, NamedTuple}()
    for α in WF_QUAD_ALPHAS
        setup = wf_quad_water_filling_setup(
            thp, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr, q_liq, q_ice, FT(α); iters,
        )
        bulk = ClimaAtmos.integrate_over_sgs(
            setup.eval, SG, q_tot_mean, T_mean, q′q′, T′T′, corr,
        )
        q_gh = bulk.q_liq + bulk.q_ice
        q_dense = wf_quad_dense_ref_bulk(
            thp, ρ, q_tot_mean, T_mean, q′q′, T′T′, corr,
            setup.μ, setup.λ, FT(α), setup.lf, setup.scale_pos; n = dense_grid,
        )
        by_α[FT(α)] = (;
            setup,
            eval = setup.eval,
            q_gh,
            q_dense,
            λ = setup.λ,
            A = setup.A,
            scale_pos = setup.scale_pos,
            branch = setup.scale_pos < one(FT) - FT(1e-6) ? :subcap : :shift,
        )
    end

    σ_q, σ_T, _ = ClimaAtmos.sgs_stddevs_and_correlation(q′q′, T′T′, corr)
    transform = ClimaAtmos.PhysicalPointTransform(
        SG.dist, T_mean, q_tot_mean, σ_T, σ_q, corr, SG.T_min, SG.q_max,
    )

    inv_sqrt_pi = one(FT) / sqrt(FT(π))
    χ = SG.a
    pts = []
    for i in 1:N, j in 1:N
        T_hat, q_hat = transform(χ[i], χ[j])
        w = SG.w[i] * SG.w[j] * inv_sqrt_pi^2
        qc = Dict{FT, FT}()
        for α in WF_QUAD_ALPHAS
            out = by_α[FT(α)].eval(T_hat, q_hat)
            qc[FT(α)] = out.q_liq + out.q_ice
        end
        push!(pts, (;
            T = T_hat,
            q = q_hat,
            w,
            qc_0 = qc[zero(FT)],
            qc_05 = qc[FT(0.5)],
            qc_1 = qc[one(FT)],
        ))
    end

    return (;
        pts,
        q_tot_mean,
        T_mean,
        q_sat_mean,
        by_α,
        res_on = by_α[one(FT)],
        res_off = by_α[zero(FT)],
        res_mid = by_α[FT(0.5)],
        q_cond_tgt,
        σ_q,
        σ_T,
        corr,
    )
end

function wf_quad_gaussian_ellipse_points(σ_q, σ_T, corr; levels = (1.0, 2.0), n = 100)
    FT = typeof(σ_q)
    ρc = clamp(corr, -one(FT), one(FT))
    cov = [σ_q^2 ρc * σ_q * σ_T; ρc * σ_q * σ_T σ_T^2]
    evals, evecs = eigen(Symmetric(cov))
    evals = max.(evals, zero(FT))
    s1, s2 = sqrt(evals[1]), sqrt(evals[2])
    θ = range(zero(FT), 2 * FT(π); length = n)
    curves = []
    for r in levels
        qp = similar(θ)
        Tp = similar(θ)
        for (k, t) in enumerate(θ)
            c, s = cos(t), sin(t)
            qp[k] = r * (s1 * c * evecs[1, 1] + s2 * s * evecs[1, 2])
            Tp[k] = r * (s1 * c * evecs[2, 1] + s2 * s * evecs[2, 2])
        end
        push!(curves, (qp, Tp))
    end
    return curves
end

function wf_quad_gaussian_ellipses_physical(T_mean, q_mean, σ_q, σ_T, corr; kwargs...)
    return [
        (q_mean .+ rq, T_mean .+ rT) for (rq, rT) in wf_quad_gaussian_ellipse_points(σ_q, σ_T, corr; kwargs...)
    ]
end

"""Axis limits: 3σ PDF around ``(T̄, q̄)`` (or GH node extent when ``σ_T ≈ 0``)."""
function wf_quad_axis_limits(T_mean, q_mean, σ_T, σ_q, T, q; nσ = 3)
    FT = typeof(T_mean)
    if σ_T > FT(1e-9)
        pad_T = nσ * σ_T
        pad_q = nσ * max(σ_q, FT(1e-10))
        return (T_mean - pad_T, T_mean + pad_T, q_mean - pad_q, q_mean + pad_q)
    end
    T_lo, T_hi = extrema(T)
    q_lo, q_hi = extrema(q)
    dT = max(T_hi - T_lo, FT(2e-4))
    dq = max(q_hi - q_lo, FT(5e-7))
    pad_T = FT(0.15) * dT + FT(1e-4)
    pad_q = FT(0.15) * dq
    return (T_lo - pad_T, T_hi + pad_T, q_lo - pad_q, q_hi + pad_q)
end

function wf_quad_weighted_qc(qc, w)
    return w .* qc
end

"""
    wf_quad_wf_boundary_q(thp, ρ, T, μ, λ, α)

Curve in ``(T, q_t)`` where ``\\hat{q}_c = 0`` before ``scale_{pos}`` (shift branch):
``q_t = q_{sat}(T) + \\mu - \\lambda/\\alpha``. For ``\\alpha = 0`` the partition does not
use ``\\Delta S'``, so return `nothing`.
"""
function wf_quad_wf_boundary_q(thp, ρ, T, μ, λ, α)
    α <= zero(α) && return nothing
    return [ClimaAtmos.TD.q_vap_saturation(thp, Ti, ρ) + μ - λ / α for Ti in T]
end

"""
    wf_quad_marker_sizes(wq; sz_lo, sz_hi, wq_floor_frac)

Makie/CairoMakie [`scatter`](https://docs.makie.org/stable/reference/plots/scatter.html) `markersize`
scales the marker's **linear** size (≈ diameter for `:circle`), **not** area — unlike Matplotlib
`scatter(..., s=...)` where `s` is area in points².

So for **area** ``\\propto w\\hat{q}_c`` (absolute contribution at the node), use
``\\texttt{markersize} \\propto \\sqrt{w\\hat{q}_c}``, normalized by ``\\max(w\\hat{q}_c)`` on
that α series. Returns **0** when ``w\\hat{q}_c`` is below `wq_floor_frac` × max (no dry ghosts).
"""
function wf_quad_marker_sizes(wq; sz_lo = 3.0, sz_hi = 18.0, wq_floor_frac = 1e-4)
    FT = eltype(wq)
    wq_max = max(maximum(wq), FT(1e-24))
    floor_v = wq_max * FT(wq_floor_frac)
  # area fraction = wq / wq_max  =>  diameter fraction = sqrt(area)
    t = sqrt.(clamp.(wq ./ wq_max, zero(FT), one(FT)))
    sz = sz_lo .+ (sz_hi - sz_lo) .* t
    return ifelse.(wq .> floor_v, sz, zero(FT))
end

"""Relative error ``|q - q_{tgt}| / q_{tgt}`` for WF bulk recovery checks."""
function wf_quad_qc_rel_err(q, q_tgt)
    q_tgt = Float64(q_tgt)
    q = Float64(q)
    return q_tgt > 0 ? abs(q - q_tgt) / q_tgt : (q == 0 ? 0.0 : Inf)
end

"""Human-readable recovery error (percent, or absolute Δ when `q_tgt` is tiny)."""
function wf_quad_fmt_recovery_err(q, q_tgt; tol = 0.02)
    q_tgt = Float64(q_tgt)
    q = Float64(q)
    Δ = abs(q - q_tgt)
    if !(q_tgt > 0)
        return q == 0 ? "ok" : "Δ=$(wf_quad_fmt_qc(Δ))"
    end
    rel = Δ / q_tgt
    if q_tgt < 1e-5 && rel > 0.5
        return "Δ=$(wf_quad_fmt_qc(Δ)) ($(round(100 * rel, digits=0))% of tiny q_tgt)"
    end
    tag = rel > tol ? "WARN" : "ok"
    return "$(tag) $(round(100 * rel, digits=1))%"
end

function wf_quad_recovery_line(label, q_gh, q_dense, q_tgt; tol = 0.02)
    return "$label GH=$(wf_quad_fmt_qc(q_gh)) ($(wf_quad_fmt_recovery_err(q_gh, q_tgt; tol)))  " *
           "dense=$(wf_quad_fmt_qc(q_dense)) ($(wf_quad_fmt_recovery_err(q_dense, q_tgt; tol)))"
end

"""Condensate [kg/kg] for panel annotations. Exact zero prints `0`; any nonzero uses sci notation."""
function wf_quad_fmt_qc(q)
    q = Float64(q)
    if !isfinite(q)
        return "nonfinite"
    elseif q == 0.0
        return "0"
    elseif abs(q) >= 1e-3
        return @sprintf("%.5f", q)
    else
        return @sprintf("%.4e", q)
    end
end

"""
Plot one panel. **Blue fill** ``α=1`` (color = ``\\hat{q}_c``); rings for ``α=0.5`` / ``α=0``.
**Marker area** ``\\propto w\\hat{q}_c`` (Makie `markersize` ``\\propto \\sqrt{w\\hat{q}_c}``). Colorbar: ``\\hat{q}_c`` at ``α=1`` only.

Black: ``q_{sat}(T)``. Red / green dashed: WF zero-``\\hat{q}_c`` lines for ``α=1`` and ``α=1/2``
(``q_t = q_{sat} + \\mu - \\lambda/\\alpha``; each ``\\lambda`` from that α setup). No sloped line for ``α=0``.
"""
function wf_quad_plot_case!(
    ax, thp, case, samp; colormap, nσ = 3,
    marker_sz_lo = 3.0, marker_sz_hi = 18.0,
)
    FT = typeof(case.μ)
    ρ = case.ρ
    T_mean = samp.T_mean
    q_mean = samp.q_tot_mean
    TD = ClimaAtmos.TD
    pts = samp.pts
    μ = case.μ
    λ_1 = samp.res_on.λ
    λ_05 = samp.res_mid.λ

    T = [p.T for p in pts]
    q = [p.q for p in pts]
    w = [p.w for p in pts]
    qc_1 = [p.qc_1 for p in pts]
    qc_05 = [p.qc_05 for p in pts]
    qc_0 = [p.qc_0 for p in pts]
    wq_1 = wf_quad_weighted_qc(qc_1, w)
    wq_05 = wf_quad_weighted_qc(qc_05, w)
    wq_0 = wf_quad_weighted_qc(qc_0, w)
    qc_max = max(maximum(qc_1), maximum(qc_05), maximum(qc_0), FT(1e-24))
    qc_range = (zero(FT), qc_max)
    # Color (α=1): q̂_c. Size (all α): area ∝ w·q̂_c via sqrt(w·q̂_c) markersize (Makie convention).
    ms_1 = wf_quad_marker_sizes(wq_1; sz_lo = marker_sz_lo, sz_hi = marker_sz_hi)
    ms_05 = wf_quad_marker_sizes(wq_05; sz_lo = marker_sz_lo, sz_hi = marker_sz_hi)
    ms_0 = wf_quad_marker_sizes(wq_0; sz_lo = marker_sz_lo, sz_hi = marker_sz_hi)

    T1, T2, q1, q2 = wf_quad_axis_limits(T_mean, q_mean, samp.σ_T, samp.σ_q, T, q; nσ)
    pad_T = FT(0.06) * (T2 - T1)
    pad_q = FT(0.06) * (q2 - q1)
    T_line = range(T1 - pad_T, T2 + pad_T; length = 120)
    q_sat = [TD.q_vap_saturation(thp, Ti, ρ) for Ti in T_line]
    q_bd_1 = wf_quad_wf_boundary_q(thp, ρ, T_line, μ, λ_1, one(FT))
    q_bd_05 = wf_quad_wf_boundary_q(thp, ρ, T_line, μ, λ_05, FT(0.5))

    lines!(ax, T_line, q_sat; color = :black, linewidth = 1.5)
    if q_bd_1 !== nothing
        lines!(ax, T_line, q_bd_1; color = :steelblue, linewidth = 1.5, linestyle = :dash)
    end
    if q_bd_05 !== nothing
        lines!(ax, T_line, q_bd_05; color = :limegreen, linewidth = 1.5, linestyle = :dash)
    end

    for (k, (qell, Tell)) in enumerate(wf_quad_gaussian_ellipses_physical(T_mean, q_mean, samp.σ_q, samp.σ_T, samp.corr))
        lines!(ax, Tell, qell; color = (:gray, 0.45), linewidth = k == 1 ? 1.2 : 0.8, linestyle = k == 1 ? :solid : :dot)
    end

    sc = scatter!(
        ax, T, q;
        color = qc_1,
        colormap,
        colorrange = qc_range,
        markersize = ms_1,
        strokewidth = 0,
        alpha = 0.9,
    )

    scatter!(
        ax, T, q;
        color = (:white, 0),
        marker = :circle,
        markersize = ms_05,
        strokecolor = :limegreen,
        strokewidth = 1.0,
    )

    scatter!(
        ax, T, q;
        color = (:white, 0),
        marker = :circle,
        markersize = ms_0,
        strokecolor = :orangered,
        strokewidth = 1.0,
    )

    scatter!(ax, [T_mean], [q_mean]; color = :black, marker = :x, markersize = 9, strokewidth = 1.2)

    xlims!(ax, T1, T2)
    ylims!(ax, q1, q2)

    ax.xlabel = "Temperature [K]"
    ax.ylabel = "Total water qₜ [kg/kg]"
    branch = samp.res_on.branch == :subcap ? "sub-cap (scale_pos=$(wf_quad_fmt_qc(samp.res_on.scale_pos)))" : "shift"
    ax.title = "$(case.title) [$branch]"
    ax.titlesize = 10
    ax.xticklabelsize = 8
    ax.yticklabelsize = 8

    q_tgt = samp.q_cond_tgt
    r1 = samp.res_on
    r05 = samp.res_mid
    r0 = samp.res_off
    A_on = r1.A
    sat_note = μ < zero(FT) ? "\nμ<0: subsaturated mean (× below q_sat at T̄)" : ""
    recovery_text = join(
        [
            wf_quad_recovery_line("α=1", r1.q_gh, r1.q_dense, q_tgt),
            wf_quad_recovery_line("α=½", r05.q_gh, r05.q_dense, q_tgt),
            wf_quad_recovery_line("α=0", r0.q_gh, r0.q_dense, q_tgt),
        ],
        "\n",
    )
    text!(
        ax, 0.03, 0.97,
        text = "λ(α=1)=$(wf_quad_fmt_qc(λ_1))  λ(α=½)=$(wf_quad_fmt_qc(λ_05))  μ=$(wf_quad_fmt_qc(μ))  A(α=1)=$(wf_quad_fmt_qc(A_on))\n" *
               "q_sat(T̄)=$(wf_quad_fmt_qc(samp.q_sat_mean))  q_tgt=$(wf_quad_fmt_qc(q_tgt))\n" *
               recovery_text * sat_note,
        space = :relative, fontsize = 7, align = (:left, :top),
    )

    return sc, qc_range
end

"""
    plot_water_filling_quadrature(out = "/tmp/water_filling_quadrature.png"; kwargs...)

One 3×3 figure. See `wf_quad_default_cases` for the nine package-aligned regimes.

Keyword args include case builder (`ρ`, `T_mean`, `lf`, `μ_pos`, `μ_neg`, `σ_q_narrow`, `σ_q_wide`,
`T′T′_narrow`, `T′T′_wide`, `T′T′_mp`, `corr_mp`), plot (`quadrature_order`, `iters`, `marker_sz_lo/hi`,
`nσ`, `figure_size`, `colormap`), and optional `cases` override vector (length 9).

Goal: partition prescribed grid-mean ``q_{cond}`` over the GH pdf. Blue fill ``α=1``; green / orange
rings ``α=0.5`` / ``α=0``. Panel text reports **GH** and **dense χ-grid** bulk recovery.

Per-panel colorbar for ``\\hat{q}_c`` at ``α=1``; marker **size** is ``w\\hat{q}_c`` (separate channel).
"""
function plot_water_filling_quadrature(
    out = "/tmp/water_filling_quadrature.png";
    cases = nothing,
    dense_grid::Int = 64,
    ρ = 1.0,
    T_mean = 280.0,
    lf = 0.5,
    μ_pos = 3e-4,
    μ_neg = -3e-3,
    σ_q_narrow = 1e-5,
    σ_q_wide = 4e-5,
    T′T′_narrow = 0.25,
    T′T′_wide = 1.0,
    T′T′_mp = 0.5,
    corr_mp = 0.6,
    quadrature_order = 5,
    iters = 1,
    marker_sz_lo = 3.0,
    marker_sz_hi = 18.0,
    nσ = 3,
    figure_size = (1500, 1280),
    colormap = nothing,
)
    FT = Float64
    thp = wf_quad_thermo_params(FT)
    if cases === nothing
        cases = wf_quad_default_cases(
            FT, thp;
            ρ, T_mean, lf, μ_pos, μ_neg, σ_q_narrow, σ_q_wide, T′T′_narrow, T′T′_wide, T′T′_mp, corr_mp,
        )
    end
    length(cases) == 9 || error("expected 9 cases for 3×3 grid, got $(length(cases))")

    samples = [wf_quad_sample_case(thp, c; quadrature_order, iters, dense_grid) for c in cases]

    # α=1: blue fill (colormap = q̂_c) + blue WF dash; α=½ green; α=0 orange rings.
    cmap = colormap === nothing ? CairoMakie.cgrad([:white, :deepskyblue, :midnightblue], 64) : colormap
    CairoMakie.activate!(type = "png")
    fig = Figure(size = figure_size, figure_padding = 10)
    # Dense 3×3 grid (axis + colorbar per cell). Do not skip columns — odd columns only
    # leave empty tracks and stretch panels vertically with huge horizontal gaps.
    for (k, (case, samp)) in enumerate(zip(cases, samples))
        r = div(k - 1, 3) + 1
        c = mod(k - 1, 3) + 1
        panel = fig[r, c] = GridLayout()
        ax = Axis(panel[1, 1])
        sc, _ = wf_quad_plot_case!(
            ax, thp, case, samp;
            colormap = cmap, marker_sz_lo, marker_sz_hi, nσ,
        )
        Colorbar(
            panel[1, 2], sc;
            label = "q̂_c [kg/kg]",
            width = 10,
            ticklabelsize = 7,
            labelsize = 8,
            tickformat = "{:.3e}",
        )
        rowgap!(panel, 4)
        colgap!(panel, 4)
    end

    Legend(
        fig[0, 1:3],
        [
            LineElement(color = :black, linewidth = 1.5),
            LineElement(color = :steelblue, linestyle = :dash, linewidth = 1.5),
            LineElement(color = :limegreen, linestyle = :dash, linewidth = 1.5),
            LineElement(color = (:gray, 0.6), linewidth = 1),
            MarkerElement(marker = :circle, color = :steelblue, markersize = 10),
            MarkerElement(marker = :circle, color = (:white, 0), strokecolor = :limegreen, markersize = 10, strokewidth = 1.2),
            MarkerElement(marker = :circle, color = (:white, 0), strokecolor = :orangered, markersize = 10, strokewidth = 1.2),
            MarkerElement(marker = :x, color = :black, markersize = 10),
        ],
        [
            "q = qₛₐₜ(T) (black)",
            "WF zero-q̂_c line, α=1 (blue dash)",
            "WF zero-q̂_c line, α=½ (green dash)",
            "Gaussian 1σ, 2σ",
            "α=1: blue fill (shade = q̂_c; size ∝ w·q̂_c)",
            "α=½: lime ring (size ∝ w·q̂_c)",
            "α=0: orange ring (size ∝ w·q̂_c)",
            "grid mean (×)",
        ];
        orientation = :horizontal,
        labelsize = 9,
        tellheight = true,
    )

    save(out, fig)
    println("Wrote ", out)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = length(ARGS) ≥ 1 ? abspath(ARGS[1]) : "/tmp/water_filling_quadrature.png"
    plot_water_filling_quadrature(out)
end
