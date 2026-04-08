# Load after `moment_recovery.jl` (needs `mathsanity_quadrature_moments_bivariate`, transforms).

using LinearAlgebra
using Printf
using Random
using Statistics

function mathsanity_cov_ellipse_xy(
    μ::AbstractVector{FT},
    Σ::AbstractMatrix{FT},
    nσ::FT;
    n::Int = 120,
) where {FT <: AbstractFloat}
    λ, V = eigen(Symmetric(Σ))
    λ = max.(λ, zero(FT))
    r = sqrt.(λ) .* nσ
    θ = range(zero(FT), convert(FT, 2π); length = n)
    circle = vcat(cos.(θ)', sin.(θ)')
    pts = V * (r .* circle) .+ μ
    return vec(pts[1, :]), vec(pts[2, :])
end

"""
`1σ` and `2σ` ellipses in physical `(q,T)` for the **naive** bivariate Gaussian with means `(μ_q, μ_T)`,
standard deviations `σ_q`, `σ_T`, and correlation `ρ` (same object the red curves reference in standardized
`mathsanity_plot_quadrature_scatter`, mapped to physical axes).

Returns `(q₁, T₁, q₂, T₂)` polylines for solid `1σ` and dashed `2σ`. Empty vectors if `σ_q == σ_T == 0`.
"""
function mathsanity_naive_gaussian_ellipse_polylines_physical(
    μ_q::FT,
    μ_T::FT,
    σ_q::FT,
    σ_T::FT,
    ρ::FT;
    npts::Int = 120,
) where {FT <: AbstractFloat}
    if iszero(σ_q) && iszero(σ_T)
        return (FT[], FT[], FT[], FT[])
    end
    ρc = clamp(ρ, -one(FT), one(FT))
    Σ = Symmetric(FT[σ_q^2 ρc * σ_q * σ_T; ρc * σ_q * σ_T σ_T^2])
    μ = FT[μ_q, μ_T]
    q1, T1 = mathsanity_cov_ellipse_xy(μ, Σ, one(FT); n = npts)
    q2, T2 = mathsanity_cov_ellipse_xy(μ, Σ, FT(2); n = npts)
    return (q1, T1, q2, T2)
end

"""Map one `(q,T)` point to `(z_q,z_T)` using turbulent `σ_q,σ_T` and the same σ floor as MC diagnostics."""
function mathsanity_zq_zT_pair(q::FT, T::FT, μ_q::FT, μ_T::FT, σ_q::FT, σ_T::FT) where {FT <: AbstractFloat}
    σf = mathsanity_sigma_floor_standardized(FT)
    σqi = max(σ_q, σf)
    σTj = max(σ_T, σf)
    return (q - μ_q) / σqi, (T - μ_T) / σTj
end

"""All standardized-series data for one MC-vs-quadrature panel (GaussianSGS, optional clamp)."""
function mathsanity_standardized_mc_quad_data(
    μ_q::FT,
    μ_T::FT,
    σ_q::FT,
    σ_T::FT,
    ρ::FT;
    N_quad::Int = 5,
    n_mc::Int = 8000,
    clamped::Bool = false,
    rng::AbstractRNG = Random.Xoshiro(42),
    lim_z::FT = FT(3.2),
    T_min::FT = zero(FT),
    q_max::FT = FT(1.0),
) where {FT <: AbstractFloat}
    qs, Ts, ωs = mathsanity_quadrature_moments_bivariate(μ_q, μ_T, σ_q, σ_T, ρ, N_quad; clamped)
    if clamped
        q_mc = Vector{FT}(undef, n_mc)
        T_mc = Vector{FT}(undef, n_mc)
        for k in 1:n_mc
            q_mc[k], T_mc[k] =
                mathsanity_gaussian_sgs_clamped(randn(rng, FT), randn(rng, FT), μ_q, μ_T, σ_q, σ_T, ρ, T_min, q_max)
        end
    else
        z1 = randn(rng, FT, n_mc)
        z2 = randn(rng, FT, n_mc)
        h = sqrt(max(zero(FT), one(FT) - ρ^2))
        q_mc = μ_q .+ σ_q .* z1
        T_mc = μ_T .+ σ_T .* (ρ .* z1 .+ h .* z2)
    end
    σf = mathsanity_sigma_floor_standardized(FT)
    σq_eff = max(σ_q, σf)
    σT_eff = max(σ_T, σf)
    zq(q) = (q - μ_q) / σq_eff
    zT(T) = (T - μ_T) / σT_eff
    zq_quad = zq.(qs)
    zT_quad = zT.(Ts)
    zq_mc = zq.(q_mc)
    zT_mc = zT.(T_mc)
    μz = FT[zero(FT), zero(FT)]
    Σz = [one(FT) ρ; ρ one(FT)]
    ωmax = maximum(ωs)
    ms = @. FT(5) + FT(26) * sqrt(ωs / ωmax)
    xlo, xhi, ylo, yhi = if clamped
        zallq = [zq_quad; zq_mc]
        zallT = [zT_quad; zT_mc]
        loq, hiq = quantile(zallq, (FT(0.01), FT(0.99)))
        loT, hiT = quantile(zallT, (FT(0.01), FT(0.99)))
        spanq = max(hiq - loq, FT(0.4))
        spanT = max(hiT - loT, FT(0.4))
        padq = FT(0.12) * spanq
        padT = FT(0.12) * spanT
        (loq - padq, hiq + padq, loT - padT, hiT + padT)
    else
        (-lim_z, lim_z, -lim_z, lim_z)
    end
    xe1, ye1 = mathsanity_cov_ellipse_xy(μz, Σz, one(FT))
    xe2, ye2 = mathsanity_cov_ellipse_xy(μz, Σz, FT(2))
    return (;
        zq_quad = zq_quad,
        zT_quad = zT_quad,
        zq_mc = zq_mc,
        zT_mc = zT_mc,
        ms = ms,
        xlo = xlo,
        xhi = xhi,
        ylo = ylo,
        yhi = yhi,
        xe1 = xe1,
        ye1 = ye1,
        xe2 = xe2,
        ye2 = ye2,
    )
end

"""Map shared standard normals `(z1,z2)` to `(q,T)` with the same linear (or clamped) map as the MC paths."""
function mathsanity_bivariate_qT_from_shared_z(
    z1::AbstractVector{FT},
    z2::AbstractVector{FT},
    μ_q::FT,
    μ_T::FT,
    σ_q::FT,
    σ_T::FT,
    ρ::FT;
    clamped::Bool = false,
    T_min::FT = zero(FT),
    q_max::FT = FT(1.0),
) where {FT <: AbstractFloat}
    n = length(z1)
    @assert length(z2) == n
    q_out = Vector{FT}(undef, n)
    T_out = Vector{FT}(undef, n)
    ρc = clamp(ρ, -one(FT), one(FT))
    if clamped
        for k in 1:n
            q_out[k], T_out[k] =
                mathsanity_gaussian_sgs_clamped(z1[k], z2[k], μ_q, μ_T, σ_q, σ_T, ρc, T_min, q_max)
        end
    else
        h = sqrt(max(zero(FT), one(FT) - ρc^2))
        @inbounds for k in 1:n
            q_out[k] = μ_q + σ_q * z1[k]
            T_out[k] = μ_T + σ_T * (ρc * z1[k] + h * z2[k])
        end
    end
    return q_out, T_out
end

"""One pass over shared `(z1,z2)`: fill naive and geometry-corrected `(q,T)` (same maps as two `mathsanity_bivariate_qT_from_shared_z` calls)."""
function mathsanity_bivariate_naive_and_corr!(
    q_n::AbstractVector{FT},
    T_n::AbstractVector{FT},
    q_c::AbstractVector{FT},
    T_c::AbstractVector{FT},
    z1::AbstractVector{FT},
    z2::AbstractVector{FT},
    μ_q::FT,
    μ_T::FT,
    σ_q::FT,
    σ_T::FT,
    ρ::FT,
    σq_c::FT,
    σT_c::FT,
    ρ_eff::FT;
    clamped::Bool = false,
    T_min::FT = zero(FT),
    q_max::FT = FT(1.0),
) where {FT <: AbstractFloat}
    n = length(z1)
    @assert length(z2) == n == length(q_n) == length(T_n) == length(q_c) == length(T_c)
    ρn = clamp(ρ, -one(FT), one(FT))
    ρc = clamp(ρ_eff, -one(FT), one(FT))
    if clamped
        @inbounds for k in 1:n
            z1k, z2k = z1[k], z2[k]
            q_n[k], T_n[k] =
                mathsanity_gaussian_sgs_clamped(z1k, z2k, μ_q, μ_T, σ_q, σ_T, ρn, T_min, q_max)
            q_c[k], T_c[k] =
                mathsanity_gaussian_sgs_clamped(z1k, z2k, μ_q, μ_T, σq_c, σT_c, ρc, T_min, q_max)
        end
    else
        hn = sqrt(max(zero(FT), one(FT) - ρn^2))
        hc = sqrt(max(zero(FT), one(FT) - ρc^2))
        @inbounds for k in 1:n
            z1k, z2k = z1[k], z2[k]
            q_n[k] = μ_q + σ_q * z1k
            T_n[k] = μ_T + σ_T * (ρn * z1k + hn * z2k)
            q_c[k] = μ_q + σq_c * z1k
            T_c[k] = μ_T + σT_c * (ρc * z1k + hc * z2k)
        end
    end
    return nothing
end

"""Copy `a[1:n]` and `b[1:n]` into `scratch`, `sort!`, then 1% / 99% quantiles (`sorted=true`, same as `Statistics.quantile`)."""
function mathsanity_sorted_union_quantile_01_99!(
    scratch::AbstractVector{FT},
    a::AbstractVector{FT},
    b::AbstractVector{FT},
    n::Int,
) where {FT <: AbstractFloat}
    copyto!(scratch, 1, a, 1, n)
    copyto!(scratch, n + 1, b, 1, n)
    v = view(scratch, 1:2n)
    sort!(v)
    lo = quantile(v, FT(0.01); sorted = true)
    hi = quantile(v, FT(0.99); sorted = true)
    return lo, hi
end

"""`(dq/dz, dT/dz)` as a direction in physical `(q,T)` for column structure (`dT/dz = (∂T/∂θ_li)(∂θ_li/∂z)`)."""
function mathsanity_column_grad_dq_dT(dq_dz::FT, dtheta_dz::FT, dT_dθ::FT) where {FT <: AbstractFloat}
    return dq_dz, dT_dθ * dtheta_dz
end

"""Subtitles/legends: `dT/dq` from exact float zeros only (`iszero(dq/dz)` / `iszero(dT/dz)`), otherwise plain division."""
function mathsanity_column_slope_category(dq_dz::FT, dtheta_dz::FT, dT_dθ::FT) where {FT <: AbstractFloat}
    dq, dT = mathsanity_column_grad_dq_dT(dq_dz, dtheta_dz, dT_dθ)
    iszero(dq) && iszero(dT) && return (:none, FT(0))
    iszero(dq) && return (:vertical, FT(0))
    iszero(dT) && return (:horizontal, FT(0))
    return (:oblique, dT / dq)
end

"""
Finite segment of `T = (dT/dq)(q - μ_q) + μ_T` inside the axis-aligned box `[qlo,qhi]×[Tlo,Thi]`.

Uses `lines!` instead of `ablines!` because CairoMakie often fails to show `ablines!` / infinite lines reliably in dense
multi-layer scenes. Branches only on **exact** `iszero(dq)` / `iszero(dT)` (singular cases).
"""
function mathsanity_column_line_segment_in_box(
    qlo::FT,
    qhi::FT,
    Tlo::FT,
    Thi::FT,
    μq::FT,
    μT::FT,
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dθ::FT,
) where {FT <: AbstractFloat}
    dq, dT = mathsanity_column_grad_dq_dT(dq_dz, dtheta_dz, dT_dθ)
    iszero(dq) && iszero(dT) && return nothing
    if iszero(dq)
        (qlo <= μq <= qhi) || return nothing
        return (μq, Tlo, μq, Thi)
    end
    if iszero(dT)
        (Tlo <= μT <= Thi) || return nothing
        return (qlo, μT, qhi, μT)
    end
    m = dT / dq
    b = μT - m * μq
    cand = Tuple{FT, FT}[]
    for qx in (qlo, qhi)
        Ty = m * qx + b
        if (Ty >= Tlo) && (Ty <= Thi)
            push!(cand, (qx, Ty))
        end
    end
    for Ty in (Tlo, Thi)
        qx = (Ty - b) / m
        if (qx >= qlo) && (qx <= qhi)
            push!(cand, (qx, Ty))
        end
    end
    length(cand) < 2 && return nothing
    best_d, q1, T1, q2, T2 = zero(FT), cand[1][1], cand[1][2], cand[1][1], cand[1][2]
    for i in 1:length(cand), j in (i + 1):length(cand)
        a, bpt = cand[i], cand[j]
        dx = bpt[1] - a[1]
        dy = bpt[2] - a[2]
        d2 = dx * dx + dy * dy
        if d2 > best_d
            best_d = d2
            q1, T1, q2, T2 = a[1], a[2], bpt[1], bpt[2]
        end
    end
    iszero(best_d) && return nothing
    return (q1, T1, q2, T2)
end

"""
Clip the segment `(q1,T1)`–`(q2,T2)` to the axis-aligned rectangle `[qlo,qhi]×[Tlo,Thi]` (Liang–Barsky).
Returns `(qa,Ta,qb,Tb)` along the same line with `Tb≥Ta` not guaranteed, or `nothing` if disjoint / degenerate after clip.
"""
function mathsanity_clip_segment_to_axis_box(
    q1::FT,
    T1::FT,
    q2::FT,
    T2::FT,
    qlo::FT,
    qhi::FT,
    Tlo::FT,
    Thi::FT,
) where {FT <: AbstractFloat}
    dq = q2 - q1
    dT = T2 - T1
    u0 = zero(FT)
    u1 = one(FT)
    p = (-dq, dq, -dT, dT)
    qv = (q1 - qlo, qhi - q1, T1 - Tlo, Thi - T1)
    @inbounds for k in 1:4
        pk, qk = p[k], qv[k]
        if iszero(pk)
            if qk < zero(FT)
                return nothing
            end
        else
            rk = qk / pk
            if pk < zero(FT)
                u0 = max(u0, rk)
            else
                u1 = min(u1, rk)
            end
        end
    end
    u0 > u1 && return nothing
    qa = q1 + u0 * dq
    Ta = T1 + u0 * dT
    qb = q1 + u1 * dq
    Tb = T1 + u1 * dT
    if isapprox(qa, qb; atol = 8 * eps(FT)) && isapprox(Ta, Tb; atol = 8 * eps(FT))
        return nothing
    end
    return (qa, Ta, qb, Tb)
end

"""
Finite segment for the column direction in `(q,T)`.

- If `column_dz === nothing`: full column line through `[qlo,qhi]×[Tlo,Thi]` (`mathsanity_column_line_segment_in_box`).
- If `column_dz > 0`: endpoints `(μ_q, μ_T) ± (Δz/2)(dq/dz, dT/dz)`, then clipped to the axis box.
- If `column_dz ≤ 0` (explicit): return `nothing`; the caller should draw a **point** at `(μ_q, μ_T)` (no vertical layer to span).
"""
function mathsanity_column_line_segment_dz_or_box(
    qlo::FT,
    qhi::FT,
    Tlo::FT,
    Thi::FT,
    μq::FT,
    μT::FT,
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dθ::FT,
    column_dz,
) where {FT <: AbstractFloat}
    if column_dz === nothing
        return mathsanity_column_line_segment_in_box(qlo, qhi, Tlo, Thi, μq, μT, dq_dz, dtheta_dz, dT_dθ)
    end
    dz = FT(column_dz)
    if dz <= zero(FT)
        return nothing
    end
    _, dT_dz = mathsanity_column_grad_dq_dT(dq_dz, dtheta_dz, dT_dθ)
    half = dz / 2
    q_a = μq - dq_dz * half
    T_a = μT - dT_dz * half
    q_b = μq + dq_dz * half
    T_b = μT + dT_dz * half
    return mathsanity_clip_segment_to_axis_box(q_a, T_a, q_b, T_b, qlo, qhi, Tlo, Thi)
end

"""One-line caption for the column slope in `(q,T)`."""
function mathsanity_column_slope_phrase(dq_dz::FT, dtheta_dz::FT, dT_dθ::FT) where {FT <: AbstractFloat}
    cat, m = mathsanity_column_slope_category(dq_dz, dtheta_dz, dT_dθ)
    if cat == :none
        return "column gradients exactly 0"
    elseif cat == :vertical
        return "dT/dq = ∞ (dq/dz == 0)"
    elseif cat == :horizontal
        return "dT/dq = 0 (dT/dz == 0)"
    else
        return @sprintf("dT/dq = %.5g K·(kg/kg)⁻¹", m)
    end
end

"""True if totals `(σ_q,tot, σ_T,tot, ρ_eff)` differ from turbulent `(σ_q, σ_T, ρ)` enough to draw teal Σ / purple GH."""
function mathsanity_sigma_geometry_differs_from_turb(
    σ_q::FT,
    σ_T::FT,
    ρ::FT,
    σqg::FT,
    σTg::FT,
    ρg::FT,
) where {FT <: AbstractFloat}
    rtol_σ = FT(5e-4)
    atol_q = max(FT(1e-12), FT(1e-6) * max(one(FT), abs(σ_q)))
    atol_T = max(FT(1e-10), FT(1e-6) * max(one(FT), abs(σ_T)))
    return !(
        isapprox(σqg, σ_q; rtol = rtol_σ, atol = atol_q) &&
        isapprox(σTg, σ_T; rtol = rtol_σ, atol = atol_T) &&
        isapprox(ρg, ρ; rtol = FT(0.002), atol = FT(0.002))
    )
end

"""`nothing` iff `dq/dz` and `dT/dz` are both exactly zero; else store parameters for Makie `ablines!`/`vlines!`/`hlines!`."""
function mathsanity_column_grad_params(
    μ_q::FT,
    μ_T::FT,
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dθ::FT,
) where {FT <: AbstractFloat}
    dq, dT = mathsanity_column_grad_dq_dT(dq_dz, dtheta_dz, dT_dθ)
    if iszero(dq) && iszero(dT)
        return nothing
    end
    return (μ_q, μ_T, dq_dz, dtheta_dz, dT_dθ)
end

"""One `(q,T)` mosaic panel: optional MC clouds (`q_naive === nothing` when `n_mc == 0`) plus axis limits.
`column_grad` holds `(μ_q, μ_T, dq_dz, dtheta_dz, dT_dθ)` for the column reference in `(q,T)`.
`sigma_*_tot` / `rho_eff_tot` are **geometry-inclusive** totals (`√var` after `(1/12)Δz²` terms) matching the coral MC map.
`quad_*` = Gauss–Hermite nodes at **turbulent** `(σ_q, σ_T, ρ)` (blue). `quad_*_geom` = same at **`(σ_tot, ρ_eff)`** when geometry
changes totals (purple); omitted when identical (e.g. `Δz=0`)."""
struct MathsanityPhysicalMosaicPanel{FT <: AbstractFloat}
    q_naive::Union{Nothing, Vector{FT}}
    T_naive::Union{Nothing, Vector{FT}}
    q_corr::Union{Nothing, Vector{FT}}
    T_corr::Union{Nothing, Vector{FT}}
    qlo::FT
    qhi::FT
    Tlo::FT
    Thi::FT
    column_grad::Union{Nothing, NTuple{5, FT}}
    sigma_q_tot::FT
    sigma_T_tot::FT
    rho_eff_tot::FT
    quad_q::Union{Nothing, Vector{FT}}
    quad_T::Union{Nothing, Vector{FT}}
    quad_ms::Union{Nothing, Vector{Float32}}
    quad_q_geom::Union{Nothing, Vector{FT}}
    quad_T_geom::Union{Nothing, Vector{FT}}
    quad_ms_geom::Union{Nothing, Vector{Float32}}
end

"""
Shared-`χ` naive vs geometry-corrected `(q,T)` MC clouds (when `n_mc > 0`) and gradient line for one mosaic panel.
Turbulent variances are `σ_q²`, `σ_T²`; corrected moments are total `var_q`, `var_T` and `ρ_eff`.
With `n_mc == 0`, skips MC clouds; set `N_quad > 0` to attach Gauss–Hermite nodes in `(q,T)` for plotting (blue dots).
"""
function mathsanity_physical_mosaic_panel_data(
    μ_q::FT,
    μ_T::FT,
    σ_q::FT,
    σ_T::FT,
    ρ::FT,
    var_q::FT,
    var_T::FT,
    ρ_eff::FT,
    dq_dz::FT,
    dtheta_dz::FT,
    dT_dθ::FT;
    n_mc::Int = 2000,
    N_quad::Int = 0,
    clamped::Bool = false,
    rng::AbstractRNG = Random.Xoshiro(42),
    T_min::FT = zero(FT),
    q_max::FT = FT(1.0),
) where {FT <: AbstractFloat}
    σq_c = sqrt(max(zero(FT), var_q))
    σT_c = sqrt(max(zero(FT), var_T))
    if n_mc <= 0
        span_q = max(max(abs(σ_q), σq_c) * FT(0.25), FT(1e-8))
        span_T = max(max(abs(σ_T), σT_c) * FT(0.25), FT(1e-4))
        pad_q = FT(0.2) * span_q
        pad_T = FT(0.2) * span_T
        column_grad = mathsanity_column_grad_params(μ_q, μ_T, dq_dz, dtheta_dz, dT_dθ)
        loq = μ_q - span_q - pad_q
        hiq = μ_q + span_q + pad_q
        loT = μ_T - span_T - pad_T
        hiT = μ_T + span_T + pad_T
        padq2 = FT(0.12) * span_q
        padT2 = FT(0.12) * span_T
        qq_v, TT_v, ms_v = if N_quad > 0
            qs, Ts, ωs = mathsanity_quadrature_moments_bivariate(μ_q, μ_T, σ_q, σ_T, ρ, N_quad; clamped)
            ωmx = maximum(ωs)
            ms = @. FT(5) + FT(26) * sqrt(ωs / ωmx)
            loq = min(loq, minimum(qs))
            hiq = max(hiq, maximum(qs))
            loT = min(loT, minimum(Ts))
            hiT = max(hiT, maximum(Ts))
            (collect(qs), collect(Ts), map(x -> Float32(x), collect(ms)))
        else
            (nothing, nothing, nothing)
        end
        qq_g, TT_g, ms_g = if N_quad > 0 && mathsanity_sigma_geometry_differs_from_turb(σ_q, σ_T, ρ, σq_c, σT_c, ρ_eff)
            qs, Ts, ωs = mathsanity_quadrature_moments_bivariate(μ_q, μ_T, σq_c, σT_c, ρ_eff, N_quad; clamped)
            ωmx = maximum(ωs)
            ms = @. FT(5) + FT(22) * sqrt(ωs / ωmx)
            loq = min(loq, minimum(qs))
            hiq = max(hiq, maximum(qs))
            loT = min(loT, minimum(Ts))
            hiT = max(hiT, maximum(Ts))
            (collect(qs), collect(Ts), map(x -> Float32(x), collect(ms)))
        else
            (nothing, nothing, nothing)
        end
        loq -= padq2
        hiq += padq2
        loT -= padT2
        hiT += padT2
        return MathsanityPhysicalMosaicPanel{FT}(
            nothing,
            nothing,
            nothing,
            nothing,
            loq,
            hiq,
            loT,
            hiT,
            column_grad,
            σq_c,
            σT_c,
            ρ_eff,
            qq_v,
            TT_v,
            ms_v,
            qq_g,
            TT_g,
            ms_g,
        )
    end

    z1 = Vector{FT}(undef, n_mc)
    z2 = Vector{FT}(undef, n_mc)
    randn!(rng, z1)
    randn!(rng, z2)
    q_n = Vector{FT}(undef, n_mc)
    T_n = Vector{FT}(undef, n_mc)
    q_c = Vector{FT}(undef, n_mc)
    T_c = Vector{FT}(undef, n_mc)
    mathsanity_bivariate_naive_and_corr!(
        q_n,
        T_n,
        q_c,
        T_c,
        z1,
        z2,
        μ_q,
        μ_T,
        σ_q,
        σ_T,
        ρ,
        σq_c,
        σT_c,
        ρ_eff;
        clamped = clamped,
        T_min = T_min,
        q_max = q_max,
    )
    scratch = Vector{FT}(undef, 2 * n_mc)
    loq, hiq = mathsanity_sorted_union_quantile_01_99!(scratch, q_n, q_c, n_mc)
    loT, hiT = mathsanity_sorted_union_quantile_01_99!(scratch, T_n, T_c, n_mc)
    span_q = max(hiq - loq, max(abs(σ_q), σq_c) * FT(0.25), FT(1e-8))
    span_T = max(hiT - loT, max(abs(σ_T), σT_c) * FT(0.25), FT(1e-4))
    pad_q = FT(0.2) * span_q
    pad_T = FT(0.2) * span_T
    column_grad = mathsanity_column_grad_params(μ_q, μ_T, dq_dz, dtheta_dz, dT_dθ)
    padq2 = FT(0.12) * span_q
    padT2 = FT(0.12) * span_T
    qq_v, TT_v, ms_v = if N_quad > 0
        qs, Ts, ωs = mathsanity_quadrature_moments_bivariate(μ_q, μ_T, σ_q, σ_T, ρ, N_quad; clamped)
        ωmx = maximum(ωs)
        ms = @. FT(5) + FT(26) * sqrt(ωs / ωmx)
        loq = min(loq, minimum(qs))
        hiq = max(hiq, maximum(qs))
        loT = min(loT, minimum(Ts))
        hiT = max(hiT, maximum(Ts))
        (collect(qs), collect(Ts), map(x -> Float32(x), collect(ms)))
    else
        (nothing, nothing, nothing)
    end
    qq_g, TT_g, ms_g = if N_quad > 0 && mathsanity_sigma_geometry_differs_from_turb(σ_q, σ_T, ρ, σq_c, σT_c, ρ_eff)
        qs, Ts, ωs = mathsanity_quadrature_moments_bivariate(μ_q, μ_T, σq_c, σT_c, ρ_eff, N_quad; clamped)
        ωmx = maximum(ωs)
        ms = @. FT(5) + FT(22) * sqrt(ωs / ωmx)
        loq = min(loq, minimum(qs))
        hiq = max(hiq, maximum(qs))
        loT = min(loT, minimum(Ts))
        hiT = max(hiT, maximum(Ts))
        (collect(qs), collect(Ts), map(x -> Float32(x), collect(ms)))
    else
        (nothing, nothing, nothing)
    end
    loq -= padq2
    hiq += padq2
    loT -= padT2
    hiT += padT2
    return MathsanityPhysicalMosaicPanel{FT}(
        q_n,
        T_n,
        q_c,
        T_c,
        loq,
        hiq,
        loT,
        hiT,
        column_grad,
        σq_c,
        σT_c,
        ρ_eff,
        qq_v,
        TT_v,
        ms_v,
        qq_g,
        TT_g,
        ms_g,
    )
end
