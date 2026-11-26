# ============================================================================
# Coupled Microphysics and Sedimentation Top-Down Column Sweep
# ============================================================================
#
# Advances cloud and precipitation condensates, water vapor, air density, and
# total energy over one step `dt` by a single top-down level-by-level sweep per
# column. Within each cell, microphysical sources, phase relaxation, thermal
# melt arrest, and vertical sedimentation act simultaneously on the cell pool
# and incoming shaft flux.

import StaticArrays as SA
import Thermodynamics as TD
import ClimaComms
import ClimaCore: DataLayouts, Fields, Geometry, Operators, Spaces
import ClimaCore.MatrixFields
import UnrolledUtilities: unrolled_foreach, unrolled_map
import CloudMicrophysics.Parameters as CMP
import CloudMicrophysics.Utilities as CMU
import CloudMicrophysics.Common as CMCo
import CloudMicrophysics.Microphysics0M as CM0
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2
import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT

# ============================================================================
# 1. Float32-safe scalar φ functions and divided differences
# ============================================================================

"""
    _phi1_neg(x)

Evaluate `φ₁(-x) = (1 - exp(-x)) / x` for `x ≥ 0` without cancellation near
`x = 0` or overflow as `x → ∞`.
"""
@inline _phi_poly_cutoff(::Type{Float32}) = 1.0f0
@inline _phi_poly_cutoff(::Type{FT}) where {FT} = FT(1 / 4)

@inline function _phi1_poly(xc::FT) where {FT}
    if FT === Float32
        return evalpoly(
            xc,
            (
                one(FT),
                -FT(1 / 2),
                FT(1 / 6),
                -FT(1 / 24),
                FT(1 / 120),
                -FT(1 / 720),
                FT(1 / 5040),
                -FT(1 / 40320),
                FT(1 / 362880),
                -FT(1 / 3628800),
            ),
        )
    end
    return evalpoly(
        xc,
        (
            one(FT),
            -FT(1 / 2),
            FT(1 / 6),
            -FT(1 / 24),
            FT(1 / 120),
            -FT(1 / 720),
            FT(1 / 5040),
            -FT(1 / 40320),
            FT(1 / 362880),
            -FT(1 / 3628800),
            FT(1 / 39916800),
            -FT(1 / 479001600),
        ),
    )
end

@inline function _phi2_poly(xc::FT) where {FT}
    if FT === Float32
        return evalpoly(
            xc,
            (
                FT(1 / 2),
                -FT(1 / 6),
                FT(1 / 24),
                -FT(1 / 120),
                FT(1 / 720),
                -FT(1 / 5040),
                FT(1 / 40320),
                -FT(1 / 362880),
                FT(1 / 3628800),
                -FT(1 / 39916800),
            ),
        )
    end
    return evalpoly(
        xc,
        (
            FT(1 / 2),
            -FT(1 / 6),
            FT(1 / 24),
            -FT(1 / 120),
            FT(1 / 720),
            -FT(1 / 5040),
            FT(1 / 40320),
            -FT(1 / 362880),
            FT(1 / 3628800),
            -FT(1 / 39916800),
            FT(1 / 479001600),
            -FT(1 / 6227020800),
        ),
    )
end

@inline function _psi1_poly(xc::FT) where {FT}
    if FT === Float32
        return evalpoly(
            xc,
            (
                FT(1 / 2),
                -FT(1 / 3),
                FT(1 / 8),
                -FT(1 / 30),
                FT(1 / 144),
                -FT(1 / 840),
                FT(1 / 5760),
                -FT(1 / 45360),
                FT(1 / 403200),
                -FT(1 / 3991680),
            ),
        )
    end
    return evalpoly(
        xc,
        (
            FT(1 / 2),
            -FT(1 / 3),
            FT(1 / 8),
            -FT(1 / 30),
            FT(1 / 144),
            -FT(1 / 840),
            FT(1 / 5760),
            -FT(1 / 45360),
            FT(1 / 403200),
            -FT(1 / 3991680),
            FT(1 / 43545600),
        ),
    )
end

@inline function _psi2_poly(xc::FT) where {FT}
    if FT === Float32
        return evalpoly(
            xc,
            (
                FT(1 / 6),
                -FT(1 / 12),
                FT(1 / 40),
                -FT(1 / 180),
                FT(1 / 1008),
                -FT(1 / 6720),
                FT(1 / 51840),
                -FT(1 / 453600),
                FT(1 / 4435200),
                -FT(1 / 47900160),
            ),
        )
    end
    return evalpoly(
        xc,
        (
            FT(1 / 6),
            -FT(1 / 12),
            FT(1 / 40),
            -FT(1 / 180),
            FT(1 / 1008),
            -FT(1 / 6720),
            FT(1 / 51840),
            -FT(1 / 453600),
            FT(1 / 4435200),
            -FT(1 / 47900160),
            FT(1 / 566092800),
        ),
    )
end

@inline function _phi1_neg(x)
    FT = typeof(x)
    xc = max(x, zero(FT))
    if xc <= _phi_poly_cutoff(FT)
        return _phi1_poly(xc)
    end
    return (FT === Float32 ? (one(FT) - exp(-xc)) : -expm1(-xc)) / xc
end

"""
    _phi2_neg(x)

Evaluate `φ₂(-x) = (x - (1 - exp(-x))) / x² = (1 - φ₁(-x)) / x` for `x ≥ 0`
without cancellation near `x = 0` or `x²` overflow at large `x`.
"""
@inline function _phi2_neg(x)
    FT = typeof(x)
    xc = max(x, zero(FT))
    if xc <= _phi_poly_cutoff(FT)
        return _phi2_poly(xc)
    end
    inv_x = one(FT) / xc
    p1 = (FT === Float32 ? (one(FT) - exp(-xc)) : -expm1(-xc)) * inv_x
    return (one(FT) - p1) * inv_x
end

# Joint `(φ₁(-x), φ₂(-x))` and `(exp(-x), φ₁(-x), φ₂(-x))` sharing a single
# exponential evaluation.
@inline function _phi1_phi2_neg(x)
    FT = typeof(x)
    xc = max(x, zero(FT))
    if xc <= _phi_poly_cutoff(FT)
        return (_phi1_poly(xc), _phi2_poly(xc))
    end
    inv_x = one(FT) / xc
    p1 = (FT === Float32 ? (one(FT) - exp(-xc)) : -expm1(-xc)) * inv_x
    return (p1, (one(FT) - p1) * inv_x)
end

@inline function _exp_phi1_phi2_neg(x)
    FT = typeof(x)
    xc = max(x, zero(FT))
    if xc == zero(FT)
        return (one(FT), one(FT), FT(0.5))
    end
    e = exp(-xc)
    if xc <= _phi_poly_cutoff(FT)
        return (e, _phi1_poly(xc), _phi2_poly(xc))
    end
    inv_x = one(FT) / xc
    p1 = (FT === Float32 ? (one(FT) - e) : -expm1(-xc)) * inv_x
    return (e, p1, (one(FT) - p1) * inv_x)
end

"""
    _dd_exp_neg(u, v)

Non-negative divided difference `(exp(-u) - exp(-v)) / (v - u)` for
`0 ≤ u ≤ v`, equal to `exp(-u) φ₁(-(v - u))` for all `v ≥ u`, including `v = u`.
"""
@inline function _dd_exp_neg(u, v)
    FT = typeof(u)
    h = max(v - u, zero(FT))
    return exp(-u) * _phi1_neg(h)
end

"""
    _dd_phi1_neg(u, v)

Non-negative divided difference `(φ₁(-u) - φ₁(-v)) / (v - u)` for `0 ≤ u ≤ v`.
Uses the convex small-argument decomposition on `v ≤ 1` and the cancellation-free
recurrence `(φ₁(-u) - exp(-u) φ₁(-(v - u))) / v` on `v > 1`.
"""
@inline function _dd_phi1_neg(u, v)
    FT = typeof(u)
    if v <= zero(FT)
        return FT(1 / 2)
    end
    h = max(v - u, zero(FT))
    if v <= one(FT)
        p1_u, p2_u = _phi1_phi2_neg(u)
        p1_h, p2_h = _phi1_phi2_neg(h)
        return (u * (p1_u * p1_h - p2_u) + h * p2_h) / max(v, floatmin(FT))
    end
    e_u, p1_u, _ = _exp_phi1_phi2_neg(u)
    p1_h = _phi1_neg(h)
    return max(zero(FT), p1_u - e_u * p1_h) / v
end

@inline function _dd_phi2_series(u::FT, v::FT) where {FT}
    c = one(FT)
    u_pow = u
    acc = FT(1 / 6)
    n_max = FT === Float32 ? 7 : 12
    inv_fact = FT(1 / 6)
    sign_k = one(FT)
    @inbounds for k in 2:n_max
        c = muladd(v, c, u_pow)
        u_pow *= u
        inv_fact /= FT(k + 2)
        sign_k = -sign_k
        acc = muladd(sign_k * inv_fact, c, acc)
    end
    return max(zero(FT), acc)
end

"""
    _dd_phi2_neg(u, v)

Non-negative divided difference `(φ₂(-u) - φ₂(-v)) / (v - u)` for `0 ≤ u ≤ v`.
Uses a symmetric power divided-difference series on `v ≤ 1/2` and the
exact recurrence `(φ₂(-u) - _dd_phi1_neg(u, v)) / v` on `v > 1/2`.
"""
@inline function _dd_phi2_neg(u, v)
    FT = typeof(u)
    if v <= FT(1 / 2)
        return _dd_phi2_series(u, v)
    end
    return max(zero(FT), _phi2_neg(u) - _dd_phi1_neg(u, v)) / v
end

# Joint `(dd_exp, dd_phi1, dd_phi2)` reusing `(exp(-u), φ₁(-u), φ₂(-u))`.
@inline function _dd_exp_phi1_phi2_neg(
    u::FT,
    v::FT,
    e_u::FT,
    p1_u::FT,
    p2_u::FT,
) where {FT <: AbstractFloat}
    h = max(v - u, zero(FT))
    p1_h, p2_h = _phi1_phi2_neg(h)
    dd_e = e_u * p1_h
    dd_p1 = if v <= zero(FT)
        FT(1 / 2)
    elseif v <= one(FT)
        (u * (p1_u * p1_h - p2_u) + h * p2_h) / max(v, floatmin(FT))
    else
        max(zero(FT), p1_u - dd_e) / v
    end
    dd_p2 = if v <= FT(1 / 2)
        _dd_phi2_series(u, v)
    else
        max(zero(FT), p2_u - dd_p1) / v
    end
    return (dd_e, dd_p1, dd_p2)
end

# Negative derivatives `-d/dx φ₁(-x) > 0` and `-d/dx φ₂(-x) > 0` for `x ≥ 0`.
@inline function _psi1_neg(x::FT) where {FT <: AbstractFloat}
    xc = max(x, zero(FT))
    if xc <= _phi_poly_cutoff(FT)
        return _psi1_poly(xc)
    end
    inv_x = one(FT) / xc
    e = exp(-xc)
    p1 = (FT === Float32 ? (one(FT) - e) : -expm1(-xc)) * inv_x
    return max(zero(FT), p1 - e) * inv_x
end

@inline function _psi2_neg(x::FT) where {FT <: AbstractFloat}
    xc = max(x, zero(FT))
    if xc <= _phi_poly_cutoff(FT)
        return _psi2_poly(xc)
    end
    inv_x = one(FT) / xc
    e = exp(-xc)
    p1 = (FT === Float32 ? (one(FT) - e) : -expm1(-xc)) * inv_x
    p2 = (one(FT) - p1) * inv_x
    psi1 = max(zero(FT), p1 - e) * inv_x
    return max(zero(FT), p2 - psi1) * inv_x
end

# ============================================================================
# 2. Divided-difference 2 × 2 Metzler matrix functions
# ============================================================================

"""
    metzler_2x2_exp_phi1_phi2(D1, D2, D12, D21, dt[, det_negA_override])
    metzler_2x2_exp_phi1_phi2(M::SMatrix{2, 2}, dt)

Closed-form `(exp(dt * M), φ₁(dt * M), φ₂(dt * M))` as `SMatrix{2, 2}` tuples
for a `2 × 2` Metzler matrix

    M = [ -D1   D21 ]
        [  D12  -D2 ]

with `D1, D2, D12, D21 ≥ 0` and non-positive real eigenvalues (`D1 * D2 ≥ D12 * D21`).
Evaluates diagonal entries as convex combinations of `f(-x_-)` and `f(-x_+)`
and off-diagonal entries via divided differences `f[-x_-, -x_+]` so that
`Δ = (D1 - D2)² + 4 D12 D21 → 0`, `M → 0`, and stiff `dt * ||M|| ≫ 1` are
smooth, `NaN`-free, non-negative, and mass-conserving in `Float32` and `Float64`.

When supplied, `det_negA_override` provides a cancellation-free algebraic form
for `det(-dt * M)` and must equal `dt² (D1 * D2 - D12 * D21)` to floating-point
rounding because the eigenvalue half-splitting `δ` is computed from
`hypot((a1 - a2) / 2, sqrt(b12 * b21))`.
"""
@inline function metzler_2x2_exp_phi1_phi2(
    D1::FT,
    D2::FT,
    D12::FT,
    D21::FT,
    dt::FT,
    det_negA_override::Union{Nothing, FT} = nothing,
) where {FT <: AbstractFloat}
    a1 = max(zero(FT), D1 * dt)
    a2 = max(zero(FT), D2 * dt)
    b21 = max(zero(FT), D12 * dt) # M[2, 1]
    b12 = max(zero(FT), D21 * dt) # M[1, 2]

    # Diagonal fast path skips eigendecomposition and divided differences.
    if (b21 <= zero(FT)) & (b12 <= zero(FT))
        z = zero(FT)
        e1, p1_1, p2_1 = _exp_phi1_phi2_neg(a1)
        e2, p1_2, p2_2 = _exp_phi1_phi2_neg(a2)
        E = SA.SMatrix{2, 2, FT, 4}(e1, z, z, e2)
        Phi1 = SA.SMatrix{2, 2, FT, 4}(p1_1, z, z, p1_2)
        Phi2 = SA.SMatrix{2, 2, FT, 4}(p2_1, z, z, p2_2)
        return (E, Phi1, Phi2)
    end

    # Triangular fast path skips hypot/sqrt when one off-diagonal vanishes.
    if (b21 <= zero(FT)) | (b12 <= zero(FT))
        e1, p1_1, p2_1 = _exp_phi1_phi2_neg(a1)
        e2, p1_2, p2_2 = _exp_phi1_phi2_neg(a2)
        if a1 <= a2
            dd_e, dd_p1, dd_p2 =
                _dd_exp_phi1_phi2_neg(a1, a2, e1, p1_1, p2_1)
        else
            dd_e, dd_p1, dd_p2 =
                _dd_exp_phi1_phi2_neg(a2, a1, e2, p1_2, p2_2)
        end
        E = SA.SMatrix{2, 2, FT, 4}(e1, dd_e * b21, dd_e * b12, e2)
        Phi1 = SA.SMatrix{2, 2, FT, 4}(p1_1, dd_p1 * b21, dd_p1 * b12, p1_2)
        Phi2 = SA.SMatrix{2, 2, FT, 4}(p2_1, dd_p2 * b21, dd_p2 * b12, p2_2)
        return (E, Phi1, Phi2)
    end

    return _metzler_2x2_exp_phi1_phi2_general(
        a1,
        a2,
        b21,
        b12,
        det_negA_override,
    )
end

@inline function _metzler_2x2_exp_phi1_phi2_general(
    a1::FT,
    a2::FT,
    b21::FT,
    b12::FT,
    det_negA_override::Union{Nothing, FT},
) where {FT <: AbstractFloat}
    μ = FT(0.5) * (a1 + a2)
    d = FT(0.5) * (a1 - a2)
    off_prod = b12 * b21
    δ = hypot(d, sqrt(off_prod))

    x_plus = μ + δ
    det_negA = if !isnothing(det_negA_override)
        max(zero(FT), det_negA_override)
    elseif (a1 >= b21) & (a2 >= b12)
        (a1 - b21) * a2 + b21 * (a2 - b12)
    elseif (a1 >= b12) & (a2 >= b21)
        (a1 - b12) * a2 + b12 * (a2 - b21)
    else
        max(zero(FT), muladd(a1, a2, -off_prod))
    end
    x_minus = ifelse(
        x_plus > zero(FT),
        min(x_plus, det_negA / max(x_plus, floatmin(FT))),
        zero(FT),
    )

    # Convex diagonal weights `w1 = (x_+ - a1) / (x_+ - x_-)` and `w2 = 1 - w1`
    # computed without subtraction via `δ - |d| = off_prod / (δ + |d|)`.
    δ_plus_absd = δ + abs(d)
    δ_minus_absd = ifelse(
        δ_plus_absd > zero(FT),
        off_prod / max(δ_plus_absd, floatmin(FT)),
        zero(FT),
    )
    two_δ = FT(2) * δ
    two_δ_safe = max(two_δ, floatmin(FT))
    w_small = δ_minus_absd / two_δ_safe
    w_large = δ_plus_absd / two_δ_safe
    w1 = ifelse(
        δ > zero(FT),
        ifelse(d >= zero(FT), w_small, w_large),
        FT(0.5),
    )
    w2 = ifelse(
        δ > zero(FT),
        ifelse(d >= zero(FT), w_large, w_small),
        FT(0.5),
    )

    e_m, p1_m, p2_m = _exp_phi1_phi2_neg(x_minus)
    e_p, p1_p, p2_p = _exp_phi1_phi2_neg(x_plus)
    dd_e, dd_p1, dd_p2 =
        _dd_exp_phi1_phi2_neg(x_minus, x_plus, e_m, p1_m, p2_m)
    E = SA.SMatrix{2, 2, FT, 4}(
        muladd(w1, e_m, w2 * e_p),
        dd_e * b21,
        dd_e * b12,
        muladd(w2, e_m, w1 * e_p),
    )
    Phi1 = SA.SMatrix{2, 2, FT, 4}(
        muladd(w1, p1_m, w2 * p1_p),
        dd_p1 * b21,
        dd_p1 * b12,
        muladd(w2, p1_m, w1 * p1_p),
    )
    Phi2 = SA.SMatrix{2, 2, FT, 4}(
        muladd(w1, p2_m, w2 * p2_p),
        dd_p2 * b21,
        dd_p2 * b12,
        muladd(w2, p2_m, w1 * p2_p),
    )

    return (E, Phi1, Phi2)
end

@inline function metzler_2x2_exp_phi1_phi2(
    M::SA.SMatrix{2, 2},
    dt,
)
    return metzler_2x2_exp_phi1_phi2(-M[1, 1], -M[2, 2], M[2, 1], M[1, 2], dt)
end

# ============================================================================
# 3. Residence-limited supply and 2-step rational-seeded Newton inversion
# ============================================================================

"""
    residence_supply(D_e, q0, S_in_eff, D_sed, dt)

Integrated mass sink `E(D_e) = D_e dt [q0 φ₁(-(D_sed + D_e) dt) + S_in_eff dt φ₂(-(D_sed + D_e) dt)]`
over `dt` for a species with initial specific content `q0`, effective inflow/production
rate `S_in_eff`, sedimentation decay `D_sed`, and sink decay `D_e`. Bounded above
by `q0 + S_in_eff * dt` as `D_e → ∞`.
"""
@inline function residence_supply(
    D_e,
    q0,
    S_in_eff,
    D_sed,
    dt,
)
    FT = typeof(D_e)
    if (D_e <= zero(FT)) || (dt <= zero(FT)) || ((q0 <= zero(FT)) && (S_in_eff <= zero(FT)))
        return zero(FT)
    end
    y = D_e * dt
    x = max(zero(FT), D_sed * dt) + y
    p1, p2 = _phi1_phi2_neg(x)
    return y * muladd(q0, p1, (S_in_eff * dt) * p2)
end

"""
    invert_residence_supply(target_raw, q0, S_in_eff, D_sed, dt)

Invert `residence_supply(D_e, q0, S_in_eff, D_sed, dt) = target` for `D_e ≥ 0`,
where `target = min(target_raw, (1 - 16 eps(FT)) * (q0 + S_in_eff * dt))`.
Starts from the Michaelis-Menten rational seed matching `E'(0)` and `E(∞)` and
applies 2 safeguarded scalar Newton steps on the strictly increasing concave `E`.
"""
@noinline function invert_residence_supply(
    target_raw,
    q0,
    S_in_eff,
    D_sed,
    dt,
)
    FT = typeof(target_raw)
    S_dt = max(zero(FT), S_in_eff * dt)
    q0_pos = max(zero(FT), q0)
    Q_sup = q0_pos + S_dt
    if !((target_raw > zero(FT)) & (Q_sup > floatmin(FT)) & (dt > zero(FT)))
        return zero(FT)
    end
    cap_frac = one(FT) - FT(16) * eps(FT)
    target = min(target_raw, cap_frac * Q_sup)

    x_s = max(zero(FT), D_sed * dt)
    f = clamp(target / Q_sup, zero(FT), cap_frac)
    one_minus_f = max(one(FT) - f, floatmin(FT))
    neg_log_1mf = if FT === Float32
        f <= 0.25f0 ?
        f * evalpoly(f, (1.0f0, 0.5f0, 1.0f0 / 3, 0.25f0, 0.2f0, 1.0f0 / 6)) :
        -log(one_minus_f)
    else
        -log1p(-f)
    end
    y_q0 = neg_log_1mf + x_s * (f / one_minus_f)
    y_S = (f * (FT(2) - f + x_s)) / one_minus_f
    y = muladd(q0_pos, y_q0, S_dt * y_S) / Q_sup

    y_lo = zero(FT)
    y_hi = floatmax(FT)
    @inbounds for _ in 1:2
        x = x_s + y
        e_x, p1_x, p2_x = _exp_phi1_phi2_neg(x)
        val = y * muladd(q0_pos, p1_x, S_dt * p2_x)
        p1_psi, p2_psi = if x <= _phi_poly_cutoff(FT)
            (_psi1_poly(x), _psi2_poly(x))
        else
            inv_x = one(FT) / x
            psi1 = max(zero(FT), p1_x - e_x) * inv_x
            (psi1, max(zero(FT), p2_x - psi1) * inv_x)
        end
        dval = muladd(
            q0_pos,
            muladd(x_s, p1_psi, e_x),
            S_dt * muladd(x_s, p2_psi, p1_psi),
        )
        converged =
            abs(val - target) <= FT(4) * eps(FT) * max(target, floatmin(FT))
        y_lo = ifelse(val <= target, max(y_lo, y), y_lo)
        y_hi = ifelse(val >= target, min(y_hi, y), y_hi)
        y_cand = y - (val - target) / max(dval, floatmin(FT))
        in_bracket = (y_cand > y_lo) & (y_cand < y_hi)
        y_fallback = ifelse(
            y_hi < floatmax(FT) * FT(0.5),
            FT(0.5) * (y_lo + y_hi),
            max(FT(2) * y, y_lo),
        )
        y = ifelse(converged, y, ifelse(in_bracket, y_cand, y_fallback))
    end
    return max(zero(FT), y / dt)
end

# ============================================================================
# 4. 1M Parameter Combination and Terminal Velocity Update Helpers
# ============================================================================

@inline _something_or(::Nothing, b) = b
@inline _something_or(a, _) = a

"""
    _combined_microphysics_1m_params(ps)

Return a unified `CMP.Microphysics1MParams` combining the bulk 1M process options
with any phase-change options stored on `microphysics_1m_implicit_params`.
"""
@inline function _combined_microphysics_1m_params(ps)
    mp = CAP.microphysics_1m_params(ps)
    hasproperty(ps, :microphysics_1m_implicit_params) || return mp
    imp = ps.microphysics_1m_implicit_params
    isnothing(imp) && return mp
    b_p = mp.processes
    i_p = imp.processes
    clf = _something_or(b_p.cloud_liquid_formation, i_p.cloud_liquid_formation)
    cif = _something_or(b_p.cloud_ice_formation, i_p.cloud_ice_formation)
    rce = _something_or(
        b_p.rain_condensation_evaporation,
        i_p.rain_condensation_evaporation,
    )
    sds = _something_or(
        b_p.snow_deposition_sublimation,
        i_p.snow_deposition_sublimation,
    )
    if clf === b_p.cloud_liquid_formation &&
       cif === b_p.cloud_ice_formation &&
       rce === b_p.rain_condensation_evaporation &&
       sds === b_p.snow_deposition_sublimation
        return mp
    end
    processes = CMP.Microphysics1MOptions(;
        cloud_liquid_formation = clf,
        cloud_ice_formation = cif,
        cloud_ice_melt = b_p.cloud_ice_melt,
        cloud_liquid_freezing = b_p.cloud_liquid_freezing,
        rain_autoconversion = b_p.rain_autoconversion,
        snow_autoconversion = b_p.snow_autoconversion,
        rain_condensation_evaporation = rce,
        snow_deposition_sublimation = sds,
        snow_melt = b_p.snow_melt,
        cloud_liquid_rain_accretion = b_p.cloud_liquid_rain_accretion,
        cloud_liquid_snow_accretion = b_p.cloud_liquid_snow_accretion,
        cloud_ice_rain_accretion = b_p.cloud_ice_rain_accretion,
        cloud_ice_snow_accretion = b_p.cloud_ice_snow_accretion,
        rain_snow_accretion = b_p.rain_snow_accretion,
        vapor_relaxation = b_p.vapor_relaxation,
    )
    b_pp = mp.process_params
    i_pp = imp.process_params
    process_params = merge(
        b_pp,
        (;
            cloud_liquid_formation = _something_or(
                b_pp.cloud_liquid_formation,
                i_pp.cloud_liquid_formation,
            ),
            cloud_ice_formation = _something_or(
                b_pp.cloud_ice_formation,
                i_pp.cloud_ice_formation,
            ),
            rain_condensation_evaporation = _something_or(
                b_pp.rain_condensation_evaporation,
                i_pp.rain_condensation_evaporation,
            ),
            snow_deposition_sublimation = _something_or(
                b_pp.snow_deposition_sublimation,
                i_pp.snow_deposition_sublimation,
            ),
        ),
    )
    return CMP.Microphysics1MParams(;
        processes,
        process_params,
        cloud = mp.cloud,
        precip = mp.precip,
        air_properties = mp.air_properties,
        terminal_velocity = mp.terminal_velocity,
        max_latent_heating_rate = mp.max_latent_heating_rate,
    )
end

@inline function _without_cloud_formation(mp::CMP.Microphysics1MParams)
    procs = mp.processes
    pp = mp.process_params
    return CMP.Microphysics1MParams(;
        processes = CMP.Microphysics1MOptions(
            nothing,
            nothing,
            procs.cloud_ice_melt,
            procs.cloud_liquid_freezing,
            procs.rain_autoconversion,
            procs.snow_autoconversion,
            procs.rain_condensation_evaporation,
            procs.snow_deposition_sublimation,
            procs.snow_melt,
            procs.cloud_liquid_rain_accretion,
            procs.cloud_liquid_snow_accretion,
            procs.cloud_ice_rain_accretion,
            procs.cloud_ice_snow_accretion,
            procs.rain_snow_accretion,
            procs.vapor_relaxation,
        ),
        process_params = merge(
            pp,
            (; cloud_liquid_formation = nothing, cloud_ice_formation = nothing),
        ),
        cloud = mp.cloud,
        precip = mp.precip,
        air_properties = mp.air_properties,
        terminal_velocity = mp.terminal_velocity,
        max_latent_heating_rate = mp.max_latent_heating_rate,
    )
end

struct VelocityConfig1M{ML, MI, MR, MS, FT, TVC, CMP1}
    mode_lcl::ML
    mode_icl::MI
    mode_rai::MR
    mode_sno::MS
    fixed_w_lcl::FT
    fixed_w_icl::FT
    fixed_w_rai::FT
    fixed_w_sno::FT
    cmc_tv::TVC
    cmp::CMP1
end

@inline function _build_velocity_config_1m(atmos, params, dev, cmp)
    cmc = CAP.microphysics_cloud_params(params)
    tv = (; cmc.liquid, cmc.ice, cmc.stokes, cmc.Ch2022)
    return VelocityConfig1M(
        atmos.terminal_velocity_liquid,
        atmos.terminal_velocity_ice,
        atmos.terminal_velocity_rain,
        atmos.terminal_velocity_snow,
        CAP.fixed_cloud_liquid_terminal_velocity(params),
        CAP.fixed_cloud_ice_terminal_velocity(params),
        CAP.fixed_rain_terminal_velocity(params),
        CAP.fixed_snow_terminal_velocity(params),
        _fold(dev, tv),
        _fold(dev, cmp),
    )
end

@inline _tv_1m_lcl(vc::VelocityConfig1M{FixedTerminalVelocity}, ρ, q) =
    vc.fixed_w_lcl
@inline function _tv_1m_lcl(
    vc::VelocityConfig1M{StokesTerminalVelocity},
    ρ,
    q,
)
    FT = typeof(ρ)
    q > FT(CMU.ϵ_numerics(FT)) || return zero(FT)
    cmc = _unwrapped(vc.cmc_tv)
    return CMNe.terminal_velocity(cmc.liquid, cmc.stokes, ρ, q)
end

@inline _tv_1m_icl(
    vc::VelocityConfig1M{<:Any, FixedTerminalVelocity},
    ρ,
    q,
) = vc.fixed_w_icl
@inline function _tv_1m_icl(
    vc::VelocityConfig1M{<:Any, Chen2022TerminalVelocity},
    ρ,
    q,
)
    FT = typeof(ρ)
    q > FT(CMU.ϵ_numerics(FT)) || return zero(FT)
    cmc = _unwrapped(vc.cmc_tv)
    return CMNe.terminal_velocity(cmc.ice, cmc.Ch2022.small_ice, ρ, q)
end

@inline _tv_1m_rai(
    vc::VelocityConfig1M{<:Any, <:Any, FixedTerminalVelocity},
    ρ,
    q,
) = vc.fixed_w_rai
@inline function _tv_1m_rai(
    vc::VelocityConfig1M{<:Any, <:Any, Blk1MTerminalVelocity},
    ρ,
    q,
)
    FT = typeof(ρ)
    q > FT(CMU.ϵ_numerics(FT)) || return zero(FT)
    cmp = _unwrapped(vc.cmp)
    return CM1.terminal_velocity(
        cmp.precip.rain,
        cmp.terminal_velocity.rain,
        ρ,
        max(q, FT(1e-8)),
    )
end
@inline function _tv_1m_rai(
    vc::VelocityConfig1M{<:Any, <:Any, Chen2022TerminalVelocity},
    ρ,
    q,
)
    FT = typeof(ρ)
    q > FT(CMU.ϵ_numerics(FT)) || return zero(FT)
    cmp = _unwrapped(vc.cmp)
    cmc = _unwrapped(vc.cmc_tv)
    return CM1.terminal_velocity(
        cmp.precip.rain,
        cmc.Ch2022.rain,
        ρ,
        max(q, FT(1e-8)),
    )
end

@inline _tv_1m_sno(
    vc::VelocityConfig1M{<:Any, <:Any, <:Any, FixedTerminalVelocity},
    ρ,
    q,
) = vc.fixed_w_sno
@inline function _tv_1m_sno(
    vc::VelocityConfig1M{<:Any, <:Any, <:Any, Blk1MTerminalVelocity},
    ρ,
    q,
)
    FT = typeof(ρ)
    q > FT(CMU.ϵ_numerics(FT)) || return zero(FT)
    cmp = _unwrapped(vc.cmp)
    return CM1.terminal_velocity(
        cmp.precip.snow,
        cmp.terminal_velocity.snow,
        ρ,
        q,
    )
end
@inline function _tv_1m_sno(
    vc::VelocityConfig1M{<:Any, <:Any, <:Any, Chen2022TerminalVelocity},
    ρ,
    q,
)
    FT = typeof(ρ)
    q > FT(CMU.ϵ_numerics(FT)) || return zero(FT)
    cmp = _unwrapped(vc.cmp)
    cmc = _unwrapped(vc.cmc_tv)
    return CM1.terminal_velocity(cmp.precip.snow, cmc.Ch2022.large_ice, ρ, q)
end

@inline function _eval_tv_1m(
    tv_fn::F,
    vc::VelocityConfig1M,
    ρ::FT,
    q::FT,
    q_scale::FT,
    ::Val{UseShaft},
    β_p::FT,
    a_s::FT,
) where {F, FT <: AbstractFloat, UseShaft}
    if UseShaft
        q_pos = max(zero(FT), q)
        φ_out = one(FT) - β_p
        φ_in = φ_out + β_p / a_s
        w_in = tv_fn(vc, ρ, φ_in * q_pos)
        w_out = tv_fn(vc, ρ, φ_out * q_pos)
        return a_s * φ_in * w_in + (one(FT) - a_s) * φ_out * w_out
    end
    return tv_fn(vc, ρ, q * q_scale)
end

"""
    _two_step_sed_velocity(tv_fn, vc, ρ, q0, S_in_eff, w_above, dz_eff, dt)

Run 2 fixed-point updates of `(w_k, q̄_sed)` seeded from `w_above` and
`S_in,eff` with `q̄_sed = q0 φ₁(-C_sed) + S_in,eff dt φ₂(-C_sed)`,
`C_sed = w_k dt / dz_eff`.
"""
@inline _two_step_sed_velocity(
    tv_fn::F,
    vc::VelocityConfig1M,
    ρ::FT,
    q0::FT,
    S_in_eff::FT,
    w_above::FT,
    dz_eff::FT,
    dt::FT,
) where {F, FT <: AbstractFloat} = _two_step_sed_velocity(
    tv_fn,
    vc,
    ρ,
    q0,
    S_in_eff,
    w_above,
    dz_eff,
    dt,
    one(FT),
    Val(false),
    zero(FT),
    one(FT),
)

@inline function _two_step_sed_velocity(
    tv_fn::F,
    vc::VelocityConfig1M,
    ρ::FT,
    q0::FT,
    S_in_eff::FT,
    w_above::FT,
    dz_eff::FT,
    dt::FT,
    q_scale::FT,
    use_shaft::Val,
    β_p::FT,
    a_s::FT,
) where {F, FT <: AbstractFloat}
    S_dt = max(zero(FT), S_in_eff * dt)
    q0_pos = max(zero(FT), q0)
    q_sup_max = q0_pos + S_dt
    if q_sup_max <= zero(FT)
        return (
            max(
                zero(FT),
                _eval_tv_1m(tv_fn, vc, ρ, zero(FT), q_scale, use_shaft, β_p, a_s),
            ),
            zero(FT),
        )
    end
    if q_sup_max <= FT(CMU.ϵ_numerics(FT))
        w_sup = max(
            zero(FT),
            _eval_tv_1m(tv_fn, vc, ρ, q_sup_max, q_scale, use_shaft, β_p, a_s),
        )
        if w_sup <= zero(FT)
            return (zero(FT), muladd(FT(0.5), S_dt, q0_pos))
        end
    end
    inv_dz = one(FT) / max(dz_eff, floatmin(FT))
    w = max(zero(FT), w_above)
    @inbounds for _ in 1:2
        C_sed = w * dt * inv_dz
        p1, p2 = _phi1_phi2_neg(C_sed)
        q_bar = muladd(q0_pos, p1, S_dt * p2)
        w = max(
            zero(FT),
            _eval_tv_1m(tv_fn, vc, ρ, q_bar, q_scale, use_shaft, β_p, a_s),
        )
    end
    C_sed = w * dt * inv_dz
    p1, p2 = _phi1_phi2_neg(C_sed)
    q_bar = muladd(q0_pos, p1, S_dt * p2)
    return (w, q_bar)
end

# ============================================================================
# 5. Warm-air gated 1M process rates, joint vapor + thermal melt relaxation,
#    and coupled 2 × 2 Metzler cell solver
# ============================================================================

@inline function _ice_dep_limits(
    opt::CMP.TemperatureDependentIceNumber,
    mp::CMP.Microphysics1MParams,
)
    fit = CMP.process_params(mp, opt)
    return (; fit.T_dep_max, fit.T_vap_nuc)
end
@inline _ice_dep_limits(_, _) = (;)

@inline _liq_hom_temp(
    opt::CMP.CloudLiquidFormation,
    mp::CMP.Microphysics1MParams,
    ::Type{FT},
) where {FT} = CMP.process_params(mp, opt).T_hom
@inline _liq_hom_temp(_, _, ::Type{FT}) where {FT} = FT(-Inf)

# TODO: remove or move into the test suite. The cell step calls
# `_coupled_1m_source_terms_with_qsat` directly; only
# `coupled_microphysics_sweep_tests.jl` uses this entry point.
"""
    coupled_1m_source_terms(mp, tps, ρ, T, w, q_v0, q_lcl, q_icl, q_rai, q_sno)

Evaluate 1-moment microphysics rates through public `CloudMicrophysics` rate
functions with the warm-air rain-ice gate (`T < T_freeze` for `:q_icl + :q_rai → :q_sno`,
routing warm `T ≥ T_freeze` cloud-ice + rain collisions to `:q_rai`).
`q_v0` is the ambient water-vapor specific content of the cell, kept invariant
when `q_rai` and `q_sno` hold residence-time shaft averages `q̄_sed`.
"""
@inline function coupled_1m_source_terms(
    mp::CMP.Microphysics1MParams,
    tps,
    ρ,
    T,
    w,
    q_v0,
    q_lcl,
    q_icl,
    q_rai,
    q_sno,
)
    FT = typeof(ρ)
    return _coupled_1m_source_terms_with_qsat(
        mp,
        tps,
        ρ,
        T,
        w,
        q_v0,
        q_lcl,
        q_icl,
        q_rai,
        q_sno,
        FT(-1),
        FT(-1),
    )
end

@inline function _coupled_1m_source_terms_with_qsat(
    mp::CMP.Microphysics1MParams,
    tps,
    ρ,
    T,
    w,
    q_v0,
    q_lcl,
    q_icl,
    q_rai,
    q_sno,
    q_sl_in,
    q_si_in,
)
    FT = typeof(ρ)
    ρ_pos = max(zero(FT), ρ)
    q_lcl_p = max(zero(FT), q_lcl)
    q_icl_p = max(zero(FT), q_icl)
    q_rai_p = max(zero(FT), q_rai)
    q_sno_p = max(zero(FT), q_sno)
    q_v_p = max(zero(FT), q_v0)
    q_tot_eff = q_v_p + q_lcl_p + q_icl_p + q_rai_p + q_sno_p

    procs = mp.processes
    micro = (;
        q_tot = q_tot_eff,
        q_lcl = q_lcl_p,
        q_icl = q_icl_p,
        q_rai = q_rai_p,
        q_sno = q_sno_p,
    )
    thermo = (; ρ = ρ_pos, T, w)

    ϵ_cm = FT(CMU.ϵ_numerics(FT))
    has_lcl = q_lcl_p > zero(FT)
    has_icl = q_icl_p > zero(FT)
    has_rai = q_rai_p > ϵ_cm
    has_sno = q_sno_p > ϵ_cm

    q_sl =
        q_sl_in >= zero(FT) ? q_sl_in :
        TD.q_vap_saturation(tps, T, ρ_pos, TD.Liquid())
    T_triple = FT(TD.Parameters.T_triple(tps))
    q_si = if q_si_in >= zero(FT)
        q_si_in
    elseif (T >= T_triple) & !has_icl & !has_sno
        q_sl
    else
        TD.q_vap_saturation(tps, T, ρ_pos, TD.Ice())
    end
    q_tol = eps(FT) * max(q_sl, q_si, floatmin(FT))

    T_hom = _liq_hom_temp(procs.cloud_liquid_formation, mp, FT)
    S_vap_lcl, τ_vap_lcl = if procs.cloud_liquid_formation isa
       CMP.CloudLiquidFormation
        pp_lcl = CMP.process_params(mp, procs.cloud_liquid_formation)
        τ_l = FT(pp_lcl.τ_relax)
        if !has_lcl & ((q_v_p <= q_sl) | (T < T_hom))
            (zero(FT), τ_l)
        else
            L_v_cm = TD.latent_heat_vapor(tps, T)
            c_pm_cm = TD.cp_m(tps, q_tot_eff, q_lcl_p + q_rai_p, q_icl_p + q_sno_p)
            dqsl_cm = CMNe.dqcld_dT(q_sl, L_v_cm, FT(TD.Parameters.R_v(tps)), T)
            Γ_l_cm = CMNe.gamma_helper(L_v_cm, c_pm_cm, dqsl_cm)
            sat_ex_l = q_v_p - q_sl
            tend_l =
                ifelse(
                    sat_ex_l < zero(FT),
                    -min(-sat_ex_l, Γ_l_cm * q_lcl_p),
                    sat_ex_l,
                ) / (τ_l * Γ_l_cm)
            S_l = ifelse((T < T_hom) & (tend_l > zero(FT)), zero(FT), tend_l)
            (S_l, τ_l)
        end
    else
        (
            CMNe.conv_q_vap_to_q_lcl(
                procs.cloud_liquid_formation,
                mp,
                tps,
                micro,
                thermo,
            ),
            CMNe.τ_vap_to_q_lcl(
                procs.cloud_liquid_formation,
                mp,
                tps,
                micro,
                thermo,
            ),
        )
    end

    icl_limits = _ice_dep_limits(procs.cloud_ice_formation, mp)
    S_vap_icl, τ_vap_icl = if !has_icl & (q_v_p < q_si - q_tol)
        # `_cloud_relax_coeff` returns zero for `S_vap_icl == 0` when
        # `|δ_i0| > q_tol`.
        (zero(FT), FT(Inf))
    elseif procs.cloud_ice_formation isa CMP.TemperatureDependentIceNumber
        sat_ex_i = q_v_p - q_si
        blocked_i = CMNe.INP_limiter(sat_ex_i, tps, T; q_lcl = q_lcl_p, icl_limits...)
        if blocked_i | (!has_icl & (sat_ex_i <= zero(FT)))
            τ_edge = if !has_icl & (abs(sat_ex_i) <= q_tol)
                fit_i = CMP.process_params(mp, procs.cloud_ice_formation)
                τ_e = FT(
                    CMNe.τ_relax(
                        mp.cloud.ice,
                        mp.air_properties,
                        fit_i,
                        q_icl_p,
                        T,
                        ρ_pos,
                    ),
                )
                ifelse(blocked_i, oftype(τ_e, Inf), τ_e)
            else
                FT(Inf)
            end
            (zero(FT), τ_edge)
        else
            fit_i = CMP.process_params(mp, procs.cloud_ice_formation)
            τ_raw = FT(
                CMNe.τ_relax(
                    mp.cloud.ice,
                    mp.air_properties,
                    fit_i,
                    q_icl_p,
                    T,
                    ρ_pos,
                ),
            )
            L_s_cm = TD.latent_heat_sublim(tps, T)
            c_pm_cm = TD.cp_m(tps, q_tot_eff, q_lcl_p + q_rai_p, q_icl_p + q_sno_p)
            dqsi_cm = CMNe.dqcld_dT(q_si, L_s_cm, FT(TD.Parameters.R_v(tps)), T)
            Γ_i_cm = CMNe.gamma_helper(L_s_cm, c_pm_cm, dqsi_cm)
            S_i =
                ifelse(
                    sat_ex_i < zero(FT),
                    -min(-sat_ex_i, Γ_i_cm * q_icl_p),
                    sat_ex_i,
                ) / (τ_raw * Γ_i_cm)
            (S_i, τ_raw)
        end
    else
        (
            CMNe.conv_q_vap_to_q_icl(
                procs.cloud_ice_formation,
                mp,
                tps,
                micro,
                thermo,
            ),
            CMNe.τ_vap_to_q_icl(
                procs.cloud_ice_formation,
                mp,
                tps,
                micro,
                thermo,
            ),
        )
    end
    icl_sink_only = CMNe.INP_limiter(
        one(FT),
        tps,
        T;
        q_lcl = q_lcl_p,
        icl_limits...,
    )

    S_acnv_lcl_rai = if has_lcl
        CM1.conv_q_lcl_to_q_rai(
            procs.rain_autoconversion,
            mp,
            tps,
            micro,
            thermo,
        )
    else
        zero(FT)
    end

    T_freeze = FT(TD.Parameters.T_freeze(tps))
    is_warm = T >= T_freeze
    is_strictly_warm = T > T_freeze

    if !(has_icl | has_rai | has_sno)
        z = zero(FT)
        S_freeze_lcl_icl = if has_lcl && !is_warm
            CM1.conv_q_lcl_to_q_icl(
                procs.cloud_liquid_freezing,
                mp,
                tps,
                micro,
                thermo,
            )
        else
            z
        end
        return (;
            S_vap_lcl,
            τ_vap_lcl,
            S_vap_icl,
            τ_vap_icl,
            icl_sink_only,
            S_acnv_lcl_rai,
            S_acnv_icl_sno = z,
            S_accr_lcl_rai = z,
            S_accr_lcl_sno_cold = z,
            S_accr_lcl_sno_warm = z,
            S_accr_melt_lcl_sno = z,
            S_accr_icl_rai_cold = z,
            S_accr_icl_rai_warm = z,
            S_accr_freeze_icl_rai = z,
            S_accr_icl_sno = z,
            S_accr_rai_sno_cold = z,
            S_accr_melt_rai_sno = z,
            S_vap_rai = z,
            S_vap_sno = z,
            c_rai_raw = z,
            c_sno_raw = z,
            S_melt_icl_lcl = z,
            S_melt_sno_rai = z,
            S_freeze_lcl_icl,
            q_sl,
            q_si,
            q_tol,
        )
    end

    sd = (;
        λ_inv_rai = has_rai ?
                    CM1.lambda_inverse(
            mp.precip.rain.pdf,
            mp.precip.rain.mass,
            q_rai_p,
            ρ_pos,
        ) : zero(FT),
        n0_rai = has_rai ? CM1.get_n0(mp.precip.rain.pdf, q_rai_p, ρ_pos) :
                 zero(FT),
        v0_rai = has_rai ? CM1.get_v0(mp.terminal_velocity.rain, ρ_pos) :
                 zero(FT),
        λ_inv_sno = has_sno ?
                    CM1.lambda_inverse(
            mp.precip.snow.pdf,
            mp.precip.snow.mass,
            q_sno_p,
            ρ_pos,
        ) : zero(FT),
        n0_sno = has_sno ? CM1.get_n0(mp.precip.snow.pdf, q_sno_p, ρ_pos) :
                 zero(FT),
        v0_sno = has_sno ? CM1.get_v0(mp.terminal_velocity.snow, ρ_pos) :
                 zero(FT),
        λ_inv_icl = has_icl ?
                    CM1.lambda_inverse(
            mp.cloud.ice.pdf,
            mp.cloud.ice.mass,
            q_icl_p,
            ρ_pos,
        ) : zero(FT),
        n0_icl = CM1.get_n0(mp.cloud.ice.pdf),
    )

    S_acnv_icl_sno = if has_icl
        CM1.conv_q_icl_to_q_sno(
            procs.snow_autoconversion,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
    else
        zero(FT)
    end

    S_accr_lcl_rai = if has_lcl & has_rai
        CM1.accretion(
            procs.cloud_liquid_rain_accretion,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
    else
        zero(FT)
    end

    lcl_sno = if has_lcl & has_sno & !isnothing(procs.cloud_liquid_snow_accretion)
        CM1.accretion(
            procs.cloud_liquid_snow_accretion,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
    else
        (; S_accr = zero(FT), S_melt = zero(FT))
    end
    S_accr_lcl_sno_cold = ifelse(is_warm, zero(FT), lcl_sno.S_accr)
    S_accr_lcl_sno_warm = ifelse(is_warm, lcl_sno.S_accr, zero(FT))
    S_accr_melt_lcl_sno = ifelse(is_warm, lcl_sno.S_melt, zero(FT))

    T_snow = mp.temperature_rain_collision_snow
    f_liq = ifelse(
        is_warm,
        one(FT),
        ifelse(
            T <= T_snow,
            zero(FT),
            (T - T_snow) / max(T_freeze - T_snow, eps(FT)),
        ),
    )

    # Warm-air rain-ice gate: cold collisions form snow and freeze `1 - f_liq`
    # of collected rain; warm collisions (`T ≥ T_freeze`) melt ice into rain.
    if has_icl & has_rai
        S_icl_rai_raw = CM1.accretion(
            procs.cloud_ice_rain_accretion,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
        S_accr_icl_rai_cold = ifelse(is_warm, zero(FT), S_icl_rai_raw)
        S_accr_icl_rai_warm = ifelse(is_warm, S_icl_rai_raw, zero(FT))
        S_accr_freeze_icl_rai = if !is_warm
            (one(FT) - f_liq) * CM1.accretion_rain_sink(
                procs.cloud_ice_rain_accretion,
                mp,
                tps,
                micro,
                thermo,
                sd,
            )
        else
            zero(FT)
        end
    else
        S_accr_icl_rai_cold = zero(FT)
        S_accr_icl_rai_warm = zero(FT)
        S_accr_freeze_icl_rai = zero(FT)
    end

    S_accr_icl_sno = if has_icl & has_sno
        CM1.accretion(
            procs.cloud_ice_snow_accretion,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
    else
        zero(FT)
    end

    if has_rai & has_sno
        rai_sno = CM1.accretion_snow_rain(
            procs.rain_snow_accretion,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
        S_accr_rai_sno_cold =
            ifelse(is_warm, zero(FT), (one(FT) - f_liq) * rai_sno.S_rai_sno)
        S_accr_melt_rai_sno = ifelse(is_warm, rai_sno.S_melt, zero(FT))
    else
        S_accr_rai_sno_cold = zero(FT)
        S_accr_melt_rai_sno = zero(FT)
    end

    if has_rai
        S_vap_rai = CM1.conv_q_rai_to_q_vap(
            procs.rain_condensation_evaporation,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
        # Rain evaporation slope is probed near saturation so WBF sign flips
        # retain the physical ventilation coefficient with at most one probe.
        c_rai_raw = if q_v_p - q_sl < -FT(4) * q_tol
            max(zero(FT), S_vap_rai / (q_v_p - q_sl))
        else
            δ_probe_l = -max(FT(1e-3) * q_sl, FT(16) * q_tol)
            micro_probe_r = (;
                q_tot = max(zero(FT), q_sl + δ_probe_l) +
                        q_lcl_p +
                        q_icl_p +
                        q_rai_p +
                        q_sno_p,
                q_lcl = q_lcl_p,
                q_icl = q_icl_p,
                q_rai = q_rai_p,
                q_sno = q_sno_p,
            )
            S_vap_rai_probe = CM1.conv_q_rai_to_q_vap(
                procs.rain_condensation_evaporation,
                mp,
                tps,
                micro_probe_r,
                thermo,
                sd,
            )
            max(zero(FT), S_vap_rai_probe / δ_probe_l)
        end
    else
        S_vap_rai = zero(FT)
        c_rai_raw = zero(FT)
    end

    if has_sno
        S_vap_sno = CM1.conv_q_sno_to_q_vap(
            procs.snow_deposition_sublimation,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
        c_sno_raw = if q_v_p - q_si < -FT(4) * q_tol
            max(zero(FT), S_vap_sno / (q_v_p - q_si))
        elseif (q_v_p - q_si > FT(4) * q_tol) &&
               !is_warm &&
               !CM1._snow_deposition_blocked(
                   procs.snow_deposition_sublimation,
                   tps,
                   micro,
                   thermo,
               )
            max(zero(FT), S_vap_sno / (q_v_p - q_si))
        else
            δ_probe_i = -max(FT(1e-3) * q_si, FT(16) * q_tol)
            micro_probe_s = (;
                q_tot = max(zero(FT), q_si + δ_probe_i) +
                        q_lcl_p +
                        q_icl_p +
                        q_rai_p +
                        q_sno_p,
                q_lcl = q_lcl_p,
                q_icl = q_icl_p,
                q_rai = q_rai_p,
                q_sno = q_sno_p,
            )
            S_vap_sno_probe = CM1.conv_q_sno_to_q_vap(
                procs.snow_deposition_sublimation,
                mp,
                tps,
                micro_probe_s,
                thermo,
                sd,
            )
            max(zero(FT), S_vap_sno_probe / δ_probe_i)
        end
    else
        S_vap_sno = zero(FT)
        c_sno_raw = zero(FT)
    end

    S_melt_icl_lcl = if has_icl && is_strictly_warm
        CM1.conv_q_icl_to_q_lcl(
            procs.cloud_ice_melt,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
    else
        zero(FT)
    end
    S_melt_sno_rai = if has_sno && is_strictly_warm
        CM1.conv_q_sno_to_q_rai(
            procs.snow_melt,
            mp,
            tps,
            micro,
            thermo,
            sd,
        )
    else
        zero(FT)
    end
    S_freeze_lcl_icl = if has_lcl && !is_warm
        CM1.conv_q_lcl_to_q_icl(
            procs.cloud_liquid_freezing,
            mp,
            tps,
            micro,
            thermo,
        )
    else
        zero(FT)
    end

    return (;
        S_vap_lcl,
        τ_vap_lcl,
        S_vap_icl,
        τ_vap_icl,
        icl_sink_only,
        S_acnv_lcl_rai,
        S_acnv_icl_sno,
        S_accr_lcl_rai,
        S_accr_lcl_sno_cold,
        S_accr_lcl_sno_warm,
        S_accr_melt_lcl_sno,
        S_accr_icl_rai_cold,
        S_accr_icl_rai_warm,
        S_accr_freeze_icl_rai,
        S_accr_icl_sno,
        S_accr_rai_sno_cold,
        S_accr_melt_rai_sno,
        S_vap_rai,
        S_vap_sno,
        c_rai_raw,
        c_sno_raw,
        S_melt_icl_lcl,
        S_melt_sno_rai,
        S_freeze_lcl_icl,
        q_sl,
        q_si,
        q_tol,
    )
end

@inline function _mm15_averaged_excess(δ0, A, invτ, dt)
    FT = typeof(A)
    x = invτ * dt
    x > eps(FT) || return δ0
    A_τ = (A * dt) / x
    return muladd(δ0 - A_τ, _phi1_neg(x), A_τ)
end

@inline function _mm15_joint_excesses(
    c_l,
    c_r,
    c_i,
    c_s,
    Γ_l,
    Γ_i,
    κ_il,
    κ_li,
    δ_l,
    δ_i,
    Δs,
    dt,
)
    FT = typeof(c_l)
    a = c_l + c_r
    b = c_i + c_s
    liq_primary = Γ_l * a >= Γ_i * b
    Γ_p = ifelse(liq_primary, Γ_l, Γ_i)
    κ = ifelse(liq_primary, κ_il, κ_li)
    C_p = ifelse(liq_primary, a, b)
    C_s = ifelse(liq_primary, b, a)
    δ_p = ifelse(liq_primary, δ_l, δ_i)
    σ = ifelse(liq_primary, one(FT), -one(FT))
    invτ = muladd(Γ_p, C_p, κ * C_s)
    A = -σ * Δs * κ * C_s
    δ_bar_p = _mm15_averaged_excess(δ_p, A, invτ, dt)
    δ_bar_s = δ_bar_p + σ * Δs
    return (
        ifelse(liq_primary, δ_bar_p, δ_bar_s),
        ifelse(liq_primary, δ_bar_s, δ_bar_p),
    )
end

@inline function _cloud_relax_coeff(
    S,
    τ,
    Γ_cm,
    δ,
    tol,
)
    FT = typeof(S)
    τ_c = clamp(τ, eps(FT), floatmax(FT))
    c = ifelse(τ < floatmax(FT), one(FT) / (τ_c * Γ_cm), zero(FT))
    off = (S == zero(FT)) & (abs(δ) > tol)
    return ifelse(off, zero(FT), c)
end

"""
    _coupled_thermal_melt_and_vapor(
        C_l, C_i, Γ_l, Γ_i, κ_il, κ_li, L_vl_over_c, L_vi_over_c,
        dqsl_dT, ν_melt, Δθ_sup_max, δ_l0, δ_i0, Δs, θ0, dt,
    )

Advance the coupled `(δ_l, θ = T - T_freeze)` relaxation over `dt` when `θ0 > 0`
and `ν_melt > 0`. If evaporative and melting cooling drive `θ(t)` across `0`
within `(0, dt)`, or if cumulative melt cooling reaches the residence-bounded
ice/snow supply `Δθ_sup_max`, locates the arrest time `t*` in 2 safeguarded
scalar Newton steps and continues pure MM15 vapor relaxation over `[t*, dt]`.
Returns `(δ̄_l, δ̄_i, f_melt)`.
"""
@noinline function _coupled_thermal_melt_and_vapor(
    C_l,
    C_i,
    Γ_l,
    Γ_i,
    κ_il,
    κ_li,
    L_vl_over_c,
    L_vi_over_c,
    dqsl_dT,
    ν_melt,
    Δθ_sup_max,
    δ_l0,
    δ_i0,
    Δs,
    θ0,
    dt,
)
    FT = typeof(C_l)
    if !(
        θ0 > zero(FT) &&
        ν_melt > zero(FT) &&
        Δθ_sup_max > zero(FT) &&
        dt > zero(FT)
    )
        δ_bar_l, δ_bar_i = _mm15_joint_excesses(
            C_l,
            zero(FT),
            C_i,
            zero(FT),
            Γ_l,
            Γ_i,
            κ_il,
            κ_li,
            δ_l0,
            δ_i0,
            Δs,
            dt,
        )
        return (δ_bar_l, δ_bar_i, zero(FT))
    end

    # `2 × 2` Metzler system for `y = (δ_l, θ)` while `θ > 0`:
    #   dδ_l/dt = -(Γ_l C_l + κ_il C_i) δ_l + (dqsl_dT ν_melt) θ - κ_il C_i Δs
    #   dθ/dt   = (L_vl/c C_l + L_vi/c C_i) δ_l - ν_melt θ + (L_vi/c C_i Δs)
    D1 = muladd(Γ_l, C_l, κ_il * C_i)
    D2 = ν_melt
    D12 = muladd(L_vl_over_c, C_l, L_vi_over_c * C_i) # M[2, 1]
    D21 = dqsl_dT * ν_melt                            # M[1, 2]
    b1 = -κ_il * C_i * Δs
    b2 = L_vi_over_c * C_i * Δs
    y0 = SA.SVector{2, FT}(δ_l0, θ0)
    b_th = SA.SVector{2, FT}(b1, b2)

    det_exact = (C_l + C_i) * ν_melt * dt * dt
    E_full, Phi1_full, Phi2_full =
        metzler_2x2_exp_phi1_phi2(D1, D2, D12, D21, dt, det_exact)
    y_end = E_full * y0 + dt * (Phi1_full * b_th)
    y_bar_full = Phi1_full * y0 + dt * (Phi2_full * b_th)

    Δθ_melt_full = ν_melt * dt * max(zero(FT), y_bar_full[2])
    if (y_end[2] >= zero(FT)) & (Δθ_melt_full <= Δθ_sup_max)
        f_melt = clamp(
            max(zero(FT), y_bar_full[2]) / max(θ0, floatmin(FT)),
            zero(FT),
            one(FT),
        )
        return (y_bar_full[1], y_bar_full[1] + Δs, f_melt)
    end

    # Step 1: locate `θ(t_θ) = 0` in `(0, dt)` when `y_end[2] < 0`.
    t_θ = dt
    y_at_tθ = y_end
    y_bar_tθ = y_bar_full
    if y_end[2] < zero(FT)
        dθ0_dt = muladd(D12, δ_l0, -ν_melt * θ0 + b2)
        t_sec = dt * (θ0 / max(θ0 - y_end[2], floatmin(FT)))
        t_init = ifelse(
            dθ0_dt < -floatmin(FT),
            min(θ0 / (-dθ0_dt), t_sec),
            t_sec,
        )
        t_lo = zero(FT)
        t_hi = dt
        t = clamp(t_init, eps(FT) * dt, dt * (one(FT) - eps(FT)))
        @inbounds for _ in 1:2
            det_t = (C_l + C_i) * ν_melt * t * t
            E_t, Phi1_t, _ =
                metzler_2x2_exp_phi1_phi2(D1, D2, D12, D21, t, det_t)
            y_t = E_t * y0 + t * (Phi1_t * b_th)
            θ_t = y_t[2]
            dθ_dt = muladd(D12, y_t[1], -ν_melt * θ_t + b2)
            converged = abs(θ_t) <= FT(4) * eps(FT) * max(θ0, floatmin(FT))
            t_lo = ifelse(θ_t >= zero(FT), max(t_lo, t), t_lo)
            t_hi = ifelse(θ_t <= zero(FT), min(t_hi, t), t_hi)
            t_cand = t - θ_t / min(dθ_dt, -floatmin(FT))
            in_bracket = (t_cand > t_lo) & (t_cand < t_hi)
            t = ifelse(
                converged,
                t,
                ifelse(in_bracket, t_cand, FT(0.5) * (t_lo + t_hi)),
            )
        end
        t_θ = t
        det_θ = (C_l + C_i) * ν_melt * t_θ * t_θ
        E_θ, Phi1_θ, Phi2_θ =
            metzler_2x2_exp_phi1_phi2(D1, D2, D12, D21, t_θ, det_θ)
        y_at_tθ = E_θ * y0 + t_θ * (Phi1_θ * b_th)
        y_bar_tθ = Phi1_θ * y0 + t_θ * (Phi2_θ * b_th)
    end

    # Step 2: if cumulative melt cooling on `[0, t_θ]` exceeds `Δθ_sup_max`,
    # locate supply exhaustion `ν_melt * t_s * y_bar(t_s)[2] = Δθ_sup_max`.
    t_stop = t_θ
    y_star = y_at_tθ
    y_bar_1 = y_bar_tθ
    Δθ_melt_tθ = ν_melt * t_θ * max(zero(FT), y_bar_tθ[2])
    if Δθ_melt_tθ > Δθ_sup_max
        t_lo = zero(FT)
        t_hi = t_θ
        t_init = min(
            Δθ_sup_max / max(ν_melt * θ0, floatmin(FT)),
            t_θ * (Δθ_sup_max / max(Δθ_melt_tθ, floatmin(FT))),
        )
        t = clamp(t_init, eps(FT) * t_θ, t_θ * (one(FT) - eps(FT)))
        @inbounds for _ in 1:2
            det_t = (C_l + C_i) * ν_melt * t * t
            E_t, Phi1_t, Phi2_t =
                metzler_2x2_exp_phi1_phi2(D1, D2, D12, D21, t, det_t)
            y_t = E_t * y0 + t * (Phi1_t * b_th)
            yb_t = Phi1_t * y0 + t * (Phi2_t * b_th)
            g_t = ν_melt * t * max(zero(FT), yb_t[2]) - Δθ_sup_max
            dg_dt = ν_melt * max(y_t[2], floatmin(FT))
            converged =
                abs(g_t) <= FT(4) * eps(FT) * max(Δθ_sup_max, floatmin(FT))
            t_lo = ifelse(g_t <= zero(FT), max(t_lo, t), t_lo)
            t_hi = ifelse(g_t >= zero(FT), min(t_hi, t), t_hi)
            t_cand = t - g_t / max(dg_dt, floatmin(FT))
            in_bracket = (t_cand > t_lo) & (t_cand < t_hi)
            t = ifelse(
                converged,
                t,
                ifelse(in_bracket, t_cand, FT(0.5) * (t_lo + t_hi)),
            )
        end
        t_stop = t
        det_1 = (C_l + C_i) * ν_melt * t_stop * t_stop
        E_1, Phi1_1, Phi2_1 =
            metzler_2x2_exp_phi1_phi2(D1, D2, D12, D21, t_stop, det_1)
        y_star = E_1 * y0 + t_stop * (Phi1_1 * b_th)
        y_bar_1 = Phi1_1 * y0 + t_stop * (Phi2_1 * b_th)
    end

    int_θ_cap = Δθ_sup_max / max(ν_melt, floatmin(FT))
    int_θ = clamp(t_stop * max(zero(FT), y_bar_1[2]), zero(FT), int_θ_cap)
    f_melt = clamp(int_θ / max(θ0 * dt, floatmin(FT)), zero(FT), one(FT))

    # Continue pure MM15 vapor relaxation on `[t_stop, dt]`.
    dt_2 = max(zero(FT), dt - t_stop)
    θ_rem_frac = clamp(max(zero(FT), y_star[2]) / max(θ0, floatmin(FT)), zero(FT), one(FT))
    Δs_2 = Δs * θ_rem_frac
    δ_l_star = y_star[1]
    δ_i_star = δ_l_star + Δs_2
    δ_bar_l2, δ_bar_i2 = _mm15_joint_excesses(
        C_l,
        zero(FT),
        C_i,
        zero(FT),
        Γ_l,
        Γ_i,
        κ_il,
        κ_li,
        δ_l_star,
        δ_i_star,
        Δs_2,
        dt_2,
    )
    inv_dt = one(FT) / max(dt, floatmin(FT))
    δ_bar_l = muladd(t_stop, y_bar_1[1], dt_2 * δ_bar_l2) * inv_dt
    δ_bar_i = muladd(t_stop, y_bar_1[1] + Δs, dt_2 * δ_bar_i2) * inv_dt
    return (δ_bar_l, δ_bar_i, f_melt)
end

"""
    coupled_1m_cell_step(
        mp, tps, ρ, T0, w_air, q_tot0, q0, S_in, q_bar_sed, D_sed, dt;
        is_constant_p = false,
    )

Advance the four 1-moment condensate pools `q0 = SVector(q_lcl, q_icl, q_rai, q_sno)`
in one cell over `dt` with sedimentation inflows `S_in`, residence-averaged pools
`q_bar_sed`, and sedimentation decay rates `D_sed = w / dz_eff`.

Returns `(; q_new, q_out, α_cap, f_melt)` where `q_new ≥ 0` are the end-of-step
specific contents (per unit pre-step `ρ`) and `q_out = D_sed .* q_bar * dt ≥ 0`
are the specific masses exported through the cell's bottom face over `dt`.
"""
@inline function _solve_1m_cloud_matrices(
    D_sed_1,
    D_sed_2,
    D_lcl_rai,
    D_icl_sno,
    D_e_lcl_s,
    D_e_icl_s,
    D_lcl_icl_s,
    D_lcl_sno_s,
    D_icl_lcl_s,
    D_icl_rai_s,
    dt,
)
    D1_c = D_sed_1 + D_e_lcl_s + D_lcl_rai + D_lcl_icl_s + D_lcl_sno_s
    D2_c = D_sed_2 + D_e_icl_s + D_icl_sno + D_icl_lcl_s + D_icl_rai_s
    return metzler_2x2_exp_phi1_phi2(D1_c, D2_c, D_lcl_icl_s, D_icl_lcl_s, dt)
end

@inline function _solve_1m_cloud_apply(
    E_c::SA.SMatrix{2, 2},
    Phi1_c::SA.SMatrix{2, 2},
    Phi2_c::SA.SMatrix{2, 2},
    q0_1,
    q0_2,
    S_in_1,
    S_in_2,
    s_vap_lcl,
    s_vap_icl,
    α_src,
    dt,
)
    FT = typeof(q0_1)
    c0 = SA.SVector{2, FT}(q0_1, q0_2)
    b_c = SA.SVector{2, FT}(
        muladd(α_src, s_vap_lcl, S_in_1),
        muladd(α_src, s_vap_icl, S_in_2),
    )
    c_new = max.(E_c * c0 + dt * (Phi1_c * b_c), zero(FT))
    c_bar = max.(Phi1_c * c0 + dt * (Phi2_c * b_c), zero(FT))
    return (c_new, c_bar)
end

# Exact solve of the 2 × 2 rain-snow system over `dt` with sedimentation
# decays `D_sed`, vapor-sink decays `D_e`, the rain → snow and snow → rain
# transfer rates, initial contents `p0`, and constant sources `b_p`. Returns the
# end-of-step and time-mean contents `(p_new, p_bar)`.
@inline function _solve_1m_precip_block(
    p0::SA.SVector{2},
    b_p::SA.SVector{2},
    D_sed_3,
    D_sed_4,
    D_e_rai,
    D_e_sno,
    D_rai_sno,
    D_sno_rai,
    dt,
)
    FT = typeof(D_sed_3)
    D1_p = D_sed_3 + D_e_rai + D_rai_sno
    D2_p = D_sed_4 + D_e_sno + D_sno_rai
    E_p, Phi1_p, Phi2_p =
        metzler_2x2_exp_phi1_phi2(D1_p, D2_p, D_rai_sno, D_sno_rai, dt)
    p_new = max.(E_p * p0 + dt * (Phi1_p * b_p), zero(FT))
    p_bar = max.(Phi1_p * p0 + dt * (Phi2_p * b_p), zero(FT))
    return (p_new, p_bar)
end

@inline function _solve_1m_precip_step(
    q0_3,
    q0_4,
    b_rai_in,
    b_sno_in_base,
    D_sed_3,
    D_sed_4,
    D_rai_sno,
    s_vap_rai,
    s_vap_sno,
    Δq_vap_rai,
    Δq_vap_sno,
    target_melt_sno_base,
    is_warm::Bool,
    α_src,
    f_pc,
    dt,
)
    FT = typeof(q0_3)
    b_sno_in = muladd(α_src, s_vap_sno, b_sno_in_base)
    D_rai_sno_s = f_pc * D_rai_sno
    # Each sink rate is inverted against the supply of its species, which
    # includes what the other precipitation species transfers to it within the
    # step: snow melt feeds rain in warm cells, and collisional freezing of
    # rain feeds snow in cold cells. The transfer enters as its step-mean rate.
    target_melt_sno = f_pc * target_melt_sno_base
    target_evap_rai = max(zero(FT), -f_pc * Δq_vap_rai)
    target_subl_sno = max(zero(FT), -f_pc * Δq_vap_sno)
    if is_warm && (target_melt_sno > zero(FT))
        D_e_sno_s =
            target_subl_sno > zero(FT) ?
            invert_residence_supply(
                target_subl_sno, q0_4, b_sno_in, D_sed_4, dt,
            ) : zero(FT)
        D_sno_rai_s = invert_residence_supply(
            target_melt_sno, q0_4, b_sno_in, D_sed_4 + D_e_sno_s, dt,
        )
        b_melt =
            residence_supply(
                D_sno_rai_s, q0_4, b_sno_in, D_sed_4 + D_e_sno_s, dt,
            ) / dt
        D_e_rai_s =
            target_evap_rai > zero(FT) ?
            invert_residence_supply(
                target_evap_rai,
                q0_3,
                b_rai_in + b_melt,
                D_sed_3 + D_rai_sno_s,
                dt,
            ) : zero(FT)
    else
        D_sno_rai_s = zero(FT)
        D_e_rai_s =
            target_evap_rai > zero(FT) ?
            invert_residence_supply(
                target_evap_rai, q0_3, b_rai_in, D_sed_3 + D_rai_sno_s, dt,
            ) : zero(FT)
        b_freeze =
            residence_supply(
                D_rai_sno_s, q0_3, b_rai_in, D_sed_3 + D_e_rai_s, dt,
            ) / dt
        D_e_sno_s =
            target_subl_sno > zero(FT) ?
            invert_residence_supply(
                target_subl_sno, q0_4, b_sno_in + b_freeze, D_sed_4, dt,
            ) : zero(FT)
    end

    p0 = SA.SVector{2, FT}(q0_3, q0_4)
    b_p = SA.SVector{2, FT}(muladd(α_src, s_vap_rai, b_rai_in), b_sno_in)
    p_new, p_bar = _solve_1m_precip_block(
        p0, b_p, D_sed_3, D_sed_4, D_e_rai_s, D_e_sno_s, D_rai_sno_s,
        D_sno_rai_s, dt,
    )

    # The transfer between the species is not uniform over the step, so the
    # realized vapor sinks `D_e q̄ dt` can exceed their targets. Scale the rates
    # down to the targets and solve again; the sinks are concave in the rates,
    # so the second solve lands between the target and the first solve.
    evap_rai = D_e_rai_s * p_bar[1] * dt
    subl_sno = D_e_sno_s * p_bar[2] * dt
    tol = one(FT) + sqrt(eps(FT))
    if (evap_rai > tol * target_evap_rai) | (subl_sno > tol * target_subl_sno)
        D_e_rai_s *= min(one(FT), target_evap_rai / max(evap_rai, floatmin(FT)))
        D_e_sno_s *= min(one(FT), target_subl_sno / max(subl_sno, floatmin(FT)))
        p_new, p_bar = _solve_1m_precip_block(
            p0, b_p, D_sed_3, D_sed_4, D_e_rai_s, D_e_sno_s, D_rai_sno_s,
            D_sno_rai_s, dt,
        )
    end
    return (p_new, p_bar)
end

# TODO: remove or move into the test suite. The sweeps call
# `_coupled_1m_cell_step` directly; only `coupled_microphysics_sweep_tests.jl`
# uses this entry point.
@inline function coupled_1m_cell_step(
    mp::CMP.Microphysics1MParams,
    tps,
    ρ,
    T0,
    w_air,
    q_tot0,
    q0::SA.SVector{4},
    S_in::SA.SVector{4},
    q_bar_sed::SA.SVector{4},
    D_sed::SA.SVector{4},
    dt;
    is_constant_p::Bool = false,
)
    return is_constant_p ?
           _coupled_1m_cell_step(
        mp,
        tps,
        ρ,
        T0,
        w_air,
        q_tot0,
        q0,
        S_in,
        q_bar_sed,
        D_sed,
        dt,
        Val(true),
    ) :
           _coupled_1m_cell_step(
        mp,
        tps,
        ρ,
        T0,
        w_air,
        q_tot0,
        q0,
        S_in,
        q_bar_sed,
        D_sed,
        dt,
        Val(false),
    )
end

@inline function _coupled_1m_cell_step(
    vmp,
    vtps,
    ρ::FT,
    T0::FT,
    w_air::FT,
    q_tot0::FT,
    q0::SA.SVector{4, FT},
    S_in::SA.SVector{4, FT},
    q_bar_sed::SA.SVector{4, FT},
    D_sed::SA.SVector{4, FT},
    dt::FT,
    ::Val{is_constant_p},
) where {FT <: AbstractFloat, is_constant_p}
    tps = _unwrapped(vtps)
    q0_p = max.(q0, zero(FT))
    S_in_p = max.(S_in, zero(FT))
    q_sed_p = max.(q_bar_sed, zero(FT))
    D_sed_p = max.(D_sed, zero(FT))
    q_cond0 = sum(q0_p)
    q_v0 = max(zero(FT), q_tot0 - q_cond0)

    # Fast path for inflow-free cells without active cloud or precipitation.
    mp = _unwrapped(vmp)
    ϵ_cm = FT(CMU.ϵ_numerics(FT))
    if (q0_p[1] <= zero(FT)) &
       (q0_p[2] <= zero(FT)) &
       (q0_p[3] <= ϵ_cm) &
       (q0_p[4] <= ϵ_cm) &
       (sum(S_in_p) <= zero(FT)) &
       (D_sed_p[3] <= zero(FT)) &
       (D_sed_p[4] <= zero(FT))
        no_cf =
            isnothing(mp.processes.cloud_liquid_formation) &
            isnothing(mp.processes.cloud_ice_formation)
        if no_cf
            z4 = zero(SA.SVector{4, FT})
            return (; q_new = q0_p, q_out = z4, α_cap = one(FT), f_melt = zero(FT))
        end
        T_freeze = FT(TD.Parameters.T_freeze(tps))
        q_smin_fast = if T0 >= T_freeze
            TD.q_vap_saturation(tps, T0, ρ, TD.Liquid())
        else
            TD.q_vap_saturation(tps, T0, ρ, TD.Ice())
        end
        if q_v0 <= q_smin_fast
            z4 = zero(SA.SVector{4, FT})
            return (; q_new = q0_p, q_out = z4, α_cap = one(FT), f_melt = zero(FT))
        end
    end

    return _coupled_1m_cell_step_active(
        vmp,
        vtps,
        ρ,
        T0,
        w_air,
        q_v0,
        q0_p,
        S_in_p,
        q_sed_p,
        D_sed_p,
        dt,
        Val(is_constant_p),
    )
end

@inline function _coupled_1m_cell_step_active(
    vmp,
    vtps,
    ρ::FT,
    T0::FT,
    w_air::FT,
    q_v0::FT,
    q0_p::SA.SVector{4, FT},
    S_in_p::SA.SVector{4, FT},
    q_sed_p::SA.SVector{4, FT},
    D_sed_p::SA.SVector{4, FT},
    dt::FT,
    ::Val{is_constant_p},
) where {FT <: AbstractFloat, is_constant_p}
    return _coupled_1m_cell_step_core(
        vmp,
        vtps,
        ρ,
        T0,
        w_air,
        q_v0,
        q0_p,
        S_in_p,
        q_sed_p,
        D_sed_p,
        dt,
        Val(is_constant_p),
        FT(-1),
        FT(-1),
    )
end

"""
    _phase_change_ΔT(L_vl, L_vi, c, Δc_l, Δc_i, Δq_liq, Δq_ice)

Temperature change of a cell at fixed energy from the realized phase changes
`Δq_liq` and `Δq_ice` (net gains of liquid and ice from vapor and from each
other, excluding sedimentation),

    ΔT = (L_vl Δq_liq + L_vi Δq_ice) / (c + Δc_l Δq_liq + Δc_i Δq_ice).

With the internal-energy latent heats `L_vl = e_v - e_l`, `L_vi = e_v - e_i`
at `T0`, the isochoric heat capacity `c = c_vm` before the phase changes, and
`Δc_l = c_vl - c_vv`, `Δc_i = c_vi - c_vv`, `ΔT` is the exact temperature
change at fixed `e_int` and `ρ` (the grid-mean cell step); the enthalpy latent
heats and `c_p` give it at fixed `h` and `p` (the EDMF subdomain steps). The
denominator is the heat capacity after the phase changes. Freezing and riming
move mass from liquid to ice and so enter with `L_vi - L_vl`, the latent heat
of fusion; melting enters with the opposite sign.
"""
@inline function _phase_change_ΔT(
    L_vl,
    L_vi,
    c,
    Δc_l,
    Δc_i,
    Δq_liq,
    Δq_ice,
)
    FT = typeof(L_vl)
    c_post = max(muladd(Δc_l, Δq_liq, muladd(Δc_i, Δq_ice, c)), floatmin(FT))
    return muladd(L_vl, Δq_liq, L_vi * Δq_ice) / c_post
end

@noinline function _coupled_1m_cell_step_core(
    vmp,
    vtps,
    ρ::FT,
    T0::FT,
    w_air::FT,
    q_v0::FT,
    q0_p::SA.SVector{4, FT},
    S_in_p::SA.SVector{4, FT},
    q_sed_p::SA.SVector{4, FT},
    D_sed_p::SA.SVector{4, FT},
    dt::FT,
    ::Val{is_constant_p},
    q_sl_in::FT,
    q_si_in::FT,
) where {FT <: AbstractFloat, is_constant_p}
    mp = _unwrapped(vmp)
    tps = _unwrapped(vtps)
    T_freeze = FT(TD.Parameters.T_freeze(tps))

    rates = _coupled_1m_source_terms_with_qsat(
        mp,
        tps,
        ρ,
        T0,
        w_air,
        q_v0,
        q_sed_p[1],
        q_sed_p[2],
        q_sed_p[3],
        q_sed_p[4],
        q_sl_in,
        q_si_in,
    )
    (; q_sl, q_si, q_tol) = rates
    q_min = max(FT(TD.Parameters.q_min(tps)), floatmin(FT))

    # Effective heat capacity and latent heats: fixed-(ρ_d, e_int) on the grid
    # mean (`is_constant_p = false`) vs fixed-(p, h) on EDMF subdomains.
    q_liq_ref = q0_p[1] + q0_p[3]
    q_ice_ref = q0_p[2] + q0_p[4]
    q_tot_ref = q_v0 + q_liq_ref + q_ice_ref
    c_pm = TD.cp_m(tps, q_tot_ref, q_liq_ref, q_ice_ref)
    c_eff = if is_constant_p
        c_pm
    else
        TD.cv_m(tps, q_tot_ref, q_liq_ref, q_ice_ref)
    end

    ϵ_cm = FT(CMU.ϵ_numerics(FT))
    no_ice_phase =
        (T0 >= FT(TD.Parameters.T_triple(tps))) &
        (q0_p[2] <= zero(FT)) &
        (q0_p[4] <= ϵ_cm) &
        (S_in_p[2] <= zero(FT)) &
        (S_in_p[4] <= zero(FT))

    L_v = TD.latent_heat_vapor(tps, T0)
    L_s = no_ice_phase ? L_v : TD.latent_heat_sublim(tps, T0)
    L_f = no_ice_phase ? zero(FT) : TD.latent_heat_fusion(tps, T0)
    R_v = FT(TD.Parameters.R_v(tps))

    e_v0 = is_constant_p ? zero(FT) : TD.internal_energy_vapor(tps, T0)
    L_vl_eff = if is_constant_p
        L_v
    else
        e_v0 - TD.internal_energy_liquid(tps, T0)
    end
    L_vi_eff = if no_ice_phase
        L_vl_eff
    elseif is_constant_p
        L_s
    else
        e_v0 - TD.internal_energy_ice(tps, T0)
    end
    L_vl_over_c = L_vl_eff / c_eff
    L_vi_over_c = L_vi_eff / c_eff
    L_f_over_c = L_f / c_eff

    dqsl_dT_cm = CMNe.dqcld_dT(q_sl, L_v, R_v, T0)
    dqsi_dT_cm = CMNe.dqcld_dT(q_si, L_s, R_v, T0)
    # `CMNe` embeds `1 + (L / c_pm) * dqcld_dT` in `S_vap_lcl/icl`; recover `1 / τ`
    # with `Γ_*_cm`, then evolve excess with fixed-`ρ` or fixed-`p` `dq_sat / dT`.
    Γ_l_cm = CMNe.gamma_helper(L_v, c_pm, dqsl_dT_cm)
    Γ_i_cm = CMNe.gamma_helper(L_s, c_pm, dqsi_dT_cm)
    if is_constant_p
        inv_RvT2 = one(FT) / max(R_v * T0 * T0, floatmin(FT))
        dqsl_dT = q_sl * L_v * inv_RvT2
        dqsi_dT = q_si * L_s * inv_RvT2
    else
        dqsl_dT = dqsl_dT_cm
        dqsi_dT = dqsi_dT_cm
    end
    Γ_l = one(FT) + L_vl_over_c * dqsl_dT
    Γ_i = one(FT) + L_vi_over_c * dqsi_dT
    κ_il = one(FT) + L_vi_over_c * dqsl_dT
    κ_li = one(FT) + L_vl_over_c * dqsi_dT

    δ_l0 = q_v0 - q_sl
    δ_i0 = q_v0 - q_si
    Δs = q_sl - q_si

    # Effective inflow+production supplies for precipitation residence bounds.
    D_acnv_lcl0 = rates.S_acnv_lcl_rai / max(q_sed_p[1], q_min)
    P_acnv_rai =
        residence_supply(D_acnv_lcl0, q0_p[1], S_in_p[1], D_sed_p[1], dt) /
        max(dt, floatmin(FT))
    D_acnv_icl0 = rates.S_acnv_icl_sno / max(q_sed_p[2], q_min)
    P_acnv_sno =
        residence_supply(D_acnv_icl0, q0_p[2], S_in_p[2], D_sed_p[2], dt) /
        max(dt, floatmin(FT))

    S_in_eff_sno = S_in_p[4] + P_acnv_sno

    # Residence-bounded warm ice → liquid melt rates for `τ_melt,T`.
    θ0 = T0 - T_freeze
    is_warm = θ0 > zero(FT)
    S_melt_icl_raw = rates.S_melt_icl_lcl + rates.S_accr_icl_rai_warm
    S_melt_icl_eff = if is_warm && (S_melt_icl_raw > zero(FT))
        D_melt_icl0 = S_melt_icl_raw / max(q_sed_p[2], q_min)
        residence_supply(D_melt_icl0, q0_p[2], S_in_p[2], D_sed_p[2], dt) /
        max(dt, floatmin(FT))
    else
        zero(FT)
    end

    S_melt_sno_raw =
        rates.S_melt_sno_rai +
        rates.S_accr_melt_lcl_sno +
        rates.S_accr_melt_rai_sno
    S_melt_sno_eff = if is_warm && (S_melt_sno_raw > zero(FT))
        D_melt_sno0 = S_melt_sno_raw / max(q_sed_p[4], q_min)
        residence_supply(D_melt_sno0, q0_p[4], S_in_eff_sno, D_sed_p[4], dt) /
        max(dt, floatmin(FT))
    else
        zero(FT)
    end
    S_melt_tot_eff = S_melt_icl_eff + S_melt_sno_eff
    P_melt_rai = if is_warm && (S_melt_tot_eff > zero(FT))
        min(
            S_melt_sno_eff +
            S_melt_icl_eff *
            (rates.S_accr_icl_rai_warm / max(S_melt_icl_raw, floatmin(FT))),
            (c_eff * θ0) / max(L_f * dt, floatmin(FT)),
        )
    else
        zero(FT)
    end
    S_in_eff_rai = S_in_p[3] + P_acnv_rai + P_melt_rai

    # Pass 1 MM15 coefficients with residence-limited precipitation evaporation.
    c_l_raw = _cloud_relax_coeff(
        rates.S_vap_lcl,
        rates.τ_vap_lcl,
        Γ_l_cm,
        δ_l0,
        q_tol,
    )
    c_i_raw = _cloud_relax_coeff(
        rates.S_vap_icl,
        rates.τ_vap_icl,
        Γ_i_cm,
        δ_i0,
        q_tol,
    )

    # Residence-limited rain evaporation coefficient `c_r^eff = E_r(D_e,r^0) / (|δ_l| dt)`.
    δ_l_mag_r = max(abs(δ_l0), FT(16) * q_tol)
    c_r_sink =
        if (rates.c_rai_raw > zero(FT)) & ((q0_p[3] > zero(FT)) | (S_in_eff_rai > zero(FT)))
            D_er0 = (rates.c_rai_raw * δ_l_mag_r) / max(q_sed_p[3], q_min)
            E_r0 = residence_supply(D_er0, q0_p[3], S_in_eff_rai, D_sed_p[3], dt)
            E_r0 / max(δ_l_mag_r * dt, floatmin(FT))
        else
            zero(FT)
        end
    c_r_pass1 = ifelse(δ_l0 <= zero(FT), c_r_sink, zero(FT))

    # Snow sublimation supply vs deposition slope.
    snow_can_deposit =
        (
            (
                mp.processes.snow_deposition_sublimation isa
                CMP.DepositionAndSublimation
            ) ||
            (
                (
                    mp.processes.snow_deposition_sublimation isa
                    CMP.DepositionWithoutCloudCondensate
                ) &&
                (max(q0_p[1], q0_p[2]) <= FT(TD.Parameters.q_min(tps)))
            )
        ) &
        (T0 <= T_freeze)
    δ_i_mag_s = max(abs(δ_i0), FT(16) * q_tol)
    c_s_sink =
        if (rates.c_sno_raw > zero(FT)) & ((q0_p[4] > zero(FT)) | (S_in_eff_sno > zero(FT)))
            D_es0 = (rates.c_sno_raw * δ_i_mag_s) / max(q_sed_p[4], q_min)
            E_s0 = residence_supply(D_es0, q0_p[4], S_in_eff_sno, D_sed_p[4], dt)
            E_s0 / max(δ_i_mag_s * dt, floatmin(FT))
        else
            zero(FT)
        end
    c_s_dep = ifelse(snow_can_deposit, rates.c_sno_raw, zero(FT))
    c_s_pass1 = ifelse(δ_i0 > zero(FT), c_s_dep, c_s_sink)

    δ_bar_l1, δ_bar_i1 = _mm15_joint_excesses(
        c_l_raw,
        c_r_pass1,
        c_i_raw,
        c_s_pass1,
        Γ_l,
        Γ_i,
        κ_il,
        κ_li,
        δ_l0,
        δ_i0,
        Δs,
        dt,
    )

    # Cloud collision & freezing donor decays (computed before cloud evaporation
    # bounds and residence inversion so `D_e_lcl/icl` see competing sinks).
    D_lcl_rai =
        (
            rates.S_acnv_lcl_rai +
            rates.S_accr_lcl_rai +
            rates.S_accr_lcl_sno_warm
        ) / max(q_sed_p[1], q_min)
    D_lcl_icl = rates.S_freeze_lcl_icl / max(q_sed_p[1], q_min)
    D_lcl_sno = rates.S_accr_lcl_sno_cold / max(q_sed_p[1], q_min)
    D_comp_lcl = D_sed_p[1] + D_lcl_rai + D_lcl_icl + D_lcl_sno

    D_icl_sno =
        (
            rates.S_acnv_icl_sno +
            rates.S_accr_icl_sno +
            rates.S_accr_icl_rai_cold
        ) / max(q_sed_p[2], q_min)
    D_comp_icl = D_sed_p[2] + D_icl_sno

    # Pass 2: refine sink/source branch selection on the step-averaged excesses
    # `(δ̄_l1, δ̄_i1)`, handling WBF sign flips (`δ_i0 < 0 < δ̄_i` or `δ_l0 > 0 > δ̄_l`).
    T_hom = _liq_hom_temp(mp.processes.cloud_liquid_formation, mp, FT)
    c_l_sink = if (δ_l0 < -q_tol) & (rates.S_vap_lcl < zero(FT))
        D_el0 = -rates.S_vap_lcl / max(q_sed_p[1], q_min)
        E_l0 = residence_supply(D_el0, q0_p[1], S_in_p[1], D_comp_lcl, dt)
        min(c_l_raw, E_l0 / max((-δ_l0) * dt, floatmin(FT)))
    else
        c_l_raw
    end
    c_l = ifelse(δ_bar_l1 < zero(FT), c_l_sink, c_l_raw)
    c_l = ifelse((T0 < T_hom) & (δ_bar_l1 > zero(FT)), zero(FT), c_l)

    c_i_sink = if (δ_i0 < -q_tol) & (rates.S_vap_icl < zero(FT))
        D_ei0 = -rates.S_vap_icl / max(q_sed_p[2], q_min)
        E_i0 = residence_supply(D_ei0, q0_p[2], S_in_p[2], D_comp_icl, dt)
        min(c_i_raw, E_i0 / max((-δ_i0) * dt, floatmin(FT)))
    else
        c_i_raw
    end
    c_i = ifelse(δ_bar_i1 < zero(FT), c_i_sink, c_i_raw)
    c_i = ifelse(rates.icl_sink_only & (δ_bar_i1 > zero(FT)), zero(FT), c_i)

    c_r = ifelse(δ_bar_l1 < zero(FT), c_r_sink, zero(FT))
    c_s = ifelse(δ_bar_i1 > zero(FT), c_s_dep, c_s_sink)

    # Coupled `(δ_l, θ = T - T_freeze)` relaxation with `θ(t*) = 0` crossing
    # and residence-supply arrest when `T0 > T_freeze` and warm melting is active.
    S_melt_tot_raw = S_melt_icl_raw + S_melt_sno_raw
    Δθ_sup_max = L_f_over_c * S_melt_tot_eff * dt
    ν_melt = ifelse(
        is_warm & (S_melt_tot_raw > zero(FT)) & (Δθ_sup_max > zero(FT)),
        (L_f_over_c * S_melt_tot_raw) / max(θ0, floatmin(FT)),
        zero(FT),
    )
    δ_bar_l, δ_bar_i, f_melt = if is_warm && ν_melt > zero(FT)
        _coupled_thermal_melt_and_vapor(
            c_l + c_r,
            c_i + c_s,
            Γ_l,
            Γ_i,
            κ_il,
            κ_li,
            L_vl_over_c,
            L_vi_over_c,
            dqsl_dT,
            ν_melt,
            Δθ_sup_max,
            δ_l0,
            δ_i0,
            Δs,
            θ0,
            dt,
        )
    else
        d_l, d_i = _mm15_joint_excesses(
            c_l,
            c_r,
            c_i,
            c_s,
            Γ_l,
            Γ_i,
            κ_il,
            κ_li,
            δ_l0,
            δ_i0,
            Δs,
            dt,
        )
        (d_l, d_i, zero(FT))
    end

    # Select sink/source coefficients from the final signs of `(δ̄_l, δ̄_i)` so
    # sign flips between pass 1 and thermal-melt arrest use the matching branch.
    c_l_fin = ifelse(
        δ_bar_l < zero(FT),
        c_l_sink,
        ifelse(T0 < T_hom, zero(FT), c_l_raw),
    )
    c_i_fin = ifelse(
        δ_bar_i < zero(FT),
        c_i_sink,
        ifelse(rates.icl_sink_only, zero(FT), c_i_raw),
    )
    Δq_vap_lcl = c_l_fin * δ_bar_l * dt
    Δq_vap_icl = c_i_fin * δ_bar_i * dt
    Δq_vap_rai = min(zero(FT), c_r_sink * δ_bar_l) * dt
    Δq_vap_sno = ifelse(δ_bar_i > zero(FT), c_s_dep, c_s_sink) * δ_bar_i * dt

    # Cloud vapor sources (`Δq ≥ 0`) vs residence-inverted evaporation (`Δq < 0`).
    s_vap_lcl = max(zero(FT), Δq_vap_lcl) / max(dt, floatmin(FT))
    D_e_lcl = if Δq_vap_lcl < zero(FT)
        invert_residence_supply(-Δq_vap_lcl, q0_p[1], S_in_p[1], D_comp_lcl, dt)
    else
        zero(FT)
    end
    s_vap_icl = max(zero(FT), Δq_vap_icl) / max(dt, floatmin(FT))
    D_e_icl = if Δq_vap_icl < zero(FT)
        invert_residence_supply(-Δq_vap_icl, q0_p[2], S_in_p[2], D_comp_icl, dt)
    else
        zero(FT)
    end
    s_vap_rai = zero(FT)
    s_vap_sno = max(zero(FT), Δq_vap_sno) / max(dt, floatmin(FT))

    # Thermally arrested cloud-ice melting (`S_melt_icl_lcl + S_accr_icl_rai_warm`).
    D_other_icl = D_comp_icl + D_e_icl
    target_melt_icl =
        min(f_melt * S_melt_icl_raw * dt, S_melt_icl_eff * dt)
    D_melt_icl = if is_warm && (target_melt_icl > zero(FT))
        invert_residence_supply(
            target_melt_icl,
            q0_p[2],
            S_in_p[2],
            D_other_icl,
            dt,
        )
    else
        zero(FT)
    end
    w_icl_lcl =
        rates.S_melt_icl_lcl / max(S_melt_icl_raw, floatmin(FT))
    w_icl_rai =
        rates.S_accr_icl_rai_warm / max(S_melt_icl_raw, floatmin(FT))
    D_icl_lcl = D_melt_icl * w_icl_lcl
    D_icl_rai_warm = D_melt_icl * w_icl_rai

    D_rai_sno =
        (rates.S_accr_rai_sno_cold + rates.S_accr_freeze_icl_rai) /
        max(q_sed_p[3], q_min)
    target_melt_sno_base = min(f_melt * S_melt_sno_raw * dt, S_melt_sno_eff * dt)

    # First-pass coupled 2 × 2 cloud and 2 × 2 precip Metzler block solves.
    E_c, Phi1_c, Phi2_c = _solve_1m_cloud_matrices(
        D_sed_p[1],
        D_sed_p[2],
        D_lcl_rai,
        D_icl_sno,
        D_e_lcl,
        D_e_icl,
        D_lcl_icl,
        D_lcl_sno,
        D_icl_lcl,
        D_icl_rai_warm,
        dt,
    )
    c_new, c_bar = _solve_1m_cloud_apply(
        E_c,
        Phi1_c,
        Phi2_c,
        q0_p[1],
        q0_p[2],
        S_in_p[1],
        S_in_p[2],
        s_vap_lcl,
        s_vap_icl,
        one(FT),
        dt,
    )
    from_c_rai = muladd(D_lcl_rai, c_bar[1], D_icl_rai_warm * c_bar[2])
    from_c_sno = muladd(D_lcl_sno, c_bar[1], D_icl_sno * c_bar[2])
    p_new, p_bar =
        if (q0_p[3] <= zero(FT)) &
           (q0_p[4] <= zero(FT)) &
           (S_in_p[3] <= zero(FT)) &
           (S_in_p[4] <= zero(FT)) &
           (from_c_rai <= zero(FT)) &
           (from_c_sno <= zero(FT)) &
           (s_vap_sno <= zero(FT))
            ((zero(FT), zero(FT)), (zero(FT), zero(FT)))
        else
            _solve_1m_precip_step(
                q0_p[3],
                q0_p[4],
                S_in_p[3] + from_c_rai,
                S_in_p[4] + from_c_sno,
                D_sed_p[3],
                D_sed_p[4],
                D_rai_sno,
                s_vap_rai,
                s_vap_sno,
                Δq_vap_rai,
                Δq_vap_sno,
                target_melt_sno_base,
                is_warm,
                one(FT),
                one(FT),
                dt,
            )
        end
    q_new = SA.SVector{4, FT}(c_new[1], c_new[2], p_new[1], p_new[2])
    q_bar = SA.SVector{4, FT}(c_bar[1], c_bar[2], p_bar[1], p_bar[2])
    q_out = (D_sed_p .* q_bar) .* dt

    # Saturation-floor (`α_cap`), vapor-positivity, and latent-heating checks on
    # the realized phase-change increments (leaving `S_in` and `D_sed` unscaled).
    Δq_mp_liq =
        (q_new[1] - q0_p[1] - S_in_p[1] * dt + q_out[1]) +
        (q_new[3] - q0_p[3] - S_in_p[3] * dt + q_out[3])
    Δq_mp_ice =
        (q_new[2] - q0_p[2] - S_in_p[2] * dt + q_out[2]) +
        (q_new[4] - q0_p[4] - S_in_p[4] * dt + q_out[4])
    Δq_vap_to_cond = Δq_mp_liq + Δq_mp_ice
    c_vap, c_liq, c_ice = if is_constant_p
        (
            FT(TD.Parameters.cp_v(tps)),
            FT(TD.Parameters.cp_l(tps)),
            FT(TD.Parameters.cp_i(tps)),
        )
    else
        (
            FT(TD.Parameters.cv_v(tps)),
            FT(TD.Parameters.cv_l(tps)),
            FT(TD.Parameters.cv_i(tps)),
        )
    end
    ΔT_1 = _phase_change_ΔT(
        L_vl_eff,
        L_vi_eff,
        c_eff,
        c_liq - c_vap,
        c_ice - c_vap,
        Δq_mp_liq,
        Δq_mp_ice,
    )

    Γ_l_lim = one(FT) + L_vl_over_c * dqsl_dT
    Γ_i_lim = one(FT) + L_vi_over_c * dqsi_dT
    q_smin, λ_min, Γ_min = ifelse(
        q_si < q_sl,
        (q_si, dqsi_dT, Γ_i_lim),
        (q_sl, dqsl_dT, Γ_l_lim),
    )

    Σ_src = (s_vap_lcl + s_vap_icl + s_vap_sno) * dt
    q_v1 = q_v0 - Δq_vap_to_cond
    gap_sat = max(zero(FT), q_smin + λ_min * ΔT_1 - q_v1)
    α_sat = ifelse(
        Σ_src > q_tol,
        clamp(one(FT) - gap_sat / (Γ_min * max(Σ_src, q_tol)), zero(FT), one(FT)),
        one(FT),
    )
    α_pos = ifelse(
        Σ_src > q_tol,
        clamp(q_v0 / max(Σ_src, q_tol), zero(FT), one(FT)),
        one(FT),
    )
    α_cap = min(α_sat, α_pos)

    ΔT_α = _phase_change_ΔT(
        L_vl_eff,
        L_vi_eff,
        c_eff,
        c_liq - c_vap,
        c_ice - c_vap,
        Δq_mp_liq - (one(FT) - α_cap) * dt * s_vap_lcl,
        Δq_mp_ice - (one(FT) - α_cap) * dt * (s_vap_icl + s_vap_sno),
    )
    ΔT_max = mp.max_latent_heating_rate * dt
    f_lim = ifelse(
        abs(ΔT_α) > ΔT_max,
        ΔT_max / max(abs(ΔT_α), floatmin(FT)),
        one(FT),
    )

    if (α_cap < one(FT)) | (f_lim < one(FT))
        α_src2 = α_cap * f_lim
        # First-pass cloud Metzler matrices are reused when phase-change decay
        # is unscaled (`f_lim == 1`).
        E_c2, Phi1_c2, Phi2_c2 = if f_lim < one(FT)
            _solve_1m_cloud_matrices(
                D_sed_p[1],
                D_sed_p[2],
                D_lcl_rai,
                D_icl_sno,
                f_lim * D_e_lcl,
                f_lim * D_e_icl,
                f_lim * D_lcl_icl,
                f_lim * D_lcl_sno,
                f_lim * D_icl_lcl,
                f_lim * D_icl_rai_warm,
                dt,
            )
        else
            (E_c, Phi1_c, Phi2_c)
        end
        c_new2, c_bar2 = _solve_1m_cloud_apply(
            E_c2,
            Phi1_c2,
            Phi2_c2,
            q0_p[1],
            q0_p[2],
            S_in_p[1],
            S_in_p[2],
            s_vap_lcl,
            s_vap_icl,
            α_src2,
            dt,
        )
        from_c_rai2 =
            muladd(D_lcl_rai, c_bar2[1], (f_lim * D_icl_rai_warm) * c_bar2[2])
        from_c_sno2 =
            muladd(f_lim * D_lcl_sno, c_bar2[1], D_icl_sno * c_bar2[2])
        p_new2, p_bar2 = _solve_1m_precip_step(
            q0_p[3],
            q0_p[4],
            S_in_p[3] + from_c_rai2,
            S_in_p[4] + from_c_sno2,
            D_sed_p[3],
            D_sed_p[4],
            D_rai_sno,
            s_vap_rai,
            s_vap_sno,
            Δq_vap_rai,
            Δq_vap_sno,
            target_melt_sno_base,
            is_warm,
            α_src2,
            f_lim,
            dt,
        )
        q_new = SA.SVector{4, FT}(c_new2[1], c_new2[2], p_new2[1], p_new2[2])
        q_bar = SA.SVector{4, FT}(c_bar2[1], c_bar2[2], p_bar2[1], p_bar2[2])
        q_out = (D_sed_p .* q_bar) .* dt
    end

    return (; q_new, q_out, α_cap, f_melt)
end

# ============================================================================
# 6. Conditional SGS Quadrature Wrapper for Coupled 1M Cell Step
# ============================================================================

struct Coupled1MSGSNodeEvaluator{IsConstantP, MP, TPS, FT}
    mp::MP
    tps::TPS
    ρ::FT
    w_air::FT
    q_rai0::FT
    q_sno0::FT
    S_in::SA.SVector{4, FT}
    q_bar_sed::SA.SVector{4, FT}
    D_sed::SA.SVector{4, FT}
    λ::FT
    ξ_liq::FT
    ξ_ice::FT
    q_lcl0::FT
    q_icl0::FT
    λ_lagrange::FT
    λ_eq::FT
    mu_S::FT
    α::FT
    β_precip::FT
    shaft_frac::FT
    dt::FT
end

@inline function (ev::Coupled1MSGSNodeEvaluator{IsConstantP})(
    T_hat,
    q_tot_hat,
    flag_p,
) where {IsConstantP}
    tps = _unwrapped(ev.tps)
    q_sat_hat = TD.q_vap_saturation(tps, T_hat, ev.ρ)
    return ev(T_hat, q_tot_hat, flag_p, q_sat_hat)
end

@noinline function (ev::Coupled1MSGSNodeEvaluator{IsConstantP})(
    T_hat,
    q_tot_hat,
    flag_p,
    q_sat_hat,
) where {IsConstantP}
    FT = typeof(ev.ρ)
    tps = _unwrapped(ev.tps)
    q_tot_h = max(zero(FT), q_tot_hat)

    S_prime = q_tot_h - q_sat_hat - ev.mu_S
    shifted_excess = max(zero(FT), muladd(ev.α, S_prime, ev.λ_lagrange))
    q_lcl_hat, q_icl_hat = sgs_local_condensate(
        ev.λ,
        shifted_excess,
        ev.ξ_liq,
        ev.ξ_ice,
        ev.q_lcl0,
        ev.q_icl0,
    )

    q_c_eq_hat = max(zero(FT), ev.λ_eq + S_prime)
    q_precip0 = ev.q_rai0 + ev.q_sno0
    S_raw_cloud = q_tot_h - q_precip0 - q_sat_hat
    S_eff_cloud = ifelse(
        q_c_eq_hat > zero(FT),
        q_c_eq_hat,
        min(zero(FT), S_raw_cloud),
    )
    q_tot_eff = max(zero(FT), q_sat_hat + S_eff_cloud + q_precip0)

    φ_p = ifelse(
        ev.β_precip > zero(FT),
        sgs_placement_factor(ev.β_precip, flag_p, ev.shaft_frac),
        one(FT),
    )
    shift = (φ_p - one(FT)) * q_precip0
    q_rai_node = ev.q_rai0 * φ_p
    q_sno_node = ev.q_sno0 * φ_p
    q_tot_node = max(
        q_lcl_hat + q_icl_hat + q_rai_node + q_sno_node,
        q_tot_eff + shift,
    )

    q0_node = SA.SVector{4, FT}(q_lcl_hat, q_icl_hat, q_rai_node, q_sno_node)
    S_in_node = max.(
        SA.SVector{4, FT}(
            ev.S_in[1],
            ev.S_in[2],
            ev.S_in[3] * φ_p,
            ev.S_in[4] * φ_p,
        ),
        zero(FT),
    )
    T_freeze = FT(TD.Parameters.T_freeze(tps))
    T_icenuc = FT(TD.Parameters.T_icenuc(tps))
    q_sl_in = T_hat >= T_freeze ? q_sat_hat : FT(-1)
    q_si_in = if T_hat <= T_icenuc
        q_sat_hat
    elseif T_hat < T_freeze
        TD.q_vap_saturation(tps, T_hat, ev.ρ, TD.Ice())
    else
        FT(-1)
    end
    q_cond_node = sum(q0_node)
    if (q_cond_node <= zero(FT)) & (sum(S_in_node) <= zero(FT))
        q_smin_hat = T_hat >= T_freeze ? q_sl_in : q_si_in
        if q_tot_node <= q_smin_hat
            z = zero(FT)
            return (;
                q_lcl_new = z,
                q_icl_new = z,
                q_rai_new = z,
                q_sno_new = z,
                q_lcl_out = z,
                q_icl_out = z,
                q_rai_out = z,
                q_sno_out = z,
            )
        end
    end
    q_v_node = max(zero(FT), q_tot_node - q_cond_node)
    q_sed_node = max.(
        SA.SVector{4, FT}(
            max(q_lcl_hat, ev.q_bar_sed[1]),
            max(q_icl_hat, ev.q_bar_sed[2]),
            ev.q_bar_sed[3] * φ_p,
            ev.q_bar_sed[4] * φ_p,
        ),
        zero(FT),
    )
    D_sed_node = max.(ev.D_sed, zero(FT))

    res = _coupled_1m_cell_step_core(
        ev.mp,
        ev.tps,
        ev.ρ,
        T_hat,
        ev.w_air,
        q_v_node,
        q0_node,
        S_in_node,
        q_sed_node,
        D_sed_node,
        ev.dt,
        Val(IsConstantP),
        q_sl_in,
        q_si_in,
    )
    return (;
        q_lcl_new = res.q_new[1],
        q_icl_new = res.q_new[2],
        q_rai_new = res.q_new[3],
        q_sno_new = res.q_new[4],
        q_lcl_out = res.q_out[1],
        q_icl_out = res.q_out[2],
        q_rai_out = res.q_out[3],
        q_sno_out = res.q_out[4],
    )
end

@inline function _clear_sgs_mixed_phase_dry(
    thp_u,
    ρ,
    T0,
    q_v0,
    T′T′,
)
    FT = typeof(ρ)
    T_freeze = FT(TD.Parameters.T_freeze(thp_u))
    T_icenuc = FT(TD.Parameters.T_icenuc(thp_u))
    σ_T = sqrt(max(T′T′, zero(FT)))
    # With zero cloud, `λ_eq == -Inf`, and no active precipitation shaft, all
    # three `S′` edges lie at `+Inf` (`S′ = 0`); a `2.5 σ_T` margin covers
    # `_freezing_split` (`p₁, p₂ > 0.02 ⇒ |t| < 2.42`).
    ΔT_bound = FT(2.5) * σ_T
    if (T0 - ΔT_bound >= T_freeze) | (T0 + ΔT_bound <= T_icenuc)
        return (true, FT(-1))
    end
    q_sat0 = TD.q_vap_saturation(thp_u, T0, ρ)
    mu_S = q_v0 - q_sat0
    Δq_ls_max = if σ_T <= ϵ_numerics(FT)
        max(zero(FT), q_sat0 - TD.q_vap_saturation(thp_u, T0, ρ, TD.Ice()))
    else
        T_lo = max(T_icenuc, T0 - ΔT_bound)
        T_hi = min(T_freeze, T0 + ΔT_bound)
        q_sl_hi = TD.q_vap_saturation(thp_u, T_hi, ρ, TD.Liquid())
        q_sl_lo = TD.q_vap_saturation(thp_u, T_lo, ρ, TD.Liquid())
        q_si_lo = TD.q_vap_saturation(thp_u, T_lo, ρ, TD.Ice())
        q_sl_hi * max(zero(FT), one(FT) - q_si_lo / max(q_sl_lo, floatmin(FT)))
    end
    return (mu_S + Δq_ls_max <= zero(FT), q_sat0)
end

@inline function _coupled_1m_cell_step_sgs(
    sgs_quad,
    sgs_opts::SGSMicrophysicsOptions{FT},
    mp,
    tps,
    ρ::FT,
    T0::FT,
    w_air::FT,
    q_tot0::FT,
    q0::SA.SVector{4, FT},
    S_in::SA.SVector{4, FT},
    q_bar_sed::SA.SVector{4, FT},
    D_sed::SA.SVector{4, FT},
    T′T′::FT,
    q′q′::FT,
    corr_Tq::FT,
    sgs_moments,
    α_sgs::FT,
    precip_frac::FT,
    dt::FT,
    ::Val{IsConstantP} = Val(false),
) where {FT <: AbstractFloat, IsConstantP}
    thp_u = _unwrapped(tps)
    q0_p = max.(q0, zero(FT))
    q_cond0 = sum(q0_p)
    S_in_p = max.(S_in, zero(FT))
    σ_S = FT(sgs_moments.sigma_S)
    q_v0 = max(zero(FT), q_tot0 - q_cond0)
    a_p = max(precip_frac, sgs_opts.precip_frac_floor)
    ϵ_cm = FT(CMU.ϵ_numerics(FT))
    q_precip0 = q0_p[3] + q0_p[4]
    q_sat0_cached = FT(-1)
    if (q0_p[1] <= zero(FT)) &
       (q0_p[2] <= zero(FT)) &
       (q_precip0 <= ϵ_cm * a_p) &
       (sum(S_in_p) <= zero(FT)) &
       (D_sed[3] <= zero(FT)) &
       (D_sed[4] <= zero(FT)) &
       (FT(sgs_moments.λ_eq) == FT(-Inf))
        is_dry, q_sat0_cached =
            _clear_sgs_mixed_phase_dry(thp_u, ρ, T0, q_v0, T′T′)
        if is_dry
            return (; q_new = q0_p, q_out = zero(SA.SVector{4, FT}))
        end
    end

    return _coupled_1m_cell_step_sgs_active(
        sgs_quad,
        sgs_opts,
        mp,
        tps,
        ρ,
        T0,
        w_air,
        q_tot0,
        q0_p,
        q_cond0,
        q_v0,
        S_in_p,
        q_bar_sed,
        D_sed,
        T′T′,
        q′q′,
        corr_Tq,
        sgs_moments,
        σ_S,
        α_sgs,
        a_p,
        q_sat0_cached,
        dt,
        Val(IsConstantP),
    )
end

@inline function _coupled_1m_cell_step_sgs_active(
    sgs_quad,
    sgs_opts::SGSMicrophysicsOptions{FT},
    mp,
    tps,
    ρ::FT,
    T0::FT,
    w_air::FT,
    q_tot0::FT,
    q0_p::SA.SVector{4, FT},
    q_cond0::FT,
    q_v0::FT,
    S_in_p::SA.SVector{4, FT},
    q_bar_sed::SA.SVector{4, FT},
    D_sed::SA.SVector{4, FT},
    T′T′::FT,
    q′q′::FT,
    corr_Tq::FT,
    sgs_moments,
    σ_S::FT,
    α_sgs::FT,
    a_p::FT,
    q_sat0_cached::FT,
    dt::FT,
    ::Val{IsConstantP},
) where {FT <: AbstractFloat, IsConstantP}
    thp_u = _unwrapped(tps)
    λ = TD.liquid_fraction(thp_u, T0, q0_p[1], q0_p[2])
    q_tot_pos = max(zero(FT), q_tot0)
    q_sat0 =
        q_sat0_cached >= zero(FT) ? q_sat0_cached :
        TD.q_vap_saturation(thp_u, T0, ρ)
    mu_S = q_tot_pos - q_sat0

    q_c = q0_p[1] + q0_p[2]
    q_rai0 = q0_p[3]
    q_sno0 = q0_p[4]
    q_precip0 = q_rai0 + q_sno0
    α_safe = max(FT(α_sgs), ϵ_numerics(FT))
    λ_lagrange = FT(sgs_moments.λ_lagrange)
    λ_eq = FT(sgs_moments.λ_eq)
    t_a = ifelse(q_c > zero(FT), -λ_lagrange / α_safe, FT(Inf))
    t_e = -λ_eq
    β_precip = sgs_opts.precip_incloud_fraction
    z_tail = FT === Float32 ? FT(15) : FT(40)
    ϵ_cm = FT(CMU.ϵ_numerics(FT))
    has_precip_shaft_split =
        (β_precip > zero(FT)) &
        (a_p < one(FT)) &
        ((q_precip0 > ϵ_cm * a_p) | ((S_in_p[3] + S_in_p[4]) > zero(FT)))
    single_node =
        (σ_S <= ϵ_numerics(FT)) &
        (T′T′ <= ϵ_numerics(FT)) &
        !has_precip_shaft_split &
        (!isfinite(t_a) | (abs(t_a) >= z_tail * σ_S)) &
        (!isfinite(t_e) | (abs(t_e) >= z_tail * σ_S))

    β_T = zero(FT)
    σ⊥ = zero(FT)
    rule = if single_node
        _compact_single_node_rule(FT)
    else
        t_p = if has_precip_shaft_split
            sgs_precip_shaft_threshold(a_p, σ_S)
        else
            FT(Inf)
        end
        β_T = sgs_temperature_regression(thp_u, T0, ρ, σ_S, T′T′, q′q′, corr_Tq)
        σ⊥ = sgs_temperature_spread_at_fixed_excess(T′T′, β_T, σ_S)
        _compact_conditional_rule(σ_S, t_a, t_e, t_p)
    end

    g = _compact_rule_mean(NodeCondensate(λ_lagrange, α_safe), rule)
    s = ifelse(
        q_c > zero(FT),
        ifelse(g > zero(FT), q_c / max(g, floatmin(FT)), one(FT)),
        zero(FT),
    )
    λ_scaled = s * λ_lagrange
    α_scaled = s * α_safe
    shaft_frac = ifelse(
        β_precip > zero(FT),
        _compact_rule_shaft_fraction(rule),
        one(FT),
    )

    T_freeze = FT(TD.Parameters.T_freeze(thp_u))
    T_icenuc = FT(TD.Parameters.T_icenuc(thp_u))
    T_min = sgs_quad.T_min
    q_max = sgs_quad.q_max
    D_sed_node = max.(D_sed, zero(FT))
    q_bar_sed_p = max.(q_bar_sed, zero(FT))

    # Positive-weight wet points share one call site to
    # `_coupled_1m_cell_step_core` so dry skips and freezing splits keep active
    # warp lanes synchronized.
    q_new_acc = zero(SA.SVector{4, FT})
    q_out_acc = zero(SA.SVector{4, FT})
    k = Int32(1)
    has_p2 = false
    w2_pend = zero(FT)
    T2_pend = zero(FT)
    Sp2_pend = zero(FT)
    sh2_pend = zero(FT)

    while true
        found = false
        w_pt = zero(FT)
        T_hat = zero(FT)
        q_v_node = zero(FT)
        q0_node = zero(SA.SVector{4, FT})
        S_in_node = zero(SA.SVector{4, FT})
        q_sed_node = zero(SA.SVector{4, FT})
        q_sl_in = FT(-1)
        q_si_in = FT(-1)

        while has_p2 | (k <= Int32(8))
            if has_p2
                has_p2 = false
                w_cand = w2_pend
                T_cand = T2_pend
                Sp_cand = Sp2_pend
                sh_cand = sh2_pend
            else
                idx = k
                k += Int32(1)
                w_nd = _select_tuple8(idx, rule.w)
                w_nd > zero(FT) || continue
                Sp_nd = _select_tuple8(idx, rule.S′)
                sh_bit =
                    ((rule.shaft_mask >> UInt32(idx - Int32(1))) & UInt8(1)) !=
                    UInt8(0)
                sh_nd = ifelse(sh_bit, one(FT), zero(FT))
                T_nd = max(T_min, muladd(β_T, Sp_nd, T0))
                (p₁, t₁, p₂, t₂) = _freezing_split(T_nd, σ⊥, T_freeze)
                w₁ = w_nd * p₁
                w₂ = w_nd * p₂
                if w₂ > zero(FT)
                    has_p2 = true
                    w2_pend = w₂
                    T2_pend = max(T_min, muladd(t₂, σ⊥, T_nd))
                    Sp2_pend = Sp_nd
                    sh2_pend = sh_nd
                end
                w₁ > zero(FT) || continue
                w_cand = w₁
                T_cand = max(T_min, muladd(t₁, σ⊥, T_nd))
                Sp_cand = Sp_nd
                sh_cand = sh_nd
            end

            q_sat_hat =
                T_cand == T0 ? q_sat0 :
                TD.q_vap_saturation(thp_u, T_cand, ρ)
            q_tot_h = clamp(q_sat_hat + mu_S + Sp_cand, zero(FT), q_max)
            S_prime = q_tot_h - q_sat_hat - mu_S
            shifted_excess = max(zero(FT), muladd(α_scaled, S_prime, λ_scaled))
            q_lcl_hat, q_icl_hat = sgs_local_condensate(
                λ,
                shifted_excess,
                sgs_opts.ξ_liq,
                sgs_opts.ξ_ice,
                q0_p[1],
                q0_p[2],
            )
            q_c_eq_hat = max(zero(FT), λ_eq + S_prime)
            S_raw_cloud = q_tot_h - q_precip0 - q_sat_hat
            S_eff_cloud = ifelse(
                q_c_eq_hat > zero(FT),
                q_c_eq_hat,
                min(zero(FT), S_raw_cloud),
            )
            q_tot_eff = max(zero(FT), q_sat_hat + S_eff_cloud + q_precip0)

            φ_p = ifelse(
                β_precip > zero(FT),
                sgs_placement_factor(β_precip, sh_cand, shaft_frac),
                one(FT),
            )
            shift = (φ_p - one(FT)) * q_precip0
            q_rai_node = q_rai0 * φ_p
            q_sno_node = q_sno0 * φ_p
            q_cond_node = q_lcl_hat + q_icl_hat + q_rai_node + q_sno_node
            q_tot_node = max(q_cond_node, q_tot_eff + shift)
            S_in_cand = SA.SVector{4, FT}(
                S_in_p[1],
                S_in_p[2],
                S_in_p[3] * φ_p,
                S_in_p[4] * φ_p,
            )

            q_sl_cand = T_cand >= T_freeze ? q_sat_hat : FT(-1)
            q_si_cand = if T_cand <= T_icenuc
                q_sat_hat
            elseif T_cand < T_freeze
                TD.q_vap_saturation(thp_u, T_cand, ρ, TD.Ice())
            else
                FT(-1)
            end
            if (q_lcl_hat <= zero(FT)) &
               (q_icl_hat <= zero(FT)) &
               (q_rai_node <= ϵ_cm) &
               (q_sno_node <= ϵ_cm) &
               (sum(S_in_cand) <= zero(FT))
                q_smin_hat = T_cand >= T_freeze ? q_sl_cand : q_si_cand
                if q_tot_node - q_cond_node <= q_smin_hat
                    if (q_rai_node > zero(FT)) | (q_sno_node > zero(FT))
                        e_r = _exp_phi1_phi2_neg(D_sed_node[3] * dt)[1]
                        e_s = _exp_phi1_phi2_neg(D_sed_node[4] * dt)[1]
                        r_pas = q_rai_node * e_r
                        s_pas = q_sno_node * e_s
                        q_new_acc =
                            q_new_acc +
                            SA.SVector{4, FT}(zero(FT), zero(FT), r_pas, s_pas) .*
                            w_cand
                        q_out_acc =
                            q_out_acc +
                            SA.SVector{4, FT}(
                                zero(FT),
                                zero(FT),
                                max(zero(FT), q_rai_node - r_pas),
                                max(zero(FT), q_sno_node - s_pas),
                            ) .* w_cand
                    end
                    continue
                end
            end

            w_pt = w_cand
            T_hat = T_cand
            q_v_node = max(zero(FT), q_tot_node - q_cond_node)
            q0_node =
                SA.SVector{4, FT}(q_lcl_hat, q_icl_hat, q_rai_node, q_sno_node)
            S_in_node = S_in_cand
            q_sed_node = SA.SVector{4, FT}(
                max(q_lcl_hat, q_bar_sed_p[1]),
                max(q_icl_hat, q_bar_sed_p[2]),
                q_bar_sed_p[3] * φ_p,
                q_bar_sed_p[4] * φ_p,
            )
            q_sl_in = q_sl_cand
            q_si_in = q_si_cand
            found = true
            break
        end
        found || break

        res = _coupled_1m_cell_step_core(
            mp,
            tps,
            ρ,
            T_hat,
            w_air,
            q_v_node,
            q0_node,
            S_in_node,
            q_sed_node,
            D_sed_node,
            dt,
            Val(IsConstantP),
            q_sl_in,
            q_si_in,
        )
        q_new_acc = q_new_acc + res.q_new .* w_pt
        q_out_acc = q_out_acc + res.q_out .* w_pt
    end

    q_new = SA.SVector{4, FT}(
        max(zero(FT), q_new_acc[1]),
        max(zero(FT), q_new_acc[2]),
        max(zero(FT), q_new_acc[3]),
        max(zero(FT), q_new_acc[4]),
    )
    q_out = SA.SVector{4, FT}(
        max(zero(FT), q_out_acc[1]),
        max(zero(FT), q_out_acc[2]),
        max(zero(FT), q_out_acc[3]),
        max(zero(FT), q_out_acc[4]),
    )

    # Guard cell-mean vapor non-negativity when node-wise equilibrium shifts
    # condense slightly more than cell-mean `q_v0`; draw excess from cloud
    # species `(q_lcl, q_icl)` first so falling precipitation is preserved.
    Δq_vap_to_cond = sum(q_new - q0_p - S_in_p .* dt + q_out)
    if Δq_vap_to_cond > q_v0
        excess = Δq_vap_to_cond - q_v0
        q_cld_sum = q_new[1] + q_new[2]
        take_cld = min(excess, q_cld_sum)
        scale_cld = ifelse(
            q_cld_sum > floatmin(FT),
            clamp(one(FT) - take_cld / max(q_cld_sum, floatmin(FT)), zero(FT), one(FT)),
            one(FT),
        )
        rem_excess = max(zero(FT), excess - take_cld)
        q_sno_adj = max(zero(FT), q_new[4] - rem_excess)
        q_new = SA.SVector{4, FT}(
            q_new[1] * scale_cld,
            q_new[2] * scale_cld,
            q_new[3],
            q_sno_adj,
        )
    end
    return (; q_new, q_out)
end

# ============================================================================
# 7. Donor-Cell Post-Microphysics Temperature and Energy Export Helpers
# ============================================================================

"""
    _donor_post_mp_temperature_and_energy_out(
        tps, ρ_old, ρe_tot_old, F_e_in, V_k, dt,
        ρ_new, ρq_tot_new, ρq_new, q_out, w_sed, uvw_air, Φ_k,
    )

Diagnose the post-microphysics donor-cell temperature `T_k` and bottom-face
sedimentation energy export `F_e_out` (`J s⁻¹` integrated over the bottom-face
area element `A_bot`, so `F_e_out / A_bot` has units `W m⁻²`) so that outgoing
condensate mass `m_out,s = ρ_old * q_out,s` leaves cell `k` carrying
`e_int,s(T_k) + Φ_k + K_s` while `ρe_tot_new = ρe_tot_old + (dt / V_k) (F_e_in - F_e_out)`
matches `TD.internal_energy(tps, T_k, ...)` of the remaining moist air.
`K_s = sedimentation_kinetic_energy(w_sed,s, uvw_air)` uses the fall speed of
the cell's own pool. Returns `(T_k, F_e_out, F_K_out)`, where `F_K_out` is the
kinetic-energy part `Σ_s m_out,s K_s` of `F_e_out`.
"""
@inline function _donor_post_mp_temperature_and_energy_out(
    tps,
    ρ_old,
    ρe_tot_old,
    F_e_in,
    V_k,
    dt,
    ρ_new,
    ρq_tot_new,
    ρq_new::SA.SVector{4},
    q_out::SA.SVector{4},
    w_sed::SA.SVector{4},
    uvw_air,
    Φ_k,
)
    FT = typeof(ρ_old)
    T_0 = FT(TD.Parameters.T_0(tps))
    cv_l = FT(TD.Parameters.cv_l(tps))
    cv_i = FT(TD.Parameters.cv_i(tps))
    e_ice_T0 = TD.internal_energy_ice(tps, T_0) # `-e_int_i0`

    inv_ρ_new = one(FT) / max(ρ_new, floatmin(FT))
    q_tot_post = max(zero(FT), ρq_tot_new * inv_ρ_new)
    q_liq_post = max(zero(FT), (ρq_new[1] + ρq_new[3]) * inv_ρ_new)
    q_ice_post = max(zero(FT), (ρq_new[2] + ρq_new[4]) * inv_ρ_new)

    e_int_air_T0 =
        TD.internal_energy(tps, T_0, q_tot_post, q_liq_post, q_ice_post)
    cv_m_post = TD.cv_m(tps, q_tot_post, q_liq_post, q_ice_post)
    K_air = norm_sqr(uvw_air) / 2

    m_out = ρ_old .* q_out
    if (m_out[1] + m_out[2] + m_out[3] + m_out[4]) <= zero(FT)
        E_total = ρe_tot_old + (dt / V_k) * F_e_in
        E_ref_T0 = ρ_new * (e_int_air_T0 + K_air + Φ_k)
        C_tot = ρ_new * cv_m_post
        T_k = T_0 + (E_total - E_ref_T0) / max(C_tot, floatmin(FT))
        return (T_k, zero(FT), zero(FT))
    end
    m_out_liq = m_out[1] + m_out[3]
    m_out_ice = m_out[2] + m_out[4]

    K_lcl = sedimentation_kinetic_energy(w_sed[1], uvw_air)
    K_icl = sedimentation_kinetic_energy(w_sed[2], uvw_air)
    K_rai = sedimentation_kinetic_energy(w_sed[3], uvw_air)
    K_sno = sedimentation_kinetic_energy(w_sed[4], uvw_air)

    mech_out =
        m_out[1] * (Φ_k + K_lcl) +
        m_out[2] * (Φ_k + K_icl) +
        m_out[3] * (Φ_k + K_rai) +
        m_out[4] * (Φ_k + K_sno)

    E_total = ρe_tot_old + (dt / V_k) * F_e_in
    E_ref_T0 =
        ρ_new * (e_int_air_T0 + K_air + Φ_k) + mech_out + m_out_ice * e_ice_T0
    C_tot = muladd(ρ_new, cv_m_post, muladd(m_out_liq, cv_l, m_out_ice * cv_i))
    T_k = T_0 + (E_total - E_ref_T0) / max(C_tot, floatmin(FT))

    e_int_liq = TD.internal_energy_liquid(tps, T_k)
    e_int_ice = TD.internal_energy_ice(tps, T_k)
    e_lcl = e_int_liq + Φ_k + K_lcl
    e_icl = e_int_ice + Φ_k + K_icl
    e_rai = e_int_liq + Φ_k + K_rai
    e_sno = e_int_ice + Φ_k + K_sno

    scale_flux = V_k / max(dt, floatmin(FT))
    F_e_out =
        scale_flux * (
            m_out[1] * e_lcl +
            m_out[2] * e_icl +
            m_out[3] * e_rai +
            m_out[4] * e_sno
        )
    F_K_out =
        scale_flux * (
            m_out[1] * K_lcl +
            m_out[2] * K_icl +
            m_out[3] * K_rai +
            m_out[4] * K_sno
        )
    return (T_k, F_e_out, F_K_out)
end

"""
    _predicted_1m_precip_production(
        mp, tps, vc, ρ, T0, w_air, q_v0, q_lcl0, q_icl0, q_sno0,
        S_in_lcl, S_in_icl, S_in_sno, w_lcl_above, w_icl_above, w_sno_above,
        dz_eff, dt, c_eff,
    )

Evaluate cloud terminal velocities `(w_lcl, w_icl)`, donor-bounded cloud
autoconversion, 2-step snow fall speed `(w_sno, q̄_sed,sno)`, and thermally
bounded snow/ice melt production `P_rai` so `w_rai` and `q̄_sed,rai` see newly
produced cloud-top and melt-layer rain within the same step.
"""
@inline _predicted_1m_precip_production(
    vmp,
    vtps,
    vc::VelocityConfig1M,
    ρ::FT,
    T0::FT,
    w_air::FT,
    q_v0::FT,
    q_lcl0::FT,
    q_icl0::FT,
    q_sno0::FT,
    S_in_lcl::FT,
    S_in_icl::FT,
    S_in_sno::FT,
    w_lcl_above::FT,
    w_icl_above::FT,
    w_sno_above::FT,
    dz_eff::FT,
    dt::FT,
    c_eff::FT,
) where {FT <: AbstractFloat} = _predicted_1m_precip_production(
    vmp,
    vtps,
    vc,
    ρ,
    T0,
    w_air,
    q_v0,
    q_lcl0,
    q_icl0,
    q_sno0,
    S_in_lcl,
    S_in_icl,
    S_in_sno,
    w_lcl_above,
    w_icl_above,
    w_sno_above,
    dz_eff,
    dt,
    c_eff,
    one(FT),
    Val(false),
    zero(FT),
    one(FT),
)

@inline function _predicted_1m_precip_production(
    vmp,
    vtps,
    vc::VelocityConfig1M,
    ρ::FT,
    T0::FT,
    w_air::FT,
    q_v0::FT,
    q_lcl0::FT,
    q_icl0::FT,
    q_sno0::FT,
    S_in_lcl::FT,
    S_in_icl::FT,
    S_in_sno::FT,
    w_lcl_above::FT,
    w_icl_above::FT,
    w_sno_above::FT,
    dz_eff::FT,
    dt::FT,
    c_eff::FT,
    cf_scale::FT,
    use_shaft::Val,
    β_p::FT,
    a_s::FT,
) where {FT <: AbstractFloat}
    if (q_lcl0 <= zero(FT)) &
       (q_icl0 <= zero(FT)) &
       (q_sno0 <= zero(FT)) &
       (S_in_lcl <= zero(FT)) &
       (S_in_icl <= zero(FT)) &
       (S_in_sno <= zero(FT))
        inv_dz = one(FT) / max(dz_eff, floatmin(FT))
        w_lcl = max(zero(FT), _tv_1m_lcl(vc, ρ, zero(FT)))
        w_icl = max(zero(FT), _tv_1m_icl(vc, ρ, zero(FT)))
        w_sno = max(
            zero(FT),
            _eval_tv_1m(_tv_1m_sno, vc, ρ, zero(FT), one(FT), use_shaft, β_p, a_s),
        )
        z = zero(FT)
        return (;
            w_lcl,
            q_bar_lcl = z,
            D_sed_lcl = w_lcl * inv_dz,
            w_icl,
            q_bar_icl = z,
            D_sed_icl = w_icl * inv_dz,
            w_sno,
            q_bar_sno = z,
            D_sed_sno = w_sno * inv_dz,
            P_sno = z,
            P_rai = z,
        )
    end
    mp = _unwrapped(vmp)
    tps = _unwrapped(vtps)
    w_lcl, q_bar_lcl = _two_step_sed_velocity(
        _tv_1m_lcl,
        vc,
        ρ,
        q_lcl0,
        S_in_lcl,
        w_lcl_above,
        dz_eff,
        dt,
        cf_scale,
        Val(false),
        zero(FT),
        one(FT),
    )
    w_icl, q_bar_icl = _two_step_sed_velocity(
        _tv_1m_icl,
        vc,
        ρ,
        q_icl0,
        S_in_icl,
        w_icl_above,
        dz_eff,
        dt,
        cf_scale,
        Val(false),
        zero(FT),
        one(FT),
    )

    q_min = max(FT(TD.Parameters.q_min(tps)), floatmin(FT))
    inv_dz = one(FT) / max(dz_eff, floatmin(FT))
    D_sed_lcl = w_lcl * inv_dz
    D_sed_icl = w_icl * inv_dz

    thermo = (; ρ = max(zero(FT), ρ), T = T0, w = w_air)
    q_sno_pos = max(zero(FT), q_sno0)
    has_cld_lcl = q_bar_lcl > zero(FT)
    has_cld_icl = q_bar_icl > zero(FT)

    D_acnv_lcl = zero(FT)
    P_acnv_rai = zero(FT)
    if has_cld_lcl
        micro_lcl = (;
            q_tot = q_v0 + q_bar_lcl + q_bar_icl + q_sno_pos,
            q_lcl = q_bar_lcl,
            q_icl = q_bar_icl,
            q_rai = zero(FT),
            q_sno = q_sno_pos,
        )
        S_acnv_rai0 = CM1.conv_q_lcl_to_q_rai(
            mp.processes.rain_autoconversion,
            mp,
            tps,
            micro_lcl,
            thermo,
        )
        D_acnv_lcl = S_acnv_rai0 / max(q_bar_lcl, q_min)
        P_acnv_rai =
            residence_supply(D_acnv_lcl, q_lcl0, S_in_lcl, D_sed_lcl, dt) /
            max(dt, floatmin(FT))
    end

    P_sno = zero(FT)
    if has_cld_icl
        micro_icl = (;
            q_tot = q_v0 + q_bar_lcl + q_bar_icl + q_sno_pos,
            q_lcl = q_bar_lcl,
            q_icl = q_bar_icl,
            q_rai = zero(FT),
            q_sno = q_sno_pos,
        )
        sd_cld =
            if mp.processes.snow_autoconversion isa
               Union{Nothing, CMP.NoSupersaturation}
                nothing
            else
                CM1.size_distr_parameters(mp, micro_icl, thermo)
            end
        S_acnv_sno0 = CM1.conv_q_icl_to_q_sno(
            mp.processes.snow_autoconversion,
            mp,
            tps,
            micro_icl,
            thermo,
            sd_cld,
        )
        D_acnv_icl = S_acnv_sno0 / max(q_bar_icl, q_min)
        P_sno =
            residence_supply(D_acnv_icl, q_icl0, S_in_icl, D_sed_icl, dt) /
            max(dt, floatmin(FT))
    end

    S_in_eff_sno = S_in_sno + P_sno
    w_sno, q_bar_sno = _two_step_sed_velocity(
        _tv_1m_sno,
        vc,
        ρ,
        q_sno0,
        S_in_eff_sno,
        w_sno_above,
        dz_eff,
        dt,
        one(FT),
        use_shaft,
        β_p,
        a_s,
    )
    D_sed_sno = w_sno * inv_dz

    T_freeze = FT(TD.Parameters.T_freeze(tps))
    P_melt_rai = if T0 > T_freeze && q_bar_sno > FT(CMU.ϵ_numerics(FT))
        micro_sno = (;
            q_tot = q_v0 + q_bar_lcl + q_bar_icl + q_bar_sno,
            q_lcl = q_bar_lcl,
            q_icl = q_bar_icl,
            q_rai = zero(FT),
            q_sno = q_bar_sno,
        )
        sd_sno = CM1.size_distr_parameters(mp, micro_sno, thermo)
        S_melt0 = CM1.conv_q_sno_to_q_rai(
            mp.processes.snow_melt,
            mp,
            tps,
            micro_sno,
            thermo,
            sd_sno,
        )
        lcl_sno = if has_cld_lcl && !isnothing(mp.processes.cloud_liquid_snow_accretion)
            CM1.accretion(
                mp.processes.cloud_liquid_snow_accretion,
                mp,
                tps,
                micro_sno,
                thermo,
                sd_sno,
            )
        else
            (; S_accr = zero(FT), S_melt = zero(FT))
        end
        S_melt_tot = S_melt0 + lcl_sno.S_melt
        D_melt = S_melt_tot / max(q_bar_sno, q_min)
        S_donor_cap =
            residence_supply(D_melt, q_sno0, S_in_eff_sno, D_sed_sno, dt) /
            max(dt, floatmin(FT))
        L_f = TD.latent_heat_fusion(tps, T0)
        S_therm_cap = (c_eff * (T0 - T_freeze)) / max(L_f * dt, floatmin(FT))
        S_accr_cap = if lcl_sno.S_accr > zero(FT)
            D_accr_lcl = lcl_sno.S_accr / max(q_bar_lcl, q_min)
            residence_supply(
                D_accr_lcl,
                q_lcl0,
                S_in_lcl,
                D_sed_lcl + D_acnv_lcl,
                dt,
            ) / max(dt, floatmin(FT))
        else
            zero(FT)
        end
        min(S_donor_cap, S_therm_cap) + S_accr_cap
    else
        zero(FT)
    end

    P_rai = P_acnv_rai + P_melt_rai
    return (;
        w_lcl,
        q_bar_lcl,
        D_sed_lcl,
        w_icl,
        q_bar_icl,
        D_sed_icl,
        w_sno,
        q_bar_sno,
        D_sed_sno,
        P_sno,
        P_rai,
    )
end

# ============================================================================
# 8. Top-Down Column Sweep Functors (Grid-Mean, SGS Quadrature, PrognosticEDMFX)
# ============================================================================

@inline function _partition_nonneg_condensates(
    q_tot_raw,
    q_lcl_raw,
    q_icl_raw,
    q_rai_raw,
    q_sno_raw,
)
    FT = typeof(q_tot_raw)
    q_tot_pos = max(zero(FT), q_tot_raw)
    q_lcl = max(zero(FT), q_lcl_raw)
    q_icl = max(zero(FT), q_icl_raw)
    q_rai = max(zero(FT), q_rai_raw)
    q_sno = max(zero(FT), q_sno_raw)
    q_sum = q_lcl + q_icl + q_rai + q_sno
    scale = ifelse(
        q_sum > q_tot_pos,
        q_tot_pos / max(q_sum, floatmin(FT)),
        one(FT),
    )
    return SA.SVector{4, FT}(
        q_lcl * scale,
        q_icl * scale,
        q_rai * scale,
        q_sno * scale,
    )
end

struct Coupled1MGridMeanSweep{
    HasSGS,
    HasPrecipFrac,
    HasPrecipShaft,
    HasIncloudSed,
    MP,
    TPS,
    VC,
    SGQ,
    SGO,
    FT,
} <: ColumnSweep
    mp::MP
    tps::TPS
    vc::VC
    sgs_quad::SGQ
    sgs_opts::SGO
    dt::FT
    f_decay::FT
    q_precip_min::FT
    cf_floor::FT
end

@inline function (
    sw::Coupled1MGridMeanSweep{
        HasSGS,
        HasPrecipFrac,
        HasPrecipShaft,
        HasIncloudSed,
    }
)
    (
        Yc_col,
        geom_col,
        pre_col,
        sgs_col,
        sfc_col,
    ) where {HasSGS, HasPrecipFrac, HasPrecipShaft, HasIncloudSed}
    FT = typeof(sw.dt)
    tps = _unwrapped(sw.tps)
    mp = _unwrapped(sw.mp)

    ρ_fv = Fields.field_values(Yc_col.ρ)
    ρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    ρql_fv = Fields.field_values(Yc_col.ρq_lcl)
    ρqi_fv = Fields.field_values(Yc_col.ρq_icl)
    ρqr_fv = Fields.field_values(Yc_col.ρq_rai)
    ρqs_fv = Fields.field_values(Yc_col.ρq_sno)

    lg_fv = Fields.field_values(geom_col.ᶜlg)
    flg_fv = Fields.field_values(geom_col.ᶠlg)
    dξ_fv = Fields.field_values(geom_col.ᶠ∂ξ³∂x³)
    Φ_fv = Fields.field_values(geom_col.ᶜΦ)

    u_fv = Fields.field_values(pre_col.ᶜu)
    wl_fv = Fields.field_values(pre_col.ᶜwₗ)
    wi_fv = Fields.field_values(pre_col.ᶜwᵢ)
    wr_fv = Fields.field_values(pre_col.ᶜwᵣ)
    ws_fv = Fields.field_values(pre_col.ᶜwₛ)
    whht_fv = Fields.field_values(pre_col.ᶜwₕhₜ)

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    F_mass_in = zero(SA.SVector{4, FT})
    F_e_in = zero(FT)
    w_above = zero(SA.SVector{4, FT})
    a_p_above = zero(FT)

    @inbounds for step_idx in 0:(nz - 1)
        k = nz - step_idx
        ρ_old = ρ_fv[k]
        inv_ρ_old = one(FT) / ρ_old
        ρe_old = ρe_fv[k]
        ρqt_old = ρqt_fv[k]

        lg_k = lg_fv[k]
        V_k = lg_k.J
        A_bot = flg_fv[k].J * dξ_fv[k]
        dz_eff = V_k / max(A_bot, floatmin(FT))
        inv_dz = one(FT) / max(dz_eff, floatmin(FT))

        q_tot0 = ρqt_old * inv_ρ_old
        q0 = _partition_nonneg_condensates(
            q_tot0,
            ρql_fv[k] * inv_ρ_old,
            ρqi_fv[k] * inv_ρ_old,
            ρqr_fv[k] * inv_ρ_old,
            ρqs_fv[k] * inv_ρ_old,
        )
        q_v0 = max(zero(FT), max(zero(FT), q_tot0) - sum(q0))

        uvw_k = Geometry.UVWVector(u_fv[k], lg_k)
        w_air = uvw_k.w
        K_air = norm_sqr(uvw_k) / 2
        Φ_k = Φ_fv[k]
        e_int_old = ρe_old * inv_ρ_old - K_air - Φ_k
        q_liq0 = q0[1] + q0[3]
        q_ice0 = q0[2] + q0[4]
        T0 = TD.air_temperature(
            tps,
            e_int_old,
            max(zero(FT), q_tot0),
            q_liq0,
            q_ice0,
        )
        c_vm0 = TD.cv_m(tps, max(zero(FT), q_tot0), q_liq0, q_ice0)

        S_in = F_mass_in .* (one(FT) / (ρ_old * V_k))

        a_p_cand = if HasPrecipFrac
            cf_k = max(zero(FT), Fields.field_values(sgs_col.ᶜcf)[k])
            step_idx == 0 ? cf_k :
            min(one(FT), max(cf_k, sw.f_decay * a_p_above))
        else
            one(FT)
        end
        β_p = HasPrecipShaft ? sw.sgs_opts.precip_incloud_fraction : zero(FT)
        a_s_k = if HasPrecipShaft
            clamp(
                max(a_p_cand, sw.sgs_opts.precip_frac_floor),
                eps(FT),
                one(FT),
            )
        else
            one(FT)
        end
        cf_scale = if HasIncloudSed
            inv(
                clamp(
                    Fields.field_values(pre_col.ᶜcf_sed)[k],
                    sw.cf_floor,
                    one(FT),
                ),
            )
        else
            one(FT)
        end

        pred = _predicted_1m_precip_production(
            sw.mp,
            sw.tps,
            sw.vc,
            ρ_old,
            T0,
            w_air,
            q_v0,
            q0[1],
            q0[2],
            q0[4],
            S_in[1],
            S_in[2],
            S_in[4],
            w_above[1],
            w_above[2],
            w_above[4],
            dz_eff,
            sw.dt,
            c_vm0,
            cf_scale,
            Val(HasPrecipShaft),
            β_p,
            a_s_k,
        )
        S_in_eff_rai = S_in[3] + pred.P_rai
        w_rai, q_bar_rai = _two_step_sed_velocity(
            _tv_1m_rai,
            sw.vc,
            ρ_old,
            q0[3],
            S_in_eff_rai,
            w_above[3],
            dz_eff,
            sw.dt,
            one(FT),
            Val(HasPrecipShaft),
            β_p,
            a_s_k,
        )
        D_sed_rai = w_rai * inv_dz

        w_k = SA.SVector{4, FT}(pred.w_lcl, pred.w_icl, w_rai, pred.w_sno)
        q_bar_sed = SA.SVector{4, FT}(
            pred.q_bar_lcl,
            pred.q_bar_icl,
            q_bar_rai,
            pred.q_bar_sno,
        )
        D_sed = SA.SVector{4, FT}(
            pred.D_sed_lcl,
            pred.D_sed_icl,
            D_sed_rai,
            pred.D_sed_sno,
        )

        # In-sweep maximum-random precipitation-fraction overlap on `q̄_sed`.
        a_p_k = if HasPrecipFrac
            q_p_sed = q_bar_sed[3] + q_bar_sed[4]
            val_ap = ifelse(q_p_sed > sw.q_precip_min, a_p_cand, zero(FT))
            Fields.field_values(sgs_col.ᶜprecip_frac)[k] = val_ap
            a_p_above = val_ap
            val_ap
        else
            one(FT)
        end

        cell_res = if HasSGS
            _coupled_1m_cell_step_sgs(
                sw.sgs_quad,
                sw.sgs_opts,
                sw.mp,
                sw.tps,
                ρ_old,
                T0,
                w_air,
                q_tot0,
                q0,
                S_in,
                q_bar_sed,
                D_sed,
                Fields.field_values(sgs_col.ᶜT′T′)[k],
                Fields.field_values(sgs_col.ᶜq′q′)[k],
                Fields.field_values(sgs_col.ᶜcorr_Tq)[k],
                Fields.field_values(sgs_col.ᶜsgs_moments)[k],
                Fields.field_values(sgs_col.ᶜα)[k],
                a_p_k,
                sw.dt,
                Val(false),
            )
        else
            _coupled_1m_cell_step(
                sw.mp,
                sw.tps,
                ρ_old,
                T0,
                w_air,
                q_tot0,
                q0,
                S_in,
                q_bar_sed,
                D_sed,
                sw.dt,
                Val(false),
            )
        end

        q_new = cell_res.q_new
        q_out = cell_res.q_out

        scale_flux = (ρ_old * V_k) / max(sw.dt, floatmin(FT))
        F_mass_out = q_out .* scale_flux
        Δρ_sed = (sw.dt / V_k) * (sum(F_mass_in) - sum(F_mass_out))

        ρ_new = ρ_old + Δρ_sed
        ρqt_new = ρqt_old + Δρ_sed
        ρq_new = q_new .* ρ_old

        F_e_out = if sum(q_out) > zero(FT)
            _, f_out, _ = _donor_post_mp_temperature_and_energy_out(
                tps,
                ρ_old,
                ρe_old,
                F_e_in,
                V_k,
                sw.dt,
                ρ_new,
                ρqt_new,
                ρq_new,
                q_out,
                w_k,
                uvw_k,
                Φ_k,
            )
            f_out
        else
            zero(FT)
        end
        ρe_new = ρe_old + (sw.dt / V_k) * (F_e_in - F_e_out)

        ρ_fv[k] = ρ_new
        ρqt_fv[k] = ρqt_new
        ρql_fv[k] = ρq_new[1]
        ρqi_fv[k] = ρq_new[2]
        ρqr_fv[k] = ρq_new[3]
        ρqs_fv[k] = ρq_new[4]
        ρe_fv[k] = ρe_new

        wl_fv[k] = w_k[1]
        wi_fv[k] = w_k[2]
        wr_fv[k] = w_k[3]
        ws_fv[k] = w_k[4]
        whht_fv[k] = WVec(F_e_out / max(A_bot * ρ_new, floatmin(FT)))

        F_mass_in = F_mass_out
        F_e_in = F_e_out
        w_above = w_k

        if k == 1
            inv_A_sfc = one(FT) / max(A_bot, floatmin(FT))
            Fields.field_values(sfc_col.rain)[] =
                -(F_mass_out[3] + F_mass_out[1]) * inv_A_sfc
            Fields.field_values(sfc_col.snow)[] =
                -(F_mass_out[4] + F_mass_out[2]) * inv_A_sfc
            Fields.field_values(sfc_col.energy)[] = -F_e_out * inv_A_sfc
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# PrognosticEDMFX single-updraft + environment coupled 1M column sweep
# ----------------------------------------------------------------------------

@inline _passive_sgs1_fvs(sgs1, passive_names) = unrolled_map(
    name -> Fields.field_values(MatrixFields.get_field(sgs1, name)),
    passive_names,
)

@inline function _rescale_passive_sgs!(passive_fvs, k::Int, scale)
    unrolled_foreach(passive_fvs) do χ1_fv
        @inbounds χ1_fv[k] *= scale
    end
    return nothing
end

"""
    _sweep_env_value(
        χ_name, ρχ, ρ, sgsʲs, ρa⁰, turbconv_model, ρχ_pos, ρa¹, χ¹, ρa⁰_pool,
    )

Return the non-negative environment specific value `χ⁰` of one cell for the
EDMFX microphysics sweeps. `χ⁰` comes from `specific_env_value` applied to the
unprojected state (`ρχ`, `ρ`, the pointwise single-updraft state `sgsʲs`, and
`ρa⁰ = ρ - ρa¹`), the reconstruction that EDMFX transport, the closures, and
`set_prognostic_edmf_precomputed_quantities_environment!` use. It is bounded by
the residual `(ρχ_pos - ρa¹ χ¹) / ρa⁰_pool` of the sweep's non-negative updraft
pool, so the two subdomain pools never hold more than the grid mean. For a
realizable state (`0 ≤ ρa¹ χ¹ ≤ ρχ`, `ρa¹` below its clamp, and environment
area above `42 a_half`), the bound is inactive and `χ⁰` equals
`max(0, specific_env_value(...))` exactly.
"""
@inline function _sweep_env_value(
    χ_name,
    ρχ,
    ρ,
    sgsʲs,
    ρa⁰,
    turbconv_model,
    ρχ_pos,
    ρa¹,
    χ¹,
    ρa⁰_pool,
)
    χ⁰ = specific_env_value(χ_name, ρχ, ρ, sgsʲs, ρa⁰, turbconv_model)
    χ⁰_max = max(zero(ρ), ρχ_pos - ρa¹ * χ¹) / ρa⁰_pool
    return min(max(zero(ρ), χ⁰), χ⁰_max)
end

# ----------------------------------------------------------------------------
# Subdomain helpers shared by the 1M and 2M EDMFX sweeps
# ----------------------------------------------------------------------------

"""
    _sweep_subdomain_masses(ρ, ρa¹, ρ¹)

Updraft and environment masses of one cell at the start of an EDMFX sweep
step: the floored updraft density `ρ1_k`, the updraft mass `ρa1_old` and area
fraction `a1_eff` (both zero unless `has_updraft`), and the environment mass
`ρa0_old = ρ - ρa1_old`, floored at `eps`. The updraft holds at most
`1 - 10⁻⁴` of the cell mass and area.
"""
@inline function _sweep_subdomain_masses(ρ, ρa1, ρ1)
    FT = typeof(ρ)
    ρ1_k = max(eps(FT), ρ1)
    ρa1_raw = clamp(ρa1, zero(FT), ρ * (one(FT) - FT(1e-4)))
    a1_k = clamp(draft_area(ρa1_raw, ρ1_k), zero(FT), one(FT) - FT(1e-4))
    has_updraft = (ρa1_raw > ϵ_numerics(FT)) & (a1_k > zero(FT))
    ρa1_old = ifelse(has_updraft, ρa1_raw, zero(FT))
    a1_eff = ifelse(has_updraft, a1_k, zero(FT))
    ρa0_old = max(eps(FT), ρ - ρa1_old)
    return (; ρ1_k, has_updraft, ρa1_old, a1_eff, ρa0_old)
end

# Updraft specific value `χ¹`, capped so that the updraft pool `ρa¹ χ¹` holds
# at most the non-negative grid-mean pool `ρχ_pos`.
@inline _cap_updraft_pool(χ1, ρχ_pos, ρa1) =
    ifelse(ρa1 * χ1 > ρχ_pos, ρχ_pos / max(ρa1, floatmin(ρa1)), χ1)

"""
    _sweep_updraft_entry_pools(ρa1_old, qt_gs_pos, qt1, q1, ρq_gs_pos)

Conservative entry projection of the updraft water of one cell: the total
water `qt1_0` and the condensate species `q1_0` (an `SVector` in the order
`lcl`, `icl`, `rai`, `sno`) are non-negative, the species sum to at most
`qt1_0`, and each updraft pool holds at most its non-negative grid-mean pool
(`qt_gs_pos`, `ρq_gs_pos`), so that the environment pools start non-negative.
"""
@inline function _sweep_updraft_entry_pools(ρa1_old, qt_gs_pos, qt1, q1, ρq_gs_pos)
    qt1_0 = _cap_updraft_pool(max(zero(qt1), qt1), qt_gs_pos, ρa1_old)
    q1_raw = _partition_nonneg_condensates(qt1_0, q1[1], q1[2], q1[3], q1[4])
    q1_0 = _cap_updraft_pool.(q1_raw, ρq_gs_pos, ρa1_old)
    return (qt1_0, q1_0)
end

"""
    _sweep_env_water_pools(
        sgsʲs_k, turbconv_model, ρ, ρa⁰_k, ρa1_old, ρa0_old,
        ρq, ρq_gs_pos, q1_0, ρqt, qt_gs_pos, qt1_0,
    )

Environment condensate species `q0_env` (an `SVector` in the order `lcl`,
`icl`, `rai`, `sno`) and total water `qt0_env ≥ sum(q0_env)` of one cell from
`_sweep_env_value`, the environment that transport, the closures, and `ᶜT⁰`
see. `ρq` and `ρqt` are the unprojected grid-mean densities, `ρq_gs_pos` and
`qt_gs_pos` their non-negative parts, and `q1_0`, `qt1_0` the projected updraft
values.
"""
@inline function _sweep_env_water_pools(
    sgsʲs_k,
    turbconv_model,
    ρ,
    ρa⁰_k,
    ρa1_old,
    ρa0_old,
    ρq,
    ρq_gs_pos,
    q1_0,
    ρqt,
    qt_gs_pos,
    qt1_0,
)
    FT = typeof(ρ)
    w_env = sgs_weight_function(ρa⁰_k / ρ, turbconv_model.a_half)
    q0_env = SA.SVector{4, FT}(
        _sweep_env_value(
            @name(q_lcl), ρq[1], ρ, sgsʲs_k, ρa⁰_k, w_env,
            ρq_gs_pos[1], ρa1_old, q1_0[1], ρa0_old,
        ),
        _sweep_env_value(
            @name(q_icl), ρq[2], ρ, sgsʲs_k, ρa⁰_k, w_env,
            ρq_gs_pos[2], ρa1_old, q1_0[2], ρa0_old,
        ),
        _sweep_env_value(
            @name(q_rai), ρq[3], ρ, sgsʲs_k, ρa⁰_k, w_env,
            ρq_gs_pos[3], ρa1_old, q1_0[3], ρa0_old,
        ),
        _sweep_env_value(
            @name(q_sno), ρq[4], ρ, sgsʲs_k, ρa⁰_k, w_env,
            ρq_gs_pos[4], ρa1_old, q1_0[4], ρa0_old,
        ),
    )
    qt0_env = max(
        sum(q0_env),
        _sweep_env_value(
            @name(q_tot), ρqt, ρ, sgsʲs_k, ρa⁰_k, w_env,
            qt_gs_pos, ρa1_old, qt1_0, ρa0_old,
        ),
    )
    return (q0_env, qt0_env)
end

"""
    _shaft_transfer_fractions(a1_above, a1_eff)

Lateral redistribution of the sedimentation shafts across the tilted updraft
boundary at the top face of a cell, in donor form, for the updraft area
fractions `a1_above` of the cell above and `a1_eff` of the cell:

  - Narrowing downward (`a1_above > a1_eff`): the updraft keeps the fraction
    `keep = a1_eff / a1_above` of its shaft and detrains `detr = 1 - keep`
    into the environment.
  - Widening downward (`a1_eff > a1_above`): the fraction
    `entr = (a1_eff - a1_above) / (1 - a1_above)` of the environment shaft
    enters the updraft, the transfer `-∂_z a ρ⁰w⁰q⁰` for every updraft area.

Otherwise `keep = 1` and `detr = entr = 0`.
"""
@inline function _shaft_transfer_fractions(a1_above, a1_eff)
    FT = typeof(a1_eff)
    keep = ifelse(
        a1_above > a1_eff,
        a1_eff / max(a1_above, floatmin(FT)),
        one(FT),
    )
    entr = ifelse(
        a1_eff > a1_above,
        clamp(
            (a1_eff - a1_above) / max(one(FT) - a1_above, FT(1e-4)),
            zero(FT),
            one(FT),
        ),
        zero(FT),
    )
    return (; keep, detr = one(FT) - keep, entr)
end

# Shaft fluxes entering the updraft and the environment of a cell from the
# updraft flux `F_1` and environment flux `F_0` leaving the cell above, with
# the fractions `tr` of `_shaft_transfer_fractions`.
@inline _shaft_into_updraft(F_1, F_0, tr) = tr.keep .* F_1 .+ tr.entr .* F_0
@inline _shaft_into_environment(F_1, F_0, tr) =
    (one(tr.entr) - tr.entr) .* F_0 .+ tr.detr .* F_1

"""
    _sweep_updraft_post_state(
        tps, ρa1_old, ρa1_new, qt1_new, q1_int_new, mse1_old, FhΦ_1,
        q1_out, w1_k, uvw1, Φ_k, V_k, dt,
    )

Updraft state of one cell after its microphysics and sedimentation step, from
the `mse` balance: the available `ρa¹ mse¹` is the old value plus the static
energy `h + Φ` flux `FhΦ_1` entering through the top face, and the temperature
`T1_post` closes it for the new air (`ρa1_new`, `qt1_new`, species
`q1_int_new`) and the condensate `ρa1_old q1_out` leaving through the bottom
face at `T1_post`.

Returns `T1_post`, the liquid and ice contents `q1_liq_p`, `q1_ice_p`, the new
`mse1_new`, the outgoing static-energy and kinetic-energy fluxes `F_hΦ_out_1`
and `F_K_out_1` (with the fall speeds `w1_k` and the updraft velocity `uvw1`),
and the updraft total energy `ρe1_post`.
"""
@inline function _sweep_updraft_post_state(
    tps,
    ρa1_old,
    ρa1_new,
    qt1_new,
    q1_int_new,
    mse1_old,
    FhΦ_1,
    q1_out,
    w1_k,
    uvw1,
    Φ_k,
    V_k,
    dt,
)
    FT = typeof(dt)
    T_0 = FT(TD.Parameters.T_0(tps))
    cv_l = FT(TD.Parameters.cv_l(tps))
    cv_i = FT(TD.Parameters.cv_i(tps))
    e_ice_T0 = TD.internal_energy_ice(tps, T_0)

    mse1_avail = ρa1_old * mse1_old + (dt / V_k) * FhΦ_1
    m1_out = ρa1_old .* q1_out
    m1_out_liq = m1_out[1] + m1_out[3]
    m1_out_ice = m1_out[2] + m1_out[4]
    q1_liq_p = q1_int_new[1] + q1_int_new[3]
    q1_ice_p = q1_int_new[2] + q1_int_new[4]
    h_air_T0 = TD.enthalpy(
        TD.internal_energy(tps, T_0, qt1_new, q1_liq_p, q1_ice_p),
        TD.gas_constant_air(tps, qt1_new, q1_liq_p, q1_ice_p),
        T_0,
    )
    cp_m1_p = TD.cp_m(tps, qt1_new, q1_liq_p, q1_ice_p)
    H_ref =
        ρa1_new * (h_air_T0 + Φ_k) +
        sum(m1_out) * Φ_k +
        m1_out_ice * e_ice_T0
    C_h1 = muladd(
        ρa1_new,
        cp_m1_p,
        muladd(m1_out_liq, cv_l, m1_out_ice * cv_i),
    )
    T1_post = T_0 + (mse1_avail - H_ref) / max(C_h1, floatmin(FT))

    e_liq1 = TD.internal_energy_liquid(tps, T1_post)
    e_ice1 = TD.internal_energy_ice(tps, T1_post)
    v_dt = V_k / max(dt, floatmin(FT))
    F_hΦ_out_1 =
        v_dt * (
            m1_out[1] * (e_liq1 + Φ_k) +
            m1_out[2] * (e_ice1 + Φ_k) +
            m1_out[3] * (e_liq1 + Φ_k) +
            m1_out[4] * (e_ice1 + Φ_k)
        )
    F_K_out_1 =
        v_dt * (
            m1_out[1] * sedimentation_kinetic_energy(w1_k[1], uvw1) +
            m1_out[2] * sedimentation_kinetic_energy(w1_k[2], uvw1) +
            m1_out[3] * sedimentation_kinetic_energy(w1_k[3], uvw1) +
            m1_out[4] * sedimentation_kinetic_energy(w1_k[4], uvw1)
        )

    e_int1_post =
        TD.internal_energy(tps, T1_post, qt1_new, q1_liq_p, q1_ice_p)
    ρe1_post = ρa1_new * (e_int1_post + norm_sqr(uvw1) / 2 + Φ_k)
    mse1_new = (mse1_avail - (dt / V_k) * F_hΦ_out_1) * (one(FT) / ρa1_new)
    return (;
        T1_post, q1_liq_p, q1_ice_p, mse1_new, F_hΦ_out_1, F_K_out_1, ρe1_post,
    )
end

"""
    _sweep_grid_mean_update(
        tps, ρ_old, ρqt_old, ρe_old, ρq_gs_pos, ρa0_old, ρa1_old,
        q0_env, q0_ode, q0_out, q1_0, q1_ode, F_in, F_out,
        ρa1_new_eff, qt1_new_eff, ρe1_post, w0_k, uvw0, Φ_k, V_k, dt,
    )

Grid-mean update of one cell from its subdomain steps, with the environment
energy export that closes the total energy budget.

The mass and total water change by the net sedimentation flux of the two
subdomains. The condensate species change by the subdomain increments
`ρa⁰ (q0_ode - q0_env) + ρa¹ (q1_ode - q1_0)`, so that a grid-mean share that
the environment value leaves out stays in place. The cell energy left after
the fluxes `F_in` enter and the updraft flux leaves, minus the updraft energy
`ρe1_post`, belongs to the environment; `_donor_post_mp_temperature_and_energy_out`
turns it into the environment temperature `T0_post` and the outgoing
environment flux `F_e_out_0` with kinetic part `F_K_out_0`.

`F_in` and `F_out` hold the fluxes through the top and bottom faces: the
condensate mass fluxes `mass_1`, `mass_0` and the static-energy and
kinetic-energy fluxes `hΦ_1`, `K_1` (and `hΦ_0`, `K_0` in `F_in`).
"""
@inline function _sweep_grid_mean_update(
    tps,
    ρ_old,
    ρqt_old,
    ρe_old,
    ρq_gs_pos,
    ρa0_old,
    ρa1_old,
    q0_env,
    q0_ode,
    q0_out,
    q1_0,
    q1_ode,
    F_in,
    F_out,
    ρa1_new_eff,
    qt1_new_eff,
    ρe1_post,
    w0_k,
    uvw0,
    Φ_k,
    V_k,
    dt,
)
    FT = typeof(dt)
    F_in_tot = sum(F_in.mass_1) + sum(F_in.mass_0)
    F_out_tot = sum(F_out.mass_1) + sum(F_out.mass_0)
    Δρ_sed = (dt / V_k) * (F_in_tot - F_out_tot)

    ρ_new = ρ_old + Δρ_sed
    ρqt_new = ρqt_old + Δρ_sed
    # The floor removes round-off of a fully depleted pool.
    ρq_new = max.(
        zero(FT),
        ρq_gs_pos .+ ρa0_old .* (q0_ode .- q0_env) .+
        ρa1_old .* (q1_ode .- q1_0),
    )

    ρa0_new = max(eps(FT), ρ_new - ρa1_new_eff)
    ρqt0_new = max(zero(FT), ρqt_new - ρa1_new_eff * qt1_new_eff)
    ρq0_new = q0_ode .* ρa0_old
    F_e_in_tot = F_in.hΦ_1 + F_in.K_1 + F_in.hΦ_0 + F_in.K_0
    F_e_out_1 = F_out.hΦ_1 + F_out.K_1
    ρe_rem = ρe_old + (dt / V_k) * (F_e_in_tot - F_e_out_1)
    ρe0_avail = ρe_rem - ρe1_post
    T0_post, F_e_out_0, F_K_out_0 = _donor_post_mp_temperature_and_energy_out(
        tps,
        ρa0_old,
        ρe0_avail,
        zero(FT),
        V_k,
        dt,
        ρa0_new,
        ρqt0_new,
        ρq0_new,
        q0_out,
        w0_k,
        uvw0,
        Φ_k,
    )
    ρe_new = ρe_rem - (dt / V_k) * F_e_out_0
    return (;
        ρ_new, ρqt_new, ρq_new, ρe_new, T0_post, F_e_out_0, F_K_out_0, F_e_out_1,
    )
end

struct Coupled1MEDMFXColumnSweep{
    HasSGS,
    HasPrecipFrac,
    HasPrecipShaft,
    HasIncloudSed,
    HasEnvAcnvVel,
    MPU,
    MP,
    TPS,
    VC,
    SGQ,
    SGO,
    PN,
    TC,
    FT,
} <: ColumnSweep
    mp_up::MPU
    mp::MP
    tps::TPS
    vc::VC
    sgs_quad::SGQ
    sgs_opts::SGO
    passive_names::PN
    turbconv_model::TC
    dt::FT
    f_decay::FT
    q_precip_min::FT
    cf_floor::FT
    c_w::FT
    w_d::FT
    a_min::FT
end

@inline function (
    sw::Coupled1MEDMFXColumnSweep{
        HasSGS,
        HasPrecipFrac,
        HasPrecipShaft,
        HasIncloudSed,
        HasEnvAcnvVel,
    }
)
    (
        Yc_col,
        geom_col,
        pre_col,
        edmf_col,
        sgs_col,
        sfc_col,
    ) where {
        HasSGS,
        HasPrecipFrac,
        HasPrecipShaft,
        HasIncloudSed,
        HasEnvAcnvVel,
    }
    FT = typeof(sw.dt)
    tps = _unwrapped(sw.tps)
    mp = _unwrapped(sw.mp)

    ρ_fv = Fields.field_values(Yc_col.ρ)
    ρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    ρql_fv = Fields.field_values(Yc_col.ρq_lcl)
    ρqi_fv = Fields.field_values(Yc_col.ρq_icl)
    ρqr_fv = Fields.field_values(Yc_col.ρq_rai)
    ρqs_fv = Fields.field_values(Yc_col.ρq_sno)

    sgs1 = Yc_col.sgsʲs.:(1)
    ρa1_fv = Fields.field_values(sgs1.ρa)
    mse1_fv = Fields.field_values(sgs1.mse)
    qt1_fv = Fields.field_values(sgs1.q_tot)
    ql1_fv = Fields.field_values(sgs1.q_lcl)
    qi1_fv = Fields.field_values(sgs1.q_icl)
    qr1_fv = Fields.field_values(sgs1.q_rai)
    qs1_fv = Fields.field_values(sgs1.q_sno)
    passive1_fvs = _passive_sgs1_fvs(sgs1, sw.passive_names)

    lg_fv = Fields.field_values(geom_col.ᶜlg)
    flg_fv = Fields.field_values(geom_col.ᶠlg)
    dξ_fv = Fields.field_values(geom_col.ᶠ∂ξ³∂x³)
    Φ_fv = Fields.field_values(geom_col.ᶜΦ)

    wl_fv = Fields.field_values(pre_col.ᶜwₗ)
    wi_fv = Fields.field_values(pre_col.ᶜwᵢ)
    wr_fv = Fields.field_values(pre_col.ᶜwᵣ)
    ws_fv = Fields.field_values(pre_col.ᶜwₛ)
    whht_fv = Fields.field_values(pre_col.ᶜwₕhₜ)

    ρ1_fv = Fields.field_values(edmf_col.ᶜρ¹)
    T1_fv = Fields.field_values(edmf_col.ᶜT¹)
    u1_fv = Fields.field_values(edmf_col.ᶜu¹)
    wl1_fv = Fields.field_values(edmf_col.ᶜwₗ¹)
    wi1_fv = Fields.field_values(edmf_col.ᶜwᵢ¹)
    wr1_fv = Fields.field_values(edmf_col.ᶜwᵣ¹)
    ws1_fv = Fields.field_values(edmf_col.ᶜwₛ¹)
    T0_env_fv = Fields.field_values(edmf_col.ᶜT⁰)
    p_fv = Fields.field_values(edmf_col.ᶜp)
    u0_fv = Fields.field_values(edmf_col.ᶜu⁰)

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    z = zero(FT)
    z4 = zero(SA.SVector{4, FT})

    F_mass_in_1 = z4
    F_mass_in_0 = z4
    F_hΦ_in_1 = z
    F_K_in_1 = z
    F_hΦ_in_0 = z
    F_K_in_0 = z
    w_above_1 = z4
    w_above_0 = z4
    a1_above = z
    a_p_above = z

    @inbounds for step_idx in 0:(nz - 1)
        k = nz - step_idx
        ρ_old = ρ_fv[k]
        ρa1_k_raw = ρa1_fv[k]
        (; ρ1_k, has_updraft, ρa1_old, a1_eff, ρa0_old) =
            _sweep_subdomain_masses(ρ_old, ρa1_k_raw, ρ1_fv[k])
        ρql_k = ρql_fv[k]
        ρqi_k = ρqi_fv[k]
        ρqr_k = ρqr_fv[k]
        ρqs_k = ρqs_fv[k]

        ρe_old = ρe_fv[k]
        ρqt_old = ρqt_fv[k]
        Φ_k = Φ_fv[k]

        lg_k = lg_fv[k]
        V_k = lg_k.J
        A_bot = flg_fv[k].J * dξ_fv[k]
        dz_eff = V_k / max(A_bot, floatmin(FT))
        inv_dz = one(FT) / max(dz_eff, floatmin(FT))

        # Conservative entry projection so environment pools start non-negative
        # whenever grid-mean pools are non-negative.
        qt_gs_pos = max(z, ρqt_old)
        ρq_k = SA.SVector{4, FT}(ρql_k, ρqi_k, ρqr_k, ρqs_k)
        ρq_gs_pos = max.(z, ρq_k)
        if ρa1_k_raw == z
            qt1_0 = z
            q1_0 = z4
            inv_ρ_old = one(FT) / max(ρ_old, eps(FT))
            q0_env = ρq_gs_pos .* inv_ρ_old
            qt0_env = max(sum(q0_env), qt_gs_pos * inv_ρ_old)
        else
            qt1_k = qt1_fv[k]
            ql1_k = ql1_fv[k]
            qi1_k = qi1_fv[k]
            qr1_k = qr1_fv[k]
            qs1_k = qs1_fv[k]
            qt1_0, q1_0 = _sweep_updraft_entry_pools(
                ρa1_old,
                qt_gs_pos,
                qt1_k,
                SA.SVector{4, FT}(ql1_k, qi1_k, qr1_k, qs1_k),
                ρq_gs_pos,
            )
            sgsʲs_k = ((;
                ρa = ρa1_k_raw,
                q_tot = qt1_k,
                q_lcl = ql1_k,
                q_icl = qi1_k,
                q_rai = qr1_k,
                q_sno = qs1_k,
            ),)
            ρa⁰_k = ρa⁰(ρ_old, sgsʲs_k, sw.turbconv_model)
            q0_env, qt0_env = _sweep_env_water_pools(
                sgsʲs_k,
                sw.turbconv_model,
                ρ_old,
                ρa⁰_k,
                ρa1_old,
                ρa0_old,
                ρq_k,
                ρq_gs_pos,
                q1_0,
                ρqt_old,
                qt_gs_pos,
                qt1_0,
            )
        end

        # Lateral redistribution across the tilted updraft boundary at face
        # `k + ½`. The updraft `mse` receives the static energy `h + Φ` of the
        # entering condensate; its kinetic energy stays in the grid-mean
        # `ρe_tot`.
        transfer = _shaft_transfer_fractions(a1_above, a1_eff)
        F_in_1 = _shaft_into_updraft(F_mass_in_1, F_mass_in_0, transfer)
        F_in_0 = _shaft_into_environment(F_mass_in_1, F_mass_in_0, transfer)
        FhΦ_1 = _shaft_into_updraft(F_hΦ_in_1, F_hΦ_in_0, transfer)

        # --- Updraft 1 step ---
        F_mass_out_1 = z4
        F_hΦ_out_1 = z
        F_K_out_1 = z
        w1_k = z4
        q_sed1 = z4
        q1_ode = q1_0
        ρa1_new_eff = z
        qt1_new_eff = z
        ρe1_post = z

        if has_updraft
            uvw1 = Geometry.UVWVector(u1_fv[k], lg_k)
            T1_0 = T1_fv[k]
            S_in_1 = F_in_1 .* (one(FT) / (ρa1_old * V_k))
            qv1_0 = max(zero(FT), qt1_0 - sum(q1_0))
            cp_m1 = TD.cp_m(tps, qt1_0, q1_0[1] + q1_0[3], q1_0[2] + q1_0[4])

            pred1 = _predicted_1m_precip_production(
                sw.mp_up,
                sw.tps,
                sw.vc,
                ρ1_k,
                T1_0,
                uvw1.w,
                qv1_0,
                q1_0[1],
                q1_0[2],
                q1_0[4],
                S_in_1[1],
                S_in_1[2],
                S_in_1[4],
                w_above_1[1],
                w_above_1[2],
                w_above_1[4],
                dz_eff,
                sw.dt,
                cp_m1,
            )
            w_rai1, q_bar_rai1 = _two_step_sed_velocity(
                _tv_1m_rai,
                sw.vc,
                ρ1_k,
                q1_0[3],
                S_in_1[3] + pred1.P_rai,
                w_above_1[3],
                dz_eff,
                sw.dt,
            )
            w1_k = SA.SVector{4, FT}(
                pred1.w_lcl,
                pred1.w_icl,
                w_rai1,
                pred1.w_sno,
            )
            q_sed1 = SA.SVector{4, FT}(
                pred1.q_bar_lcl,
                pred1.q_bar_icl,
                q_bar_rai1,
                pred1.q_bar_sno,
            )
            D_sed1 = w1_k .* inv_dz

            res1 = _coupled_1m_cell_step(
                sw.mp_up,
                sw.tps,
                ρ1_k,
                T1_0,
                uvw1.w,
                qt1_0,
                q1_0,
                S_in_1,
                q_sed1,
                D_sed1,
                sw.dt,
                Val(true),
            )
            q1_ode = res1.q_new
            q1_out = res1.q_out

            scale_1 = (ρa1_old * V_k) / max(sw.dt, floatmin(FT))
            F_mass_out_1 = q1_out .* scale_1
            Δρa1_sed = (sw.dt / V_k) * (sum(F_in_1) - sum(F_mass_out_1))
            # `has_updraft` gates on `ϵ_numerics(FT)` (`< eps(Float32)`), so
            # flooring at `floatmin(FT)` preserves positive division without
            # inflating `ρa1_new` or diluting `mse1` in `Float32` trace drafts.
            ρa1_new = max(floatmin(FT), ρa1_old + Δρa1_sed)

            # Update updraft `mse`, `q_tot`, and intensive condensates with
            # `r_mass = Δρa1_sed / (ρa1_old * dt)` dilution.
            inv_ρa1_new = one(FT) / ρa1_new
            qt1_new = max(zero(FT), (ρa1_old * qt1_0 + Δρa1_sed) * inv_ρa1_new)
            q1_int_new = (q1_ode .* ρa1_old) .* inv_ρa1_new

            # Post-microphysics updraft state from the `mse` balance.
            post1 = _sweep_updraft_post_state(
                tps,
                ρa1_old,
                ρa1_new,
                qt1_new,
                q1_int_new,
                mse1_fv[k],
                FhΦ_1,
                q1_out,
                w1_k,
                uvw1,
                Φ_k,
                V_k,
                sw.dt,
            )
            (; T1_post, q1_liq_p, q1_ice_p, mse1_new) = post1
            F_hΦ_out_1 = post1.F_hΦ_out_1
            F_K_out_1 = post1.F_K_out_1
            ρe1_post = post1.ρe1_post
            ρa1_new_eff = ρa1_new
            qt1_new_eff = qt1_new

            ρa1_fv[k] = ρa1_new
            mse1_fv[k] = mse1_new
            qt1_fv[k] = qt1_new
            ql1_fv[k] = q1_int_new[1]
            qi1_fv[k] = q1_int_new[2]
            qr1_fv[k] = q1_int_new[3]
            qs1_fv[k] = q1_int_new[4]
            _rescale_passive_sgs!(passive1_fvs, k, ρa1_old * inv_ρa1_new)
            T1_fv[k] = T1_post
            ρ1_fv[k] = max(
                eps(FT),
                TD.air_density(
                    tps,
                    T1_post,
                    p_fv[k],
                    qt1_new,
                    q1_liq_p,
                    q1_ice_p,
                ),
            )
        end

        wl1_fv[k] = w1_k[1]
        wi1_fv[k] = w1_k[2]
        wr1_fv[k] = w1_k[3]
        ws1_fv[k] = w1_k[4]

        # --- Environment 0 step ---
        uvw0 = Geometry.UVWVector(u0_fv[k], lg_k)
        T0_env = T0_env_fv[k]
        p_k = p_fv[k]
        ρ0_env = max(
            eps(FT),
            TD.air_density(
                tps,
                T0_env,
                p_k,
                qt0_env,
                q0_env[1] + q0_env[3],
                q0_env[2] + q0_env[4],
            ),
        )
        S_in_0 = F_in_0 .* (one(FT) / (ρa0_old * V_k))
        qv0_env = max(zero(FT), qt0_env - sum(q0_env))
        cp_m0 = TD.cp_m(
            tps,
            qt0_env,
            q0_env[1] + q0_env[3],
            q0_env[2] + q0_env[4],
        )

        a_p_cand = if HasPrecipFrac
            cf_k = max(zero(FT), Fields.field_values(sgs_col.ᶜcf)[k])
            step_idx == 0 ? cf_k :
            min(one(FT), max(cf_k, sw.f_decay * a_p_above))
        else
            one(FT)
        end
        β_p = HasPrecipShaft ? sw.sgs_opts.precip_incloud_fraction : z
        a_s_k = if HasPrecipShaft
            clamp(
                max(a_p_cand, sw.sgs_opts.precip_frac_floor),
                eps(FT),
                one(FT),
            )
        else
            one(FT)
        end
        cf_scale = if HasIncloudSed
            inv(
                clamp(
                    Fields.field_values(pre_col.ᶜcf_sed)[k],
                    sw.cf_floor,
                    one(FT),
                ),
            )
        else
            one(FT)
        end
        w_mp0 = if HasEnvAcnvVel && ((q0_env[1] > z) | (q0_env[2] > z))
            sgs_m0 = Fields.field_values(sgs_col.ᶜsgs_moments)[k]
            f_neq0 = env_detrained_condensate_fraction(
                q0_env[1],
                q0_env[2],
                FT(sgs_m0.sigma_S),
                FT(sgs_m0.λ_eq),
            )
            a1_pos = if has_updraft
                max(z, draft_area(ρa1_k_raw, ρ1_k))
            else
                z
            end
            w1_pos = if has_updraft
                max(z, Geometry.UVWVector(u1_fv[k], lg_k).w)
            else
                z
            end
            w_up0 = (a1_pos * w1_pos) / max(a1_pos, sw.a_min)
            env_autoconversion_velocity(
                uvw0.w,
                w_up0,
                f_neq0,
                sw.c_w,
                sw.w_d,
            )
        else
            uvw0.w
        end

        pred0 = _predicted_1m_precip_production(
            sw.mp,
            sw.tps,
            sw.vc,
            ρ0_env,
            T0_env,
            w_mp0,
            qv0_env,
            q0_env[1],
            q0_env[2],
            q0_env[4],
            S_in_0[1],
            S_in_0[2],
            S_in_0[4],
            w_above_0[1],
            w_above_0[2],
            w_above_0[4],
            dz_eff,
            sw.dt,
            cp_m0,
            cf_scale,
            Val(HasPrecipShaft),
            β_p,
            a_s_k,
        )
        w_rai0, q_bar_rai0 = _two_step_sed_velocity(
            _tv_1m_rai,
            sw.vc,
            ρ0_env,
            q0_env[3],
            S_in_0[3] + pred0.P_rai,
            w_above_0[3],
            dz_eff,
            sw.dt,
            one(FT),
            Val(HasPrecipShaft),
            β_p,
            a_s_k,
        )
        w0_k = SA.SVector{4, FT}(pred0.w_lcl, pred0.w_icl, w_rai0, pred0.w_sno)
        q_sed0 = SA.SVector{4, FT}(
            pred0.q_bar_lcl,
            pred0.q_bar_icl,
            q_bar_rai0,
            pred0.q_bar_sno,
        )
        D_sed0 = w0_k .* inv_dz

        a_p_k = if HasPrecipFrac
            q_p_sed =
                (
                    ρa0_old * (q_sed0[3] + q_sed0[4]) +
                    ρa1_old * (q_sed1[3] + q_sed1[4])
                ) / max(ρ_old, floatmin(FT))
            val_ap = ifelse(q_p_sed > sw.q_precip_min, a_p_cand, zero(FT))
            Fields.field_values(sgs_col.ᶜprecip_frac)[k] = val_ap
            a_p_above = val_ap
            val_ap
        else
            one(FT)
        end

        res0 = if HasSGS
            _coupled_1m_cell_step_sgs(
                sw.sgs_quad,
                sw.sgs_opts,
                sw.mp,
                sw.tps,
                ρ0_env,
                T0_env,
                w_mp0,
                qt0_env,
                q0_env,
                S_in_0,
                q_sed0,
                D_sed0,
                Fields.field_values(sgs_col.ᶜT′T′)[k],
                Fields.field_values(sgs_col.ᶜq′q′)[k],
                Fields.field_values(sgs_col.ᶜcorr_Tq)[k],
                Fields.field_values(sgs_col.ᶜsgs_moments)[k],
                Fields.field_values(sgs_col.ᶜα)[k],
                a_p_k,
                sw.dt,
                Val(true),
            )
        else
            _coupled_1m_cell_step(
                sw.mp,
                sw.tps,
                ρ0_env,
                T0_env,
                w_mp0,
                qt0_env,
                q0_env,
                S_in_0,
                q_sed0,
                D_sed0,
                sw.dt,
                Val(true),
            )
        end

        q0_ode = res0.q_new
        q0_out = res0.q_out
        scale_0 = (ρa0_old * V_k) / max(sw.dt, floatmin(FT))
        F_mass_out_0 = q0_out .* scale_0

        # Grid-mean extensive updates and total energy conservation.
        (; ρ_new, ρqt_new, ρq_new, ρe_new, T0_post, F_e_out_0, F_K_out_0, F_e_out_1) =
            _sweep_grid_mean_update(
                tps,
                ρ_old,
                ρqt_old,
                ρe_old,
                ρq_gs_pos,
                ρa0_old,
                ρa1_old,
                q0_env,
                q0_ode,
                q0_out,
                q1_0,
                q1_ode,
                (;
                    mass_1 = F_mass_in_1,
                    mass_0 = F_mass_in_0,
                    hΦ_1 = F_hΦ_in_1,
                    K_1 = F_K_in_1,
                    hΦ_0 = F_hΦ_in_0,
                    K_0 = F_K_in_0,
                ),
                (;
                    mass_1 = F_mass_out_1,
                    mass_0 = F_mass_out_0,
                    hΦ_1 = F_hΦ_out_1,
                    K_1 = F_K_out_1,
                ),
                ρa1_new_eff,
                qt1_new_eff,
                ρe1_post,
                w0_k,
                uvw0,
                Φ_k,
                V_k,
                sw.dt,
            )
        T0_env_fv[k] = T0_post

        ρ_fv[k] = ρ_new
        ρqt_fv[k] = ρqt_new
        ρql_fv[k] = ρq_new[1]
        ρqi_fv[k] = ρq_new[2]
        ρqr_fv[k] = ρq_new[3]
        ρqs_fv[k] = ρq_new[4]
        ρe_fv[k] = ρe_new

        # Mass-weighted grid-mean terminal velocities for diagnostics.
        ρw_gs = (q0_ode .* w0_k) .* ρa0_old + (q1_ode .* w1_k) .* ρa1_old
        wl_fv[k] = if ρq_new[1] > ϵ_numerics(FT)
            ρw_gs[1] / ρq_new[1]
        else
            _tv_1m_lcl(sw.vc, ρ_new, zero(FT))
        end
        wi_fv[k] = if ρq_new[2] > ϵ_numerics(FT)
            ρw_gs[2] / ρq_new[2]
        else
            _tv_1m_icl(sw.vc, ρ_new, zero(FT))
        end
        wr_fv[k] = if ρq_new[3] > ϵ_numerics(FT)
            ρw_gs[3] / ρq_new[3]
        else
            _tv_1m_rai(sw.vc, ρ_new, zero(FT))
        end
        ws_fv[k] = if ρq_new[4] > ϵ_numerics(FT)
            ρw_gs[4] / ρq_new[4]
        else
            _tv_1m_sno(sw.vc, ρ_new, zero(FT))
        end
        whht_fv[k] =
            WVec((F_e_out_0 + F_e_out_1) / max(A_bot * ρ_new, floatmin(FT)))

        F_mass_in_1 = F_mass_out_1
        F_mass_in_0 = F_mass_out_0
        F_hΦ_in_1 = F_hΦ_out_1
        F_K_in_1 = F_K_out_1
        F_hΦ_in_0 = F_e_out_0 - F_K_out_0
        F_K_in_0 = F_K_out_0
        w_above_1 = w1_k
        w_above_0 = w0_k
        a1_above = a1_eff

        if k == 1
            inv_A_sfc = one(FT) / max(A_bot, floatmin(FT))
            F_rain_tot =
                F_mass_out_0[3] +
                F_mass_out_0[1] +
                F_mass_out_1[3] +
                F_mass_out_1[1]
            F_snow_tot =
                F_mass_out_0[4] +
                F_mass_out_0[2] +
                F_mass_out_1[4] +
                F_mass_out_1[2]
            Fields.field_values(sfc_col.rain)[] = -F_rain_tot * inv_A_sfc
            Fields.field_values(sfc_col.snow)[] = -F_snow_tot * inv_A_sfc
            Fields.field_values(sfc_col.energy)[] =
                -(F_e_out_0 + F_e_out_1) * inv_A_sfc
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# GPU active-column compaction for coupled 1M column sweeps
# ----------------------------------------------------------------------------
#
# Columns are partitioned on the device into five active regimes (updraft,
# warm cloud, cold cloud, warm precipitation shaft, cold shaft/supersaturated
# clear air) plus clear columns (bucket 0). Clear columns write their
# diagnostic defaults during the prescan and skip the sweep; active columns
# are compacted into a `DataLayouts.IJHMask` by a device prefix sum (`cumsum!`)
# with zero host synchronization so each warp processes homogeneous regimes.

const _COUPLED_1M_COMPACT_BUCKETS = 5

struct _Coupled1MColumnCompactWorkspace{M, F, C}
    mask::M
    flags::F
    offsets::F
    cls::C
    Ni::Int
    Nj::Int
    Ncols::Int
end

const _COUPLED_1M_COMPACT_CACHE = Dict{Tuple{DataType, Int, Int, Int}, Any}()
const _COUPLED_1M_COMPACT_LOCK = ReentrantLock()

function _get_coupled_1m_compact_workspace(ρ_data::DataLayouts.VIJHWithF)
    _, Ni, Nj, Nh = size(ρ_data)
    key = (typeof(parent(ρ_data)), Ni, Nj, Nh)
    return @lock _COUPLED_1M_COMPACT_LOCK get!(_COUPLED_1M_COMPACT_CACHE, key) do
        Ncols = Ni * Nj * Nh
        mask = DataLayouts.IJHMask(ρ_data)
        flags =
            similar(parent(ρ_data), Int32, _COUPLED_1M_COMPACT_BUCKETS * Ncols)
        offsets = similar(flags)
        cls = similar(parent(ρ_data), UInt8, Ncols)
        _Coupled1MColumnCompactWorkspace(mask, flags, offsets, cls, Ni, Nj, Ncols)
    end
end

@inline function _write_bucket_flags!(
    flags,
    cls,
    col_idx::Int,
    Ncols::Int,
    bucket::UInt8,
)
    @inbounds begin
        cls[col_idx] = bucket
        flags[col_idx] = Int32(bucket == UInt8(1))
        flags[Ncols + col_idx] = Int32(bucket == UInt8(2))
        flags[2 * Ncols + col_idx] = Int32(bucket == UInt8(3))
        flags[3 * Ncols + col_idx] = Int32(bucket == UInt8(4))
        flags[4 * Ncols + col_idx] = Int32(bucket == UInt8(5))
    end
    return nothing
end

struct _Coupled1MColumnScatter{O, C, M} <: Function
    offsets::O
    cls::C
    mask::M
    Ni::Int
    Nj::Int
    Ncols::Int
end
Adapt.@adapt_structure _Coupled1MColumnScatter

@inline function (sc::_Coupled1MColumnScatter)(
    index::CartesianIndex{3},
    _col,
)
    i = index[1]
    j = index[2]
    h = index[3]
    col_idx = i + sc.Ni * ((j - 1) + sc.Nj * (h - 1))
    b = Int(@inbounds sc.cls[col_idx])
    if b > 0
        pos = Int(@inbounds sc.offsets[(b - 1) * sc.Ncols + col_idx])
        @inbounds sc.mask.i_map[pos] = i
        @inbounds sc.mask.j_map[pos] = j
        @inbounds sc.mask.h_map[pos] = h
    end
    if col_idx == sc.Ncols
        @inbounds sc.mask.N[1] =
            Int(sc.offsets[_COUPLED_1M_COMPACT_BUCKETS * sc.Ncols])
    end
    return nothing
end

struct _Coupled1MEDMFXPrescan{SW, F, C} <: Function
    sw::SW
    flags::F
    cls::C
    Ni::Int
    Nj::Int
    Ncols::Int
end
Adapt.@adapt_structure _Coupled1MEDMFXPrescan

@inline function (ps::_Coupled1MEDMFXPrescan{SW})(
    index::CartesianIndex{3},
    Yc_col,
    geom_col,
    pre_col,
    edmf_col,
    sgs_col,
    sfc_col,
) where {
    HasSGS,
    HasPrecipFrac,
    HasPrecipShaft,
    HasIncloudSed,
    HasEnvAcnvVel,
    SW <: Coupled1MEDMFXColumnSweep{
        HasSGS,
        HasPrecipFrac,
        HasPrecipShaft,
        HasIncloudSed,
        HasEnvAcnvVel,
    },
}
    sw = ps.sw
    FT = typeof(sw.dt)
    tps = _unwrapped(sw.tps)
    T_freeze = FT(TD.Parameters.T_freeze(tps))
    T_warm = T_freeze - FT(2)
    z = zero(FT)
    a_p_clear = if HasSGS
        max(HasPrecipFrac ? z : one(FT), sw.sgs_opts.precip_frac_floor)
    else
        one(FT)
    end
    β_precip = HasSGS ? sw.sgs_opts.precip_incloud_fraction : z

    ρ_fv = Fields.field_values(Yc_col.ρ)
    ρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    ρql_fv = Fields.field_values(Yc_col.ρq_lcl)
    ρqi_fv = Fields.field_values(Yc_col.ρq_icl)
    ρqr_fv = Fields.field_values(Yc_col.ρq_rai)
    ρqs_fv = Fields.field_values(Yc_col.ρq_sno)
    ρa1_fv = Fields.field_values(Yc_col.sgsʲs.:(1).ρa)
    qt1_fv = Fields.field_values(Yc_col.sgsʲs.:(1).q_tot)
    ρ1_fv = Fields.field_values(edmf_col.ᶜρ¹)
    T0_env_fv = Fields.field_values(edmf_col.ᶜT⁰)
    p_fv = Fields.field_values(edmf_col.ᶜp)

    nz = Spaces.nlevels(axes(Yc_col.ρ))
    has_up = false
    has_cld = false
    has_prc = false
    has_clear_act = false
    has_warm_act = false

    @inbounds for k in 1:nz
        ρ_old = ρ_fv[k]
        ρ1_k = max(eps(FT), ρ1_fv[k])
        ρa1_raw = clamp(ρa1_fv[k], z, ρ_old * (one(FT) - FT(1e-4)))
        a1_k = clamp(draft_area(ρa1_raw, ρ1_k), z, one(FT) - FT(1e-4))
        if (ρa1_raw > ϵ_numerics(FT)) & (a1_k > z)
            has_up = true
            break
        end
        ρql = ρql_fv[k]
        ρqi = ρqi_fv[k]
        ρqr = ρqr_fv[k]
        ρqs = ρqs_fv[k]
        if (ρ_old < eps(FT)) |
           (ρql < z) |
           (ρqi < z) |
           (ρqr < z) |
           (ρqs < z)
            has_clear_act = true
        elseif (ρql > z) | (ρqi > z)
            has_cld = true
            has_warm_act |= (T0_env_fv[k] >= T_warm)
        elseif (ρqr > z) | (ρqs > z)
            has_prc = true
            has_warm_act |= (T0_env_fv[k] >= T_warm)
        elseif !(has_cld | has_prc | has_clear_act)
            # The environment `q_tot` of the sweep without an updraft pool.
            sgsʲs_k = ((; ρa = ρa1_fv[k], q_tot = qt1_fv[k]),)
            qt0_env = _sweep_env_value(
                @name(q_tot),
                ρqt_fv[k],
                ρ_old,
                sgsʲs_k,
                ρa⁰(ρ_old, sgsʲs_k, sw.turbconv_model),
                sw.turbconv_model,
                max(z, ρqt_fv[k]),
                z,
                z,
                max(eps(FT), ρ_old),
            )
            T0_env = T0_env_fv[k]
            ρ0_env = max(
                eps(FT),
                TD.air_density(tps, T0_env, p_fv[k], qt0_env, z, z),
            )
            if HasSGS
                sgs_m = Fields.field_values(sgs_col.ᶜsgs_moments)[k]
                if FT(sgs_m.λ_eq) != FT(-Inf)
                    has_clear_act = true
                    has_warm_act |= (T0_env >= T_warm)
                else
                    is_dry, _ = _clear_sgs_mixed_phase_dry(
                        tps,
                        ρ0_env,
                        T0_env,
                        qt0_env,
                        Fields.field_values(sgs_col.ᶜT′T′)[k],
                    )
                    if !is_dry
                        has_clear_act = true
                        has_warm_act |= (T0_env >= T_warm)
                    end
                end
            else
                q_smin = if T0_env >= T_freeze
                    TD.q_vap_saturation(tps, T0_env, ρ0_env, TD.Liquid())
                else
                    TD.q_vap_saturation(tps, T0_env, ρ0_env, TD.Ice())
                end
                if qt0_env > q_smin
                    has_clear_act = true
                    has_warm_act |= (T0_env >= T_warm)
                end
            end
        end
    end

    bucket = if has_up
        UInt8(1)
    elseif has_cld
        has_warm_act ? UInt8(2) : UInt8(3)
    elseif has_prc
        (has_warm_act || (@inbounds T0_env_fv[1] >= T_warm)) ? UInt8(4) :
        UInt8(5)
    elseif has_clear_act
        UInt8(5)
    else
        UInt8(0)
    end

    if bucket == UInt8(0)
        T_0 = FT(TD.Parameters.T_0(tps))
        ρe_fv = Fields.field_values(Yc_col.ρe_tot)
        lg_fv = Fields.field_values(geom_col.ᶜlg)
        Φ_fv = Fields.field_values(geom_col.ᶜΦ)
        wl_fv = Fields.field_values(pre_col.ᶜwₗ)
        wi_fv = Fields.field_values(pre_col.ᶜwᵢ)
        wr_fv = Fields.field_values(pre_col.ᶜwᵣ)
        ws_fv = Fields.field_values(pre_col.ᶜwₛ)
        whht_fv = Fields.field_values(pre_col.ᶜwₕhₜ)
        wl1_fv = Fields.field_values(edmf_col.ᶜwₗ¹)
        wi1_fv = Fields.field_values(edmf_col.ᶜwᵢ¹)
        wr1_fv = Fields.field_values(edmf_col.ᶜwᵣ¹)
        ws1_fv = Fields.field_values(edmf_col.ᶜwₛ¹)
        u0_fv = Fields.field_values(edmf_col.ᶜu⁰)
        whht_z = WVec(z)
        @inbounds for k in 1:nz
            ρ_old = ρ_fv[k]
            inv_ρ = one(FT) / max(ρ_old, floatmin(FT))
            q_tot_post = max(z, ρqt_fv[k] * inv_ρ)
            e_int_air_T0 = TD.internal_energy(tps, T_0, q_tot_post, z, z)
            cv_m_post = TD.cv_m(tps, q_tot_post, z, z)
            uvw0 = Geometry.UVWVector(u0_fv[k], lg_fv[k])
            K_air = norm_sqr(uvw0) / 2
            E_ref_T0 = ρ_old * (e_int_air_T0 + K_air + Φ_fv[k])
            T0_env_fv[k] =
                T_0 + (ρe_fv[k] - E_ref_T0) / max(ρ_old * cv_m_post, floatmin(FT))
            wl_fv[k] = _tv_1m_lcl(sw.vc, ρ_old, z)
            wi_fv[k] = _tv_1m_icl(sw.vc, ρ_old, z)
            wr_fv[k] = _tv_1m_rai(sw.vc, ρ_old, z)
            ws_fv[k] = _tv_1m_sno(sw.vc, ρ_old, z)
            whht_fv[k] = whht_z
            wl1_fv[k] = z
            wi1_fv[k] = z
            wr1_fv[k] = z
            ws1_fv[k] = z
            if HasPrecipFrac
                Fields.field_values(sgs_col.ᶜprecip_frac)[k] = z
            end
        end
        @inbounds begin
            Fields.field_values(sfc_col.rain)[] = z
            Fields.field_values(sfc_col.snow)[] = z
            Fields.field_values(sfc_col.energy)[] = z
        end
    end

    col_idx = index[1] + ps.Ni * ((index[2] - 1) + ps.Nj * (index[3] - 1))
    _write_bucket_flags!(ps.flags, ps.cls, col_idx, ps.Ncols, bucket)
    return nothing
end

struct _Coupled1MGridMeanPrescan{SW, F, C} <: Function
    sw::SW
    flags::F
    cls::C
    Ni::Int
    Nj::Int
    Ncols::Int
end
Adapt.@adapt_structure _Coupled1MGridMeanPrescan

@inline function (ps::_Coupled1MGridMeanPrescan{SW})(
    index::CartesianIndex{3},
    Yc_col,
    geom_col,
    pre_col,
    sgs_col,
    sfc_col,
) where {
    HasSGS,
    HasPrecipFrac,
    HasPrecipShaft,
    HasIncloudSed,
    SW <: Coupled1MGridMeanSweep{
        HasSGS,
        HasPrecipFrac,
        HasPrecipShaft,
        HasIncloudSed,
    },
}
    sw = ps.sw
    FT = typeof(sw.dt)
    tps = _unwrapped(sw.tps)
    T_freeze = FT(TD.Parameters.T_freeze(tps))
    T_warm = T_freeze - FT(2)
    z = zero(FT)
    a_p_clear = if HasSGS
        max(HasPrecipFrac ? z : one(FT), sw.sgs_opts.precip_frac_floor)
    else
        one(FT)
    end
    β_precip = HasSGS ? sw.sgs_opts.precip_incloud_fraction : z

    ρ_fv = Fields.field_values(Yc_col.ρ)
    ρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    ρql_fv = Fields.field_values(Yc_col.ρq_lcl)
    ρqi_fv = Fields.field_values(Yc_col.ρq_icl)
    ρqr_fv = Fields.field_values(Yc_col.ρq_rai)
    ρqs_fv = Fields.field_values(Yc_col.ρq_sno)
    lg_fv = Fields.field_values(geom_col.ᶜlg)
    Φ_fv = Fields.field_values(geom_col.ᶜΦ)
    u_fv = Fields.field_values(pre_col.ᶜu)

    nz = Spaces.nlevels(axes(Yc_col.ρ))
    has_cld = false
    has_prc = false
    has_clear_act = false
    has_warm_act = false

    @inbounds for k in 1:nz
        ρ_old = ρ_fv[k]
        ρql = ρql_fv[k]
        ρqi = ρqi_fv[k]
        ρqr = ρqr_fv[k]
        ρqs = ρqs_fv[k]
        if (ρ_old < eps(FT)) |
           (ρql < z) |
           (ρqi < z) |
           (ρqr < z) |
           (ρqs < z)
            has_clear_act = true
        elseif (ρql > z) | (ρqi > z)
            has_cld = true
            if !has_warm_act
                inv_ρ = one(FT) / ρ_old
                q_tot0 = max(z, ρqt_fv[k] * inv_ρ)
                q0 = _partition_nonneg_condensates(
                    q_tot0,
                    ρql * inv_ρ,
                    ρqi * inv_ρ,
                    ρqr * inv_ρ,
                    ρqs * inv_ρ,
                )
                uvw_k = Geometry.UVWVector(u_fv[k], lg_fv[k])
                e_int = ρe_fv[k] * inv_ρ - norm_sqr(uvw_k) / 2 - Φ_fv[k]
                T0 = TD.air_temperature(
                    tps,
                    e_int,
                    q_tot0,
                    q0[1] + q0[3],
                    q0[2] + q0[4],
                )
                has_warm_act = T0 >= T_warm
            end
            has_warm_act && break
        elseif (ρqr > z) | (ρqs > z)
            has_prc = true
            if !has_warm_act
                inv_ρ = one(FT) / ρ_old
                q_tot0 = max(z, ρqt_fv[k] * inv_ρ)
                q0 = _partition_nonneg_condensates(
                    q_tot0,
                    z,
                    z,
                    ρqr * inv_ρ,
                    ρqs * inv_ρ,
                )
                uvw_k = Geometry.UVWVector(u_fv[k], lg_fv[k])
                e_int = ρe_fv[k] * inv_ρ - norm_sqr(uvw_k) / 2 - Φ_fv[k]
                T0 = TD.air_temperature(tps, e_int, q_tot0, q0[3], q0[4])
                has_warm_act = T0 >= T_warm
            end
        elseif !(has_cld | has_prc | has_clear_act)
            inv_ρ = one(FT) / ρ_old
            q_v0 = max(z, ρqt_fv[k] * inv_ρ)
            uvw_k = Geometry.UVWVector(u_fv[k], lg_fv[k])
            e_int = ρe_fv[k] * inv_ρ - norm_sqr(uvw_k) / 2 - Φ_fv[k]
            T0 = TD.air_temperature(tps, e_int, q_v0, z, z)
            if HasSGS
                sgs_m = Fields.field_values(sgs_col.ᶜsgs_moments)[k]
                if FT(sgs_m.λ_eq) != FT(-Inf)
                    has_clear_act = true
                    has_warm_act |= (T0 >= T_warm)
                else
                    is_dry, _ = _clear_sgs_mixed_phase_dry(
                        tps,
                        ρ_old,
                        T0,
                        q_v0,
                        Fields.field_values(sgs_col.ᶜT′T′)[k],
                    )
                    if !is_dry
                        has_clear_act = true
                        has_warm_act |= (T0 >= T_warm)
                    end
                end
            else
                q_smin = if T0 >= T_freeze
                    TD.q_vap_saturation(tps, T0, ρ_old, TD.Liquid())
                else
                    TD.q_vap_saturation(tps, T0, ρ_old, TD.Ice())
                end
                if q_v0 > q_smin
                    has_clear_act = true
                    has_warm_act |= (T0 >= T_warm)
                end
            end
        end
    end

    bucket = if has_cld
        has_warm_act ? UInt8(2) : UInt8(3)
    elseif has_prc
        has_warm_act ? UInt8(4) : UInt8(5)
    elseif has_clear_act
        UInt8(5)
    else
        UInt8(0)
    end

    if bucket == UInt8(0)
        wl_fv = Fields.field_values(pre_col.ᶜwₗ)
        wi_fv = Fields.field_values(pre_col.ᶜwᵢ)
        wr_fv = Fields.field_values(pre_col.ᶜwᵣ)
        ws_fv = Fields.field_values(pre_col.ᶜwₛ)
        whht_fv = Fields.field_values(pre_col.ᶜwₕhₜ)
        whht_z = WVec(z)
        @inbounds for k in 1:nz
            ρ_old = ρ_fv[k]
            wl_fv[k] = max(z, _tv_1m_lcl(sw.vc, ρ_old, z))
            wi_fv[k] = max(z, _tv_1m_icl(sw.vc, ρ_old, z))
            wr_fv[k] = max(z, _tv_1m_rai(sw.vc, ρ_old, z))
            ws_fv[k] = max(z, _tv_1m_sno(sw.vc, ρ_old, z))
            whht_fv[k] = whht_z
            if HasPrecipFrac
                Fields.field_values(sgs_col.ᶜprecip_frac)[k] = z
            end
        end
        @inbounds begin
            Fields.field_values(sfc_col.rain)[] = z
            Fields.field_values(sfc_col.snow)[] = z
            Fields.field_values(sfc_col.energy)[] = z
        end
    end

    col_idx = index[1] + ps.Ni * ((index[2] - 1) + ps.Nj * (index[3] - 1))
    _write_bucket_flags!(ps.flags, ps.cls, col_idx, ps.Ncols, bucket)
    return nothing
end

function _run_coupled_1m_edmfx_compacted!(
    ws::_Coupled1MColumnCompactWorkspace,
    sw::Coupled1MEDMFXColumnSweep,
    Yc,
    geom,
    pre,
    edmf,
    sgs_fields,
    sfc,
)
    prescan = _Coupled1MEDMFXPrescan(sw, ws.flags, ws.cls, ws.Ni, ws.Nj, ws.Ncols)
    DataLayouts.foreach_column(
        prescan,
        Yc,
        geom,
        pre,
        edmf,
        sgs_fields,
        sfc;
        enumerate = Val(true),
        shmem = Val(:none),
    )
    cumsum!(ws.offsets, ws.flags)
    scatter =
        _Coupled1MColumnScatter(ws.offsets, ws.cls, ws.mask, ws.Ni, ws.Nj, ws.Ncols)
    DataLayouts.foreach_column(
        scatter,
        Fields.field_values(Yc.ρ);
        enumerate = Val(true),
        shmem = Val(:none),
    )
    DataLayouts.foreach_column(
        sw,
        Yc,
        geom,
        pre,
        edmf,
        sgs_fields,
        sfc;
        mask = ws.mask,
        shmem = Val(:none),
    )
    return nothing
end

function _run_coupled_1m_gridmean_compacted!(
    ws::_Coupled1MColumnCompactWorkspace,
    sw::Coupled1MGridMeanSweep,
    Yc,
    geom,
    pre,
    sgs_fields,
    sfc,
)
    prescan = _Coupled1MGridMeanPrescan(sw, ws.flags, ws.cls, ws.Ni, ws.Nj, ws.Ncols)
    DataLayouts.foreach_column(
        prescan,
        Yc,
        geom,
        pre,
        sgs_fields,
        sfc;
        enumerate = Val(true),
        shmem = Val(:none),
    )
    cumsum!(ws.offsets, ws.flags)
    scatter =
        _Coupled1MColumnScatter(ws.offsets, ws.cls, ws.mask, ws.Ni, ws.Nj, ws.Ncols)
    DataLayouts.foreach_column(
        scatter,
        Fields.field_values(Yc.ρ);
        enumerate = Val(true),
        shmem = Val(:none),
    )
    DataLayouts.foreach_column(
        sw,
        Yc,
        geom,
        pre,
        sgs_fields,
        sfc;
        mask = ws.mask,
        shmem = Val(:none),
    )
    return nothing
end

# ============================================================================
# 9. Equilibrium 0M Column Sweep and Subcycled 2M Sweeps
# ============================================================================

struct Coupled0MGridMeanSweep{HasSGS, HasEvap, CM0P, TPS, SGQ, FT} <: ColumnSweep
    cm0::CM0P
    tps::TPS
    sgs_quad::SGQ
    dt::FT
    diag_w::FT
    diag_reset::Bool
end

# Weighted accumulation of a step-mean diagnostic over the substeps of a step.
@inline _acc_0m_diag(old, new, w, reset) = ifelse(reset, w * new, old + w * new)

# Top-down scan order with below-cloud evaporation, which needs the
# precipitation flux from above; bottom-up order otherwise.
@inline _scan_0m_levels(::Val{true}, nz) = nz:-1:1
@inline _scan_0m_levels(::Val{false}, nz) = 1:1:nz

"""
    (sw::Coupled0MGridMeanSweep{HasSGS, HasEvap})(Yc_col, geom_col, u_col,
        sgs_col, sfc_col, evap_col)

0-moment precipitation removal over `sw.dt` in one column of the grid mean.
Writes the surface rain, snow, and energy fluxes to `sfc_col`, the column
integrals of the cell sinks less the below-cloud evaporation, so total water
and total energy are conserved with the surface fluxes.

With `HasEvap`, the column is scanned top-down. Each cell first removes its
condensate excess, which joins the rain flux at `T ≥ T_freeze` and the snow
flux otherwise, and then takes up the evaporation of the flux entering from
above, `precipitation_evaporation_step` against the cell's grid-mean state.
The evaporated water returns to vapor with the flux-mean specific energy of
its phase. The evaporation rates and their column integrals accumulate into
`evap_col` and `sfc_col` with weight `sw.diag_w`, restarting when
`sw.diag_reset`.
"""
@inline function (sw::Coupled0MGridMeanSweep{HasSGS, HasEvap})(
    Yc_col,
    geom_col,
    u_col,
    sgs_col,
    sfc_col,
    evap_col,
) where {HasSGS, HasEvap}
    FT = typeof(sw.dt)
    tps = _unwrapped(sw.tps)
    cm0 = _unwrapped(sw.cm0)
    T_freeze = FT(TD.Parameters.T_freeze(tps))

    ρ_fv = Fields.field_values(Yc_col.ρ)
    ρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    lg_fv = Fields.field_values(geom_col.ᶜlg)
    flg_fv = Fields.field_values(geom_col.ᶠlg)
    dξ_fv = Fields.field_values(geom_col.ᶠ∂ξ³∂x³)
    Φ_fv = Fields.field_values(geom_col.ᶜΦ)
    u_fv = Fields.field_values(u_col)
    p_fv = Fields.field_values(evap_col.ᶜp)
    er_fv = Fields.field_values(evap_col.ρ_evap_rai)
    es_fv = Fields.field_values(evap_col.ρ_evap_sno)
    dq_fv = Fields.field_values(evap_col.ᶜρ_dq_tot_dt)
    de_fv = Fields.field_values(evap_col.ᶜρ_de_tot_dt)

    nz = Spaces.nlevels(axes(Yc_col.ρ))
    A_sfc = @inbounds flg_fv[1].J * dξ_fv[1]
    inv_A_sfc = one(FT) / max(A_sfc, floatmin(FT))
    p_sfc = @inbounds p_fv[1]

    rain_acc = zero(FT)
    snow_acc = zero(FT)
    energy_acc = zero(FT)
    evap_rai_acc = zero(FT)
    evap_sno_acc = zero(FT)
    flux = (zero(FT), zero(FT), zero(FT), zero(FT))

    @inbounds for k in _scan_0m_levels(Val(HasEvap), nz)
        ρ_old = ρ_fv[k]
        ρqt_old = ρqt_fv[k]
        ρe_old = ρe_fv[k]
        lg_k = lg_fv[k]
        V_k = lg_k.J
        Φ_k = Φ_fv[k]
        K_air = norm_sqr(Geometry.UVWVector(u_fv[k], lg_k)) / 2
        q_tot_pos = max(zero(FT), ρqt_old / ρ_old)
        e_int = ρe_old / ρ_old - K_air - Φ_k
        sa = TD.saturation_adjustment(tps, TD.ρe(), ρ_old, e_int, q_tot_pos)
        T_k = sa.T

        mp0 = if HasSGS
            microphysics_tendencies_0m(
                sw.sgs_quad,
                cm0,
                tps,
                ρ_old,
                T_k,
                q_tot_pos,
                Fields.field_values(sgs_col.ᶜT′T′)[k],
                Fields.field_values(sgs_col.ᶜq′q′)[k],
                Fields.field_values(sgs_col.ᶜcorr_Tq)[k],
                Φ_k,
                sw.dt,
            )
        else
            microphysics_tendencies_0m(
                cm0,
                tps,
                ρ_old,
                T_k,
                q_tot_pos,
                sa.q_liq,
                sa.q_ice,
                Φ_k,
                sw.dt,
            )
        end

        ρ_dq_dt = ρ_old * mp0.dq_tot_dt
        ρ_de_dt = ρ_dq_dt * mp0.e_tot_hlpr
        w_vol = V_k * inv_A_sfc
        is_warm = T_k >= T_freeze

        ρE_r = zero(FT)
        ρE_s = zero(FT)
        ρe_evap = zero(FT)
        if HasEvap
            q_vap = max(zero(FT), q_tot_pos - sa.q_liq - sa.q_ice)
            out = precipitation_evaporation_step(
                cm0.evaporation,
                tps,
                sw.dt,
                flux,
                (
                    ifelse(is_warm, -ρ_dq_dt, zero(FT)),
                    ifelse(is_warm, zero(FT), -ρ_dq_dt),
                    ifelse(is_warm, -ρ_de_dt, zero(FT)),
                    ifelse(is_warm, zero(FT), -ρ_de_dt),
                    w_vol,
                    ρ_old,
                    ρ_old,
                    T_k,
                    p_fv[k] / p_sfc,
                    q_vap,
                ),
            )
            flux = (out[1], out[2], out[3], out[4])
            ρE_r = out[5]
            ρE_s = out[6]
            ρe_evap = out[7] + out[8]
            er_fv[k] = _acc_0m_diag(er_fv[k], ρE_r, sw.diag_w, sw.diag_reset)
            es_fv[k] = _acc_0m_diag(es_fv[k], ρE_s, sw.diag_w, sw.diag_reset)
            evap_rai_acc += ρE_r * w_vol
            evap_sno_acc += ρE_s * w_vol
        end

        ρ_dq_net = ρ_dq_dt + ρE_r + ρE_s
        ρ_de_net = ρ_de_dt + ρe_evap
        dq_fv[k] = _acc_0m_diag(dq_fv[k], ρ_dq_net, sw.diag_w, sw.diag_reset)
        de_fv[k] = _acc_0m_diag(de_fv[k], ρ_de_net, sw.diag_w, sw.diag_reset)
        Δρ = ρ_dq_net * sw.dt
        ρ_fv[k] = ρ_old + Δρ
        ρqt_fv[k] = ρqt_old + Δρ
        ρe_fv[k] = ρe_old + ρ_de_net * sw.dt

        rain_acc += (ifelse(is_warm, ρ_dq_dt, zero(FT)) + ρE_r) * w_vol
        snow_acc += (ifelse(is_warm, zero(FT), ρ_dq_dt) + ρE_s) * w_vol
        energy_acc += ρ_de_net * w_vol
    end

    @inbounds begin
        Fields.field_values(sfc_col.rain)[] = rain_acc
        Fields.field_values(sfc_col.snow)[] = snow_acc
        Fields.field_values(sfc_col.energy)[] = energy_acc
        if HasEvap
            evr = Fields.field_values(sfc_col.evap_rai)
            evs = Fields.field_values(sfc_col.evap_sno)
            evr[] = _acc_0m_diag(evr[], evap_rai_acc, sw.diag_w, sw.diag_reset)
            evs[] = _acc_0m_diag(evs[], evap_sno_acc, sw.diag_w, sw.diag_reset)
        end
    end
    return nothing
end

struct Coupled0MEDMFXSweep{HasSGS, HasEvap, CM0P, CM0U, TPS, SGQ, PN, FT} <:
       ColumnSweep
    cm0::CM0P
    cm0_up::CM0U
    tps::TPS
    sgs_quad::SGQ
    passive_names::PN
    dt::FT
    diag_w::FT
    diag_reset::Bool
end

"""
    (sw::Coupled0MEDMFXSweep{HasSGS, HasEvap})(Yc_col, geom_col, edmf_col,
        sgs_col, sfc_col, evap_col)

0-moment precipitation removal over `sw.dt` in one column with a prognostic
updraft. The environment uses the `precip` parameters of `sw.cm0`, the updraft
the `precip_updraft` parameters in `sw.cm0_up`, whose timescale is
`precipitation_timescale_updraft_ratio` times the environment timescale. Each
subdomain's removal falls as rain or snow by its own temperature. The column
integral of the updraft removal accumulates into `sfc_col.updraft` with weight
`sw.diag_w`, restarting when `sw.diag_reset`.

With `HasEvap`, the column is scanned top-down and the precipitation flux
evaporates into the environment (`precipitation_evaporation_step` with the
environment density, mass per unit volume `ρa⁰`, temperature, and vapor), as in
`Coupled0MGridMeanSweep`. The evaporated water and its energy enter the grid
mean only, so the updraft prognostics are unchanged and the environment takes
them up.
"""
@inline function (sw::Coupled0MEDMFXSweep{HasSGS, HasEvap})(
    Yc_col,
    geom_col,
    edmf_col,
    sgs_col,
    sfc_col,
    evap_col,
) where {HasSGS, HasEvap}
    FT = typeof(sw.dt)
    tps = _unwrapped(sw.tps)
    cm0 = _unwrapped(sw.cm0)
    cm0_up = _unwrapped(sw.cm0_up)
    T_freeze = FT(TD.Parameters.T_freeze(tps))

    ρ_fv = Fields.field_values(Yc_col.ρ)
    ρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    sgs1 = Yc_col.sgsʲs.:(1)
    ρa1_fv = Fields.field_values(sgs1.ρa)
    mse1_fv = Fields.field_values(sgs1.mse)
    qt1_fv = Fields.field_values(sgs1.q_tot)
    passive1_fvs = _passive_sgs1_fvs(sgs1, sw.passive_names)

    lg_fv = Fields.field_values(geom_col.ᶜlg)
    flg_fv = Fields.field_values(geom_col.ᶠlg)
    dξ_fv = Fields.field_values(geom_col.ᶠ∂ξ³∂x³)
    Φ_fv = Fields.field_values(geom_col.ᶜΦ)

    ρ1_fv = Fields.field_values(edmf_col.ᶜρ¹)
    T1_fv = Fields.field_values(edmf_col.ᶜT¹)
    ql1_fv = Fields.field_values(edmf_col.ᶜq_liq¹)
    qi1_fv = Fields.field_values(edmf_col.ᶜq_ice¹)
    # The environment state comes from `specific_env_value` in
    # `set_prognostic_edmf_precomputed_quantities_environment!`, which
    # `refresh_pre_microphysics_thermo!` runs before the sweep.
    T0_fv = Fields.field_values(edmf_col.ᶜT⁰)
    p_fv = Fields.field_values(edmf_col.ᶜp)
    qt0_fv = Fields.field_values(edmf_col.ᶜq_tot_nonneg⁰)
    ql0_fv = Fields.field_values(edmf_col.ᶜq_liq⁰)
    qi0_fv = Fields.field_values(edmf_col.ᶜq_ice⁰)
    er_fv = Fields.field_values(evap_col.ρ_evap_rai)
    es_fv = Fields.field_values(evap_col.ρ_evap_sno)
    dq_fv = Fields.field_values(evap_col.ᶜρ_dq_tot_dt)
    de_fv = Fields.field_values(evap_col.ᶜρ_de_tot_dt)

    nz = Spaces.nlevels(axes(Yc_col.ρ))
    A_sfc = @inbounds flg_fv[1].J * dξ_fv[1]
    inv_A_sfc = one(FT) / max(A_sfc, floatmin(FT))
    p_sfc = @inbounds p_fv[1]

    rain_acc = zero(FT)
    snow_acc = zero(FT)
    energy_acc = zero(FT)
    updraft_acc = zero(FT)
    evap_rai_acc = zero(FT)
    evap_sno_acc = zero(FT)
    flux = (zero(FT), zero(FT), zero(FT), zero(FT))

    @inbounds for k in _scan_0m_levels(Val(HasEvap), nz)
        ρ_old = ρ_fv[k]
        ρqt_old = ρqt_fv[k]
        ρe_old = ρe_fv[k]
        V_k = lg_fv[k].J
        Φ_k = Φ_fv[k]

        ρa1_old = clamp(ρa1_fv[k], zero(FT), ρ_old * (one(FT) - FT(1e-4)))
        ρa0_old = max(eps(FT), ρ_old - ρa1_old)

        qt1_pos = max(zero(FT), qt1_fv[k])
        mp1 = microphysics_tendencies_0m(
            cm0_up,
            tps,
            max(eps(FT), ρ1_fv[k]),
            T1_fv[k],
            qt1_pos,
            ql1_fv[k],
            qi1_fv[k],
            Φ_k,
            sw.dt,
        )
        T0_k = T0_fv[k]
        qt0_pos = qt0_fv[k]
        ρ0_k = max(
            eps(FT),
            TD.air_density(tps, T0_k, p_fv[k], qt0_pos, ql0_fv[k], qi0_fv[k]),
        )
        mp0 = if HasSGS
            microphysics_tendencies_0m(
                sw.sgs_quad,
                cm0,
                tps,
                ρ0_k,
                T0_k,
                qt0_pos,
                Fields.field_values(sgs_col.ᶜT′T′)[k],
                Fields.field_values(sgs_col.ᶜq′q′)[k],
                Fields.field_values(sgs_col.ᶜcorr_Tq)[k],
                Φ_k,
                sw.dt,
            )
        else
            microphysics_tendencies_0m(
                cm0,
                tps,
                ρ0_k,
                T0_k,
                qt0_pos,
                ql0_fv[k],
                qi0_fv[k],
                Φ_k,
                sw.dt,
            )
        end

        Δq1 = max(-min(qt1_pos, FT(0.9)), mp1.dq_tot_dt * sw.dt)
        dq1_dt_eff = Δq1 / max(sw.dt, floatmin(FT))
        ρ_dq_dt0 = ρa0_old * mp0.dq_tot_dt
        ρ_dq_dt1 = ifelse(ρa1_old > ϵ_numerics(FT), ρa1_old * dq1_dt_eff, zero(FT))
        ρ_dq_dt = ρ_dq_dt0 + ρ_dq_dt1
        ρ_de_dt0 = ρ_dq_dt0 * mp0.e_tot_hlpr
        ρ_de_dt1 = ρ_dq_dt1 * mp1.e_tot_hlpr
        ρ_de_dt = ρ_de_dt0 + ρ_de_dt1

        # Each subdomain's removal falls as rain or snow according to its
        # own temperature.
        w_vol = V_k * inv_A_sfc
        is_warm0 = T0_k >= T_freeze
        is_warm1 = T1_fv[k] >= T_freeze
        ρ_dq_dt_rai =
            ifelse(is_warm0, ρ_dq_dt0, zero(FT)) +
            ifelse(is_warm1, ρ_dq_dt1, zero(FT))
        ρ_dq_dt_sno =
            ifelse(is_warm0, zero(FT), ρ_dq_dt0) +
            ifelse(is_warm1, zero(FT), ρ_dq_dt1)

        ρE_r = zero(FT)
        ρE_s = zero(FT)
        ρe_evap = zero(FT)
        if HasEvap
            ρ_de_dt_rai =
                ifelse(is_warm0, ρ_de_dt0, zero(FT)) +
                ifelse(is_warm1, ρ_de_dt1, zero(FT))
            ρ_de_dt_sno =
                ifelse(is_warm0, zero(FT), ρ_de_dt0) +
                ifelse(is_warm1, zero(FT), ρ_de_dt1)
            q_vap0 = max(zero(FT), qt0_pos - ql0_fv[k] - qi0_fv[k])
            out = precipitation_evaporation_step(
                cm0.evaporation,
                tps,
                sw.dt,
                flux,
                (
                    -ρ_dq_dt_rai,
                    -ρ_dq_dt_sno,
                    -ρ_de_dt_rai,
                    -ρ_de_dt_sno,
                    w_vol,
                    ρ0_k,
                    ρa0_old,
                    T0_k,
                    p_fv[k] / p_sfc,
                    q_vap0,
                ),
            )
            flux = (out[1], out[2], out[3], out[4])
            ρE_r = out[5]
            ρE_s = out[6]
            ρe_evap = out[7] + out[8]
            er_fv[k] = _acc_0m_diag(er_fv[k], ρE_r, sw.diag_w, sw.diag_reset)
            es_fv[k] = _acc_0m_diag(es_fv[k], ρE_s, sw.diag_w, sw.diag_reset)
            evap_rai_acc += ρE_r * w_vol
            evap_sno_acc += ρE_s * w_vol
        end

        ρ_dq_net = ρ_dq_dt + ρE_r + ρE_s
        ρ_de_net = ρ_de_dt + ρe_evap
        dq_fv[k] = _acc_0m_diag(dq_fv[k], ρ_dq_net, sw.diag_w, sw.diag_reset)
        de_fv[k] = _acc_0m_diag(de_fv[k], ρ_de_net, sw.diag_w, sw.diag_reset)
        Δρ = ρ_dq_net * sw.dt
        ρ_fv[k] = ρ_old + Δρ
        ρqt_fv[k] = ρqt_old + Δρ
        ρe_fv[k] = ρe_old + ρ_de_net * sw.dt

        if ρa1_old > ϵ_numerics(FT)
            fac1 = one(FT) + Δq1
            ρa1_fv[k] = ρa1_old * fac1
            qt1_fv[k] = max(zero(FT), (qt1_pos + Δq1) / fac1)
            mse1_fv[k] = (mse1_fv[k] + Δq1 * mp1.e_tot_hlpr) / fac1
            _rescale_passive_sgs!(passive1_fvs, k, one(FT) / fac1)
        end

        rain_acc += (ρ_dq_dt_rai + ρE_r) * w_vol
        snow_acc += (ρ_dq_dt_sno + ρE_s) * w_vol
        energy_acc += ρ_de_net * w_vol
        updraft_acc += ρ_dq_dt1 * w_vol
    end

    @inbounds begin
        Fields.field_values(sfc_col.rain)[] = rain_acc
        Fields.field_values(sfc_col.snow)[] = snow_acc
        Fields.field_values(sfc_col.energy)[] = energy_acc
        upd = Fields.field_values(sfc_col.updraft)
        upd[] = _acc_0m_diag(upd[], updraft_acc, sw.diag_w, sw.diag_reset)
        if HasEvap
            evr = Fields.field_values(sfc_col.evap_rai)
            evs = Fields.field_values(sfc_col.evap_sno)
            evr[] = _acc_0m_diag(evr[], evap_rai_acc, sw.diag_w, sw.diag_reset)
            evs[] = _acc_0m_diag(evs[], evap_sno_acc, sw.diag_w, sw.diag_reset)
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Subcycled Donor-Implicit Euler Sweep for 2M
# ----------------------------------------------------------------------------

"""
    _donor_implicit_number_step(n0, S_in_n, dn_dt, D_sed_n, dt_sub)

Advance a number concentration over `dt_sub` with donor-implicit Euler: the
sedimentation inflow `S_in_n` and the positive part of the microphysics
tendency `dn_dt` are sources, and sedimentation `D_sed_n` and the negative part
of `dn_dt` are implicit decays. `dn_dt` is the total
`BMT.bulk_microphysics_tendencies` number tendency, which includes the
relaxation toward the mean-mass limits
(`CM2.number_tendency_from_mass_limits`). Returns `(n_new, n_out)`, with
`n_out` the number exported through the bottom face.
"""
@inline function _donor_implicit_number_step(
    n0,
    S_in_n,
    dn_dt,
    D_sed_n,
    dt_sub,
)
    FT = typeof(n0)
    n_star = max(zero(FT), n0) + max(zero(FT), S_in_n) * dt_sub
    dn_pos = max(zero(FT), dn_dt)
    D_mp_n = max(zero(FT), -dn_dt) / max(n_star, floatmin(FT))
    n_new =
        (n_star + dn_pos * dt_sub) /
        (one(FT) + (max(zero(FT), D_sed_n) + D_mp_n) * dt_sub)
    n_out = max(zero(FT), D_sed_n) * n_new * dt_sub
    return (n_new, n_out)
end

@inline function _conservative_donor_implicit_mass_step(
    q0::SA.SVector{4},
    S_in::SA.SVector{4},
    dq_mp_dt::SA.SVector{4},
    D_sed::SA.SVector{4},
    q_v0,
    q_min,
    dt_sub,
)
    FT = typeof(q_v0)
    q_star = max.(q0, zero(FT)) .+ max.(S_in, zero(FT)) .* dt_sub

    # Step sink species (`dq_mp_dt < 0`) with donor decay to measure exact
    # realized donor losses before feeding source species.
    loss_sum = zero(FT)
    gain_raw_sum = zero(FT)
    q_sink = zero(SA.SVector{4, FT})
    @inbounds for i in 1:4
        if dq_mp_dt[i] < zero(FT)
            D_mp = (-dq_mp_dt[i]) / max(q_star[i], q_min)
            qi_new = q_star[i] / (one(FT) + (D_sed[i] + D_mp) * dt_sub)
            q_sink = Base.setindex(q_sink, qi_new, i)
            loss_sum += D_mp * qi_new * dt_sub
        else
            gain_raw_sum += dq_mp_dt[i] * dt_sub
        end
    end

    vap_supply = min(max(zero(FT), q_v0), max(zero(FT), sum(dq_mp_dt) * dt_sub))
    gain_cap = loss_sum + vap_supply
    γ_src = ifelse(
        gain_raw_sum > floatmin(FT),
        min(one(FT), gain_cap / max(gain_raw_sum, floatmin(FT))),
        zero(FT),
    )

    q_new = q_sink
    q_out = zero(SA.SVector{4, FT})
    @inbounds for i in 1:4
        qi_new = if dq_mp_dt[i] < zero(FT)
            q_sink[i]
        else
            (q_star[i] + γ_src * dq_mp_dt[i] * dt_sub) /
            (one(FT) + D_sed[i] * dt_sub)
        end
        q_new = Base.setindex(q_new, max(zero(FT), qi_new), i)
        q_out = Base.setindex(q_out, D_sed[i] * max(zero(FT), qi_new) * dt_sub, i)
    end
    return (q_new, q_out, γ_src)
end

struct Coupled2MGridMeanSweep{CM2P, CM1P, CMC, TPS, PAP, FT} <: ColumnSweep
    cm2p::CM2P
    cm1p::CM1P
    cmc::CMC
    tps::TPS
    pap::PAP
    dt_sub::FT
    n_sub::Int
end

@inline function (sw::Coupled2MGridMeanSweep)(
    Yc_col,
    geom_col,
    pre_col,
    sfc_col,
    aero_col,
)
    FT = typeof(sw.dt_sub)
    tps = _unwrapped(sw.tps)
    cm2p = _unwrapped(sw.cm2p)
    cm1p = _unwrapped(sw.cm1p)
    cmc = _unwrapped(sw.cmc)
    sb = cm2p.warm_rain.seifert_beheng
    q_min = max(FT(TD.Parameters.q_min(tps)), floatmin(FT))

    ρ_fv = Fields.field_values(Yc_col.ρ)
    ρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    ρql_fv = Fields.field_values(Yc_col.ρq_lcl)
    ρqi_fv = Fields.field_values(Yc_col.ρq_icl)
    ρqr_fv = Fields.field_values(Yc_col.ρq_rai)
    ρqs_fv = Fields.field_values(Yc_col.ρq_sno)
    ρnl_fv = Fields.field_values(Yc_col.ρn_lcl)
    ρnr_fv = Fields.field_values(Yc_col.ρn_rai)

    lg_fv = Fields.field_values(geom_col.ᶜlg)
    flg_fv = Fields.field_values(geom_col.ᶠlg)
    dξ_fv = Fields.field_values(geom_col.ᶠ∂ξ³∂x³)
    Φ_fv = Fields.field_values(geom_col.ᶜΦ)

    u_fv = Fields.field_values(pre_col.ᶜu)
    p_fv = Fields.field_values(pre_col.ᶜp)
    wl_fv = Fields.field_values(pre_col.ᶜwₗ)
    wi_fv = Fields.field_values(pre_col.ᶜwᵢ)
    wr_fv = Fields.field_values(pre_col.ᶜwᵣ)
    ws_fv = Fields.field_values(pre_col.ᶜwₛ)
    wnl_fv = Fields.field_values(pre_col.ᶜwₙₗ)
    wnr_fv = Fields.field_values(pre_col.ᶜwₙᵣ)
    whht_fv = Fields.field_values(pre_col.ᶜwₕhₜ)

    nz = Spaces.nlevels(axes(Yc_col.ρ))
    rain_sfc_sum = zero(FT)
    snow_sfc_sum = zero(FT)
    energy_sfc_sum = zero(FT)

    @inbounds for _ in 1:sw.n_sub
        F_mass_in = zero(SA.SVector{4, FT})
        F_nl_in = zero(FT)
        F_nr_in = zero(FT)
        F_e_in = zero(FT)

        for step_idx in 0:(nz - 1)
            k = nz - step_idx
            ρ_old = ρ_fv[k]
            inv_ρ_old = one(FT) / ρ_old
            ρe_old = ρe_fv[k]
            ρqt_old = ρqt_fv[k]
            lg_k = lg_fv[k]
            V_k = lg_k.J
            A_bot = flg_fv[k].J * dξ_fv[k]
            dz_eff = V_k / max(A_bot, floatmin(FT))
            inv_dz = one(FT) / max(dz_eff, floatmin(FT))
            inv_ρV = one(FT) / (ρ_old * V_k)

            q_tot0 = ρqt_old * inv_ρ_old
            q0 = _partition_nonneg_condensates(
                q_tot0,
                ρql_fv[k] * inv_ρ_old,
                ρqi_fv[k] * inv_ρ_old,
                ρqr_fv[k] * inv_ρ_old,
                ρqs_fv[k] * inv_ρ_old,
            )
            q_v0 = max(zero(FT), max(zero(FT), q_tot0) - sum(q0))
            nl0 = max(zero(FT), ρnl_fv[k] * inv_ρ_old)
            nr0 = max(zero(FT), ρnr_fv[k] * inv_ρ_old)

            uvw_k = Geometry.UVWVector(u_fv[k], lg_k)
            K_air = norm_sqr(uvw_k) / 2
            Φ_k = Φ_fv[k]
            e_int_old = ρe_old * inv_ρ_old - K_air - Φ_k
            T0 = TD.air_temperature(
                tps,
                e_int_old,
                max(zero(FT), q_tot0),
                q0[1] + q0[3],
                q0[2] + q0[4],
            )

            S_in = F_mass_in .* inv_ρV
            S_nl_in = F_nl_in * inv_ρV
            S_nr_in = F_nr_in * inv_ρV

            # Terminal velocities on local + incoming pools.
            q_lcl_eff = q0[1] + S_in[1] * sw.dt_sub
            n_lcl_eff = nl0 + S_nl_in * sw.dt_sub
            w_nl, w_l = CM2.cloud_terminal_velocity(
                sb.pdf_c,
                cmc.stokes,
                q_lcl_eff,
                ρ_old,
                ρ_old * n_lcl_eff,
            )

            q_rai_eff = q0[3] + S_in[3] * sw.dt_sub
            n_rai_eff = nr0 + S_nr_in * sw.dt_sub
            w_nr, w_r = CM2.rain_terminal_velocity(
                sb,
                cmc.Ch2022.rain,
                q_rai_eff,
                ρ_old,
                ρ_old * n_rai_eff,
            )

            q_icl_eff = q0[2] + S_in[2] * sw.dt_sub
            q_sno_eff = q0[4] + S_in[4] * sw.dt_sub
            w_i = CMNe.terminal_velocity(
                cmc.ice,
                cmc.Ch2022.small_ice,
                ρ_old,
                q_icl_eff,
            )
            w_s = CM1.terminal_velocity(
                cm1p.precip.snow,
                cm1p.terminal_velocity.snow,
                ρ_old,
                q_sno_eff,
            )

            bmt = BMT.bulk_microphysics_tendencies(
                BMT.Microphysics2Moment(),
                cm2p,
                tps,
                ρ_old,
                T0,
                max(zero(FT), q_tot0),
                q0[1],
                nl0,
                q0[3],
                nr0,
            )
            dq_mp = SA.SVector{4, FT}(
                bmt.dq_lcl_dt,
                bmt.dq_ice_dt,
                bmt.dq_rai_dt,
                zero(FT),
            )
            w_k = SA.SVector{4, FT}(w_l, w_i, w_r, w_s)
            D_sed = w_k .* inv_dz

            q_new, q_out, _ = _conservative_donor_implicit_mass_step(
                q0,
                S_in,
                dq_mp,
                D_sed,
                q_v0,
                q_min,
                sw.dt_sub,
            )
            dn_act = zero(FT)
            if !(sw.pap isa Nothing)
                pap = _unwrapped(sw.pap)
                dn_act = aerosol_activation_sources(
                    cmc.activation,
                    Fields.field_values(aero_col.seasalt_num)[k],
                    Fields.field_values(aero_col.seasalt_mean_radius)[k],
                    Fields.field_values(aero_col.sulfate_num)[k],
                    max(zero(FT), q_tot0),
                    q0[1] + q0[3],
                    q0[2] + q0[4],
                    nl0 + nr0,
                    ρ_old,
                    uvw_k.w,
                    cm2p,
                    tps,
                    T0,
                    p_fv[k],
                    sw.dt_sub,
                    pap,
                )
            end
            nl_new, nl_out = _donor_implicit_number_step(
                nl0,
                S_nl_in,
                bmt.dn_lcl_dt + dn_act,
                w_nl * inv_dz,
                sw.dt_sub,
            )
            nr_new, nr_out = _donor_implicit_number_step(
                nr0,
                S_nr_in,
                bmt.dn_rai_dt,
                w_nr * inv_dz,
                sw.dt_sub,
            )
            wnl_fv[k] = w_nl
            wnr_fv[k] = w_nr

            scale_flux = (ρ_old * V_k) / max(sw.dt_sub, floatmin(FT))
            F_mass_out = q_out .* scale_flux
            Δρ_sed = (sw.dt_sub / V_k) * (sum(F_mass_in) - sum(F_mass_out))

            ρ_new = ρ_old + Δρ_sed
            ρqt_new = ρqt_old + Δρ_sed
            ρq_new = q_new .* ρ_old

            _, F_e_out, _ = _donor_post_mp_temperature_and_energy_out(
                tps,
                ρ_old,
                ρe_old,
                F_e_in,
                V_k,
                sw.dt_sub,
                ρ_new,
                ρqt_new,
                ρq_new,
                q_out,
                w_k,
                uvw_k,
                Φ_k,
            )
            ρe_new = ρe_old + (sw.dt_sub / V_k) * (F_e_in - F_e_out)

            ρ_fv[k] = ρ_new
            ρqt_fv[k] = ρqt_new
            ρql_fv[k] = ρq_new[1]
            ρqi_fv[k] = ρq_new[2]
            ρqr_fv[k] = ρq_new[3]
            ρqs_fv[k] = ρq_new[4]
            ρnl_fv[k] = ρ_old * nl_new
            ρnr_fv[k] = ρ_old * nr_new
            ρe_fv[k] = ρe_new

            wl_fv[k] = w_k[1]
            wi_fv[k] = w_k[2]
            wr_fv[k] = w_k[3]
            ws_fv[k] = w_k[4]
            whht_fv[k] = WVec(F_e_out / max(A_bot * ρ_new, floatmin(FT)))

            F_mass_in = F_mass_out
            F_nl_in = nl_out * scale_flux
            F_nr_in = nr_out * scale_flux
            F_e_in = F_e_out

            if k == 1
                inv_A_sfc = one(FT) / max(A_bot, floatmin(FT))
                rain_sfc_sum += -(F_mass_out[3] + F_mass_out[1]) * inv_A_sfc
                snow_sfc_sum += -(F_mass_out[4] + F_mass_out[2]) * inv_A_sfc
                energy_sfc_sum += -F_e_out * inv_A_sfc
            end
        end
    end

    inv_nsub = one(FT) / FT(sw.n_sub)
    @inbounds begin
        Fields.field_values(sfc_col.rain)[] = rain_sfc_sum * inv_nsub
        Fields.field_values(sfc_col.snow)[] = snow_sfc_sum * inv_nsub
        Fields.field_values(sfc_col.energy)[] = energy_sfc_sum * inv_nsub
    end
    return nothing
end

struct Coupled2MEDMFXSweep{CM2P, CM1P, CMC, TPS, PAP, PN, TC, FT} <:
       ColumnSweep
    cm2p::CM2P
    cm1p::CM1P
    cmc::CMC
    tps::TPS
    pap::PAP
    passive_names::PN
    turbconv_model::TC
    dt_sub::FT
    n_sub::Int
end

@inline function (sw::Coupled2MEDMFXSweep)(
    Yc_col,
    geom_col,
    pre_col,
    edmf_col,
    sfc_col,
    aero_col,
)
    FT = typeof(sw.dt_sub)
    tps = _unwrapped(sw.tps)
    cm2p = _unwrapped(sw.cm2p)
    cm1p = _unwrapped(sw.cm1p)
    cmc = _unwrapped(sw.cmc)
    sb = cm2p.warm_rain.seifert_beheng
    q_min = max(FT(TD.Parameters.q_min(tps)), floatmin(FT))

    ρ_fv = Fields.field_values(Yc_col.ρ)
    ρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    ρql_fv = Fields.field_values(Yc_col.ρq_lcl)
    ρqi_fv = Fields.field_values(Yc_col.ρq_icl)
    ρqr_fv = Fields.field_values(Yc_col.ρq_rai)
    ρqs_fv = Fields.field_values(Yc_col.ρq_sno)
    ρnl_fv = Fields.field_values(Yc_col.ρn_lcl)
    ρnr_fv = Fields.field_values(Yc_col.ρn_rai)

    sgs1 = Yc_col.sgsʲs.:(1)
    ρa1_fv = Fields.field_values(sgs1.ρa)
    mse1_fv = Fields.field_values(sgs1.mse)
    qt1_fv = Fields.field_values(sgs1.q_tot)
    ql1_fv = Fields.field_values(sgs1.q_lcl)
    qi1_fv = Fields.field_values(sgs1.q_icl)
    qr1_fv = Fields.field_values(sgs1.q_rai)
    qs1_fv = Fields.field_values(sgs1.q_sno)
    nl1_fv = Fields.field_values(sgs1.n_lcl)
    nr1_fv = Fields.field_values(sgs1.n_rai)
    passive1_fvs = _passive_sgs1_fvs(sgs1, sw.passive_names)

    lg_fv = Fields.field_values(geom_col.ᶜlg)
    flg_fv = Fields.field_values(geom_col.ᶠlg)
    dξ_fv = Fields.field_values(geom_col.ᶠ∂ξ³∂x³)
    Φ_fv = Fields.field_values(geom_col.ᶜΦ)

    wl_fv = Fields.field_values(pre_col.ᶜwₗ)
    wi_fv = Fields.field_values(pre_col.ᶜwᵢ)
    wr_fv = Fields.field_values(pre_col.ᶜwᵣ)
    ws_fv = Fields.field_values(pre_col.ᶜwₛ)
    wnl_fv = Fields.field_values(pre_col.ᶜwₙₗ)
    wnr_fv = Fields.field_values(pre_col.ᶜwₙᵣ)
    whht_fv = Fields.field_values(pre_col.ᶜwₕhₜ)

    ρ1_fv = Fields.field_values(edmf_col.ᶜρ¹)
    T1_fv = Fields.field_values(edmf_col.ᶜT¹)
    u1_fv = Fields.field_values(edmf_col.ᶜu¹)
    T0_env_fv = Fields.field_values(edmf_col.ᶜT⁰)
    p_fv = Fields.field_values(edmf_col.ᶜp)
    u0_fv = Fields.field_values(edmf_col.ᶜu⁰)

    nz = Spaces.nlevels(axes(Yc_col.ρ))
    rain_sfc_sum = zero(FT)
    snow_sfc_sum = zero(FT)
    energy_sfc_sum = zero(FT)

    @inbounds for _ in 1:sw.n_sub
        F_mass_in_1 = zero(SA.SVector{4, FT})
        F_mass_in_0 = zero(SA.SVector{4, FT})
        F_nl_in_1 = zero(FT)
        F_nr_in_1 = zero(FT)
        F_nl_in_0 = zero(FT)
        F_nr_in_0 = zero(FT)
        F_hΦ_in_1 = zero(FT)
        F_K_in_1 = zero(FT)
        F_hΦ_in_0 = zero(FT)
        F_K_in_0 = zero(FT)
        a1_above = zero(FT)

        for step_idx in 0:(nz - 1)
            k = nz - step_idx
            ρ_old = ρ_fv[k]
            ρe_old = ρe_fv[k]
            ρqt_old = ρqt_fv[k]
            Φ_k = Φ_fv[k]
            lg_k = lg_fv[k]
            V_k = lg_k.J
            A_bot = flg_fv[k].J * dξ_fv[k]
            dz_eff = V_k / max(A_bot, floatmin(FT))
            inv_dz = one(FT) / max(dz_eff, floatmin(FT))

            (; ρ1_k, has_updraft, ρa1_old, a1_eff, ρa0_old) =
                _sweep_subdomain_masses(ρ_old, ρa1_fv[k], ρ1_fv[k])

            # Conservative entry projection of the updraft water and number.
            qt_gs_pos = max(zero(FT), ρqt_old)
            ρq_k = SA.SVector{4, FT}(ρql_fv[k], ρqi_fv[k], ρqr_fv[k], ρqs_fv[k])
            ρq_gs_pos = max.(zero(FT), ρq_k)
            ρnl_gs_pos = max(zero(FT), ρnl_fv[k])
            ρnr_gs_pos = max(zero(FT), ρnr_fv[k])
            qt1_0, q1_0 = _sweep_updraft_entry_pools(
                ρa1_old,
                qt_gs_pos,
                qt1_fv[k],
                SA.SVector{4, FT}(ql1_fv[k], qi1_fv[k], qr1_fv[k], qs1_fv[k]),
                ρq_gs_pos,
            )
            nl1_0 =
                _cap_updraft_pool(max(zero(FT), nl1_fv[k]), ρnl_gs_pos, ρa1_old)
            nr1_0 =
                _cap_updraft_pool(max(zero(FT), nr1_fv[k]), ρnr_gs_pos, ρa1_old)

            sgsʲs_k = ((;
                ρa = ρa1_fv[k],
                q_tot = qt1_fv[k],
                q_lcl = ql1_fv[k],
                q_icl = qi1_fv[k],
                q_rai = qr1_fv[k],
                q_sno = qs1_fv[k],
                n_lcl = nl1_fv[k],
                n_rai = nr1_fv[k],
            ),)
            ρa⁰_k = ρa⁰(ρ_old, sgsʲs_k, sw.turbconv_model)
            q0_env, qt0_env = _sweep_env_water_pools(
                sgsʲs_k,
                sw.turbconv_model,
                ρ_old,
                ρa⁰_k,
                ρa1_old,
                ρa0_old,
                ρq_k,
                ρq_gs_pos,
                q1_0,
                ρqt_old,
                qt_gs_pos,
                qt1_0,
            )
            nl0_env = _sweep_env_value(
                @name(n_lcl), ρnl_fv[k], ρ_old, sgsʲs_k, ρa⁰_k,
                sw.turbconv_model, ρnl_gs_pos, ρa1_old, nl1_0, ρa0_old,
            )
            nr0_env = _sweep_env_value(
                @name(n_rai), ρnr_fv[k], ρ_old, sgsʲs_k, ρa⁰_k,
                sw.turbconv_model, ρnr_gs_pos, ρa1_old, nr1_0, ρa0_old,
            )

            # Lateral redistribution across the tilted updraft boundary at
            # face `k + ½` of mass, number, and static energy `h + Φ`.
            transfer = _shaft_transfer_fractions(a1_above, a1_eff)
            F_in_1 = _shaft_into_updraft(F_mass_in_1, F_mass_in_0, transfer)
            F_in_0 = _shaft_into_environment(F_mass_in_1, F_mass_in_0, transfer)
            Fnl_1 = _shaft_into_updraft(F_nl_in_1, F_nl_in_0, transfer)
            Fnr_1 = _shaft_into_updraft(F_nr_in_1, F_nr_in_0, transfer)
            Fnl_0 = _shaft_into_environment(F_nl_in_1, F_nl_in_0, transfer)
            Fnr_0 = _shaft_into_environment(F_nr_in_1, F_nr_in_0, transfer)
            FhΦ_1 = _shaft_into_updraft(F_hΦ_in_1, F_hΦ_in_0, transfer)

            uvw1 = Geometry.UVWVector(u1_fv[k], lg_k)
            F_mass_out_1 = zero(SA.SVector{4, FT})
            F_nl_out_1 = zero(FT)
            F_nr_out_1 = zero(FT)
            F_hΦ_out_1 = zero(FT)
            F_K_out_1 = zero(FT)
            w1_k = zero(SA.SVector{4, FT})
            wnl1 = zero(FT)
            wnr1 = zero(FT)
            q1_ode = q1_0
            nl1_ode = nl1_0
            nr1_ode = nr1_0
            ρa1_new_eff = zero(FT)
            qt1_new_eff = zero(FT)
            ρe1_post = zero(FT)

            if has_updraft
                inv_m1 = one(FT) / (ρa1_old * V_k)
                S_in_1 = F_in_1 .* inv_m1
                S_nl_1 = Fnl_1 * inv_m1
                S_nr_1 = Fnr_1 * inv_m1
                qv1_0 = max(zero(FT), qt1_0 - sum(q1_0))

                wnl1, wl1 = CM2.cloud_terminal_velocity(
                    sb.pdf_c,
                    cmc.stokes,
                    q1_0[1] + S_in_1[1] * sw.dt_sub,
                    ρ1_k,
                    ρ1_k * (nl1_0 + S_nl_1 * sw.dt_sub),
                )
                wnr1, wr1 = CM2.rain_terminal_velocity(
                    sb,
                    cmc.Ch2022.rain,
                    q1_0[3] + S_in_1[3] * sw.dt_sub,
                    ρ1_k,
                    ρ1_k * (nr1_0 + S_nr_1 * sw.dt_sub),
                )
                wi1 = CMNe.terminal_velocity(
                    cmc.ice,
                    cmc.Ch2022.small_ice,
                    ρ1_k,
                    q1_0[2] + S_in_1[2] * sw.dt_sub,
                )
                ws1 = CM1.terminal_velocity(
                    cm1p.precip.snow,
                    cm1p.terminal_velocity.snow,
                    ρ1_k,
                    q1_0[4] + S_in_1[4] * sw.dt_sub,
                )
                w1_k = SA.SVector{4, FT}(wl1, wi1, wr1, ws1)
                D_sed1 = w1_k .* inv_dz

                bmt1 = BMT.bulk_microphysics_tendencies(
                    BMT.Microphysics2Moment(),
                    cm2p,
                    tps,
                    ρ1_k,
                    T1_fv[k],
                    qt1_0,
                    q1_0[1],
                    nl1_0,
                    q1_0[3],
                    nr1_0,
                )
                dq_mp1 = SA.SVector{4, FT}(
                    bmt1.dq_lcl_dt,
                    bmt1.dq_ice_dt,
                    bmt1.dq_rai_dt,
                    zero(FT),
                )
                q1_ode, q1_out, _ = _conservative_donor_implicit_mass_step(
                    q1_0,
                    S_in_1,
                    dq_mp1,
                    D_sed1,
                    qv1_0,
                    q_min,
                    sw.dt_sub,
                )
                dn_act_1 = zero(FT)
                if !(sw.pap isa Nothing)
                    pap = _unwrapped(sw.pap)
                    dn_act_1 = aerosol_activation_sources(
                        cmc.activation,
                        Fields.field_values(aero_col.seasalt_num)[k],
                        Fields.field_values(aero_col.seasalt_mean_radius)[k],
                        Fields.field_values(aero_col.sulfate_num)[k],
                        qt1_0,
                        q1_0[1] + q1_0[3],
                        q1_0[2],
                        nl1_0 + nr1_0,
                        ρ1_k,
                        max(zero(FT), uvw1.w),
                        cm2p,
                        tps,
                        T1_fv[k],
                        p_fv[k],
                        sw.dt_sub,
                        pap,
                    )
                end
                nl1_ode, nl1_out = _donor_implicit_number_step(
                    nl1_0,
                    S_nl_1,
                    bmt1.dn_lcl_dt + dn_act_1,
                    wnl1 * inv_dz,
                    sw.dt_sub,
                )
                nr1_ode, nr1_out = _donor_implicit_number_step(
                    nr1_0,
                    S_nr_1,
                    bmt1.dn_rai_dt,
                    wnr1 * inv_dz,
                    sw.dt_sub,
                )

                scale_1 = (ρa1_old * V_k) / max(sw.dt_sub, floatmin(FT))
                F_mass_out_1 = q1_out .* scale_1
                F_nl_out_1 = nl1_out * scale_1
                F_nr_out_1 = nr1_out * scale_1

                Δρa1_sed = (sw.dt_sub / V_k) * (sum(F_in_1) - sum(F_mass_out_1))
                ρa1_new = max(floatmin(FT), ρa1_old + Δρa1_sed)
                inv_ρa1_new = one(FT) / ρa1_new
                qt1_new =
                    max(zero(FT), (ρa1_old * qt1_0 + Δρa1_sed) * inv_ρa1_new)
                q1_int_new = (q1_ode .* ρa1_old) .* inv_ρa1_new

                post1 = _sweep_updraft_post_state(
                    tps,
                    ρa1_old,
                    ρa1_new,
                    qt1_new,
                    q1_int_new,
                    mse1_fv[k],
                    FhΦ_1,
                    q1_out,
                    w1_k,
                    uvw1,
                    Φ_k,
                    V_k,
                    sw.dt_sub,
                )
                (; T1_post, q1_liq_p, q1_ice_p, mse1_new) = post1
                F_hΦ_out_1 = post1.F_hΦ_out_1
                F_K_out_1 = post1.F_K_out_1
                ρe1_post = post1.ρe1_post
                ρa1_new_eff = ρa1_new
                qt1_new_eff = qt1_new

                ρa1_fv[k] = ρa1_new
                mse1_fv[k] = mse1_new
                qt1_fv[k] = qt1_new
                ql1_fv[k] = q1_int_new[1]
                qi1_fv[k] = q1_int_new[2]
                qr1_fv[k] = q1_int_new[3]
                qs1_fv[k] = q1_int_new[4]
                nl1_fv[k] = nl1_ode * ρa1_old * inv_ρa1_new
                nr1_fv[k] = nr1_ode * ρa1_old * inv_ρa1_new
                _rescale_passive_sgs!(passive1_fvs, k, ρa1_old * inv_ρa1_new)
                T1_fv[k] = T1_post
                ρ1_fv[k] = max(
                    eps(FT),
                    TD.air_density(
                        tps,
                        T1_post,
                        p_fv[k],
                        qt1_new,
                        q1_liq_p,
                        q1_ice_p,
                    ),
                )
            end

            # Environment 0 step
            uvw0 = Geometry.UVWVector(u0_fv[k], lg_k)
            T0_env = T0_env_fv[k]
            ρ0_env = max(
                eps(FT),
                TD.air_density(
                    tps,
                    T0_env,
                    p_fv[k],
                    qt0_env,
                    q0_env[1] + q0_env[3],
                    q0_env[2] + q0_env[4],
                ),
            )
            inv_m0 = one(FT) / (ρa0_old * V_k)
            S_in_0 = F_in_0 .* inv_m0
            S_nl_0 = Fnl_0 * inv_m0
            S_nr_0 = Fnr_0 * inv_m0
            qv0_env = max(zero(FT), qt0_env - sum(q0_env))

            wnl0, wl0 = CM2.cloud_terminal_velocity(
                sb.pdf_c,
                cmc.stokes,
                q0_env[1] + S_in_0[1] * sw.dt_sub,
                ρ0_env,
                ρ0_env * (nl0_env + S_nl_0 * sw.dt_sub),
            )
            wnr0, wr0 = CM2.rain_terminal_velocity(
                sb,
                cmc.Ch2022.rain,
                q0_env[3] + S_in_0[3] * sw.dt_sub,
                ρ0_env,
                ρ0_env * (nr0_env + S_nr_0 * sw.dt_sub),
            )
            wi0 = CMNe.terminal_velocity(
                cmc.ice,
                cmc.Ch2022.small_ice,
                ρ0_env,
                q0_env[2] + S_in_0[2] * sw.dt_sub,
            )
            ws0 = CM1.terminal_velocity(
                cm1p.precip.snow,
                cm1p.terminal_velocity.snow,
                ρ0_env,
                q0_env[4] + S_in_0[4] * sw.dt_sub,
            )
            w0_k = SA.SVector{4, FT}(wl0, wi0, wr0, ws0)
            D_sed0 = w0_k .* inv_dz

            bmt0 = BMT.bulk_microphysics_tendencies(
                BMT.Microphysics2Moment(),
                cm2p,
                tps,
                ρ0_env,
                T0_env,
                qt0_env,
                q0_env[1],
                nl0_env,
                q0_env[3],
                nr0_env,
            )
            dq_mp0 = SA.SVector{4, FT}(
                bmt0.dq_lcl_dt,
                bmt0.dq_ice_dt,
                bmt0.dq_rai_dt,
                zero(FT),
            )
            q0_ode, q0_out, _ = _conservative_donor_implicit_mass_step(
                q0_env,
                S_in_0,
                dq_mp0,
                D_sed0,
                qv0_env,
                q_min,
                sw.dt_sub,
            )
            dn_act_0 = zero(FT)
            if !(sw.pap isa Nothing)
                pap = _unwrapped(sw.pap)
                dn_act_0 = aerosol_activation_sources(
                    cmc.activation,
                    Fields.field_values(aero_col.seasalt_num)[k],
                    Fields.field_values(aero_col.seasalt_mean_radius)[k],
                    Fields.field_values(aero_col.sulfate_num)[k],
                    qt0_env,
                    q0_env[1] + q0_env[3],
                    q0_env[2] + q0_env[4],
                    nl0_env + nr0_env,
                    ρ0_env,
                    uvw0.w,
                    cm2p,
                    tps,
                    T0_env,
                    p_fv[k],
                    sw.dt_sub,
                    pap,
                )
            end
            nl0_ode, nl0_out = _donor_implicit_number_step(
                nl0_env,
                S_nl_0,
                bmt0.dn_lcl_dt + dn_act_0,
                wnl0 * inv_dz,
                sw.dt_sub,
            )
            nr0_ode, nr0_out = _donor_implicit_number_step(
                nr0_env,
                S_nr_0,
                bmt0.dn_rai_dt,
                wnr0 * inv_dz,
                sw.dt_sub,
            )

            scale_0 = (ρa0_old * V_k) / max(sw.dt_sub, floatmin(FT))
            F_mass_out_0 = q0_out .* scale_0
            F_nl_out_0 = nl0_out * scale_0
            F_nr_out_0 = nr0_out * scale_0

            (; ρ_new, ρqt_new, ρq_new, ρe_new, T0_post, F_e_out_0, F_K_out_0, F_e_out_1) =
                _sweep_grid_mean_update(
                    tps,
                    ρ_old,
                    ρqt_old,
                    ρe_old,
                    ρq_gs_pos,
                    ρa0_old,
                    ρa1_old,
                    q0_env,
                    q0_ode,
                    q0_out,
                    q1_0,
                    q1_ode,
                    (;
                        mass_1 = F_mass_in_1,
                        mass_0 = F_mass_in_0,
                        hΦ_1 = F_hΦ_in_1,
                        K_1 = F_K_in_1,
                        hΦ_0 = F_hΦ_in_0,
                        K_0 = F_K_in_0,
                    ),
                    (;
                        mass_1 = F_mass_out_1,
                        mass_0 = F_mass_out_0,
                        hΦ_1 = F_hΦ_out_1,
                        K_1 = F_K_out_1,
                    ),
                    ρa1_new_eff,
                    qt1_new_eff,
                    ρe1_post,
                    w0_k,
                    uvw0,
                    Φ_k,
                    V_k,
                    sw.dt_sub,
                )
            T0_env_fv[k] = T0_post
            ρnl_new = max(
                zero(FT),
                ρnl_gs_pos +
                ρa0_old * (nl0_ode - nl0_env) +
                ρa1_old * (nl1_ode - nl1_0),
            )
            ρnr_new = max(
                zero(FT),
                ρnr_gs_pos +
                ρa0_old * (nr0_ode - nr0_env) +
                ρa1_old * (nr1_ode - nr1_0),
            )

            ρ_fv[k] = ρ_new
            ρqt_fv[k] = ρqt_new
            ρql_fv[k] = ρq_new[1]
            ρqi_fv[k] = ρq_new[2]
            ρqr_fv[k] = ρq_new[3]
            ρqs_fv[k] = ρq_new[4]
            ρnl_fv[k] = ρnl_new
            ρnr_fv[k] = ρnr_new
            ρe_fv[k] = ρe_new

            ρw_gs = (q0_ode .* w0_k) .* ρa0_old + (q1_ode .* w1_k) .* ρa1_old
            wl_fv[k] = ifelse(ρq_new[1] > zero(FT), ρw_gs[1] / ρq_new[1], zero(FT))
            wi_fv[k] = ifelse(ρq_new[2] > zero(FT), ρw_gs[2] / ρq_new[2], zero(FT))
            wr_fv[k] = ifelse(ρq_new[3] > zero(FT), ρw_gs[3] / ρq_new[3], zero(FT))
            ws_fv[k] = ifelse(ρq_new[4] > zero(FT), ρw_gs[4] / ρq_new[4], zero(FT))
            wnl_fv[k] = ifelse(
                ρnl_new > zero(FT),
                (ρa0_old * nl0_ode * wnl0 + ρa1_old * nl1_ode * wnl1) / ρnl_new,
                zero(FT),
            )
            wnr_fv[k] = ifelse(
                ρnr_new > zero(FT),
                (ρa0_old * nr0_ode * wnr0 + ρa1_old * nr1_ode * wnr1) / ρnr_new,
                zero(FT),
            )
            whht_fv[k] =
                WVec((F_e_out_0 + F_e_out_1) / max(A_bot * ρ_new, floatmin(FT)))

            F_mass_in_1 = F_mass_out_1
            F_mass_in_0 = F_mass_out_0
            F_nl_in_1 = F_nl_out_1
            F_nr_in_1 = F_nr_out_1
            F_nl_in_0 = F_nl_out_0
            F_nr_in_0 = F_nr_out_0
            F_hΦ_in_1 = F_hΦ_out_1
            F_K_in_1 = F_K_out_1
            F_hΦ_in_0 = F_e_out_0 - F_K_out_0
            F_K_in_0 = F_K_out_0
            a1_above = a1_eff

            if k == 1
                inv_A_sfc = one(FT) / max(A_bot, floatmin(FT))
                rain_sfc_sum +=
                    -(
                        F_mass_out_0[3] +
                        F_mass_out_0[1] +
                        F_mass_out_1[3] +
                        F_mass_out_1[1]
                    ) * inv_A_sfc
                snow_sfc_sum +=
                    -(
                        F_mass_out_0[4] +
                        F_mass_out_0[2] +
                        F_mass_out_1[4] +
                        F_mass_out_1[2]
                    ) * inv_A_sfc
                energy_sfc_sum += -(F_e_out_0 + F_e_out_1) * inv_A_sfc
            end
        end
    end

    inv_nsub = one(FT) / FT(sw.n_sub)
    @inbounds begin
        Fields.field_values(sfc_col.rain)[] = rain_sfc_sum * inv_nsub
        Fields.field_values(sfc_col.snow)[] = snow_sfc_sum * inv_nsub
        Fields.field_values(sfc_col.energy)[] = energy_sfc_sum * inv_nsub
    end
    return nothing
end

# ============================================================================
# 10. Unified Top-Level Entry Point: `step_coupled_microphysics_sedimentation!`
# ============================================================================

"""
    step_coupled_microphysics_sedimentation!(
        Y,
        p,
        dt;
        dt_sub_2m = 30,
        skip_updraft_cloud_formation = false,
        diag_weight = 1,
        diag_reset = true,
    )

Advance cloud microphysics and hydrometeor sedimentation jointly over `dt` by a
single top-down sweep per column across `DryModel`, `EquilibriumMicrophysics0M`,
`NonEquilibriumMicrophysics1M`, and `NonEquilibriumMicrophysics2M` (with and
without `PrognosticEDMFX`; SGS quadrature with `0M` and `1M`).

Updates `Y.c` (and `Y.c.sgsʲs` under `PrognosticEDMFX`) in place, preserving
dry-air mass cell-wise and column-integrated total water and total energy up to
the bottom-face surface fluxes written to `p.precomputed.surface_rain_flux`,
`p.precomputed.surface_snow_flux`, and
`p.precomputed.col_integrated_precip_energy_tendency`.

When `skip_updraft_cloud_formation` is `true` under
`NonEquilibriumMicrophysics1M` with `PrognosticEDMFX`, updraft cloud liquid and
ice phase relaxation is omitted because it is coupled into the updraft scalar
transport sweep in `column_physics_step.jl`.

Under `EquilibriumMicrophysics0M`, the below-cloud evaporation diagnostics
(`p.precomputed.ᶜprecip_evap`, `col_integrated_precip_evap_rai` and `_sno`)
and the column-integrated updraft precipitation
(`p.precomputed.col_integrated_updraft_precip`) accumulate with weight
`diag_weight`, restarting when `diag_reset`, so substeps of one step give
their step mean.

Under `PrognosticEDMFX`, a single updraft subdomain (`n_updrafts == 1`) is
supported on `0M`, `1M`, and `2M`; multi-updraft configurations raise an error.
`NonEquilibriumMicrophysics2MP3` throws an `ArgumentError`: the sweep has no P3
ice step.
"""
function step_coupled_microphysics_sedimentation!(
    Y,
    p,
    dt;
    dt_sub_2m::Real = 30,
    skip_updraft_cloud_formation::Bool = false,
    diag_weight::Real = 1,
    diag_reset::Bool = true,
)
    FT = eltype(p.params)
    dt_ft = FT(dt)
    dt_ft <= zero(FT) && return nothing
    if p.atmos.microphysics_model isa EquilibriumMicrophysics0M
        return _step_coupled_microphysics_sedimentation_0m!(
            Y,
            p,
            dt_ft,
            p.atmos.turbconv_model,
            FT(diag_weight),
            diag_reset,
        )
    end
    if p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M
        return _step_coupled_microphysics_sedimentation_1m!(
            Y,
            p,
            dt_ft,
            p.atmos.turbconv_model,
            skip_updraft_cloud_formation,
        )
    end
    return _step_coupled_microphysics_sedimentation!(
        Y,
        p,
        dt_ft,
        p.atmos.microphysics_model,
        p.atmos.turbconv_model,
        FT(dt_sub_2m),
    )
end

function _step_coupled_microphysics_sedimentation!(
    Y,
    p,
    dt,
    ::DryModel,
    _,
    _,
)
    FT = typeof(dt)
    hasproperty(p.precomputed, :ᶜwₕhₜ) &&
        fill!(parent(p.precomputed.ᶜwₕhₜ), zero(FT))
    return nothing
end

function _step_coupled_microphysics_sedimentation_0m!(
    Y,
    p,
    dt,
    turbconv_model,
    diag_w,
    diag_reset::Bool,
)
    FT = typeof(dt)
    hasproperty(p.precomputed, :ᶜwₕhₜ) &&
        fill!(parent(p.precomputed.ᶜwₕhₜ), zero(FT))
    dev = ClimaComms.device(Y.c)
    tps = _fold(dev, CAP.thermodynamics_params(p.params))
    cm0_host = CAP.microphysics_0m_params(p.params)
    has_evap = !isnothing(cm0_host.evaporation)
    cm0 = _fold(dev, cm0_host)
    sgs_quad = p.atmos.sgs_quadrature
    has_sgs = !not_quadrature(sgs_quad)

    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠlg = Fields.local_geometry_field(Y.f)
    geom = (; ᶜlg, ᶠlg, p.core.ᶠ∂ξ³∂x³, p.core.ᶜΦ)
    sfc = (;
        rain = p.precomputed.surface_rain_flux,
        snow = p.precomputed.surface_snow_flux,
        energy = p.precomputed.col_integrated_precip_energy_tendency,
        evap_rai = p.precomputed.col_integrated_precip_evap_rai,
        evap_sno = p.precomputed.col_integrated_precip_evap_sno,
        updraft = p.precomputed.col_integrated_updraft_precip,
    )
    (; ᶜprecip_evap, ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt) = p.precomputed
    evap = (;
        p.precomputed.ᶜp,
        ᶜprecip_evap.ρ_evap_rai,
        ᶜprecip_evap.ρ_evap_sno,
        ᶜρ_dq_tot_dt,
        ᶜρ_de_tot_dt,
    )
    sgs_fields =
        has_sgs ?
        (;
            p.precomputed.ᶜT′T′,
            p.precomputed.ᶜq′q′,
            p.precomputed.ᶜcorr_Tq,
        ) : ()

    if turbconv_model isa PrognosticEDMFX
        n_mass_flux_subdomains(turbconv_model) == 1 ||
            error("Coupled EDMFX microphysics sweep supports a single updraft")
        edmf = (;
            ᶜρ¹ = p.precomputed.ᶜρʲs.:(1),
            ᶜT¹ = p.precomputed.ᶜTʲs.:(1),
            ᶜq_liq¹ = p.precomputed.ᶜq_liqʲs.:(1),
            ᶜq_ice¹ = p.precomputed.ᶜq_iceʲs.:(1),
            p.precomputed.ᶜT⁰,
            p.precomputed.ᶜp,
            p.precomputed.ᶜq_tot_nonneg⁰,
            p.precomputed.ᶜq_liq⁰,
            p.precomputed.ᶜq_ice⁰,
        )
        passive_names = passive_sgs_tracer_names(Y)
        # Updraft (convective) condensate precipitates with its own timescale.
        cm0_up = _fold(
            dev,
            CMP.Microphysics0MParams(; precip = cm0_host.precip_updraft),
        )
        sw = Coupled0MEDMFXSweep{
            has_sgs,
            has_evap,
            typeof(cm0),
            typeof(cm0_up),
            typeof(tps),
            typeof(sgs_quad),
            typeof(passive_names),
            FT,
        }(
            cm0,
            cm0_up,
            tps,
            sgs_quad,
            passive_names,
            dt,
            diag_w,
            diag_reset,
        )
        DataLayouts.foreach_column(
            sw,
            Y.c,
            geom,
            edmf,
            sgs_fields,
            sfc,
            evap;
            shmem = Val(:none),
        )
    else
        sw = Coupled0MGridMeanSweep{
            has_sgs,
            has_evap,
            typeof(cm0),
            typeof(tps),
            typeof(sgs_quad),
            FT,
        }(
            cm0,
            tps,
            sgs_quad,
            dt,
            diag_w,
            diag_reset,
        )
        DataLayouts.foreach_column(
            sw,
            Y.c,
            geom,
            p.precomputed.ᶜu,
            sgs_fields,
            sfc,
            evap;
            shmem = Val(:none),
        )
    end
    return nothing
end

@inline function _build_1m_edmf_sweep(
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_quad,
    sgs_opts,
    passive_names,
    turbconv_model,
    dt::FT,
    f_decay::FT,
    q_precip_min::FT,
    cf_floor::FT,
    c_w::FT,
    w_d::FT,
    a_min::FT,
    ::Val{HasSGS},
    ::Val{HasPF},
    ::Val{HasPS},
    ::Val{HasICS},
    ::Val{HasEA},
) where {FT, HasSGS, HasPF, HasPS, HasICS, HasEA}
    return Coupled1MEDMFXColumnSweep{
        HasSGS,
        HasPF,
        HasPS,
        HasICS,
        HasEA,
        typeof(vcmp_up),
        typeof(vcmp),
        typeof(vtps),
        typeof(vc),
        typeof(sgs_quad),
        typeof(sgs_opts),
        typeof(passive_names),
        typeof(turbconv_model),
        FT,
    }(
        vcmp_up,
        vcmp,
        vtps,
        vc,
        sgs_quad,
        sgs_opts,
        passive_names,
        turbconv_model,
        dt,
        f_decay,
        q_precip_min,
        cf_floor,
        c_w,
        w_d,
        a_min,
    )
end

@inline function _build_1m_gridmean_sweep(
    vcmp,
    vtps,
    vc,
    sgs_quad,
    sgs_opts,
    dt::FT,
    f_decay::FT,
    q_precip_min::FT,
    cf_floor::FT,
    ::Val{HasSGS},
    ::Val{HasPF},
    ::Val{HasPS},
    ::Val{HasICS},
) where {FT, HasSGS, HasPF, HasPS, HasICS}
    return Coupled1MGridMeanSweep{
        HasSGS,
        HasPF,
        HasPS,
        HasICS,
        typeof(vcmp),
        typeof(vtps),
        typeof(vc),
        typeof(sgs_quad),
        typeof(sgs_opts),
        FT,
    }(
        vcmp,
        vtps,
        vc,
        sgs_quad,
        sgs_opts,
        dt,
        f_decay,
        q_precip_min,
        cf_floor,
    )
end

function _launch_coupled_1m_sweep!(
    Y,
    p,
    dt::FT,
    turbconv_model,
    sgs_quad,
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_opts,
    q_precip_min::FT,
    f_decay::FT,
    cf_floor::FT,
    c_w::FT,
    w_d::FT,
    a_min::FT,
    ::Val{HasSGS},
    ::Val{HasPF},
    ::Val{HasPS},
    ::Val{HasICS},
    ::Val{HasEA},
) where {FT, HasSGS, HasPF, HasPS, HasICS, HasEA}
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠlg = Fields.local_geometry_field(Y.f)
    geom = (; ᶜlg, ᶠlg, p.core.ᶠ∂ξ³∂x³, p.core.ᶜΦ)
    sfc = (;
        rain = p.precomputed.surface_rain_flux,
        snow = p.precomputed.surface_snow_flux,
        energy = p.precomputed.col_integrated_precip_energy_tendency,
    )
    sgs_fields = if HasSGS && HasPF
        (;
            ᶜcf = p.precomputed.ᶜcloud_fraction,
            p.precomputed.ᶜprecip_frac,
            p.precomputed.ᶜT′T′,
            p.precomputed.ᶜq′q′,
            p.precomputed.ᶜcorr_Tq,
            p.precomputed.ᶜsgs_moments,
            p.precomputed.ᶜα,
        )
    elseif HasSGS
        (;
            p.precomputed.ᶜT′T′,
            p.precomputed.ᶜq′q′,
            p.precomputed.ᶜcorr_Tq,
            p.precomputed.ᶜsgs_moments,
            p.precomputed.ᶜα,
        )
    elseif HasPF
        (;
            ᶜcf = p.precomputed.ᶜcloud_fraction,
            p.precomputed.ᶜprecip_frac,
        )
    else
        ()
    end

    ρ_data = Fields.field_values(Y.c.ρ)
    use_compact =
        (ClimaComms.device(Y.c) isa ClimaComms.CUDADevice) &&
        (ρ_data isa DataLayouts.VIJHWithF) &&
        (DataLayouts.nelems(ρ_data) > 0)

    if turbconv_model isa PrognosticEDMFX
        n_mass_flux_subdomains(turbconv_model) == 1 ||
            error("Coupled EDMFX microphysics sweep supports a single updraft")
        ᶜcf_sed = if HasICS
            hasproperty(p.precomputed, :ᶜcloud_fraction⁰) ?
            p.precomputed.ᶜcloud_fraction⁰ : p.precomputed.ᶜcloud_fraction
        else
            p.precomputed.ᶜwᵣ
        end
        pre = (;
            p.precomputed.ᶜwₗ,
            p.precomputed.ᶜwᵢ,
            p.precomputed.ᶜwᵣ,
            p.precomputed.ᶜwₛ,
            p.precomputed.ᶜwₕhₜ,
            ᶜcf_sed,
        )
        edmf = (;
            ᶜρ¹ = p.precomputed.ᶜρʲs.:(1),
            ᶜT¹ = p.precomputed.ᶜTʲs.:(1),
            ᶜu¹ = p.precomputed.ᶜuʲs.:(1),
            ᶜwₗ¹ = p.precomputed.ᶜwₗʲs.:(1),
            ᶜwᵢ¹ = p.precomputed.ᶜwᵢʲs.:(1),
            ᶜwᵣ¹ = p.precomputed.ᶜwᵣʲs.:(1),
            ᶜwₛ¹ = p.precomputed.ᶜwₛʲs.:(1),
            p.precomputed.ᶜT⁰,
            p.precomputed.ᶜp,
            p.precomputed.ᶜu⁰,
        )
        passive_names = passive_sgs_tracer_names(Y)
        sw = _build_1m_edmf_sweep(
            vcmp_up,
            vcmp,
            vtps,
            vc,
            sgs_quad,
            sgs_opts,
            passive_names,
            turbconv_model,
            dt,
            f_decay,
            q_precip_min,
            cf_floor,
            c_w,
            w_d,
            a_min,
            Val(HasSGS),
            Val(HasPF),
            Val(HasPS),
            Val(HasICS),
            Val(HasEA),
        )
        DataLayouts.foreach_column(
            sw,
            Y.c,
            geom,
            pre,
            edmf,
            sgs_fields,
            sfc;
            shmem = Val(:none),
        )
    else
        ᶜcf_sed =
            HasICS ? p.precomputed.ᶜcloud_fraction : p.precomputed.ᶜwᵣ
        pre = (;
            p.precomputed.ᶜu,
            p.precomputed.ᶜwₗ,
            p.precomputed.ᶜwᵢ,
            p.precomputed.ᶜwᵣ,
            p.precomputed.ᶜwₛ,
            p.precomputed.ᶜwₕhₜ,
            ᶜcf_sed,
        )
        sw = _build_1m_gridmean_sweep(
            vcmp,
            vtps,
            vc,
            sgs_quad,
            sgs_opts,
            dt,
            f_decay,
            q_precip_min,
            cf_floor,
            Val(HasSGS),
            Val(HasPF),
            Val(HasPS),
            Val(HasICS),
        )
        if use_compact
            ws = _get_coupled_1m_compact_workspace(ρ_data)
            _run_coupled_1m_gridmean_compacted!(
                ws,
                sw,
                Y.c,
                geom,
                pre,
                sgs_fields,
                sfc,
            )
        else
            DataLayouts.foreach_column(
                sw,
                Y.c,
                geom,
                pre,
                sgs_fields,
                sfc;
                shmem = Val(:none),
            )
        end
    end
    return nothing
end

# Static boolean-to-Val cascade avoids passing >4 Union{Val{true},Val{false}}
# arguments at once, keeping dynamic dispatch and CPU heap allocations at 0 B.
@inline function _dispatch_1m_ea!(
    Y,
    p,
    dt,
    turbconv_model,
    sgs_quad,
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_opts,
    q_precip_min,
    f_decay,
    cf_floor,
    c_w,
    w_d,
    a_min,
    ::Val{false},
    v_pf::Val,
    v_ps::Val,
    v_ics::Val,
    ::Bool,
)
    return _launch_coupled_1m_sweep!(
        Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
        q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
        Val(false), v_pf, v_ps, v_ics, Val(false),
    )
end
@inline function _dispatch_1m_ea!(
    Y,
    p,
    dt,
    turbconv_model,
    sgs_quad,
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_opts,
    q_precip_min,
    f_decay,
    cf_floor,
    c_w,
    w_d,
    a_min,
    v_sgs::Val{true},
    v_pf::Val,
    v_ps::Val,
    v_ics::Val,
    has_ea::Bool,
)
    return if has_ea
        _launch_coupled_1m_sweep!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            v_sgs, v_pf, v_ps, v_ics, Val(true),
        )
    else
        _launch_coupled_1m_sweep!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            v_sgs, v_pf, v_ps, v_ics, Val(false),
        )
    end
end
@inline function _dispatch_1m_ics!(
    Y,
    p,
    dt,
    turbconv_model,
    sgs_quad,
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_opts,
    q_precip_min,
    f_decay,
    cf_floor,
    c_w,
    w_d,
    a_min,
    v_sgs::Val,
    v_pf::Val,
    v_ps::Val,
    has_ics::Bool,
    has_ea::Bool,
)
    return if has_ics
        _dispatch_1m_ea!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            v_sgs, v_pf, v_ps, Val(true), has_ea,
        )
    else
        _dispatch_1m_ea!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            v_sgs, v_pf, v_ps, Val(false), has_ea,
        )
    end
end
@inline _dispatch_1m_ps!(
    Y,
    p,
    dt,
    turbconv_model,
    sgs_quad,
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_opts,
    q_precip_min,
    f_decay,
    cf_floor,
    c_w,
    w_d,
    a_min,
    v_sgs::Val{false},
    v_pf::Val,
    ::Bool,
    has_ics::Bool,
    has_ea::Bool,
) = _dispatch_1m_ics!(
    Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
    q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
    v_sgs, v_pf, Val(false), has_ics, has_ea,
)
@inline function _dispatch_1m_ps!(
    Y,
    p,
    dt,
    turbconv_model,
    sgs_quad,
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_opts,
    q_precip_min,
    f_decay,
    cf_floor,
    c_w,
    w_d,
    a_min,
    v_sgs::Val{true},
    v_pf::Val,
    has_ps::Bool,
    has_ics::Bool,
    has_ea::Bool,
)
    return if has_ps
        _dispatch_1m_ics!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            v_sgs, v_pf, Val(true), has_ics, has_ea,
        )
    else
        _dispatch_1m_ics!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            v_sgs, v_pf, Val(false), has_ics, has_ea,
        )
    end
end
@inline function _dispatch_1m_pf!(
    Y,
    p,
    dt,
    turbconv_model,
    sgs_quad,
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_opts,
    q_precip_min,
    f_decay,
    cf_floor,
    c_w,
    w_d,
    a_min,
    v_sgs::Val,
    has_pf::Bool,
    has_ps::Bool,
    has_ics::Bool,
    has_ea::Bool,
)
    return if has_pf
        _dispatch_1m_ps!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            v_sgs, Val(true), has_ps, has_ics, has_ea,
        )
    else
        _dispatch_1m_ps!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            v_sgs, Val(false), has_ps, has_ics, has_ea,
        )
    end
end

@inline function _dispatch_coupled_1m_sweep!(
    Y,
    p,
    dt,
    turbconv_model,
    sgs_quad,
    vcmp_up,
    vcmp,
    vtps,
    vc,
    sgs_opts,
    q_precip_min,
    f_decay,
    cf_floor,
    c_w,
    w_d,
    a_min,
    has_sgs::Bool,
    has_pf::Bool,
    has_ps::Bool,
    has_ics::Bool,
    has_ea::Bool,
)
    return if has_sgs
        _dispatch_1m_pf!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            Val(true), has_pf, has_ps, has_ics, has_ea,
        )
    else
        _dispatch_1m_pf!(
            Y, p, dt, turbconv_model, sgs_quad, vcmp_up, vcmp, vtps, vc, sgs_opts,
            q_precip_min, f_decay, cf_floor, c_w, w_d, a_min,
            Val(false), has_pf, has_ps, has_ics, has_ea,
        )
    end
end

_step_coupled_microphysics_sedimentation!(
    Y,
    p,
    dt::FT,
    ::NonEquilibriumMicrophysics1M,
    turbconv_model,
    _,
) where {FT} = _step_coupled_microphysics_sedimentation_1m!(
    Y,
    p,
    dt,
    turbconv_model,
    false,
)

function _step_coupled_microphysics_sedimentation_1m!(
    Y,
    p,
    dt,
    turbconv_model,
    skip_updraft_cloud_formation::Bool,
)
    FT = typeof(dt)
    dev = ClimaComms.device(Y.c)
    sgs_quad = p.atmos.sgs_quadrature
    has_sgs = !not_quadrature(sgs_quad)
    has_pf = hasproperty(p.precomputed, :ᶜprecip_frac)
    thp = CAP.thermodynamics_params(p.params)
    cmp_full = _combined_microphysics_1m_params(p.params)
    vc = _build_velocity_config_1m(p.atmos.water, p.params, dev, cmp_full)
    vtps = _fold(dev, thp)
    vcmp = _fold(dev, cmp_full)
    sgs_opts =
        has_sgs ? sgs_microphysics_options(p.params) :
        SGSMicrophysicsOptions(zero(FT), zero(FT), zero(FT), zero(FT))
    q_precip_min = FT(TD.Parameters.q_min(thp))
    f_decay =
        has_pf ?
        clamp(FT(CAP.sgs_precip_overlap_decay(p.params)), zero(FT), one(FT)) :
        zero(FT)
    cf_floor = FT(CAP.sgs_cloud_sedimentation_cf_floor(p.params))
    has_ics = (cf_floor < one(FT)) && hasproperty(p.precomputed, :ᶜcloud_fraction)
    has_ps = has_sgs && (sgs_opts.precip_incloud_fraction > zero(FT))
    c_w = FT(CAP.sgs_env_autoconversion_updraft_velocity_weight(p.params))
    w_d = FT(CAP.sgs_env_autoconversion_detrained_velocity(p.params))
    a_min =
        turbconv_model isa PrognosticEDMFX ?
        FT(CAP.min_area(CAP.turbconv_params(p.params))) : zero(FT)
    has_ea =
        has_sgs &&
        (turbconv_model isa PrognosticEDMFX) &&
        uses_env_autoconversion_velocity(cmp_full, p.params)

    if skip_updraft_cloud_formation && (turbconv_model isa PrognosticEDMFX)
        vcmp_up = _fold(dev, _without_cloud_formation(cmp_full))
        _dispatch_coupled_1m_sweep!(
            Y,
            p,
            dt,
            turbconv_model,
            sgs_quad,
            vcmp_up,
            vcmp,
            vtps,
            vc,
            sgs_opts,
            q_precip_min,
            f_decay,
            cf_floor,
            c_w,
            w_d,
            a_min,
            has_sgs,
            has_pf,
            has_ps,
            has_ics,
            has_ea,
        )
    else
        _dispatch_coupled_1m_sweep!(
            Y,
            p,
            dt,
            turbconv_model,
            sgs_quad,
            vcmp,
            vcmp,
            vtps,
            vc,
            sgs_opts,
            q_precip_min,
            f_decay,
            cf_floor,
            c_w,
            w_d,
            a_min,
            has_sgs,
            has_pf,
            has_ps,
            has_ics,
            has_ea,
        )
    end
    return nothing
end

function _step_coupled_microphysics_sedimentation!(
    Y,
    p,
    dt,
    ::NonEquilibriumMicrophysics2M,
    turbconv_model,
    dt_sub_max,
)
    FT = typeof(dt)
    not_quadrature(p.atmos.sgs_quadrature) || error(
        "SGS quadrature is not implemented for 2-moment microphysics",
    )
    dev = ClimaComms.device(Y.c)
    tps = _fold(dev, CAP.thermodynamics_params(p.params))
    cm2p = _fold(dev, CAP.microphysics_2m_params(p.params))
    cm1p = _fold(dev, CAP.microphysics_1m_params(p.params))
    cmc = _fold(dev, CAP.microphysics_cloud_params(p.params))
    has_aerosols =
        hasproperty(p, :tracers) &&
        hasproperty(p.tracers, :prescribed_aerosols_field)
    pap, aero = if has_aerosols
        pap_raw = p.params.prescribed_aerosol_params
        seasalt_num = p.scratch.ᶜtemp_scalar
        seasalt_mean_radius = p.scratch.ᶜtemp_scalar_2
        sulfate_num = p.scratch.ᶜtemp_scalar_3
        compute_prescribed_aerosol_properties!(
            seasalt_num,
            seasalt_mean_radius,
            sulfate_num,
            p.tracers.prescribed_aerosols_field,
            pap_raw,
        )
        _fold(dev, pap_raw), (; seasalt_num, seasalt_mean_radius, sulfate_num)
    else
        nothing, ()
    end

    n_sub = max(1, Int(cld(dt, max(dt_sub_max, eps(FT)))))
    dt_sub = dt / FT(n_sub)

    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠlg = Fields.local_geometry_field(Y.f)
    geom = (; ᶜlg, ᶠlg, p.core.ᶠ∂ξ³∂x³, p.core.ᶜΦ)
    sfc = (;
        rain = p.precomputed.surface_rain_flux,
        snow = p.precomputed.surface_snow_flux,
        energy = p.precomputed.col_integrated_precip_energy_tendency,
    )

    if turbconv_model isa PrognosticEDMFX
        n_mass_flux_subdomains(turbconv_model) == 1 ||
            error("Coupled EDMFX microphysics sweep supports a single updraft")
        pre = (;
            p.precomputed.ᶜwₗ,
            p.precomputed.ᶜwᵢ,
            p.precomputed.ᶜwᵣ,
            p.precomputed.ᶜwₛ,
            p.precomputed.ᶜwₙₗ,
            p.precomputed.ᶜwₙᵣ,
            p.precomputed.ᶜwₕhₜ,
        )
        edmf = (;
            ᶜρ¹ = p.precomputed.ᶜρʲs.:(1),
            ᶜT¹ = p.precomputed.ᶜTʲs.:(1),
            ᶜu¹ = p.precomputed.ᶜuʲs.:(1),
            p.precomputed.ᶜT⁰,
            p.precomputed.ᶜp,
            p.precomputed.ᶜu⁰,
        )
        passive_names = passive_sgs_tracer_names(Y)
        sw = Coupled2MEDMFXSweep(
            cm2p,
            cm1p,
            cmc,
            tps,
            pap,
            passive_names,
            turbconv_model,
            dt_sub,
            n_sub,
        )
        DataLayouts.foreach_column(
            sw,
            Y.c,
            geom,
            pre,
            edmf,
            sfc,
            aero;
            shmem = Val(:none),
        )
    else
        pre = (;
            p.precomputed.ᶜu,
            p.precomputed.ᶜp,
            p.precomputed.ᶜwₗ,
            p.precomputed.ᶜwᵢ,
            p.precomputed.ᶜwᵣ,
            p.precomputed.ᶜwₛ,
            p.precomputed.ᶜwₙₗ,
            p.precomputed.ᶜwₙᵣ,
            p.precomputed.ᶜwₕhₜ,
        )
        sw = Coupled2MGridMeanSweep{
            typeof(cm2p),
            typeof(cm1p),
            typeof(cmc),
            typeof(tps),
            typeof(pap),
            FT,
        }(
            cm2p,
            cm1p,
            cmc,
            tps,
            pap,
            dt_sub,
            n_sub,
        )
        DataLayouts.foreach_column(
            sw,
            Y.c,
            geom,
            pre,
            sfc,
            aero;
            shmem = Val(:none),
        )
    end
    return nothing
end

function _step_coupled_microphysics_sedimentation!(
    Y,
    p,
    dt,
    ::NonEquilibriumMicrophysics2MP3,
    turbconv_model,
    dt_sub_max,
)
    throw(
        ArgumentError(
            "The coupled microphysics sweep does not support " *
            "NonEquilibriumMicrophysics2MP3 (2-moment microphysics with P3 " *
            "ice); use NonEquilibriumMicrophysics2M",
        ),
    )
end
