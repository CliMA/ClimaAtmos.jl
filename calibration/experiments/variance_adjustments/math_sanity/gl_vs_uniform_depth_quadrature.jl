# Self-contained demo (no ClimaAtmos load): compare **Gauss–Legendre** vs other **quadrature**
# rules in `z` for the **layer mean** of `f(z)` on `z ∈ [z_lo, z_hi]` with **uniform** `dz`
# (same target as column-tensor outer average). Every rule here is **only** a finite sum
# `∑ w_k f(z_k)` — including the **composite trapezoid** stencil (uniform `z`-nodes with
# trapezoid weights). That trapezoid sum is **not** used in ClimaAtmos production; it appears
# here as an optional **pedagogical** comparison when physics is **face-heavy** (see script).
#
# The comparison rule is **not** “only sample z = cell center.” It splits `[z_lo, z_hi]`
# into **N equal-thickness slabs** and evaluates `f` at the **geometric center of each
# slab** (standard name: *midpoint of each subinterval*). Those centers are equally
# spaced with spacing `H/N`, offset by half a slab from `z_lo`/`z_hi`. Weights are `1/N`
# each — the equal-weight-per-slab picture.
#
#   mean[f] := (1 / (z_hi - z_lo)) ∫_{z_lo}^{z_hi} f(z) dz
#            = (1/2) ∫_{-1}^{1} f(z(τ)) dτ,   z(τ) = (z_hi-z_lo)/2 * τ + (z_hi+z_lo)/2
#
# Run from anywhere:
#   julia gl_vs_uniform_depth_quadrature.jl

using Printf

const FT = Float64
const Z_LO = FT(1933.1)
const Z_HI = FT(2021.23)

function gauss_legendre_01(::Type{FT}, N::Int) where {FT}
    half = FT(1) / FT(2)
    if N == 1
        return (FT[half], FT[1])
    elseif N == 2
        a = one(FT) / sqrt(FT(3))
        return (FT[(1 - a) * half, (1 + a) * half], FT[half, half])
    elseif N == 3
        a = sqrt(FT(3) / FT(5))
        return (
            FT[(1 - a) * half, half, (1 + a) * half],
            FT[FT(5) / 18, FT(4) / FT(9), FT(5) / 18],
        )
    elseif N == 4
        a1 = sqrt(FT(3) / FT(7) - FT(2) / FT(7) * sqrt(FT(6) / FT(5)))
        a2 = sqrt(FT(3) / FT(7) + FT(2) / FT(7) * sqrt(FT(6) / FT(5)))
        w1 = (FT(18) + sqrt(FT(30))) / FT(36)
        w2 = (FT(18) - sqrt(FT(30))) / FT(36)
        return (
            FT[(1 - a2) * half, (1 - a1) * half, (1 + a1) * half, (1 + a2) * half],
            FT[w2 * half, w1 * half, w1 * half, w2 * half],
        )
    elseif N == 5
        a1 = FT(1) / FT(3) * sqrt(FT(5) - FT(2) * sqrt(FT(10) / FT(7)))
        a2 = FT(1) / FT(3) * sqrt(FT(5) + FT(2) * sqrt(FT(10) / FT(7)))
        w0 = FT(128) / FT(225)
        w1 = (FT(322) + FT(13) * sqrt(FT(70))) / FT(900)
        w2 = (FT(322) - FT(13) * sqrt(FT(70))) / FT(900)
        return (
            FT[(1 - a2) * half, (1 - a1) * half, half, (1 + a1) * half, (1 + a2) * half],
            FT[w2 * half, w1 * half, w0 * half, w1 * half, w2 * half],
        )
    else
        error("N must be in 1:5, got $N")
    end
end

function gauss_legendre_neg1_1(::Type{FT}, N::Int) where {FT}
    x, w01 = gauss_legendre_01(FT, N)
    two = FT(2)
    t = map(xv -> two * xv - one(FT), x)
    w = map(wv -> two * wv, w01)
    return (t, w)
end

@inline function z_of_τ(z_lo::FT, z_hi::FT, τ::FT) where {FT}
    half_L = (z_hi - z_lo) / FT(2)
    mid = (z_lo + z_hi) / FT(2)
    return mid + half_L * τ
end

"""
Centers of `N` equal-width `z`-slabs on `[z_lo, z_hi]` (composite midpoint / rectangle rule).

`z_k = z_lo + (k - 1/2) * (H/N)` is the **midpoint of the k-th subinterval**, not “only mid-cell.”
"""
function equal_slab_centers_z(z_lo::FT, z_hi::FT, N::Int) where {FT}
    L = z_hi - z_lo
    return FT[z_lo + (FT(2k - 1) / FT(2N)) * L for k in 1:N]
end

"""Layer mean via GL on τ ∈ [-1,1] (same as `wk = z_w[k]/2` in column tensor)."""
function layer_mean_gl(f, z_lo::FT, z_hi::FT, N::Int) where {FT}
    τs, zw = gauss_legendre_neg1_1(FT, N)
    s = zero(FT)
    @inbounds for k in eachindex(τs)
        z = z_of_τ(z_lo, z_hi, τs[k])
        s += (zw[k] / FT(2)) * f(z)
    end
    return s
end

"""Composite midpoint in `z`: `(1/N) ∑ f(z_k)` with `z_k` at each equal-slab center."""
function layer_mean_equal_slab_centers(f, z_lo::FT, z_hi::FT, N::Int) where {FT}
    zs = equal_slab_centers_z(z_lo, z_hi, N)
    return sum(f(z) for z in zs) / N
end

"""
    layer_mean_quad_trapezoid_uniform_nodes(f, z_lo, z_hi, N)

**Quadrature** for the layer mean `(1/H) ∫ f dz` using **uniform** nodes
`z_j = z_lo + j·H/(N-1)`, `j = 0,…,N-1`, and **composite trapezoid weights** (same sum you
would write in a homework “trapezoid rule” — it is not a separate magic from GL; both are
`∑ w_k f(z_k)` with different `(z_k, w_k)`).

**Not** used in `SubgridColumnTensor` / production ClimaAtmos; included only to contrast
**face-inclusive** stencils (always evaluates `f(z_lo)` and `f(z_hi)`) with **interior-only**
Gauss–Legendre abscissas on the mapped `τ ∈ (-1,1)`.
"""
function layer_mean_quad_trapezoid_uniform_nodes(f, z_lo::FT, z_hi::FT, N::Int) where {FT}
    N < 2 && error("trapezoid quadrature needs N ≥ 2")
    L = z_hi - z_lo
    h = L / FT(N - 1)
    s = f(z_lo) + f(z_hi)
    @inbounds for j in 1:(N - 2)
        s += FT(2) * f(z_lo + h * FT(j))
    end
    return s / (FT(2) * FT(N - 1))
end

# --- exact layer means on [z_lo, z_hi] ---

function exact_mean_sin(z_lo::FT, z_hi::FT) where {FT}
    L = z_hi - z_lo
    return (cos(z_lo) - cos(z_hi)) / L
end

function exact_mean_quadratic(z_lo::FT, z_hi::FT) where {FT}
    # f(z) = z^2
    L = z_hi - z_lo
    return (z_hi^3 - z_lo^3) / (FT(3) * L)
end

function exact_mean_cubic(z_lo::FT, z_hi::FT) where {FT}
    # f(z) = z^3
    L = z_hi - z_lo
    return (z_hi^4 - z_lo^4) / (FT(4) * L)
end

"""
    exact_mean_indicator_top(z_lo, z_hi, frac_top)

Layer mean of `1(z ≥ z_th)` with `z_th = z_hi - frac_top * (z_hi-z_lo)` (supersaturation only
in the top `frac_top` fraction of the layer). Exact mean = `frac_top` for `0 ≤ frac_top ≤ 1`.
"""
function exact_mean_indicator_top(z_lo::FT, z_hi::FT, frac_top::FT) where {FT}
    L = z_hi - z_lo
    return frac_top # (z_hi - z_th) / L with z_th = z_hi - frac_top*L
end

"""Layer mean of `t^Q + (1-t)^Q` with `t=(z-z_lo)/H` (dry middle, cloud signal near **both** faces)."""
function exact_mean_edge_cloud_u(Q::FT) where {FT}
    return FT(2) / (Q + one(FT))
end

"""Print `N = 3` nodes, weights, `f`, and `w·f` for GL vs slab — same integrand, **different** `t` samples."""
function print_n3_edge_cloud_audit(z_lo::FT, z_hi::FT, Q::FT) where {FT}
    L = z_hi - z_lo
    f = z -> begin
        t = (z - z_lo) / L
        return t^Q + (one(FT) - t)^Q
    end
    exact = exact_mean_edge_cloud_u(Q)
    println(@sprintf("--- N=3 audit (f = t^%.0f + (1-t)^%.0f): GL vs slab use different heights, not only weights ---", Q, Q))

    τs, zw = gauss_legendre_neg1_1(FT, 3)
    println(@sprintf("%4s  %9s  %10s  %10s  %12s  %12s", "GL", "τ", "t", "w(mean)", "f", "w·f"))
    s_gl = zero(FT)
    for k in 1:3
        τ = τs[k]
        z = z_of_τ(z_lo, z_hi, τ)
        t = (z - z_lo) / L
        w = zw[k] / FT(2)
        fv = f(z)
        c = w * fv
        s_gl += c
        println(@sprintf("%4d  %9.5f  %10.5f  %10.6f  %12.4e  %12.4e", k, τ, t, w, fv, c))
    end
    println(@sprintf("GL row-sum = %.8f  (exact = %.8f, |err| = %.3e)\n", s_gl, exact, abs(s_gl - exact)))

    zs = equal_slab_centers_z(z_lo, z_hi, 3)
    w_s = FT(1) / FT(3)
    println(@sprintf("%4s  %9s  %10s  %10s  %12s  %12s", "slab", "", "t", "w(mean)", "f", "w·f"))
    s_sl = zero(FT)
    for k in 1:3
        z = zs[k]
        t = (z - z_lo) / L
        fv = f(z)
        c = w_s * fv
        s_sl += c
        println(@sprintf("%4d  %9s  %10.5f  %10.6f  %12.4e  %12.4e", k, "—", t, w_s, fv, c))
    end
    println(@sprintf("slab row-sum = %.8f  (exact = %.8f, |err| = %.3e)\n", s_sl, exact, abs(s_sl - exact)))
    println("Outer `t` for GL are closer to 0 and 1 than slab outers → larger `f` there; that can")
    println("swamp the fact GL outer **weights** (5/18) are smaller than slab’s 1/3.\n")
    return nothing
end

function print_comparison(name, f, exact_mean, z_lo, z_hi)
    println("=== $name  (exact layer mean = $exact_mean) ===")
    @printf("%3s  %14s  %14s  %14s\n", "N", "|GL err|", "|slab ctr|", "ctr/GL")
    for N in 1:5
        e_gl = abs(layer_mean_gl(f, z_lo, z_hi, N) - exact_mean)
        e_mid = abs(layer_mean_equal_slab_centers(f, z_lo, z_hi, N) - exact_mean)
        ratio = e_gl > 0 ? e_mid / e_gl : (e_mid > 0 ? Inf : 1.0)
        rstr = isfinite(ratio) ? @sprintf("%.4g", ratio) : "Inf"
        @printf("%3d  %14.6e  %14.6e  %14s\n", N, e_gl, e_mid, rstr)
    end
    println()
    return nothing
end

"""GL vs slab-centers vs trapezoid quadrature (uniform nodes); trapezoid only for `N ≥ 2`."""
function print_comparison_three(name, f, exact_mean, z_lo, z_hi)
    println("=== $name  (exact layer mean = $exact_mean) ===")
    @printf(
        "%3s  %12s  %12s  %12s  %10s  %10s\n",
        "N",
        "|GL|",
        "|slab|",
        "|trap|",
        "slab/GL",
        "trap/GL",
    )
    for N in 1:5
        e_gl = abs(layer_mean_gl(f, z_lo, z_hi, N) - exact_mean)
        e_mid = abs(layer_mean_equal_slab_centers(f, z_lo, z_hi, N) - exact_mean)
        e_tr = N < 2 ? FT(NaN) : abs(layer_mean_quad_trapezoid_uniform_nodes(f, z_lo, z_hi, N) - exact_mean)
        r_mg = e_gl > 0 ? e_mid / e_gl : (e_mid > 0 ? Inf : 1.0)
        r_tg = N < 2 || e_gl > 0 ? (N < 2 ? NaN : e_tr / e_gl) : (e_tr > 0 ? Inf : 1.0)
        r_mg_s = isfinite(r_mg) ? @sprintf("%.3g", r_mg) : "Inf"
        r_tg_s = N < 2 ? "  —" : (isfinite(r_tg) ? @sprintf("%8.3g", r_tg) : "     Inf")
        if N < 2
            @printf("%3d  %12.4e  %12.4e  %12s  %10s  %10s\n", N, e_gl, e_mid, "—", r_mg_s, "  —")
        else
            @printf("%3d  %12.4e  %12.4e  %12.4e  %10s  %10s\n", N, e_gl, e_mid, e_tr, r_mg_s, r_tg_s)
        end
    end
    println()
    return nothing
end

function main()
    z_lo, z_hi = Z_LO, Z_HI
    println("Physical interval: z ∈ [$z_lo, $z_hi], length = $(z_hi - z_lo)")
    println("Target: mean[f] = (1/(z_hi-z_lo)) ∫ f(z) dz\n")

    print_comparison(
        "f(z) = sin(z)",
        sin,
        exact_mean_sin(z_lo, z_hi),
        z_lo,
        z_hi,
    )
    print_comparison(
        "f(z) = z^2  (quadratic)",
        z -> z^2,
        exact_mean_quadratic(z_lo, z_hi),
        z_lo,
        z_hi,
    )
    print_comparison(
        "f(z) = z^3  (cubic)",
        z -> z^3,
        exact_mean_cubic(z_lo, z_hi),
        z_lo,
        z_hi,
    )

    # --- Cloud-like 1D proxies (not full (T,q)); no “linear ramp” story — sharp or U-shaped ---
    println("=== Cloud-like `z` profiles (usable 1D proxies only) ===")
    println("Columns: GL (τ-nodes), equal slab centers, **trapezoid quadrature** = uniform z-nodes +")
    println("  trapezoid weights (still `∑ w_k f(z_k)`; **not** in production ClimaAtmos).\n")

    # Dry middle, condensate near **both** cell edges: t=(z-z_lo)/H, f = t^Q + (1-t)^Q, Q ≫ 1.
    let Q = FT(14)
        L = z_hi - z_lo
        f_u = z -> begin
            t = (z - z_lo) / L
            return t^Q + (one(FT) - t)^Q
        end
        print_comparison_three(
            @sprintf("edge clouds, dry middle: f=t^%.0f+(1-t)^%.0f, t=(z-z_lo)/H", Q, Q),
            f_u,
            exact_mean_edge_cloud_u(Q),
            z_lo,
            z_hi,
        )
        print_n3_edge_cloud_audit(z_lo, z_hi, Q)
    end

    # Smooth, strictly increasing “condensate proxy” t = (z-z_lo)/H ∈ [0,1], f = t^P (no dead zone).
    let P = FT(12)
        exact_pow = one(FT) / (P + one(FT))
        f_pow = z -> begin
            L = z_hi - z_lo
            t = (z - z_lo) / L
            return t^P
        end
        print_comparison_three(
            @sprintf("smooth top-weight: f(z)=((z-z_lo)/H)^%.0f, exact mean = 1/%.0f", P, P + 1),
            f_pow,
            exact_pow,
            z_lo,
            z_hi,
        )
    end

    # Sharper but still resolvable: indicator on **top 40%** — thick enough that N=4,5 hit it.
    let frac_top = FT(0.40)
        L = z_hi - z_lo
        z_th = z_hi - frac_top * L
        f_step = z -> z >= z_th ? one(FT) : zero(FT)
        print_comparison_three(
            @sprintf("step cloud: 1(z ≥ z_th), top %.0f%% (z_th = %.6g)", 100 * frac_top, z_th),
            f_step,
            exact_mean_indicator_top(z_lo, z_hi, frac_top),
            z_lo,
            z_hi,
        )
    end

    println("How to read this:")
    println("  • **Edge clouds / dry middle** (`t^Q+(1-t)^Q`): mostly dry center, peaks at **both**")
    println("    faces — not a linear ramp. It is still **smooth**, so GL is often **best** here too;")
    println("    hitting `z_lo,z_hi` every time (trap stencil) does **not** automatically win.")
    println("  • **Step cloud (40%)**: discontinuous / slab-alignment — slab or trap can beat GL for some N.")
    println("  • **Smooth t^12 (top-only)**: GL typically dominates slab/trap.\n")

    println("GL: same node/weight tables as `gauss_legendre_neg1_1` in `sgs_quadrature.jl`,")
    println("    with z(τ) = (z_hi-z_lo)/2 * τ + (z_hi+z_lo)/2.")
    println("Slab centers: N equal-thickness slabs in z, one sample at each slab center, weight 1/N.")
    return nothing
end

main()
