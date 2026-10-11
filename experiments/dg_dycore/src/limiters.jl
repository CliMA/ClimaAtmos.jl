#=
Per-RK-stage bound-preserving positivity limiter (Zhang–Shu 2010), applied
through ClimaTimeSteppers' `lim!` hook. The horizontal DG ρq_tot transport is
not sign-preserving at GLL nodes (Gibbs undershoot ⇒ q_tot < 0, poisoning the
moist thermodynamics), and the entropy-conservative Waruszewski flux needs
ρ > 0 and p > 0 for its ln_mean(ρ)/√(γp/ρ). `PositivityLimiter` scales the whole
conserved node vector toward the WJ-weighted element mean — convex,
element-mean-preserving projections (NOT pointwise clamps): ρ and ρq_tot each
toward their own mean to meet ρ ≥ ρ_min and ρq_tot ≥ 0, then the whole vector
by one θ ∈ [0,1] to meet p ≥ p_min. Moist only.
=#

# Pressure of a scaled conserved node `U`, matching the tendency EOS exactly:
# horizontal KE from the (scaled) Cartesian momenta; the vertical KE + Φ carried
# unscaled in `off`; non-throwing moist_p_dyn. The limiter's constraint `g`.
struct ZSPressure{TP}
    tp::TP
end
@inline function (g::ZSPressure)(U, off)
    K_h = (U.ρu1^2 + U.ρu2^2 + U.ρu3^2) / (2 * U.ρ^2)
    return moist_p_dyn(g.tp, U.ρ, U.ρe / U.ρ - K_h - off, U.ρq_tot / U.ρ).p
end

# `lim!` hook (ClimaTimeSteppers calls f.lim!(U, p, t, u_ref) each stage i≠1).
# u_ref is unused: PositivityLimiter takes its bounds from U's own element mean.
function lim_fddg!(U, m::DGModel, t, u_ref)
    m.prob.moisture == :dry && return nothing
    Yc = U.c
    (; Ic) = m.ops
    ᶜΦ = m.fields.ᶜΦ
    # off = w_c²/2 + Φ (w_c = ρw→centers / ρ), matching e_int = ρe/ρ − K − Φ.
    w_c = @. Ic(Geometry.WVector(U.f.ρw)).components.data.:1 / Yc.ρ
    off = @. w_c^2 / 2 + ᶜΦ
    (; ρ, ρe, ρu1, ρu2, ρu3, ρq_tot) = Yc
    Limiters.apply_limiter!(
        (; ρ, ρe, ρu1, ρu2, ρu3, ρq_tot),
        off,
        m.fields.positivity_limiter,
    )
    return nothing
end
