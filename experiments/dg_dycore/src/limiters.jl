#=
Per-RK-stage bound-preserving positivity limiter (Zhang–Shu 2010), applied
through ClimaTimeSteppers' `lim!` hook. The horizontal DG ρq_tot transport is
not sign-preserving at GLL nodes (Gibbs undershoot ⇒ q_tot < 0, poisoning the
moist thermodynamics), and the entropy-conservative Waruszewski flux needs
ρ > 0 and p > 0 for its ln_mean(ρ)/√(γp/ρ). `PositivityLimiter` scales the whole
conserved node vector toward the WJ-weighted element mean by one θ ∈ [0,1] — a
convex, element-mean-preserving projection (NOT a pointwise clamp) — enforcing
ρ ≥ ρ_min, ρq_tot ≥ 0 and p ≥ p_min simultaneously. Moist only.
=#

# Pressure of a scaled conserved node, matching the tendency EOS exactly:
# horizontal KE from the (scaled) Cartesian momenta; the vertical KE + Φ carried
# unscaled in `off`; non-throwing moist_p_dyn.
@inline function zs_pressure(tp, ρ, ρe, ρu1, ρu2, ρu3, ρq, off)
    K_h = (ρu1^2 + ρu2^2 + ρu3^2) / (2 * ρ^2)
    return moist_p_dyn(tp, ρ, ρe / ρ - K_h - off, ρq / ρ).p
end

# `lim!` hook (ClimaTimeSteppers calls f.lim!(U, p, t, u_ref) each stage i≠1).
# u_ref is unused: PositivityLimiter takes its bounds from U's own element mean.
function lim_fddg!(U, m::DGModel, t, u_ref)
    m.prob.moisture == :dry && return nothing
    Yc = U.c
    tp = m.fields.thermo_params
    (; Ic) = m.ops
    ᶜΦ = m.fields.ᶜΦ
    # off = w_c²/2 + Φ (w_c = ρw→centers / ρ), matching e_int = ρe/ρ − K − Φ.
    w_c = @. Ic(Geometry.WVector(U.f.ρw)).components.data.:1 / Yc.ρ
    off = @. w_c^2 / 2 + ᶜΦ
    pfn = (ρ, ρe, ρu1, ρu2, ρu3, ρq, o) ->
        zs_pressure(tp, ρ, ρe, ρu1, ρu2, ρu3, ρq, o)
    Limiters.apply_positivity_limiter!(
        m.fields.positivity_limiter,
        pfn,
        (Yc.ρ, Yc.ρe, Yc.ρu1, Yc.ρu2, Yc.ρu3, Yc.ρq_tot),
        off,
    )
    return nothing
end
