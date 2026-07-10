#####
##### Forward-backward acoustic inner sub-stepper. See `docs/src/acoustic_substepping.md`.
#####
##### `AcousticForwardBackward` is a ClimaTimeSteppers `TimeSteppingAlgorithm`
##### used as the inner algorithm of the acoustic-substepping `Multirate`, in
##### place of the IMEX-ARK inner sub-cycle. Each sub-step advances the horizontal
##### momentum forward, the scalars backward against the updated momentum, one
##### off-centered vertically-implicit Newton iteration, and an end-of-sub-step
##### divergence filter. The generic step-exchange skeleton (forcing freeze,
##### sub-cycle loop, outer combination) is unchanged and lives in
##### ClimaTimeSteppers.
#####

import ClimaTimeSteppers as CTS
import LinearAlgebra

"""
    AcousticForwardBackward(θ)

Inner sub-stepper for acoustic substepping: a forward-backward advance with an
off-centered vertically-implicit solve and an end-of-sub-step divergence filter.
Used as the inner algorithm of [`AcousticMultirate`](@ref) when
`acoustic_substep_scheme: forward_backward`.

`θ = (1 + β_off) / 2` is the off-centering weight of the vertically-implicit
solve; `θ = 1` is backward Euler.
"""
struct AcousticForwardBackward{FT, C} <: CTS.TimeSteppingAlgorithm
    θ::FT
    # Model `constrain_state!`, applied after each sub-step. Set by
    # `AcousticMultirate` in `init_cache`; defaults to a no-op.
    constrain_state!::C
end
AcousticForwardBackward(θ) = AcousticForwardBackward(θ, Returns(nothing))

"""
    frozen_forcing(f)

Return the frozen slow-forcing pair `(G, G_lim)` carried by the inner sub-cycle
function `f`. For the step-exchange host, `f` is a
`ClimaTimeSteppers.DualOffsetODEFunction`.
"""
frozen_forcing(f::CTS.DualOffsetODEFunction) = (f.G, f.G_lim)

"""
    AcousticFBCache

Cache for an [`AcousticForwardBackward`](@ref) sub-step: the inner implicit
operator `L_v` and its Jacobian, the aliased frozen forcing `(G, G_lim)`, the
reused DSS/constraint/cache hooks, the divergence-filter form and viscosity, the
off-centering weight, and the sub-step buffers.
"""
struct AcousticFBCache{L, J, B, DSS, CS, CI, D, FT, S, U}
    L_v::L
    jac::J
    G::B
    G_lim::B
    dss!::DSS
    constrain_state!::CS
    cache_imp!::CI
    damping_form::D
    ν_d::FT
    θ::FT
    A::B
    B_buf::B
    R::B
    ΔY::B
    ᶜδ::S
    ᶜu::U
end

function CTS.init_cache(prob, alg::AcousticForwardBackward; dt, kwargs...)
    f = prob.f
    u0 = prob.u0
    FT = eltype(u0)
    dof = f.T_exp_T_lim!
    G, G_lim = frozen_forcing(dof)
    fast_tend = dof.f
    return AcousticFBCache(
        f.T_imp!.f,
        f.T_imp!.jac_prototype,
        G,
        G_lim,
        f.dss!,
        alg.constrain_state!,
        f.cache_imp!,
        fast_tend.damping_form,
        FT(fast_tend.ν_d),
        FT(alg.θ),
        zero(u0),
        zero(u0),
        zero(u0),
        zero(u0),
        similar(u0.c, FT),
        similar(u0.c, C123{FT}),
    )
end

# End-of-sub-step divergence filter on uₕ (Klemp 2018 placement), evaluated at
# the post-solve state as a forward-Euler increment Δτ ν_d wgradₕ(δ), with δ the
# strong-form velocity divergence selected by `damping_form`.
function acoustic_fb_divergence_filter!(cache, Y, dtτ, ::HorizontalDivergenceDamping)
    ᶜδ = cache.ᶜδ
    @. ᶜδ = divₕ(Y.c.uₕ)
    @. Y.c.uₕ += dtτ * cache.ν_d * wgradₕ(ᶜδ)
    return nothing
end
function acoustic_fb_divergence_filter!(cache, Y, dtτ, ::FullDivergenceDamping)
    ᶜu = cache.ᶜu
    ᶜδ = cache.ᶜδ
    @. ᶜu = C123(Y.c.uₕ) + ᶜinterp(C123(Y.f.u₃))
    ᶠuₕ³ = compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)
    @. ᶜδ = divₕ(ᶜu)
    @. ᶜδ += ᶜdivᵥ(ᶠuₕ³ + CT3(Y.f.u₃))
    @. Y.c.uₕ += dtτ * cache.ν_d * wgradₕ(ᶜδ)
    return nothing
end

function CTS.step_u!(integrator, cache::AcousticFBCache)
    (; L_v, jac, G, dss!, constrain_state!, cache_imp!, θ) = cache
    A, B, R, ΔY = cache.A, cache.B_buf, cache.R, cache.ΔY
    Y = integrator.u
    p = integrator.p
    t = integrator.t
    dtτ = float(integrator.dt)
    dtγ = θ * dtτ

    # S1. Vertical implicit tendency at the old level (skipped at θ = 1).
    A .= zero(eltype(A))
    θ < 1 && L_v(A, Y, p, t)

    # S2. Horizontal momentum forward, with the frozen slow momentum forcing.
    B .= zero(eltype(B))
    horizontal_acoustic_momentum_tendency!(B, Y, p, t)
    kinetic_energy_gradient_uₕ_tendency!(B, Y, p, t)
    @. Y.c.uₕ += dtτ * (B.c.uₕ + G.c.uₕ)

    # S3. Assemble uₕ and refresh the sub-cycle cache.
    dss!(Y, p, t)
    constrain_state!(Y, p, t)
    cache_imp!(Y, p, t)

    # S4. Scalars backward against the updated momentum, plus the vertical-K
    # predictor, the frozen slow forcing, and the implicit predictor `A`. The
    # increment is assembled for every block; `uₕ` is then overridden to `A.c.uₕ`
    # (the implicit-diffusion predictor, zero for the restricted inner) so the
    # frozen momentum forcing reaches `uₕ` only once, in S2.
    B .= zero(eltype(B))
    horizontal_acoustic_scalar_tendency!(B, Y, p, t)
    kinetic_energy_gradient_u₃_tendency!(B, Y, p, t)
    @. R = B + G + A
    @. R.c.uₕ = A.c.uₕ
    @. Y += dtτ * R

    # S5. Off-centered vertical implicit solve, one Newton iteration.
    cache_imp!(Y, p, t)
    update_jacobian!(jac, Y, p, dtγ, t)
    B .= zero(eltype(B))
    L_v(B, Y, p, t)
    @. R = dtγ * (A - B)
    LinearAlgebra.ldiv!(ΔY, jac, R)
    @. Y += ΔY

    # S6. Time-adjusted divergence filter on uₕ, at the post-solve state.
    cache.ν_d > 0 && acoustic_fb_divergence_filter!(cache, Y, dtτ, cache.damping_form)

    # S7. Final assembly and cache for the next sub-step.
    dss!(Y, p, t)
    constrain_state!(Y, p, t)
    cache_imp!(Y, p, t)

    return integrator.u
end
