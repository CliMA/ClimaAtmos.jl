#####
##### Acoustic substepping timestepper. See `docs/src/acoustic_substepping.md`.
#####
##### The generic step-exchange multirate method (the frozen-forcing pass, the
##### inner IMEX sub-cycle, the outer combination, and the outer implicit
##### complement sequencing) is in ClimaTimeSteppers, as the
##### `LieSplitOuter`/`TrapezoidalSplitOuter` outer family. This file supplies
##### the acoustic fast tendency, the forcing-freeze operation, the sub-step
##### count, the restricted implicit operator, the outer complement, and the
##### config wiring.
#####

import ClimaTimeSteppers as CTS
import ClimaUtilities.TimeManager: ITime, seconds
import Dates

"""
    ImplicitVertical()
    ExplicitVertical()

Treatment of the vertical acoustic terms within an acoustic sub-step.
`ImplicitVertical` keeps the vertically-implicit (HEVI) acoustic solve;
`ExplicitVertical` advances the vertical acoustic terms explicitly.
"""
struct ImplicitVertical end
struct ExplicitVertical end

"""
    HorizontalDivergenceDamping()
    FullDivergenceDamping()

Divergence used by the acoustic-substepping divergence damping.
`HorizontalDivergenceDamping` damps the horizontal divergence of the horizontal
velocity `divₕ(uₕ)`; `FullDivergenceDamping` damps the three-dimensional
divergence `δ = divₕ(ᶜu) + ᶜdivᵥ(ᶠu³)`.
"""
struct HorizontalDivergenceDamping end
struct FullDivergenceDamping end

"""
    divergence_damping_tendency!(Yₜ, Y, p, form, ν_d)

Add the acoustic-substepping divergence-damping tendency `ν_d wgradₕ(δ)` to `uₕ`,
with the damped divergence `δ` selected by `form`.
"""
function divergence_damping_tendency!(
    Yₜ,
    Y,
    p,
    ::HorizontalDivergenceDamping,
    ν_d,
)
    @. Yₜ.c.uₕ += ν_d * wgradₕ(divₕ(Y.c.uₕ))
    return nothing
end
function divergence_damping_tendency!(Yₜ, Y, p, ::FullDivergenceDamping, ν_d)
    (; ᶜu, ᶠu³) = p.precomputed
    ᶜδ = p.scratch.ᶜtemp_scalar_2
    # Accumulate the horizontal (spectral) and vertical (column-stencil)
    # divergences separately.
    @. ᶜδ = divₕ(ᶜu)
    @. ᶜδ += ᶜdivᵥ(ᶠu³)
    @. Yₜ.c.uₕ += ν_d * wgradₕ(ᶜδ)
    return nothing
end

"""
    AcousticSubstepTendency(vertical, ν_d, damping_form)

Fast (sub-cycled) explicit tendency: the horizontal acoustic terms, the grid-mean
momentum advection (kinetic-energy gradients and rotational momentum flux) (and,
for `ExplicitVertical`, the vertical acoustic terms) and the divergence damping.

# Fields

  - `vertical`: vertical-acoustic treatment, `ExplicitVertical` or `ImplicitVertical`.
  - `ν_d`: divergence-damping viscosity [m² s⁻¹]; `0` disables divergence damping.
  - `damping_form`: divergence-damping form, `HorizontalDivergenceDamping` or
    `FullDivergenceDamping`.

The call form `(f::AcousticSubstepTendency)(Yₜ, Yₜ_lim, Y, p, t)` writes the
sub-step tendency into `Yₜ`, `Yₜ_lim` in place. The frozen slow forcing is added
by the inner integrator's forcing wrapper.
"""
struct AcousticSubstepTendency{V, FT, D}
    vertical::V
    ν_d::FT
    damping_form::D
end
function (f::AcousticSubstepTendency)(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    horizontal_acoustic_tendency!(Yₜ, Y, p, t)
    kinetic_energy_gradient_tendency!(Yₜ, Y, p, t)
    rotational_momentum_flux_tendency!(Yₜ, Y, p, t)
    if f.vertical isa ExplicitVertical
        implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    end
    if f.ν_d > 0
        divergence_damping_tendency!(Yₜ, Y, p, f.damping_form, f.ν_d)
    end
    return nothing
end

"""
    GridMeanAcousticTendency()

Inner implicit operator of the inner/outer implicit split of acoustic substepping.

Zeroes `Yₜ`, then accumulates the vertical grid-mean acoustic subset with
`grid_mean_acoustic_tendency!`, so fields outside that subset have a zero
tendency, matching `implicit_tendency!`.
"""
struct GridMeanAcousticTendency end
function (::GridMeanAcousticTendency)(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    grid_mean_acoustic_tendency!(Yₜ, Y, p, t)
    return nothing
end

"""
    OuterImplicitTendency(T_imp_full, buf)

Outer implicit tendency of the inner/outer implicit split of acoustic
substepping: the full implicit tendency minus the inner (vertical grid-mean
acoustic) subset, `T_imp_full − grid_mean_acoustic_tendency!`.

# Fields

  - `T_imp_full`: the full implicit tendency function (`implicit_tendency!`).
  - `buf`: scratch `FieldVector` holding the inner subset.
"""
struct OuterImplicitTendency{T, B}
    T_imp_full::T
    buf::B
end
function (o::OuterImplicitTendency)(Yₜ, Y, p, t)
    o.T_imp_full(Yₜ, Y, p, t)
    o.buf .= zero(eltype(o.buf))
    grid_mean_acoustic_tendency!(o.buf, Y, p, t)
    Yₜ .-= o.buf
    return nothing
end

"""
    AcousticMultirate(inner_alg, n_sub, vertical, outer_stages = 2, β_d = 0.0,
        implicit_split = false, damping_form = FullDivergenceDamping())

A multirate `TimeSteppingAlgorithm` that sub-cycles the acoustic modes. See
`docs/src/acoustic_substepping.md`.

# Fields

  - `inner_alg`: algorithm used for the acoustic sub-cycle (an `IMEXAlgorithm` for
    `ImplicitVertical`, an explicit algorithm for `ExplicitVertical`).
  - `n_sub`: sub-steps per outer step; `0` selects it from the acoustic CFL.
  - `vertical`: vertical-acoustic treatment.
  - `outer_stages`: order of the outer combination, `1` or `2`.
  - `β_d`: divergence-damping coefficient (`ν_d = β_d c_ref² fast_dt`).
  - `implicit_split`: when `true` (and `vertical isa ImplicitVertical`), use the
    inner/outer implicit split: the inner implicit solve is restricted to the
    vertical grid-mean acoustic block (`grid_mean_acoustic_tendency!`) and the
    remaining implicit terms are solved by an outer implicit half-step per outer
    step. When `false`, the inner solve uses the full `implicit_tendency!`.
  - `damping_form`: divergence-damping form, `HorizontalDivergenceDamping` or
    `FullDivergenceDamping`.
"""
struct AcousticMultirate{A, V, D} <: CTS.TimeSteppingAlgorithm
    inner_alg::A
    n_sub::Int
    vertical::V
    outer_stages::Int
    β_d::Float64
    implicit_split::Bool
    damping_form::D
end
AcousticMultirate(
    inner_alg,
    n_sub,
    vertical,
    outer_stages = 2,
    β_d = 0.0,
    implicit_split = false,
) = AcousticMultirate(
    inner_alg,
    n_sub,
    vertical,
    outer_stages,
    β_d,
    implicit_split,
    FullDivergenceDamping(),
)

"""
    AcousticMultirateCache(f, cts_cache)

Cache for an [`AcousticMultirate`](@ref) step. `f` is the full model
`ClimaODEFunction`; `cts_cache` is the ClimaTimeSteppers step-exchange
`Multirate` cache that drives the step.
"""
struct AcousticMultirateCache{F, C}
    f::F
    cts_cache::C
end

CTS.step_u!(integrator, cache::AcousticMultirateCache) =
    CTS.step_u!(integrator, cache.cts_cache)

# Reference sound speed √(γ R_d T_ref). T_ref is an upper-bound reference
# temperature [K] so that c ≤ c_ref and the auto sub-step count is conservative.
function reference_sound_speed(p)
    R_d = CAP.R_d(p.params)
    cp_d = CAP.cp_d(p.params)
    cv_d = cp_d - R_d
    T_ref = oftype(R_d, 300)
    return sqrt(cp_d / cv_d * R_d * T_ref)
end

# Auto sub-step count from the horizontal acoustic CFL:
# n_sub = ceil(dt / (cfl_safety · Δx_node / c_ref)).
function auto_n_sub(dt, Δx, c_ref)
    cfl_safety = oftype(c_ref, 0.5)
    safe_dt = cfl_safety * Δx / c_ref
    return max(1, ceil(Int, seconds(dt) / safe_dt))
end

# Smallest sub-step count ≥ `n_min` for which `dt / n_sub` is exact. For an
# `ITime`, a divisor of the counter at `dt`'s own period is preferred so
# `fast_dt` keeps that period; when `n_min` exceeds the counter no such divisor
# exists and the count is raised until the division is exact at a finer period.
exact_n_sub(dt, n_min) = n_min
function exact_n_sub(dt::ITime, n_min)
    counter = dt.counter
    for n in n_min:counter
        iszero(rem(counter, n)) && return n
    end
    dt_ns = counter * Dates.tons(dt.period)
    n = n_min
    while !iszero(rem(dt_ns, n))
        n += 1
    end
    return n
end

# G = T_exp(u) - sub-cycled acoustic terms(u), evaluated and left in `G`/`G_lim`.
# The sub-cycled terms are the horizontal acoustic tendency and the grid-mean
# momentum advection (kinetic-energy gradients and rotational momentum flux), so
# `G` holds exactly the tendency not re-evaluated inside the sub-cycle. Refreshes
# the cache `p` to be consistent with `u`.
function acoustic_slow_forcing!(G, G_lim, A_buf, f, u, p, t)
    f.cache!(u, p, t)
    G .= zero(eltype(G))
    G_lim .= zero(eltype(G_lim))
    f.T_exp_T_lim!(G, G_lim, u, p, t)
    A_buf .= zero(eltype(A_buf))
    horizontal_acoustic_tendency!(A_buf, u, p, t)
    kinetic_energy_gradient_tendency!(A_buf, u, p, t)
    rotational_momentum_flux_tendency!(A_buf, u, p, t)
    G .-= A_buf
    return nothing
end

# Advance the outer implicit complement by `halfdt` from the current `u`, in
# place. The outer integrator's time and step fields carry the period chosen in
# `init_cache`; `t` and `halfdt` are promoted to it before assignment, since a
# coarser incoming period does not convert to a finer field period on its own.
function outer_half!(outer, u, p, t, halfdt)
    outer.u .= u
    outer.t = first(promote(t, outer.t))
    CTS.set_dt!(outer, first(promote(halfdt, outer.dt)))
    empty!(outer.tstops)
    CTS.step!(outer)
    u .= outer.u
    return nothing
end

function CTS.init_cache(prob, alg::AcousticMultirate; dt, kwargs...)
    (; u0, p) = prob
    f = prob.f
    FT = eltype(u0)
    Δx = FT(Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(u0.c))))
    c_ref = FT(reference_sound_speed(p))
    requested_n_sub = alg.n_sub == 0 ? auto_n_sub(dt, Δx, c_ref) : alg.n_sub
    n_sub = exact_n_sub(dt, requested_n_sub)
    fast_dt = dt / n_sub
    # Divergence-damping viscosity ν_d = β_d c_ref² fast_dt.
    ν_d = FT(alg.β_d) * c_ref^2 * FT(seconds(fast_dt))
    fast! = AcousticSubstepTendency(alg.vertical, ν_d, alg.damping_form)
    A_buf = zero(u0)
    freeze!(G, G_lim, U, p, t) = acoustic_slow_forcing!(G, G_lim, A_buf, f, U, p, t)
    implicit_split = alg.implicit_split && alg.vertical isa ImplicitVertical
    # In the inner/outer implicit split the inner implicit solve is restricted to
    # the vertical grid-mean acoustic block, so it factorizes the small
    # acoustic-only `AcousticJacobian` rather than the full implicit Jacobian;
    # otherwise the inner solve uses the full `implicit_tendency!` from `f.T_imp!`.
    inner_T_imp! =
        if implicit_split
            CTS.ODEFunction(
                GridMeanAcousticTendency();
                jac_prototype = Jacobian(AcousticJacobian(), u0, p.atmos),
                Wfact = update_jacobian!,
            )
        elseif alg.vertical isa ImplicitVertical
            f.T_imp!
        else
            nothing
        end
    # The fast sub-cycle is a full `ClimaODEFunction`: the acoustic fast tendency
    # (with the frozen forcing added by the outer step), the restricted or full
    # implicit operator, and the reused implicit cache, limiter, DSS, and state
    # constraint. `constrain_state!` is applied once per outer step, to the
    # combined end-of-step state; constraint application inside the sub-cycle
    # is not supported.
    f_fast = CTS.ClimaODEFunction(;
        T_exp_T_lim! = fast!,
        T_imp! = inner_T_imp!,
        cache! = f.cache!,
        cache_imp! = f.cache_imp!,
        lim! = f.lim!,
        dss! = f.dss!,
        constrain_state! = f.constrain_state!,
        update_constrain_state = f.update_constrain_state,
        initialize_imp! = f.initialize_imp!,
    )
    # The outer implicit half-step solves the full implicit tendency minus the
    # inner subset, once per outer step, pairing that residual with the matching
    # `AcousticComplementJacobian` (the full Jacobian blocks with the vertical
    # grid-mean acoustic derivatives zeroed). Its inner integrator and the
    # sequencing are owned here; the outer method only calls the complement.
    outer_complement =
        if implicit_split
            outer_f = CTS.ClimaODEFunction(;
                T_imp! = CTS.ODEFunction(
                    OuterImplicitTendency(f.T_imp!.f, zero(u0));
                    jac_prototype = Jacobian(
                        AcousticComplementJacobian(f.T_imp!.jac_prototype.alg),
                        u0,
                        p.atmos,
                    ),
                    Wfact = update_jacobian!,
                ),
                cache! = f.cache_imp!,
                cache_imp! = f.cache_imp!,
                lim! = f.lim!,
                dss! = f.dss!,
                initialize_imp! = f.initialize_imp!,
            )
            # The complement advances by the outer half-step: `dt` for a
            # first-order outer combination, `dt / 2` for second order. Build the
            # integrator at a period fine enough to represent that half-step.
            outer_halfdt = alg.outer_stages == 1 ? dt : dt / 2
            outer_dt = first(promote(fast_dt, outer_halfdt))
            outer_integ = CTS.init(
                CTS.ODEProblem(outer_f, copy(u0), prob.tspan, p),
                alg.inner_alg;
                dt = outer_dt,
                saveat = (),
                save_everystep = false,
            )
            (u, p, t, halfdt) -> outer_half!(outer_integ, u, p, t, halfdt)
        else
            nothing
        end
    outer =
        alg.outer_stages == 1 ? CTS.LieSplitOuter(outer_complement) :
        CTS.TrapezoidalSplitOuter(outer_complement)
    split_prob = CTS.SplitODEProblem(f_fast, freeze!, u0, prob.tspan, p)
    cts_cache = CTS.init_cache(
        split_prob,
        CTS.Multirate(alg.inner_alg, outer);
        dt,
        fast_dt,
    )
    return AcousticMultirateCache(f, cts_cache)
end
