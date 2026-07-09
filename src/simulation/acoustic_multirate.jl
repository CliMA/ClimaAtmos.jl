#####
##### Acoustic substepping timestepper. See `docs/src/acoustic_substepping.md`.
#####

import ClimaTimeSteppers as CTS

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
    AcousticSubstepTendency(G, G_lim, vertical, ν_d, damping_form)

Fast (sub-cycled) explicit tendency: the horizontal acoustic terms and the
kinetic-energy gradients (and, for `ExplicitVertical`, the vertical acoustic
terms), the divergence damping, and the frozen slow forcing `G`.

# Fields

  - `G`, `G_lim`: frozen slow forcing for the unlimited and limited tendencies,
    mutated in place by the outer driver between sub-cycles.
  - `vertical`: vertical-acoustic treatment, `ExplicitVertical` or `ImplicitVertical`.
  - `ν_d`: divergence-damping viscosity [m² s⁻¹]; `0` disables divergence damping.
  - `damping_form`: divergence-damping form, `HorizontalDivergenceDamping` or
    `FullDivergenceDamping`.

The call form `(f::AcousticSubstepTendency)(Yₜ, Yₜ_lim, Y, p, t)` writes the
sub-step tendency into `Yₜ`, `Yₜ_lim` in place.
"""
struct AcousticSubstepTendency{G, GL, V, FT, D}
    G::G
    G_lim::GL
    vertical::V
    ν_d::FT
    damping_form::D
end
function (f::AcousticSubstepTendency)(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    horizontal_acoustic_tendency!(Yₜ, Y, p, t)
    kinetic_energy_gradient_tendency!(Yₜ, Y, p, t)
    if f.vertical isa ExplicitVertical
        implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    end
    if f.ν_d > 0
        divergence_damping_tendency!(Yₜ, Y, p, f.damping_form, f.ν_d)
    end
    Yₜ .+= f.G
    Yₜ_lim .+= f.G_lim
    return nothing
end

"""
    AcousticMultirate(inner_alg, n_sub, vertical, outer_stages = 2, β_d = 0.0,
        damping_form = FullDivergenceDamping())

A multirate `TimeSteppingAlgorithm` that sub-cycles the acoustic modes. See
`docs/src/acoustic_substepping.md`.

# Fields

  - `inner_alg`: algorithm used for the acoustic sub-cycle (an `IMEXAlgorithm` for
    `ImplicitVertical`, an explicit algorithm for `ExplicitVertical`).
  - `n_sub`: sub-steps per outer step; `0` selects it from the acoustic CFL.
  - `vertical`: vertical-acoustic treatment.
  - `outer_stages`: order of the outer combination, `1` or `2`.
  - `β_d`: divergence-damping coefficient (`ν_d = β_d c_ref² fast_dt`).
  - `damping_form`: divergence-damping form, `HorizontalDivergenceDamping` or
    `FullDivergenceDamping`.
"""
struct AcousticMultirate{A, V, D} <: CTS.TimeSteppingAlgorithm
    inner_alg::A
    n_sub::Int
    vertical::V
    outer_stages::Int
    β_d::Float64
    damping_form::D
end
AcousticMultirate(inner_alg, n_sub, vertical, outer_stages = 2, β_d = 0.0) =
    AcousticMultirate(
        inner_alg,
        n_sub,
        vertical,
        outer_stages,
        β_d,
        FullDivergenceDamping(),
    )

# Sub-step size dt / n_sub. For `ITime`, refine to nanoseconds before dividing
# so the result stays exact and positive even when the period is coarse relative
# to `n_sub` (e.g. dt = 600 s with period 1 s and n_sub = 1000).
sub_timestep(dt, n_sub) = dt / n_sub
function sub_timestep(dt::ITime, n_sub::Integer)
    dt_ns = dt.counter * Dates.tons(dt.period)
    return ITime(div(dt_ns, n_sub); period = Dates.Nanosecond(1), epoch = dt.epoch)
end

# Express a time in nanoseconds. The inner integrator runs in the (nanosecond)
# period of its sub-step, so outer times must be refined to match before they
# are assigned to it.
refine_ns(t) = t
refine_ns(t::ITime) =
    ITime(t.counter * Dates.tons(t.period); period = Dates.Nanosecond(1), epoch = t.epoch)

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
    return max(1, ceil(Int, float(dt) / safe_dt))
end

struct AcousticMultirateCache{F, B, II, A}
    f::F                 # the outer ClimaODEFunction
    G::B                 # frozen slow forcing (mutated; aliased into the inner forcing)
    G_lim::B             # frozen slow limited forcing (mutated; aliased)
    G2::B                # second-stage slow forcing
    G2_lim::B            # second-stage slow limited forcing
    A_buf::B             # scratch for the acoustic tendency
    U0::B                # step-start state (restart point for the corrector)
    n_sub::Int           # resolved sub-step count (auto-selected when alg.n_sub = 0)
    inner_integ::II      # inner CTS integrator for the acoustic sub-cycle
    alg::A
end

function CTS.init_cache(prob, alg::AcousticMultirate; dt, kwargs...)
    (; u0, p) = prob
    f = prob.f
    FT = eltype(u0)
    Δx = FT(Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(u0.c))))
    c_ref = FT(reference_sound_speed(p))
    n_sub = alg.n_sub == 0 ? auto_n_sub(dt, Δx, c_ref) : alg.n_sub
    fast_dt = sub_timestep(dt, n_sub)
    G = zero(u0)
    G_lim = zero(u0)
    # Divergence-damping viscosity ν_d = β_d c_ref² fast_dt.
    ν_d = FT(alg.β_d) * c_ref^2 * FT(float(fast_dt))
    forcing =
        AcousticSubstepTendency(G, G_lim, alg.vertical, ν_d, alg.damping_form)
    T_imp! = alg.vertical isa ImplicitVertical ? f.T_imp! : nothing
    # The sub-cycle uses the lighter implicit cache (acoustic precomputed
    # quantities: velocities, pressure, enthalpy). The full (slow-physics) cache
    # is refreshed once per outer stage by `acoustic_slow_forcing!`.
    inner_f = CTS.ClimaODEFunction(;
        T_exp_T_lim! = forcing,
        T_imp!,
        cache! = f.cache_imp!,
        cache_imp! = f.cache_imp!,
        lim! = f.lim!,
        dss! = f.dss!,
        initialize_imp! = f.initialize_imp!,
    )
    inner_prob = CTS.ODEProblem(inner_f, copy(u0), prob.tspan, p)
    inner_integ = CTS.init(
        inner_prob,
        alg.inner_alg;
        dt = fast_dt,
        saveat = (),
        save_everystep = false,
    )
    return AcousticMultirateCache(
        f, G, G_lim, zero(u0), zero(u0), zero(u0), zero(u0),
        n_sub, inner_integ, alg,
    )
end

# G = T_exp(u) - sub-cycled acoustic terms(u), evaluated and left in `G`/`G_lim`.
# The sub-cycled terms are the horizontal acoustic tendency and the kinetic-energy
# gradients, so `G` holds exactly the tendency not re-evaluated inside the
# sub-cycle. Refreshes the cache `p` to be consistent with `u`.
function acoustic_slow_forcing!(G, G_lim, A_buf, f, u, p, t)
    f.cache!(u, p, t)
    G .= zero(eltype(G))
    G_lim .= zero(eltype(G_lim))
    f.T_exp_T_lim!(G, G_lim, u, p, t)
    A_buf .= zero(eltype(A_buf))
    horizontal_acoustic_tendency!(A_buf, u, p, t)
    kinetic_energy_gradient_tendency!(A_buf, u, p, t)
    G .-= A_buf
    return nothing
end

# Sub-cycle the inner acoustic problem from `u_start` over `[t, t + dt]` with the
# currently-set frozen forcing. The tstop at `t + dt` makes the final sub-step
# land exactly on the step end. Returns the sub-cycled state (`inner.u`).
function acoustic_subcycle!(inner, f, u_start, p, t, dt, n_sub)
    f.cache_imp!(u_start, p, t)
    t_end = refine_ns(t + dt)
    inner.u .= u_start
    inner.t = refine_ns(t)
    CTS.set_dt!(inner, sub_timestep(dt, n_sub))
    empty!(inner.tstops)
    CTS.add_tstop!(inner, t_end)
    while inner.t < t_end
        CTS.step!(inner)
    end
    return inner.u
end

function CTS.step_u!(integrator, cache::AcousticMultirateCache)
    (; f, G, G_lim, G2, G2_lim, A_buf, U0, n_sub, inner_integ, alg) = cache
    u = integrator.u
    p = integrator.p
    t = integrator.t
    dt = integrator.dt

    if alg.outer_stages == 1
        # First-order: freeze the slow forcing over the whole step.
        acoustic_slow_forcing!(G, G_lim, A_buf, f, u, p, t)
        acoustic_subcycle!(inner_integ, f, u, p, t, dt, n_sub)
        u .= inner_integ.u
    else
        # Second-order: average the slow forcing between step start and a
        # predicted step end (trapezoidal splitting).
        U0 .= u
        acoustic_slow_forcing!(G, G_lim, A_buf, f, U0, p, t)
        acoustic_subcycle!(inner_integ, f, U0, p, t, dt, n_sub)
        acoustic_slow_forcing!(G2, G2_lim, A_buf, f, inner_integ.u, p, t + dt)
        @. G = (G + G2) / 2
        @. G_lim = (G_lim + G2_lim) / 2
        acoustic_subcycle!(inner_integ, f, U0, p, t, dt, n_sub)
        u .= inner_integ.u
    end

    # The sub-cycle refreshes only the implicit (acoustic) cache; refresh the full
    # cache once so diagnostics and the next step see the slow precomputed
    # quantities at the updated state.
    f.cache!(u, p, t + dt)
    return u
end
