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
    FastKineticEnergy()
    FrozenKineticEnergy()

Treatment of the kinetic-energy-gradient momentum terms within an acoustic
sub-step. `FastKineticEnergy` re-evaluates them every sub-step (removing the
dominant split-explicit resonance source); `FrozenKineticEnergy` holds them in
the frozen slow forcing.
"""
struct FastKineticEnergy end
struct FrozenKineticEnergy end

"""
    HorizontalDivergenceDamping()
    FullDivergenceDamping()
    PerturbationDivergenceDamping()

Divergence used by the acoustic-substepping divergence damping.
`HorizontalDivergenceDamping` damps the horizontal divergence of the horizontal
velocity `divₕ(uₕ)`; `FullDivergenceDamping` damps the three-dimensional
divergence `δ = divₕ(ᶜu) + ᶜdivᵥ(ᶠu³)`; `PerturbationDivergenceDamping` damps its
deviation `δ − δ⁰` from the sub-cycle-start value.
"""
struct HorizontalDivergenceDamping end
struct FullDivergenceDamping end
struct PerturbationDivergenceDamping end

"""
    divergence_damping_tendency!(Yₜ, Y, p, form, ν_d, ᶜδ₀)

Add the acoustic-substepping divergence-damping tendency `ν_d wgradₕ(δ)` to `uₕ`,
with the damped divergence `δ` selected by `form`. `ᶜδ₀` is the sub-cycle-start
divergence used only by `PerturbationDivergenceDamping`.
"""
function divergence_damping_tendency!(
    Yₜ,
    Y,
    p,
    ::HorizontalDivergenceDamping,
    ν_d,
    ᶜδ₀,
)
    @. Yₜ.c.uₕ += ν_d * wgradₕ(divₕ(Y.c.uₕ))
    return nothing
end
function divergence_damping_tendency!(Yₜ, Y, p, ::FullDivergenceDamping, ν_d, ᶜδ₀)
    (; ᶜu, ᶠu³) = p.precomputed
    ᶜδ = p.scratch.ᶜtemp_scalar_2
    # The horizontal spectral and vertical column-stencil divergences do not fuse
    # into one broadcast, so accumulate them separately.
    @. ᶜδ = divₕ(ᶜu)
    @. ᶜδ += ᶜdivᵥ(ᶠu³)
    @. Yₜ.c.uₕ += ν_d * wgradₕ(ᶜδ)
    return nothing
end
function divergence_damping_tendency!(
    Yₜ,
    Y,
    p,
    ::PerturbationDivergenceDamping,
    ν_d,
    ᶜδ₀,
)
    (; ᶜu, ᶠu³) = p.precomputed
    ᶜδ = p.scratch.ᶜtemp_scalar_2
    @. ᶜδ = divₕ(ᶜu)
    @. ᶜδ += ᶜdivᵥ(ᶠu³) - ᶜδ₀
    @. Yₜ.c.uₕ += ν_d * wgradₕ(ᶜδ)
    return nothing
end

"""
    AcousticSubstepTendency(G, G_lim, vertical, ν_d, kinetic_energy, damping_form, ᶜδ₀)

Fast (sub-cycled) explicit tendency: the horizontal acoustic terms (and, for
`ExplicitVertical`, the vertical acoustic terms), the divergence damping, and the
frozen slow forcing `G`.

# Fields

  - `G`, `G_lim`: frozen slow forcing for the unlimited and limited tendencies,
    mutated in place by the outer driver between sub-cycles.
  - `vertical`: vertical-acoustic treatment, `ExplicitVertical` or `ImplicitVertical`.
  - `ν_d`: divergence-damping viscosity [m² s⁻¹]; `0` disables divergence damping.
  - `kinetic_energy`: `FastKineticEnergy` or `FrozenKineticEnergy`.
  - `damping_form`: divergence-damping form, one of `HorizontalDivergenceDamping`,
    `FullDivergenceDamping`, `PerturbationDivergenceDamping`.
  - `ᶜδ₀`: sub-cycle-start divergence for `PerturbationDivergenceDamping`, else `nothing`.

The call form `(f::AcousticSubstepTendency)(Yₜ, Yₜ_lim, Y, p, t)` writes the
sub-step tendency into `Yₜ`, `Yₜ_lim` in place.
"""
struct AcousticSubstepTendency{G, GL, V, FT, K, D, C}
    G::G
    G_lim::GL
    vertical::V
    ν_d::FT
    kinetic_energy::K
    damping_form::D
    ᶜδ₀::C
end
function (f::AcousticSubstepTendency)(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    horizontal_acoustic_tendency!(Yₜ, Y, p, t)
    if f.kinetic_energy isa FastKineticEnergy
        kinetic_energy_gradient_tendency!(Yₜ, Y, p, t)
    end
    if f.vertical isa ExplicitVertical
        implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    end
    if f.ν_d > 0
        divergence_damping_tendency!(Yₜ, Y, p, f.damping_form, f.ν_d, f.ᶜδ₀)
    end
    Yₜ .+= f.G
    Yₜ_lim .+= f.G_lim
    return nothing
end

"""
    GridMeanAcousticTendency()

Inner implicit operator of the inner/outer implicit split of acoustic substepping.

`grid_mean_acoustic_tendency!` accumulates into `Yₜ` (mirroring the subset convention of
`implicit_vertical_advection_tendency!`). As a standalone implicit tendency it must first
zero `Yₜ`, so fields outside the grid-mean acoustic block get a zero tendency and the
implicit-solve residual is not polluted by a stale buffer, matching `implicit_tendency!`.
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
        implicit_split = false, kinetic_energy = FastKineticEnergy(),
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
  - `implicit_split`: when `true` (and `vertical isa ImplicitVertical`), use the
    inner/outer implicit split: the inner implicit solve is restricted to the
    vertical grid-mean acoustic block (`grid_mean_acoustic_tendency!`) and the
    remaining implicit terms are solved by an outer implicit half-step per outer
    step. When `false`, the inner solve uses the full `implicit_tendency!`.
  - `kinetic_energy`: `FastKineticEnergy` (re-evaluated in the sub-cycle) or
    `FrozenKineticEnergy` (held in the slow forcing).
  - `damping_form`: divergence-damping form, one of `HorizontalDivergenceDamping`,
    `FullDivergenceDamping`, `PerturbationDivergenceDamping`.
"""
struct AcousticMultirate{A, V, K, D} <: CTS.TimeSteppingAlgorithm
    inner_alg::A
    n_sub::Int
    vertical::V
    outer_stages::Int
    β_d::Float64
    implicit_split::Bool
    kinetic_energy::K
    damping_form::D
end
AcousticMultirate(
    inner_alg,
    n_sub,
    vertical,
    outer_stages = 2,
    β_d = 0.0,
    implicit_split = false,
) =
    AcousticMultirate(
        inner_alg,
        n_sub,
        vertical,
        outer_stages,
        β_d,
        implicit_split,
        FastKineticEnergy(),
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

# Mean horizontal node spacing of a space [m].
horizontal_node_spacing(space) =
    Spaces.node_horizontal_length_scale(Spaces.horizontal_space(space))

# Auto sub-step count from the horizontal acoustic CFL:
# n_sub = ceil(dt / (cfl_safety · Δx_node / c_ref)).
function auto_n_sub(dt, Δx, c_ref)
    cfl_safety = oftype(c_ref, 0.5)
    safe_dt = cfl_safety * Δx / c_ref
    return max(1, ceil(Int, float(dt) / safe_dt))
end

"""
    acoustic_hyperdiffusion_scale(scaling, hyperdiff, Δx_node, dt_outer)

Compute the factor applied to the vorticity hyperdiffusion coefficient under
acoustic substepping. See `docs/src/acoustic_substepping.md`.

For `scaling == "auto"` the factor is `min(1, Δt_hd_limit / (2 dt_outer))` with
`Δt_hd_limit = 2 Δx_node / (F ν₄_vorticity_coeff β⁴)`,
`F = max(divergence_damping_factor, 1 / prandtl_number)`, and `β = 4`. A real
`scaling` is returned unchanged, so `1` reproduces the unscaled coefficient.
"""
acoustic_hyperdiffusion_scale(scaling::Real, hyperdiff, Δx_node, dt_outer) =
    oftype(dt_outer, scaling)

function acoustic_hyperdiffusion_scale(
    scaling::AbstractString,
    hyperdiff,
    Δx_node,
    dt_outer,
)
    scaling == "auto" ||
        error("Unknown acoustic_substep_hyperdiffusion_scaling: $scaling")
    # Maximum-wavenumber prefactor for degree-3 spectral elements; see
    # docs/src/acoustic_substepping.md for the calibration.
    β = oftype(dt_outer, 4)
    safety = oftype(dt_outer, 2)
    F = max(hyperdiff.divergence_damping_factor, inv(hyperdiff.prandtl_number))
    Δt_hd_limit = 2 * Δx_node / (F * hyperdiff.ν₄_vorticity_coeff * β^4)
    return min(one(dt_outer), Δt_hd_limit / (safety * dt_outer))
end

"""
    scale_hyperdiffusion_under_acoustic_substepping(model, grid, parsed_args)

Return `model` with its vorticity hyperdiffusion coefficient scaled for acoustic
substepping, folding the factor from [`acoustic_hyperdiffusion_scale`](@ref) into
the stored coefficient so the hyperdiffusion tendency is unchanged at runtime.

The model is returned unchanged when substepping is disabled
(`acoustic_substeps = 0`), when it carries no hyperdiffusion, or when the factor
is `1`.
"""
function scale_hyperdiffusion_under_acoustic_substepping(model, grid, parsed_args)
    string(parsed_args["acoustic_substeps"]) == "0" && return model
    hyperdiff = model.numerics.hyperdiff
    hyperdiff isa Hyperdiffusion || return model
    FT = typeof(hyperdiff.ν₄_vorticity_coeff)
    Δx_node = FT(horizontal_node_spacing(get_spaces(grid).center_space))
    dt_outer = FT(time_to_seconds(parsed_args["dt"]))
    s = acoustic_hyperdiffusion_scale(
        parsed_args["acoustic_substep_hyperdiffusion_scaling"],
        hyperdiff,
        Δx_node,
        dt_outer,
    )
    s == one(FT) && return model
    scaled = Hyperdiffusion{FT}(;
        ν₄_vorticity_coeff = s * hyperdiff.ν₄_vorticity_coeff,
        divergence_damping_factor = hyperdiff.divergence_damping_factor,
        prandtl_number = hyperdiff.prandtl_number,
    )
    return replace_field(
        model,
        :numerics,
        replace_field(model.numerics, :hyperdiff, scaled),
    )
end

# Reconstruct an immutable struct with a single field replaced.
function replace_field(obj::T, name::Symbol, value) where {T}
    args = ntuple(fieldcount(T)) do i
        fieldname(T, i) === name ? value : getfield(obj, i)
    end
    return T.name.wrapper(args...)
end

struct AcousticMultirateCache{F, B, C, II, OI, A}
    f::F                 # the outer ClimaODEFunction
    G::B                 # frozen slow forcing (mutated; aliased into the inner forcing)
    G_lim::B             # frozen slow limited forcing (mutated; aliased)
    G2::B                # second-stage slow forcing
    G2_lim::B            # second-stage slow limited forcing
    A_buf::B             # scratch for the acoustic tendency
    U0::B                # step-start state (restart point for the corrector)
    ᶜδ₀::C               # sub-cycle-start divergence (aliased into the inner forcing; nothing unless the perturbation damping form)
    n_sub::Int           # resolved sub-step count (auto-selected when alg.n_sub = 0)
    inner_integ::II      # inner CTS integrator for the acoustic sub-cycle
    outer_integ::OI      # outer CTS integrator for the inner/outer implicit split (nothing unless implicit_split)
    alg::A
end

function CTS.init_cache(prob, alg::AcousticMultirate; dt, kwargs...)
    (; u0, p) = prob
    f = prob.f
    FT = eltype(u0)
    Δx = FT(horizontal_node_spacing(axes(u0.c)))
    c_ref = FT(reference_sound_speed(p))
    n_sub = alg.n_sub == 0 ? auto_n_sub(dt, Δx, c_ref) : alg.n_sub
    fast_dt = sub_timestep(dt, n_sub)
    G = zero(u0)
    G_lim = zero(u0)
    # Divergence-damping viscosity ν_d = β_d c_ref² fast_dt.
    ν_d = FT(alg.β_d) * c_ref^2 * FT(float(fast_dt))
    # Sub-cycle-start divergence for the perturbation damping form; unused (and
    # left `nothing`) for the other forms and when damping is disabled.
    ᶜδ₀ =
        alg.damping_form isa PerturbationDivergenceDamping && ν_d > 0 ?
        zero(u0.c.ρ) : nothing
    forcing = AcousticSubstepTendency(
        G,
        G_lim,
        alg.vertical,
        ν_d,
        alg.kinetic_energy,
        alg.damping_form,
        ᶜδ₀,
    )
    implicit_split = alg.implicit_split && alg.vertical isa ImplicitVertical
    # In the inner/outer implicit split the inner implicit solve is restricted to
    # the vertical grid-mean acoustic block, so it factorizes the small
    # acoustic-only `AcousticJacobian` rather than the full implicit Jacobian;
    # otherwise the inner solve uses the full `implicit_tendency!` carried by
    # `f.T_imp!`.
    T_imp! =
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
    # The sub-cycle only needs the acoustic precomputed quantities (velocities,
    # pressure, enthalpy), so use the lighter implicit cache rather than the full
    # cache. The full cache (slow physics) is refreshed once per outer stage by
    # `acoustic_slow_forcing!`.
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
    # The outer implicit half-step solves the full implicit tendency minus the
    # inner subset, once per outer step. It has no explicit tendency, so it needs
    # only the implicit cache; the full (slow-physics) cache is refreshed by
    # `acoustic_slow_forcing!` and the end-of-step `f.cache!`.
    outer_integ =
        if implicit_split
            outer_f = CTS.ClimaODEFunction(;
                T_exp_T_lim! = nothing,
                T_imp! = CTS.ODEFunction(
                    OuterImplicitTendency(f.T_imp!.f, zero(u0));
                    jac_prototype = f.T_imp!.jac_prototype,
                    Wfact = f.T_imp!.Wfact,
                ),
                cache! = f.cache_imp!,
                cache_imp! = f.cache_imp!,
                lim! = f.lim!,
                dss! = f.dss!,
                initialize_imp! = f.initialize_imp!,
            )
            CTS.init(
                CTS.ODEProblem(outer_f, copy(u0), prob.tspan, p),
                alg.inner_alg;
                dt = fast_dt,
                saveat = (),
                save_everystep = false,
            )
        else
            nothing
        end
    return AcousticMultirateCache(
        f, G, G_lim, zero(u0), zero(u0), zero(u0), zero(u0), ᶜδ₀,
        n_sub, inner_integ, outer_integ, alg,
    )
end

# G = T_exp(u) - sub-cycled acoustic terms(u), evaluated and left in `G`/`G_lim`.
# The sub-cycled terms are the horizontal acoustic tendency and, for
# `FastKineticEnergy`, the kinetic-energy gradients, so `G` holds exactly the
# tendency not re-evaluated inside the sub-cycle. Refreshes the cache `p` to be
# consistent with `u`.
function acoustic_slow_forcing!(G, G_lim, A_buf, f, u, p, t, kinetic_energy)
    f.cache!(u, p, t)
    G .= zero(eltype(G))
    G_lim .= zero(eltype(G_lim))
    f.T_exp_T_lim!(G, G_lim, u, p, t)
    A_buf .= zero(eltype(A_buf))
    horizontal_acoustic_tendency!(A_buf, u, p, t)
    if kinetic_energy isa FastKineticEnergy
        kinetic_energy_gradient_tendency!(A_buf, u, p, t)
    end
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

# Fill the perturbation-form reference divergence δ⁰ = divₕ(ᶜu) + ᶜdivᵥ(ᶠu³) from
# the current (sub-cycle-start) precomputed velocities. Called after the
# sub-cycle-start `acoustic_slow_forcing!` refreshes the cache at that state. A
# no-op when `ᶜδ₀` is `nothing` (any damping form other than the perturbation
# form, or damping disabled).
fill_δ₀!(::Nothing, p) = nothing
function fill_δ₀!(ᶜδ₀, p)
    (; ᶜu, ᶠu³) = p.precomputed
    @. ᶜδ₀ = divₕ(ᶜu)
    @. ᶜδ₀ += ᶜdivᵥ(ᶠu³)
    return nothing
end

# Advance the outer implicit problem by `halfdt` from the current `u`, in place.
function outer_half!(outer, u, p, t, halfdt)
    outer.u .= u
    outer.t = refine_ns(t)
    # The outer integrator runs in the (nanosecond) sub-step period, so refine the
    # half-step to match before assigning it (identity for `Float64` and for an
    # already-refined `ITime`).
    CTS.set_dt!(outer, refine_ns(halfdt))
    empty!(outer.tstops)
    CTS.step!(outer)
    u .= outer.u
    return nothing
end

function CTS.step_u!(integrator, cache::AcousticMultirateCache)
    (;
        f,
        G,
        G_lim,
        G2,
        G2_lim,
        A_buf,
        U0,
        ᶜδ₀,
        n_sub,
        inner_integ,
        outer_integ,
        alg,
    ) = cache
    u = integrator.u
    p = integrator.p
    t = integrator.t
    dt = integrator.dt
    ke = alg.kinetic_energy

    if alg.implicit_split && alg.vertical isa ImplicitVertical
        if alg.outer_stages == 1
            # First-order (Lie) inner/outer implicit split: one outer implicit
            # solve over the full `dt`, then one inner acoustic sub-cycle with the
            # slow forcing frozen at the step start.
            outer_half!(outer_integ, u, p, t, dt)
            acoustic_slow_forcing!(G, G_lim, A_buf, f, u, p, t, ke)
            fill_δ₀!(ᶜδ₀, p)
            acoustic_subcycle!(inner_integ, f, u, p, t, dt, n_sub)
            u .= inner_integ.u
            f.cache!(u, p, t + dt)
            return u
        end
        # Second-order (Strang) inner/outer implicit split: symmetric composition
        # of an outer implicit half-step, the inner acoustic sub-cycle over the
        # full `dt`, and a second outer implicit half-step.
        half = sub_timestep(dt, 2)
        outer_half!(outer_integ, u, p, t, half)
        U0 .= u
        acoustic_slow_forcing!(G, G_lim, A_buf, f, U0, p, t, ke)
        fill_δ₀!(ᶜδ₀, p)
        acoustic_subcycle!(inner_integ, f, U0, p, t, dt, n_sub)
        acoustic_slow_forcing!(G2, G2_lim, A_buf, f, inner_integ.u, p, t + dt, ke)
        @. G = (G + G2) / 2
        @. G_lim = (G_lim + G2_lim) / 2
        acoustic_subcycle!(inner_integ, f, U0, p, t, dt, n_sub)
        u .= inner_integ.u
        outer_half!(outer_integ, u, p, t + half, half)
        f.cache!(u, p, t + dt)
        return u
    end

    if alg.outer_stages == 1
        # First-order: freeze the slow forcing over the whole step.
        acoustic_slow_forcing!(G, G_lim, A_buf, f, u, p, t, ke)
        fill_δ₀!(ᶜδ₀, p)
        acoustic_subcycle!(inner_integ, f, u, p, t, dt, n_sub)
        u .= inner_integ.u
    else
        # Second-order: average the slow forcing between step start and a
        # predicted step end (trapezoidal splitting).
        U0 .= u
        acoustic_slow_forcing!(G, G_lim, A_buf, f, U0, p, t, ke)
        fill_δ₀!(ᶜδ₀, p)
        acoustic_subcycle!(inner_integ, f, U0, p, t, dt, n_sub)
        acoustic_slow_forcing!(G2, G2_lim, A_buf, f, inner_integ.u, p, t + dt, ke)
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
