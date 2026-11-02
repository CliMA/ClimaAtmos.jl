#=
Cartesian-momentum two-point and interface fluxes for the flux-form
horizontal DG dynamics (dg_equation_form = :fddg). The volume fluxes are
pressure-free advective fluxes (Kennedy-Gruber, divergence form, and
Kennedy-Gruber with the two-point Exner force of `kg_kpep`); the pressure
force is the Exner-split gradient. The interface dissipations use the model's
equation of state (`roe_eos`, `moist_es`) or a uniform Rusanov rate. State
fields: `ρ`, `ρe`, `e`, `p`, `uv` (orthonormal horizontal velocity), `u1`,
`u2`, `u3` (Cartesian components of the horizontal velocity), `E1`, `E2`,
`E3` (tangential projections of the Cartesian unit vectors), `λ` (Rusanov
speed); `pm = p − p_ref` and `ρm = ρ − ρ_ref` are the deviations from the
hydrostatic reference used in the Roe wave amplitudes.
=#

"""
    kennedy_gruber_advective_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux of the advection, with momentum in global
Cartesian components (Souza et al. 2023): `F_ρ = {ρ}{u_n}`,
`F_ρe = ({ρ}{e} + {p}){u_n}`, `F_ρu_c = {ρ}{u_c}{u_n}`, with `{u_n}` the
average of the nodal contravariant velocities. The momentum flux is `{u_c}`
times a symmetric mass flux, so the advection is kinetic-energy preserving.
It carries no pressure: the horizontal pressure force is the Exner-split
gradient of the vector-invariant form, a gradient of scalars that is
pointwise exact on constants, so over terrain it never invokes the metric
identity `Σ_k ∂_ξk(Ja^k ⋅ ê_c) = 0` that a two-point pressure flux needs and
that the horizontal collocation derivative and the vertical FD operator do
not satisfy jointly. Volume flux of `dg_volume_flux = kg_exner`.
"""
function kennedy_gruber_advective_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    return (
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn,
        ρu2 = ρ̄ * ū2 * ūn,
        ρu3 = ρ̄ * ū3 * ūn,
    )
end

"""
    divergence_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Divergence-form two-point flux for `dg_volume_flux = div_exner`: the mass
flux is the average of the nodal contravariant mass fluxes,
``F_ρ = {ρ u^ξ}`` (the scalar flux of the vector-invariant form, Ducros
et al. 2000), and the momentum and energy ride it,
``F_{ρu_c} = F_ρ {u_c}`` and ``F_E = F_ρ {h_{tot}}``. The momentum flux is
``{u_c}`` times a symmetric mass flux, so the advection is
kinetic-energy preserving; and because the mass flux is an average of
nodal fluxes, the nodal gradients of ``κ`` and ``Φ`` with their central
lifts exchange kinetic with potential energy exactly (the gradient--mass
pairing of docs/dg_vi_entropy_interface.tex), which the Kennedy--Gruber
mass flux ``{ρ}{u^ξ}`` does not. No pressure terms: the pressure force is
the Exner-split gradient, as for `kg_exner`. Same state fields as
[`kennedy_gruber_advective_cartesian_flux`](@ref).
"""
function divergence_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    F = (y_a.ρ * (y_a.uv' * nvec_a) + y_b.ρ * (y_b.uv' * nvec_b)) / 2
    h̄ = (y_a.e + y_a.p / y_a.ρ + y_b.e + y_b.p / y_b.ρ) / 2
    return (
        ρ = F,
        ρe = F * h̄,
        ρu1 = F * (y_a.u1 + y_b.u1) / 2,
        ρu2 = F * (y_a.u2 + y_b.u2) / 2,
        ρu3 = F * (y_a.u3 + y_b.u3) / 2,
    )
end

"""
    KennedyGruberKPEPFlux(cp_d)

Two-point flux of `dg_volume_flux = kg_kpep`, a callable
`(nvec_a, nvec_b, y_a, y_b)`: [`kennedy_gruber_advective_cartesian_flux`](@ref)
plus, in the momentum slots, the whole Exner force in the two-point form of
Souza et al. (2023),

    {ρ} [ (ψ_b − ψ_a)/2 + c_pd {θ_v′} (Π_b − Π_a)/2 ] (ê_c ⋅ J∇ξ)_a,

with the own-node metric. `ψ` is `Φ − Φ_r(p)`, plus the non-horizontal
kinetic energy when the vertical exchange is vector-invariant; `θ` is
`θ_v − θ_vr(p)` and `Π` the Exner function (state fields `ψ`, `θ`, `Π`).
Per unit mass the pressure part is the split Exner gradient of the
vector-invariant form. With the Kennedy--Gruber mass flux `{ρ}{u^ξ}` the
geopotential part exchanges kinetic and potential energy exactly (KPEP),
which the nodal gradient `−ρ_a ∇ψ_a` does not; weighting the pressure part
by the same `{ρ}` keeps its cancellation with the geopotential part at the
hydrostatic reference, so the rest state stays balanced. The term vanishes
for `y_a == y_b`, so the boundary lifts of the flux-differencing kernel are
unchanged; the interface part is [`KPEPFaceLift`](@ref), not the
antisymmetric numerical flux.
"""
struct KennedyGruberKPEPFlux{FT}
    cp_d::FT
end
function (f::KennedyGruberKPEPFlux)(nvec_a, nvec_b, y_a, y_b)
    F = kennedy_gruber_advective_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    g =
        (y_a.ρ + y_b.ρ) / 2 * (
            (y_b.ψ - y_a.ψ) / 2 +
            f.cp_d * (y_a.θ + y_b.θ) / 2 * (y_b.Π - y_a.Π) / 2
        )
    return (
        ρ = F.ρ,
        ρe = F.ρe,
        ρu1 = F.ρu1 + g * (y_a.E1' * nvec_a),
        ρu2 = F.ρu2 + g * (y_a.E2' * nvec_a),
        ρu3 = F.ρu3 + g * (y_a.E3' * nvec_a),
    )
end

"""
    KPEPFaceLift(cp_d)

Face part of [`KennedyGruberKPEPFlux`](@ref), a callable
`(normal, (ψ⁻, θ⁻, Π⁻, ρ⁻), (ψ⁺, θ⁺, Π⁺, ρ⁺))` applied on each side as a
lifting: `{ρ} [ (ψ⁺ − ψ⁻)/2 + c_pd {θ} (Π⁺ − Π⁻)/2 ] n̂`, the central gradient
lift of the vector-invariant form with `{ρ}` in place of the own-side
density. Returns a `UVVector`.
"""
struct KPEPFaceLift{FT}
    cp_d::FT
end
(f::KPEPFaceLift)(normal, (ψ⁻, θ⁻, Π⁻, ρ⁻), (ψ⁺, θ⁺, Π⁺, ρ⁺)) =
    (
        (ρ⁻ + ρ⁺) / 2 *
        ((ψ⁺ - ψ⁻) / 2 + f.cp_d * (θ⁻ + θ⁺) / 2 * (Π⁺ - Π⁻) / 2)
    ) * normal

"""
    CentralMinusDissipation(central, dissipation)

Interface flux `central(normal, normal, y⁻, y⁺) − dissipation(normal, y⁻, y⁺)/2`
for a two-point `central` flux and a dissipation `(normal, y⁻, y⁺) -> D`
over the slots (ρ, ρe, ρu1, ρu2, ρu3).
"""
struct CentralMinusDissipation{C, D}
    central::C
    dissipation::D
end
function (f::CentralMinusDissipation)(normal, (y⁻,), (y⁺,))
    F = f.central(normal, normal, y⁻, y⁺)
    D = f.dissipation(normal, y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ / 2,
        ρe = F.ρe - D.ρe / 2,
        ρu1 = F.ρu1 - D.ρu1 / 2,
        ρu2 = F.ρu2 - D.ρu2 / 2,
        ρu3 = F.ρu3 - D.ρu3 / 2,
    )
end
rusanov_dissipation_cartesian(normal, y⁻, y⁺) =
    rusanov_dissipation_cartesian(y⁻, y⁺)

"""
    kennedy_gruber_advective_rusanov_cartesian(normal, argvals⁻, argvals⁺)

[`kennedy_gruber_advective_cartesian_flux`](@ref) as the central part plus
the Rusanov penalty `λ⟦q⟧/2` on the conserved variables
([`rusanov_dissipation_cartesian`](@ref)).
"""
function kennedy_gruber_advective_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_advective_cartesian_flux(normal, normal, y⁻, y⁺)
    D = rusanov_dissipation_cartesian(y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ / 2,
        ρe = F.ρe - D.ρe / 2,
        ρu1 = F.ρu1 - D.ρu1 / 2,
        ρu2 = F.ρu2 - D.ρu2 / 2,
        ρu3 = F.ρu3 - D.ρu3 / 2,
    )
end

# Rusanov dissipation `λ ⟦q⟧` of the conserved variables, λ = max(λ⁻, λ⁺)
# (momentum jumps via `ρ * u_c`). The interface flux subtracts half of it.
function rusanov_dissipation_cartesian(y⁻, y⁺)
    λ = max(y⁻.λ, y⁺.λ)
    return (
        ρ = λ * (y⁺.ρ - y⁻.ρ),
        ρe = λ * (y⁺.ρe - y⁻.ρe),
        ρu1 = λ * (y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1),
        ρu2 = λ * (y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2),
        ρu3 = λ * (y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3),
    )
end

"""
    eos_pressure(thermo_params, ρ, e_int, q_tot, q_liq, q_ice)

Pressure of the moist equation of state from Thermodynamics.jl at a fixed
phase partition, `p = ρ R_m(q) T(e_int, q)`. Differentiable in every argument.
"""
@inline function eos_pressure(thermo_params, ρ, e_int, q_tot, q_liq, q_ice)
    T = TD.air_temperature(thermo_params, e_int, q_tot, q_liq, q_ice)
    return TD.air_pressure(thermo_params, T, ρ, q_tot, q_liq, q_ice)
end

"""
    eos_pressure_derivatives(thermo_params, ρ, e_int, q_tot, q_liq, q_ice)

`(; p, p_ρ, p_e, p_qt, p_ql, p_qi)`: [`eos_pressure`](@ref) and its partial
derivatives with respect to density, specific internal energy, and the three
water species at fixed phase partition, by forward-mode automatic
differentiation (one dual-number evaluation per derivative). These are the
only equation-of-state inputs the Roe eigenstructure needs: the sound speed
is ``c² = p_ρ + p p_e / ρ²`` and ``κ = p_e / ρ`` replaces ``γ − 1`` in the
energy component of the contact wave. For dry air with constant specific
heats this recovers ``c² = γ p / ρ`` and ``κ = γ − 1``.
"""
@inline function eos_pressure_derivatives(thermo_params, ρ, e_int, q_tot, q_liq, q_ice)
    p = eos_pressure(thermo_params, ρ, e_int, q_tot, q_liq, q_ice)
    p_ρ = ForwardDiff.derivative(
        x -> eos_pressure(thermo_params, x, e_int, q_tot, q_liq, q_ice),
        ρ,
    )
    p_e = ForwardDiff.derivative(
        x -> eos_pressure(thermo_params, ρ, x, q_tot, q_liq, q_ice),
        e_int,
    )
    p_qt = ForwardDiff.derivative(
        x -> eos_pressure(thermo_params, ρ, e_int, x, q_liq, q_ice),
        q_tot,
    )
    p_ql = ForwardDiff.derivative(
        x -> eos_pressure(thermo_params, ρ, e_int, q_tot, x, q_ice),
        q_liq,
    )
    p_qi = ForwardDiff.derivative(
        x -> eos_pressure(thermo_params, ρ, e_int, q_tot, q_liq, x),
        q_ice,
    )
    return (; p, p_ρ, p_e, p_qt, p_ql, p_qi)
end

"""
    RoeEOSDissipation(thermo_params)

Callable `(normal, y⁻, y⁺) -> D` giving the Roe-type dissipation
`Σ_k |λ_k| α_k r_k` of (ρ, ρe, ρu⃗-Cartesian) across a face, with the acoustic
waves damped at `|û_n ± ĉ|` and the contact and shear waves at
`max(|û_n|, ĉ/20)` (a Harten-type floor that keeps a fraction of the Rusanov
rate on contact jumps at near-stagnant faces), using the equation of state of
the model: the sound speed, the contact-wave energy component
``B = Ĥ − ĉ²/κ̂``, and the species waves come from
[`eos_pressure_derivatives`](@ref) at the √ρ-averaged state and phase
partition. A jump of a water species at fixed `ρ`, `p`, `u` changes the
internal energy by ``−(p_q/p_e) Δq``; that energy, ``−ρ̂ (p_q/p_e) Δq`` per
species, is dissipated in the `ρe` slot at the rate `λq` (the rate of the
tracer penalty that moves the species mass, `λqc` for the prognostic
condensates, zero when they are diagnostic), so the species mass and its
energy move together across the face and a humidity front does not generate
spurious temperature jumps. Additional state fields: `e_int`, `q_tot`,
`q_liq`, `q_ice`, `λq`, `λqc`.

The derivatives are taken at the face's diagnosed phase partition, so the
sound speed is the frozen-composition one. Not a Roe-average in the sense of
``ΔF = Â Δq``; for a dissipation term only consistency and the sign of ``|Â|``
matter, and both hold at the averaged state.
"""
struct RoeEOSDissipation{TP}
    thermo_params::TP
end

function (d::RoeEOSDissipation)(normal, y⁻, y⁺)
    tp = d.thermo_params
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û1 = a⁻ * y⁻.u1 + a⁺ * y⁺.u1
    û2 = a⁻ * y⁻.u2 + a⁺ * y⁺.u2
    û3 = a⁻ * y⁻.u3 + a⁺ * y⁺.u3
    Ĥ = a⁻ * (y⁻.e + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e + y⁺.p / y⁺.ρ)
    # averaged thermodynamic state and its pressure derivatives
    ê = a⁻ * y⁻.e_int + a⁺ * y⁺.e_int
    q̂t = a⁻ * y⁻.q_tot + a⁺ * y⁺.q_tot
    q̂l = a⁻ * y⁻.q_liq + a⁺ * y⁺.q_liq
    q̂i = a⁻ * y⁻.q_ice + a⁺ * y⁺.q_ice
    (; p, p_ρ, p_e, p_qt, p_ql, p_qi) =
        eos_pressure_derivatives(tp, ρ̂, ê, q̂t, q̂l, q̂i)
    ĉ = sqrt(p_ρ + p * p_e / ρ̂^2)
    κ̂ = p_e / ρ̂
    ûn = û1 * n1 + û2 * n2 + û3 * n3
    Δρ = y⁺.ρm - y⁻.ρm
    Δp = y⁺.pm - y⁻.pm
    Δu1 = y⁺.u1 - y⁻.u1
    Δu2 = y⁺.u2 - y⁻.u2
    Δu3 = y⁺.u3 - y⁻.u3
    Δun = Δu1 * n1 + Δu2 * n2 + Δu3 * n3
    α₊ = (Δp + ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₋ = (Δp - ρ̂ * ĉ * Δun) / (2 * ĉ^2)
    α₀ = Δρ - Δp / ĉ^2
    s₊ = abs(ûn + ĉ)
    s₋ = abs(ûn - ĉ)
    s₀ = max(abs(ûn), ĉ / 20)
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
    B = Ĥ - ĉ^2 / κ̂
    # species waves: energy carried by the water the tracer penalties move
    λq = max(y⁻.λq, y⁺.λq)
    λqc = max(y⁻.λqc, y⁺.λqc)
    Dρe_q =
        -(ρ̂ / p_e) * (
            λq * p_qt * (y⁺.q_tot - y⁻.q_tot) +
            λqc * (
                p_ql * (y⁺.q_liq - y⁻.q_liq) + p_qi * (y⁺.q_ice - y⁻.q_ice)
            )
        )
    Dρ = s₊ * α₊ + s₋ * α₋ + s₀ * α₀
    Dρu1 =
        s₊ * α₊ * (û1 + ĉ * n1) + s₋ * α₋ * (û1 - ĉ * n1) +
        s₀ * (α₀ * û1 + ρ̂ * Δut1)
    Dρu2 =
        s₊ * α₊ * (û2 + ĉ * n2) + s₋ * α₋ * (û2 - ĉ * n2) +
        s₀ * (α₀ * û2 + ρ̂ * Δut2)
    Dρu3 =
        s₊ * α₊ * (û3 + ĉ * n3) + s₋ * α₋ * (û3 - ĉ * n3) +
        s₀ * (α₀ * û3 + ρ̂ * Δut3)
    Dρe =
        s₊ * α₊ * (Ĥ + ĉ * ûn) + s₋ * α₋ * (Ĥ - ĉ * ûn) +
        s₀ * (α₀ * B + ρ̂ * (û1 * Δut1 + û2 * Δut2 + û3 * Δut3)) + Dρe_q
    return (ρ = Dρ, ρe = Dρe, ρu1 = Dρu1, ρu2 = Dρu2, ρu3 = Dρu3)
end

"""
    RoeEOSInterface(central, thermo_params)

Interface flux `central(normal, normal, y⁻, y⁺) − D/2` with `D` the
[`RoeEOSDissipation`](@ref); `central` is the central part of the volume
flux family (`kennedy_gruber_advective_cartesian_flux` or
`divergence_cartesian_flux`). Selected by `dg_interface_flux = roe_eos`.
"""
struct RoeEOSInterface{C, TP}
    central::C
    dissipation::RoeEOSDissipation{TP}
end
RoeEOSInterface(central, thermo_params) =
    RoeEOSInterface(central, RoeEOSDissipation(thermo_params))

function (f::RoeEOSInterface)(normal, (y⁻,), (y⁺,))
    F = f.central(normal, normal, y⁻, y⁺)
    D = f.dissipation(normal, y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ / 2,
        ρe = F.ρe - D.ρe / 2,
        ρu1 = F.ρu1 - D.ρu1 / 2,
        ρu2 = F.ρu2 - D.ρu2 / 2,
        ρu3 = F.ρu3 - D.ρu3 / 2,
    )
end

# Floor on the total specific humidity inside the moist entropy: the vapor
# mixing entropy has a `q ln q` term whose derivative diverges as q → 0, and
# `0 ⋅ ln 0` is NaN. Only the entropy evaluation is floored; the state is not.
const moist_entropy_q_min = 1e-10

"""
    moist_math_entropy(thermo_params, q, q_liq, q_ice, Φ)

Mathematical entropy ``U = −ρ s`` of the moist equation of state as a
function of the conserved vector `q = (ρ, ρu₁, ρu₂, ρu₃, ρe, [ρq_tot])`
(`SVector` of length 5 or 6), with the condensate partition `(q_liq, q_ice)`
and the geopotential `Φ` held fixed. `s` is Thermodynamics.jl's
`entropy` at the pressure and temperature the model diagnoses from
`(ρ, e_int, q)`. Thermodynamic stability makes `U` convex in `q`, so the
entropy variables ``v = ∂U/∂q`` and the Hessian ``∂²U/∂q²`` (symmetric
positive definite) exist and can be taken by forward-mode autodiff. Used to
verify the entropy-sink property of [`MoistESDissipation`](@ref).
"""
@inline function moist_math_entropy(
    thermo_params,
    q::SA.SVector{N},
    q_liq,
    q_ice,
    Φ,
) where {N}
    ρ = q[1]
    u1 = q[2] / ρ
    u2 = q[3] / ρ
    u3 = q[4] / ρ
    e = q[5] / ρ
    q_tot = N == 6 ? max(q[6] / ρ, oftype(ρ, moist_entropy_q_min)) : zero(ρ)
    e_int = e - (u1^2 + u2^2 + u3^2) / 2 - Φ
    T = TD.air_temperature(thermo_params, e_int, q_tot, q_liq, q_ice)
    p = TD.air_pressure(thermo_params, T, ρ, q_tot, q_liq, q_ice)
    s = TD.entropy(thermo_params, p, T, q_tot, q_liq, q_ice)
    return -ρ * s
end

"""
    MoistESDissipation{N}(thermo_params)

Callable `(normal, y⁻, y⁺) -> D` giving the Lax-Friedrichs dissipation
``D = λ Δq`` on the conserved vector `q = (ρ, ρu⃗, ρe[, ρq_tot])` of the moist
system (`N = 5` dry, `N = 6` with total water). For any convex entropy
``U(q)`` with entropy variables ``v = ∂U/∂q``, monotonicity of the gradient
gives ``Δvᵀ D = λ Δvᵀ Δq ≥ 0``, so every face is a sink of every convex
entropy of the system at once, in particular of the moist physical entropy
[`moist_math_entropy`](@ref) with its latent heat and vapor mixing terms
(verified with autodiff entropy variables in
`test/prognostic_equations/dg_moist_es_tests.jl`). Because ρ, ρe and ρq_tot
are dissipated at one rate, a humidity front at uniform temperature,
pressure and velocity is damped toward the mixture of the two states, which
has the same temperature and pressure: the water mass and the energy it
carries cross the face together. The ρq_tot slot replaces the generic
tracer penalty of total water; the condensates keep theirs.

The entropy-variable form ``λ H Δv`` with ``H = (∂²U/∂q²)⁻¹`` (the dry `es`
form) is sign-definite for the same reason but needs the inverse Hessian,
which is singular in the water direction as ``q_tot → 0`` (the mixing
entropy has infinite curvature there) and cannot be solved in `Float32`;
the two agree to second order in the jump.
"""
struct MoistESDissipation{N, TP}
    thermo_params::TP
end
MoistESDissipation{N}(thermo_params::TP) where {N, TP} =
    MoistESDissipation{N, TP}(thermo_params)

@inline conserved_vector(::Val{5}, y) =
    SA.SVector(y.ρ, y.ρ * y.u1, y.ρ * y.u2, y.ρ * y.u3, y.ρe)
@inline conserved_vector(::Val{6}, y) =
    SA.SVector(y.ρ, y.ρ * y.u1, y.ρ * y.u2, y.ρ * y.u3, y.ρe, y.ρ * y.q_tot)

function (d::MoistESDissipation{N})(normal, y⁻, y⁺) where {N}
    λ = max(y⁻.λ, y⁺.λ)
    D = λ * (conserved_vector(Val(N), y⁺) - conserved_vector(Val(N), y⁻))
    return N == 6 ?
           (ρ = D[1], ρe = D[5], ρu1 = D[2], ρu2 = D[3], ρu3 = D[4], ρq_tot = D[6]) :
           (ρ = D[1], ρe = D[5], ρu1 = D[2], ρu2 = D[3], ρu3 = D[4])
end

"""
    MoistESInterface{N}(central, thermo_params)

Interface flux `central − D/2` with `D` the [`MoistESDissipation`](@ref);
with `N = 6` the returned tuple has a `ρq_tot` slot holding the water
dissipation alone (its central flux stays in the tracer completion).
Selected by `dg_interface_flux = moist_es`.
"""
struct MoistESInterface{N, C, TP}
    central::C
    dissipation::MoistESDissipation{N, TP}
end
MoistESInterface{N}(central::C, thermo_params::TP) where {N, C, TP} =
    MoistESInterface{N, C, TP}(central, MoistESDissipation{N}(thermo_params))

function (f::MoistESInterface{N})(normal, (y⁻,), (y⁺,)) where {N}
    F = f.central(normal, normal, y⁻, y⁺)
    D = f.dissipation(normal, y⁻, y⁺)
    base = (
        ρ = F.ρ - D.ρ / 2,
        ρe = F.ρe - D.ρe / 2,
        ρu1 = F.ρu1 - D.ρu1 / 2,
        ρu2 = F.ρu2 - D.ρu2 / 2,
        ρu3 = F.ρu3 - D.ρu3 / 2,
    )
    return N == 6 ? merge(base, (; ρq_tot = -D.ρq_tot / 2)) : base
end

# Two-point volume flux with a zero `ρq_tot` slot, so that it writes into the
# six-slot residual of the `moist_es` path (the water volume term is the
# tracer split divergence).
with_water_slot(volume_flux) =
    (nvec_a, nvec_b, y_a, y_b) ->
        merge(volume_flux(nvec_a, nvec_b, y_a, y_b), (; ρq_tot = zero(y_a.ρ)))
