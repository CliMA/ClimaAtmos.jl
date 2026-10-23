#=
Cartesian-momentum two-point/interface fluxes for the FDDG horizontal
dynamics (dg_equation_form = :fddg): Kennedy-Gruber and Waruszewski volume
fluxes, Roe / entropy-variable interface dissipation. Vendored verbatim from
the pre-reconciliation ClimaCore DG flux library (physics fluxes live with
the model, not the discretization infrastructure). State-tuple contracts are
documented per function; `pm` is the momentum pressure slot (p − p_ref under
the perturbation formulation).
=#

const γ_dry = 7 / 5

"""
    kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point flux for the full (ρ, ρe, ρu⃗) system with momentum
carried in GLOBAL CARTESIAN components (Souza et al. 2023): the basis is
constant, so component-wise flux differencing retains the KEP property with
no curvature source terms. Contravariant nodal fluxes are averaged (each
node's own metric vector).

State fields required: `ρ`, `e`, `p`, `uv` (velocity, local orthonormal
horizontal frame), `u1`, `u2`, `u3` (Cartesian velocity components), and
`E1`, `E2`, `E3` (the tangential projections of the Cartesian unit vectors
ê₁, ê₂, ê₃, each as a `UVVector` — position-dependent on the sphere but
state-independent). The pressure flux for component ``c`` is
``\\{p\\}\\{ê_c ⋅ nvec\\}``.
"""
function kennedy_gruber_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ē = (y_a.e + y_b.e) / 2
    p̄ = (y_a.p + y_b.p) / 2
    # Momentum pressure: `pm` = p (full conservative) or p' = p − p_ref
    # (stratified conservative, well-balanced over topography). Energy keeps
    # the full thermodynamic p in the enthalpy flux.
    p̄m = (y_a.pm + y_b.pm) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    return (
        ρ = ρ̄ * ūn,
        ρe = (ρ̄ * ē + p̄) * ūn,
        ρu1 = ρ̄ * ū1 * ūn + p̄m * Ē1n,
        ρu2 = ρ̄ * ū2 * ūn + p̄m * Ē2n,
        ρu3 = ρ̄ * ū3 * ūn + p̄m * Ē3n,
    )
end

"""
    kennedy_gruber_advective_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

[`kennedy_gruber_cartesian_flux`](@ref) without the momentum pressure flux
`{p_m}{ê_c ⋅ nvec}`: mass and enthalpy fluxes are unchanged, the momentum
flux is the advective part `{ρ}{u_c}{u_n}` only. For `dg_volume_flux = kg_exner`, where the horizontal pressure force is the Exner-split gradient
of the vector-invariant form: a gradient of scalars is pointwise exact on
constants, so the terrain metric identity `Σ_k ∂_ξk(Ja^k ⋅ ê_c) = 0` that
the two-point pressure flux needs, and that the horizontal collocation
derivative and the vertical FD operator do not satisfy jointly, is never
invoked. Same state fields.
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
[`kennedy_gruber_cartesian_flux`](@ref).
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
    ln_mean(x, y)

Numerically-stable logarithmic mean ``(x-y)/(\\log x - \\log y)`` (Ismail & Roe
2009): switches to the convergent Taylor series in ``f^2=((x-y)/(x+y))^2`` when
``x≈y`` to avoid the ``0/0`` cancellation. The log mean is the building block of
entropy-conservative fluxes (it is what makes ``⟦w⟧·F^\\# = ⟦ψ⟧`` hold exactly).
"""
@inline function ln_mean(x, y)
    ε = oftype(x, 1e-4)
    f² = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)  # ((x−y)/(x+y))²
    # series coefficients in the input precision: Float64 literals here make
    # the ternary return Union{Float32, Float64}, which forces dynamic
    # NamedTuple construction inside GPU kernels
    c₃ = oftype(x, 2 / 3)
    c₅ = oftype(x, 2 / 5)
    c₇ = oftype(x, 2 / 7)
    return f² < ε ?
           (x + y) / (2 + f² * (c₃ + f² * (c₅ + f² * c₇))) :
           (y - x) / log(y / x)
end

"""
    waruszewski_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Waruszewski et al. (2022, JCP 468:111507) entropy-conservative + WELL-BALANCED
two-point flux for the (ρ, ρe, ρu⃗) system WITH GRAVITY, in global Cartesian
momentum components. This is the only flux here that is EC *and* machine-precision
well-balanced over terrain SIMULTANEOUSLY: the geopotential is handled by a
non-conservative fluctuation term ``½ρ̂⟦φ⟧`` in the momentum flux — NOT by a
reference split. It satisfies the generalized (non-conservative) Tadmor condition
``β⁻·D(a;b) − β⁺·D(b;a) = ⟦u_kη⟧`` with the geopotential-augmented entropy
variables (β₁ carries the ``+2φb`` term; see [`entropy_variables`](@ref)).

Differs from Ranocha: the EC pressure is Chandrashekar's ``p* = {{ρ}}/(2{{b}})``,
``b = ρ/(2p)`` (not ``{{p}}``); the internal energy uses the log-mean of ``b``;
and the momentum pressure slot is ``p* + ½ρ̂⟦φ⟧`` with ``ρ̂ = {{b}}{{ρ}}_log/b⁻``
(NON-symmetric — uses the own/self state ``b⁻``, which is well-defined here since
the kernel passes the self node first). Verified: at ``y_a=y_b`` it reduces to the
physical fluxes, and the Tadmor residual over a geopotential jump is ~1e-15.

Hybrid adaptation: the horizontal DG advects only the horizontal momentum, so the
vertical kinetic energy ``w_c²/2`` rides as a passive potential bundled with ``φ``
in ``e*`` (via ``Ψ = e − e_int − K_h``), while the gravity fluctuation uses the
geopotential ``φ`` alone (state field `φ`). State fields: `ρ`, `e`, `p`, `uv`,
`u1`,`u2`,`u3`, `E1`,`E2`,`E3`, `φ`.
"""
function waruszewski_cartesian_flux(nvec_a, nvec_b, y_a, y_b)
    γd = oftype(y_a.ρ, γ_dry)
    ba = y_a.ρ / (2 * y_a.p)                         # inverse temperature b⁻ (self)
    bb = y_b.ρ / (2 * y_b.p)
    ρln = ln_mean(y_a.ρ, y_b.ρ)
    bln = ln_mean(ba, bb)
    b̄ = (ba + bb) / 2
    ρ̄ = (y_a.ρ + y_b.ρ) / 2
    ūn = (y_a.uv' * nvec_a + y_b.uv' * nvec_b) / 2
    mn = ρln * ūn                                    # (ρuₖ)* = ρ^ln {{u}}
    ū1 = (y_a.u1 + y_b.u1) / 2
    ū2 = (y_a.u2 + y_b.u2) / 2
    ū3 = (y_a.u3 + y_b.u3) / 2
    p_star = ρ̄ / (2 * b̄)                             # Chandrashekar p* = {{ρ}}/2{{b}}
    ρ̂ = b̄ * ρln / ba                                # NON-symmetric (self b⁻)
    jφ = y_b.φ - y_a.φ                               # ⟦φ⟧
    pgrav = p_star + ρ̂ * jφ / 2                      # momentum pressure slot
    Ē1n = (y_a.E1' * nvec_a + y_b.E1' * nvec_b) / 2
    Ē2n = (y_a.E2' * nvec_a + y_b.E2' * nvec_b) / 2
    Ē3n = (y_a.E3' * nvec_a + y_b.E3' * nvec_b) / 2
    # internal energy log-mean 1/(2(γ−1)b^ln); horizontal KEP kinetic cross term;
    # passive potential Ψ = φ + w_c²/2 = e − e_int − K_h (advected like {{φ}}).
    e_int = 1 / (2 * (γd - 1) * bln)
    K̃ = (y_a.u1 * y_b.u1 + y_a.u2 * y_b.u2 + y_a.u3 * y_b.u3) / 2
    Ψa = y_a.e - y_a.p / ((γd - 1) * y_a.ρ) - (y_a.u1^2 + y_a.u2^2 + y_a.u3^2) / 2
    Ψb = y_b.e - y_b.p / ((γd - 1) * y_b.ρ) - (y_b.u1^2 + y_b.u2^2 + y_b.u3^2) / 2
    e_star = e_int + (Ψa + Ψb) / 2 + K̃
    return (
        ρ = mn,
        ρe = e_star * mn + ūn * p_star,
        ρu1 = mn * ū1 + pgrav * Ē1n,
        ρu2 = mn * ū2 + pgrav * Ē2n,
        ρu3 = mn * ū3 + pgrav * Ē3n,
    )
end

"""
    kennedy_gruber_rusanov_cartesian(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe, ρu⃗-Cartesian) system:
[`kennedy_gruber_cartesian_flux`](@ref) as the central part plus a Rusanov
penalty scaled by the state field `λ` (jumps of the conserved variables;
momentum jumps via `ρ * u_c`). Additional state fields: `ρe`, `λ`.
"""
function kennedy_gruber_rusanov_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    D = rusanov_dissipation_cartesian(y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ / 2,
        ρe = F.ρe - D.ρe / 2,
        ρu1 = F.ρu1 - D.ρu1 / 2,
        ρu2 = F.ρu2 - D.ρu2 / 2,
        ρu3 = F.ρu3 - D.ρu3 / 2,
    )
end

"""
    kennedy_gruber_advective_rusanov_cartesian(normal, argvals⁻, argvals⁺)

[`kennedy_gruber_advective_cartesian_flux`](@ref) as the central part plus
the Rusanov penalty of [`kennedy_gruber_rusanov_cartesian`](@ref).
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
    kennedy_gruber_roe_cartesian(normal, argvals⁻, argvals⁺)

Interface flux for the (ρ, ρe, ρu⃗-Cartesian) system:
[`kennedy_gruber_cartesian_flux`](@ref) as the central part plus ROE-TYPE
wave-selective dissipation (Souza et al. 2023 interface choice): acoustic
waves are damped at ``|û_n ± ĉ|`` but entropy and shear jumps at
``max(|û_n|, ĉ/20)`` — so stationary balanced structure (contact/shear
jumps with ``u_n ≈ 0``) receives ~5% of Rusanov's uniform ``|u| + c``
dissipation (the Harten-type floor is required: see inline comment).
The energy eigen-component uses ``B = Ĥ - ĉ²/(γ-1)``, which absorbs the
geopotential and vertical-kinetic contributions of ``ρe`` without needing
them separately (Φ is single-valued at the face). Same state fields as
[`kennedy_gruber_rusanov_cartesian`](@ref); requires `γ` jumps consistent
with `p`/`e` (dry ideal gas).
"""
function kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    D = roe_dissipation_cartesian(normal, y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ / 2,
        ρe = F.ρe - D.ρe / 2,
        ρu1 = F.ρu1 - D.ρu1 / 2,
        ρu2 = F.ρu2 - D.ρu2 / 2,
        ρu3 = F.ρu3 - D.ρu3 / 2,
    )
end

"""
    kennedy_gruber_advective_roe_cartesian(normal, argvals⁻, argvals⁺)

[`kennedy_gruber_advective_cartesian_flux`](@ref) as the central part plus
the Roe-type dissipation of [`kennedy_gruber_roe_cartesian`](@ref). The
dissipation is a penalty on the jumps and does not depend on the central
part, so the acoustic waves are still damped at ``|û_n ± ĉ|`` through
their pressure and normal-velocity components even though the pressure
force itself is the Exner-split gradient.
"""
function kennedy_gruber_advective_roe_cartesian(normal, (y⁻,), (y⁺,))
    F = kennedy_gruber_advective_cartesian_flux(normal, normal, y⁻, y⁺)
    D = roe_dissipation_cartesian(normal, y⁻, y⁺)
    return (
        ρ = F.ρ - D.ρ / 2,
        ρe = F.ρe - D.ρe / 2,
        ρu1 = F.ρu1 - D.ρu1 / 2,
        ρu2 = F.ρu2 - D.ρu2 / 2,
        ρu3 = F.ρu3 - D.ρu3 / 2,
    )
end

# Roe-type dissipation `Σ_k |λ_k| α_k r_k` of (ρ, ρe, ρu⃗-Cartesian) across
# the face (see [`kennedy_gruber_roe_cartesian`](@ref)). The interface flux
# subtracts half of it.
function roe_dissipation_cartesian(normal, y⁻, y⁺)
    # keep the working precision of the state (Float32 fields stay Float32)
    γd = oftype(y⁻.ρ, γ_dry)
    # face normal in Cartesian components (E_c single-valued at the node)
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    # Roe-averaged state
    s⁻ = sqrt(y⁻.ρ)
    s⁺ = sqrt(y⁺.ρ)
    ρ̂ = s⁻ * s⁺
    a⁻ = s⁻ / (s⁻ + s⁺)
    a⁺ = 1 - a⁻
    û1 = a⁻ * y⁻.u1 + a⁺ * y⁺.u1
    û2 = a⁻ * y⁻.u2 + a⁺ * y⁺.u2
    û3 = a⁻ * y⁻.u3 + a⁺ * y⁺.u3
    Ĥ = a⁻ * (y⁻.e + y⁻.p / y⁻.ρ) + a⁺ * (y⁺.e + y⁺.p / y⁺.ρ)
    ĉ = a⁻ * sqrt(γd * y⁻.p / y⁻.ρ) + a⁺ * sqrt(γd * y⁺.p / y⁺.ρ)
    ûn = û1 * n1 + û2 * n2 + û3 * n3
    # jumps and wave amplitudes. Both the pressure jump (acoustic) and the
    # density jump (entropy/contact, via α₀) use the PERTURBATION fields pm =
    # p − p_ref and ρm = ρ − ρ_ref, so every wave amplitude vanishes at the
    # hydrostatic reference state — the interface dissipation is then
    # well-balanced over terrain (no spurious mass flux / vertical velocity at
    # rest). Away from rest, ρm/pm carry the flow perturbations, so the
    # dissipation still damps genuine contact/acoustic structure.
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
    # Harten-type entropy floor on the contact/shear speed: pure |û_n|
    # leaves density jumps in near-stagnant columns (e.g. the model top)
    # undamped, and the min-ρ cell can drain unchecked (observed: secular
    # min-ρ collapse from day ~2.3 of a perturbed baroclinic wave at
    # zelem = 30). ε = 0.05 retains 5% of the Rusanov ρ-jump dissipation
    # while keeping the spurious forcing of balanced jets ~20× below
    # Rusanov. The acoustic speeds need no floor (|û_n| ≪ ĉ here).
    s₀ = max(abs(ûn), ĉ / 20)
    Δut1 = Δu1 - Δun * n1
    Δut2 = Δu2 - Δun * n2
    Δut3 = Δu3 - Δun * n3
    B = Ĥ - ĉ^2 / (γd - 1)
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
        s₀ * (α₀ * B + ρ̂ * (û1 * Δut1 + û2 * Δut2 + û3 * Δut3))
    return (ρ = Dρ, ρe = Dρe, ρu1 = Dρu1, ρu2 = Dρu2, ρu3 = Dρu3)
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

Callable `(normal, y⁻, y⁺) -> D` giving the Roe-type dissipation of
[`roe_dissipation_cartesian`](@ref) with the dry-gas constants replaced by the
equation of state of the model: the sound speed, the contact-wave energy
component ``B = Ĥ − ĉ²/κ̂``, and the species waves come from
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
[`RoeEOSDissipation`](@ref); `central` is one of the two-point volume fluxes
(`kennedy_gruber_cartesian_flux`, `kennedy_gruber_advective_cartesian_flux`,
`waruszewski_cartesian_flux`). Selected by `dg_interface_flux = roe_eos`.
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

function waruszewski_roe_cartesian(normal, (y⁻,), (y⁺,))
    Fw = waruszewski_cartesian_flux(normal, normal, y⁻, y⁺)
    Fkg = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    Fd = kennedy_gruber_roe_cartesian(normal, (y⁻,), (y⁺,))
    return (
        ρ = Fw.ρ + (Fd.ρ - Fkg.ρ),
        ρe = Fw.ρe + (Fd.ρe - Fkg.ρe),
        ρu1 = Fw.ρu1 + (Fd.ρu1 - Fkg.ρu1),
        ρu2 = Fw.ρu2 + (Fd.ρu2 - Fkg.ρu2),
        ρu3 = Fw.ρu3 + (Fd.ρu3 - Fkg.ρu3),
    )
end

function waruszewski_es_cartesian(normal, (y⁻,), (y⁺,))
    Fw = waruszewski_cartesian_flux(normal, normal, y⁻, y⁺)
    D = entropy_stable_dissipation(y⁻, y⁺)
    return (
        ρ = Fw.ρ - D.ρ,
        ρe = Fw.ρe - D.ρe,
        ρu1 = Fw.ρu1 - D.ρu1,
        ρu2 = Fw.ρu2 - D.ρu2,
        ρu3 = Fw.ρu3 - D.ρu3,
    )
end

"""
    entropy_variables(ρ, u1, u2, u3, p)

Entropy variables ``w = ∂S/∂U`` for the ideal-gas Euler system with the
mathematical (convex) entropy ``S = -ρs/(γ-1)``, ``s = \\log p - γ\\log ρ``
(thermal frame). With ``β = ρ/(2p)``,

    w = ((γ-s)/(γ-1) - β|u|²,  2βu1,  2βu2,  2βu3,  -2β).

Additive constants in `s` drop under the jump `⟦w⟧`, so they are irrelevant to
the dissipation built from these.
"""
@inline function entropy_variables(ρ, u1, u2, u3, p)
    γd = oftype(ρ, γ_dry)
    β = ρ / (2 * p)
    s = log(p) - γd * log(ρ)
    wρ = (γd - s) / (γd - 1) - β * (u1^2 + u2^2 + u3^2)
    return (wρ, 2 * β * u1, 2 * β * u2, 2 * β * u3, -2 * β)
end

"""
    entropy_stable_dissipation(y⁻, y⁺)

Lax-Friedrichs dissipation in ENTROPY variables, ``½ λ Ĥ ⟦w⟧``, where
``Ĥ = ∂U/∂w`` is the (symmetric positive-definite) entropy Jacobian at the
arithmetic-mean state and ``λ = \\max(|u|+c)``. Because `Ĥ` is SPD,
``⟦w⟧·(Ĥ⟦w⟧) ≥ 0``, so subtracting this from ANY entropy-conservative
(e.g. [`waruszewski_cartesian_flux`](@ref)) or kinetic-energy-preserving
([`kennedy_gruber_cartesian_flux`](@ref)) central flux gives a discrete entropy
inequality (entropy stability) — the guarantee that conserved-variable
Rusanov/Roe penalties do not provide. To leading order `Ĥ⟦w⟧ = ⟦U⟧`, so this is
an entropy-consistent Rusanov. The geopotential (single-valued at the shared
node, `⟦Φ⟧ = 0`) is handled by forming `Ĥ⟦w⟧` in the thermal frame and shifting
the energy component by `Φ·(mass dissipation)` — an identity-preserving change of
variables. Returns the conserved-variable dissipation `(ρ, ρe, ρu1, ρu2, ρu3)`.
The `Ĥ = ∂U/∂w` form is verified numerically (symmetry, SPD, `Ĥ·(∂w/∂U)=I`).
"""
@inline function entropy_stable_dissipation(y⁻, y⁺)
    γd = oftype(y⁻.ρ, γ_dry)
    w⁻ = entropy_variables(y⁻.ρ, y⁻.u1, y⁻.u2, y⁻.u3, y⁻.p)
    w⁺ = entropy_variables(y⁺.ρ, y⁺.u1, y⁺.u2, y⁺.u3, y⁺.p)
    v1 = w⁺[1] - w⁻[1]
    v2 = w⁺[2] - w⁻[2]
    v3 = w⁺[3] - w⁻[3]
    v4 = w⁺[4] - w⁻[4]
    v5 = w⁺[5] - w⁻[5]
    # arithmetic-mean state for Ĥ = ∂U/∂w
    ρ = (y⁻.ρ + y⁺.ρ) / 2
    u1 = (y⁻.u1 + y⁺.u1) / 2
    u2 = (y⁻.u2 + y⁺.u2) / 2
    u3 = (y⁻.u3 + y⁺.u3) / 2
    p = (y⁻.p + y⁺.p) / 2
    k = (u1^2 + u2^2 + u3^2) / 2
    E = p / ((γd - 1) * ρ) + k            # thermal total energy per mass
    H = E + p / ρ                         # thermal enthalpy per mass
    c2 = γd * p / ρ
    # Ĥ v (thermal frame), Ĥ = ∂U/∂w SPD
    HvR = ρ * v1 + ρ * u1 * v2 + ρ * u2 * v3 + ρ * u3 * v4 + ρ * E * v5
    Hv1 =
        ρ * u1 * v1 + (ρ * u1^2 + p) * v2 + ρ * u1 * u2 * v3 +
        ρ * u1 * u3 * v4 + ρ * u1 * H * v5
    Hv2 =
        ρ * u2 * v1 + ρ * u1 * u2 * v2 + (ρ * u2^2 + p) * v3 +
        ρ * u2 * u3 * v4 + ρ * u2 * H * v5
    Hv3 =
        ρ * u3 * v1 + ρ * u1 * u3 * v2 + ρ * u2 * u3 * v3 +
        (ρ * u3^2 + p) * v4 + ρ * u3 * H * v5
    HvE =
        ρ * E * v1 + ρ * u1 * H * v2 + ρ * u2 * H * v3 + ρ * u3 * H * v4 +
        (ρ * H^2 - c2 * p / (γd - 1)) * v5
    λ = max(y⁻.λ, y⁺.λ)
    # geopotential (single-valued at the node ⇒ Φ⁻ = Φ⁺); shift thermal→total
    Φ = y⁻.e - y⁻.p / ((γd - 1) * y⁻.ρ) - (y⁻.u1^2 + y⁻.u2^2 + y⁻.u3^2) / 2
    half = λ / 2
    Dρ = half * HvR
    return (
        ρ = Dρ,
        ρe = half * HvE + Φ * Dρ,
        ρu1 = half * Hv1,
        ρu2 = half * Hv2,
        ρu3 = half * Hv3,
    )
end
