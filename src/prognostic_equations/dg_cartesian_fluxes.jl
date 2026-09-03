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
    λ = max(y⁻.λ, y⁺.λ)
    F = kennedy_gruber_cartesian_flux(normal, normal, y⁻, y⁺)
    return (
        ρ = F.ρ - λ / 2 * (y⁺.ρ - y⁻.ρ),
        ρe = F.ρe - λ / 2 * (y⁺.ρe - y⁻.ρe),
        ρu1 = F.ρu1 - λ / 2 * (y⁺.ρ * y⁺.u1 - y⁻.ρ * y⁻.u1),
        ρu2 = F.ρu2 - λ / 2 * (y⁺.ρ * y⁺.u2 - y⁻.ρ * y⁻.u2),
        ρu3 = F.ρu3 - λ / 2 * (y⁺.ρ * y⁺.u3 - y⁻.ρ * y⁻.u3),
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
    # jumps and wave amplitudes. The pressure jump uses the momentum pressure
    # `pm` (= p for full conservative, = p' for stratified) so the acoustic
    # amplitudes vanish at rest even over topography. (The entropy amplitude α₀
    # still uses the full Δρ, so stratified Roe leaves an O(Δρ_ref) contact-wave
    # residual over terrain — stable, not machine-precision; LMARS avoids it.)
    Δρ = y⁺.ρ - y⁻.ρ
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
    return (
        ρ = F.ρ - Dρ / 2,
        ρe = F.ρe - Dρe / 2,
        ρu1 = F.ρu1 - Dρu1 / 2,
        ρu2 = F.ρu2 - Dρu2 / 2,
        ρu3 = F.ρu3 - Dρu3 / 2,
    )
end

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
([`ranocha_cartesian_flux`](@ref)) or kinetic-energy-preserving
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
