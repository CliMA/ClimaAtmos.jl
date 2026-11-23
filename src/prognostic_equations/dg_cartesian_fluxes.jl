#=
Cartesian-momentum two-point and interface fluxes for the flux-form
horizontal DG dynamics (dg_equation_form = :fddg). The volume fluxes are
pressure-free advective fluxes (Kennedy-Gruber, divergence form, and
Kennedy-Gruber with the two-point Exner force of `kg_kpep`); the pressure
force is the Exner-split gradient. The interface dissipations use the model's
equation of state (`roe_eos`, `moist_es`), the low-Mach approximate Riemann
solver (`lmars`), or a uniform Rusanov rate. State fields: `ρ`, `ρe`, `e`,
`p`, `uv` (orthonormal horizontal velocity), `u1`, `u2`, `u3` (Cartesian
components of the horizontal velocity), `E1`, `E2`, `E3` (tangential
projections of the Cartesian unit vectors), `λ` (Rusanov speed), `c` (sound
speed); `pm = p − p_ref` and `ρm = ρ − ρ_ref` are the deviations from the
hydrostatic reference used in the Roe and LMARS wave amplitudes.
=#

"""
    kennedy_gruber_advective_cartesian_flux(nvec_a, nvec_b, y_a, y_b)

Kennedy-Gruber two-point advective flux of (ρ, ρe, Cartesian ρu), without
pressure. Volume flux of `dg_volume_flux = kg_exner`.
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

Two-point flux with mass flux `{ρ u^ξ}` (average of nodal mass fluxes) carrying
`{u_c}` and `{h_tot}`, without pressure. Volume flux of `dg_volume_flux = div_exner`.
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

Kennedy-Gruber advective flux plus the `{ρ}`-weighted two-point Exner and
geopotential force in the momentum slots. Volume flux of `dg_volume_flux = kg_kpep`.
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

Face lifting of the `KennedyGruberKPEPFlux` force,
`{ρ} [⟦ψ⟧/2 + c_pd {θ} ⟦Π⟧/2] n̂`, as a `UVVector`.
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

Interface flux `central − dissipation/2` over the slots (ρ, ρe, ρu1, ρu2, ρu3).
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

Kennedy-Gruber central flux minus half the Rusanov dissipation `λ⟦q⟧`.
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

Pressure [Pa] of the moist equation of state at a fixed phase partition.
"""
@inline function eos_pressure(thermo_params, ρ, e_int, q_tot, q_liq, q_ice)
    T = TD.air_temperature(thermo_params, e_int, q_tot, q_liq, q_ice)
    return TD.air_pressure(thermo_params, T, ρ, q_tot, q_liq, q_ice)
end

"""
    eos_pressure_derivatives(thermo_params, ρ, e_int, q_tot, q_liq, q_ice)

`(; p, p_ρ, p_e, p_qt, p_ql, p_qi)`: `eos_pressure` and its partial derivatives
in density, specific internal energy, and the water species, by forward-mode AD.
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

Roe-type wave dissipation `Σ |λ_k| α_k r_k` of (ρ, ρe, Cartesian ρu) at a face,
with the sound speed and species waves from the model's equation of state.
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

Interface flux `central − D/2` with `D` the `RoeEOSDissipation`
(`dg_interface_flux = roe_eos`).
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

"""
    LMARSDissipation(ε)

Dissipation of the low-Mach approximate Riemann solver (Chen et al. 2013),
with contact speed floor `ε c` (`dg_interface_flux = lmars`).
"""
struct LMARSDissipation{FT}
    ε::FT
end

"""
    lmars_carried_rate(F, y⁻, y⁺, ε)

Upwind rate `max(|F|, ε {ρ}{c})` [kg m⁻² s⁻¹] of quantities carried by the
LMARS interface mass flux `F`.
"""
@inline lmars_carried_rate(F, y⁻, y⁺, ε) =
    max(abs(F), ε * (y⁻.ρ + y⁺.ρ) / 2 * (y⁻.c + y⁺.c) / 2)

function (d::LMARSDissipation)(normal, y⁻, y⁺)
    n1 = y⁻.E1' * normal
    n2 = y⁻.E2' * normal
    n3 = y⁻.E3' * normal
    ρ̄ = (y⁻.ρ + y⁺.ρ) / 2
    c̄ = (y⁻.c + y⁺.c) / 2
    un⁻ = y⁻.uv' * normal
    un⁺ = y⁺.uv' * normal
    Δp = y⁺.pm - y⁻.pm
    u_star = (un⁻ + un⁺) / 2 - Δp / (2 * ρ̄ * c̄)
    s = max(abs(u_star), d.ε * c̄)
    Dρ = Δp / c̄ + s * (y⁺.ρm - y⁻.ρm)
    r = lmars_carried_rate(ρ̄ * (un⁻ + un⁺) / 2 - Dρ / 2, y⁻, y⁺, d.ε)
    h⁻ = y⁻.e + y⁻.p / y⁻.ρ
    h⁺ = y⁺.e + y⁺.p / y⁺.ρ
    a = ρ̄ * c̄ * (un⁺ - un⁻)
    return (
        ρ = Dρ,
        ρe = Dρ * (h⁻ + h⁺) / 2 + r * (h⁺ - h⁻),
        ρu1 = Dρ * (y⁻.u1 + y⁺.u1) / 2 + r * (y⁺.u1 - y⁻.u1) + a * n1,
        ρu2 = Dρ * (y⁻.u2 + y⁺.u2) / 2 + r * (y⁺.u2 - y⁻.u2) + a * n2,
        ρu3 = Dρ * (y⁻.u3 + y⁺.u3) / 2 + r * (y⁺.u3 - y⁻.u3) + a * n3,
    )
end

"""
    carried_upwind_rate(interface, F, y⁻, y⁺)

Upwind rate [kg m⁻² s⁻¹] of quantities carried by the interface mass flux `F`:
the LMARS rate for `lmars`, zero otherwise.
"""
carried_upwind_rate(interface, F, y⁻, y⁺) = zero(F)
carried_upwind_rate(
    f::CentralMinusDissipation{<:Any, <:LMARSDissipation},
    F,
    y⁻,
    y⁺,
) = lmars_carried_rate(F, y⁻, y⁺, f.dissipation.ε)

# Floor on the total specific humidity inside the moist entropy: the vapor
# mixing entropy has a `q ln q` term whose derivative diverges as q → 0, and
# `0 ⋅ ln 0` is NaN. Only the entropy evaluation is floored; the state is not.
const moist_entropy_q_min = 1e-10

"""
    moist_math_entropy(thermo_params, q, q_liq, q_ice, Φ)

Mathematical entropy `−ρ s` of the moist equation of state as a function of the
conserved vector `q = (ρ, ρu₁, ρu₂, ρu₃, ρe, [ρq_tot])`.
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

Lax-Friedrichs dissipation `λ Δq` on the conserved vector `(ρ, ρu⃗, ρe[, ρq_tot])`
(`N = 5` dry, `N = 6` with total water).
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

Interface flux `central − D/2` with `D` the `MoistESDissipation`
(`dg_interface_flux = moist_es`).
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
