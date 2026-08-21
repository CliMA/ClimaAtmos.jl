#=
Entropy correction (Chan, Ranocha, Park, Lampert, Ching, Edoh 2026,
"Nodal DG for non-ideal EOS: pressure equilibrium preservation and entropy
correction", arXiv:2608.14506) — FCT variant, Section 5.

Goal: a minimally-dissipative, provably entropy-stable stabilization for the KEP
volume flux, so the FDDG core is stable WITHOUT ∇⁴ hyperdiffusion (κ₄ ≡ 0).
Per element, blend the high-order KEP flux with a low-order entropy-stable
(central + local-Lax-Friedrichs) flux by the minimum element-wise coefficient θ
that keeps the cell entropy residual δ_k ≥ 0 (paper Eq. 27-31).

THIS FILE = the foundational slice: the entropy pair (variables + potential) for
the dry-ideal-gas core, the low-order entropy-stable flux, and the volume entropy
residual δ_k used to verify the machinery. The full θ-blend wiring into
`compute_tendency_fddg!` is the next increment.

Atmosphere-specific care:
  - `ρe` includes the geopotential (ρe ⊃ ρΦ) and uses a T_tri energy reference
    (`pres_ρe`: T = (ρe/ρ − K − Φ)/cv_d + T_tri). The entropy variables are
    built from the ACTUAL T and p, and the density component picks up +Φ/T from
    ∂e_int/∂ρ carrying the geopotential (derived: v_ρ = (g − K + Φ)/T).
  - The correction targets the HORIZONTAL EXPLICIT advective flux; gravity/
    Coriolis/Held-Suarez sources and the vertical implicit acoustics do not enter
    δ_k here (documented limitation — the entropy inequality is advective-only).
=#

# Entropy variables v = ∂S/∂u for S = −ρs (physical entropy), dry ideal gas,
# with the geopotential-in-energy and T_tri-reference corrections. Returns the
# five components (density, three Cartesian momenta, total energy).
@inline function entropy_variables(c::DGConstants, ρ, ρu1, ρu2, ρu3, ρe, Φ)
    u1 = ρu1 / ρ
    u2 = ρu2 / ρ
    u3 = ρu3 / ρ
    K = (u1^2 + u2^2 + u3^2) / 2
    T = (ρe / ρ - K - Φ) / c.cv_d + c.T_tri          # actual temperature
    p = ρ * c.R_d * T
    s = c.cv_d * log(p / ρ^c.γ)                      # specific entropy (+const)
    g = c.cp_d * T - c.cv_d * c.T_tri - T * s        # Gibbs (shifted-energy form)
    invT = 1 / T
    return (
        vρ = (g - K + Φ) * invT,
        vu1 = u1 * invT,
        vu2 = u2 * invT,
        vu3 = u3 * invT,
        vε = -invT,
    )
end

# Entropy potential in direction of the (physical) velocity: ψ = p·u/T (paper
# Eq. 5, ψ_i = v^T f_i − F_i = p v_i / T). Returned as the three Cartesian
# components p·u_c/T for the element-boundary term in δ_k.
@inline function entropy_potential(c::DGConstants, ρ, ρu1, ρu2, ρu3, ρe, Φ)
    u1 = ρu1 / ρ
    u2 = ρu2 / ρ
    u3 = ρu3 / ρ
    K = (u1^2 + u2^2 + u3^2) / 2
    T = (ρe / ρ - K - Φ) / c.cv_d + c.T_tri
    p = ρ * c.R_d * T
    pT = p / T
    return (ψ1 = pT * u1, ψ2 = pT * u2, ψ3 = pT * u3)
end

# Contract entropy variables with a conserved-variable 5-tuple (the tendency or a
# flux contribution): v^T · r. Used for the global entropy rate dS/dt and for the
# θ blend numerator v^T (r_low − r_high).
@inline entropy_contract(v, r) =
    v.vρ * r.ρ + v.vu1 * r.ρu1 + v.vu2 * r.ρu2 + v.vu3 * r.ρu3 + v.vε * r.ρe

# --- Low-order entropy-stable curvilinear volume flux — INCORRECT in the DENSE
#     flux-differencing (kept for reference / the sparsified rewrite).
#
#     VERIFIED WRONG (entropy_probe: Ṡ_low = +3e14, anti-dissipative): an
#     antisymmetric LxF jump −(λ/2)[[U]] put through the dense skew-symmetric SBP
#     operator Q (with Q·1 = 0) telescopes to (Q·u)_i — a FIRST-derivative
#     (advective) term, NOT a diffusion — so it cannot be entropy-dissipative.
#     The paper's low-order scheme (Sec 5.2, refs [63,55]) rides SPARSIFIED SBP
#     operators (a subcell finite-volume structure); the LxF dissipation only
#     becomes a genuine diffusion on the nearest-neighbor sparsified operator.
#     TODO: implement the sparsified low-order operator; this dense form is a
#     placeholder. Central (arithmetic mean of the physical Euler fluxes) + LxF
#     as below. ---
function low_order_es_flux_curvilinear(nvec_a, nvec_b, y_a, y_b)
    γd = oftype(y_a.ρ, Operators.γ_dry)
    una = y_a.uvw' * nvec_a
    unb = y_b.uvw' * nvec_b
    # arithmetic-mean (central) physical flux, each side contracted with its own
    # metric normal (symmetric, consistent two-point flux)
    Ea1n = y_a.Ec1' * nvec_a
    Ea2n = y_a.Ec2' * nvec_a
    Ea3n = y_a.Ec3' * nvec_a
    Eb1n = y_b.Ec1' * nvec_b
    Eb2n = y_b.Ec2' * nvec_b
    Eb3n = y_b.Ec3' * nvec_b
    Fρ = (y_a.ρ * una + y_b.ρ * unb) / 2
    Fρe =
        ((y_a.ρ * y_a.e + y_a.p) * una + (y_b.ρ * y_b.e + y_b.p) * unb) / 2
    Fρu1 =
        (y_a.ρ * y_a.u1 * una + y_a.p * Ea1n + y_b.ρ * y_b.u1 * unb + y_b.p * Eb1n) / 2
    Fρu2 =
        (y_a.ρ * y_a.u2 * una + y_a.p * Ea2n + y_b.ρ * y_b.u2 * unb + y_b.p * Eb2n) / 2
    Fρu3 =
        (y_a.ρ * y_a.u3 * una + y_a.p * Ea3n + y_b.ρ * y_b.u3 * unb + y_b.p * Eb3n) / 2
    # local Lax-Friedrichs wavespeed (Davis): |u·n̂| + c, scaled by ‖nvec‖ so the
    # dissipation matches the metric-flux normalization. Use the average normal.
    n̄ = (nvec_a + nvec_b) / 2
    nrm = sqrt(n̄' * n̄)
    ca = sqrt(γd * y_a.p / y_a.ρ)
    cb = sqrt(γd * y_b.p / y_b.ρ)
    λ = max(abs(una) + ca * nrm, abs(unb) + cb * nrm)
    h = λ / 2
    return (
        ρ = Fρ - h * (y_b.ρ - y_a.ρ),
        ρe = Fρe - h * (y_b.ρe - y_a.ρe),
        ρu1 = Fρu1 - h * (y_b.ρ * y_b.u1 - y_a.ρ * y_a.u1),
        ρu2 = Fρu2 - h * (y_b.ρ * y_b.u2 - y_a.ρ * y_a.u2),
        ρu3 = Fρu3 - h * (y_b.ρ * y_b.u3 - y_a.ρ * y_a.u3),
    )
end
Operators._fd_metric_style(::typeof(low_order_es_flux_curvilinear)) =
    Val{:curvilinear}()
