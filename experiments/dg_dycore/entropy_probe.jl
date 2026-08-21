#=
Entropy-machinery probe for the Chan et al. entropy correction (foundational
slice). Verifies the entropy pair + low-order entropy-stable flux by evaluating
the HORIZONTAL volume-flux divergence with (a) the high-order KEP flux and (b)
the low-order (central+LxF) flux, contracting each with the entropy variables,
and reporting the global entropy production Ṡ = Σ_nodes v(u)ᵀ·(dU).

Checks:
  - Ṡ_low ≤ Ṡ_high  (the low-order flux is more entropy-dissipative — required
    for the FCT blend to be able to enforce δ_k ≥ 0), and
  - Σ v(u)ᵀ (r_low − r_high) < 0  (the θ-blend denominator has the right sign).

    julia --project=experiments/dg_dycore experiments/dg_dycore/entropy_probe.jl <config>
=#

include(joinpath(@__DIR__, "driver.jl"))  # includes DGDycore + `using`
import ClimaCore: Fields, Geometry, Operators

isempty(ARGS) && error("usage: entropy_probe.jl <config.yml>")
prob = problem_from_yaml(ARGS[1])
sim = DGSimulation(prob)
m = sim.model
Y = copy(sim.Y₀)
FT = DGDycore.float_type(m)
c = m.c

Yc = Y.c
ρ = Yc.ρ
ρe = Yc.ρe
ᶜΦ = m.fields.ᶜΦ
(; eE1, eE2, eE3, eN1, eN2, eN3, eR1, eR2, eR3) = m.fields
lgeom_c = Fields.local_geometry_field(m.spaces.hv_center_space)

# velocities / thermo, mirroring compute_tendency_fddg!
u1 = @. Yc.ρu1 / ρ
u2 = @. Yc.ρu2 / ρ
u3 = @. Yc.ρu3 / ρ
uE = @. u1 * eE1 + u2 * eE2 + u3 * eE3
uN = @. u1 * eN1 + u2 * eN2 + u3 * eN3
w_c = @. FT(0) * uE   # near-rest start; horizontal advective entropy probe
K = @. (uE^2 + uN^2 + w_c^2) / 2
p = @. DGDycore.pres_ρe(c, ρe, K, ᶜΦ, ρ)
e = @. ρe / ρ
λ = @. sqrt(uE^2 + uN^2) + sqrt(c.γ * p / ρ)
uvw = @. Geometry.UVWVector(uE, uN, w_c)
Ec1 = @. Geometry.UVWVector(eE1, eN1, eR1)
Ec2 = @. Geometry.UVWVector(eE2, eN2, eR2)
Ec3 = @. Geometry.UVWVector(eE3, eN3, eR3)

y = map(
    (ρi, ρei, ei, pi, uvwi, u1i, u2i, u3i, Ec1i, Ec2i, Ec3i, λi, Φi) -> (;
        ρ = ρi, ρe = ρei, e = ei, p = pi, uvw = uvwi,
        u1 = u1i, u2 = u2i, u3 = u3i,
        Ec1 = Ec1i, Ec2 = Ec2i, Ec3 = Ec3i, λ = λi, Φ = Φi,
        p_ref = 0.0, ρ_ref = 0.0,
    ),
    ρ, ρe, e, p, uvw, u1, u2, u3, Ec1, Ec2, Ec3, λ, ᶜΦ,
)

zero_dy() = map(
    _ -> (ρ = 0.0, ρe = 0.0, ρu1 = 0.0, ρu2 = 0.0, ρu3 = 0.0),
    ρ,
)
r_high = zero_dy()
r_low = zero_dy()
Operators.add_flux_differencing_divergence!(
    Operators.kennedy_gruber_cartesian_flux_curvilinear, r_high, y)
Operators.add_flux_differencing_divergence!(
    DGDycore.low_order_es_flux_curvilinear, r_low, y)

# entropy variables (scalar field ops), contracted with the WJ-weighted volume
# tendency. v_ρ carries the +Φ/T geopotential correction; T uses the T_tri ref.
T = @. (ρe / ρ - K - ᶜΦ) / c.cv_d + c.T_tri
s = @. c.cv_d * log(p / ρ^c.γ)
g = @. c.cp_d * T - c.cv_d * c.T_tri - T * s
invT = @. 1 / T
vρ = @. (g - K + ᶜΦ) * invT
vu1 = @. u1 * invT
vu2 = @. u2 * invT
vu3 = @. u3 * invT
vε = @. -invT
contract(r) = @. vρ * r.ρ + vu1 * r.ρu1 + vu2 * r.ρu2 + vu3 * r.ρu3 + vε * r.ρe
Ṡ_high = contract(r_high)                          # = WJ·vᵀ(dU) per node
Ṡ_low = contract(r_low)
dΔ = @. Ṡ_low - Ṡ_high                             # vᵀ(r_low − r_high), per node

sm(f) = sum(parent(f))
@info "Global entropy production Ṡ = Σ vᵀ·(volume tendency)" S_high = sm(Ṡ_high) S_low =
    sm(Ṡ_low) low_more_dissipative = sm(Ṡ_low) <= sm(Ṡ_high)
@info "θ-blend denominator sign check" sum_v_rlow_minus_rhigh = sm(dΔ) max_abs_dΔ =
    maximum(abs, parent(dΔ))
