#=
Held–Suarez (1994) forcing — thin wrappers over ClimaAtmos's own
implementation (src/parameterized_tendencies/radiation/held_suarez.jl):
the ρe relaxation reuses `CA.held_suarez_forcing_tendency_ρe_tot`
verbatim; the Rayleigh drag cannot (it acts on `uₕ::Covariant12`, our
flux-form momentum is Cartesian), so the drag reuses `CA.height_factor`
with the same k_f = 1/day, applied component-wise.

Notes:
- σ = p/(MSLP·exp(-g·z_sfc/(R_d·T_sfc))) with z_sfc from the (possibly
  terrain-warped) face space, matching CA's compute_σ. On flat topography
  z_sfc = 0 and this reduces to the ClimaCore examples' σ = p/p₀ under the
  parity TOML (MSLP = 1e5); the ρe relaxation reads z_sfc from the field's
  own space inside CA, so both terms stay consistent with `topography`.
- Pointwise additive forcing: the drag is a sign-definite KE sink, so the
  KEP advective core is untouched.
- HS constants come from ClimaParams (ΔT_y_dry 60, T_equator_dry 315,
  Δθ_z 10, T_min_hs 200, σ_b 0.7, day 86400, p_ref_theta) — the examples'
  hardcoded values match the defaults.
=#

function hs_forcing_fddg!(dYc, ρ, p, u1, u2, u3, m::DGModel)
    params = m.params
    T_sfc = m.fields.T_sfc

    # Newtonian temperature relaxation on ρe (ClimaAtmos function; returns
    # a lazy broadcast). The ᶜuₕ argument is unused by the dry ρe forcing.
    dYc.ρe .+= CA.held_suarez_forcing_tendency_ρe_tot(
        ρ,
        nothing,
        p,
        params,
        T_sfc,
        CA.DryModel(),
        Val(:held_suarez),
    )

    # Rayleigh low-level drag on the tangential Cartesian momentum
    k_f = 1 / CAP.day(params)
    σ_b = CAP.σ_b(params)
    MSLP = CAP.MSLP(params)
    grav = CAP.grav(params)
    R_d = CAP.R_d(params)
    z_sfc = Fields.level(
        Fields.coordinate_field(m.spaces.hv_face_space).z,
        Fields.half,
    )
    σ = @. p / (MSLP * exp(-grav * z_sfc / (R_d * T_sfc)))
    @. dYc.ρu1 -= k_f * CA.height_factor(σ, σ_b) * ρ * u1
    @. dYc.ρu2 -= k_f * CA.height_factor(σ, σ_b) * ρ * u2
    @. dYc.ρu3 -= k_f * CA.height_factor(σ, σ_b) * ρ * u3
    return dYc
end
