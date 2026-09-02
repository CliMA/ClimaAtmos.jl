#####
##### Hydrostatic reference state
#####
##### A dry, hydrostatically balanced reference state `(T_r, θ_vr, Φ_r)`
##### parameterized by pressure alone. Subtracting it from the split-form
##### pressure-gradient term removes the near-cancellation between the
##### pressure gradient and gravity, which dominates the truncation error of
##### the horizontal momentum tendency (see `horizontal_dynamics_tendency!` in
##### `advection.jl` and `implicit_vertical_advection_tendency!` in
##### `implicit_tendency.jl`).
#####

# TODO: Make sure this is consistent with the function in Thermodynamics.jl.
# Once PR is merged in ClimaParams, import from there.
# https://github.com/CliMA/ClimaParams.jl/pull/253
"""
    s_ref

Exponent of the Exner function in the reference temperature profile
`T_r(p) = T_min + (T_sfc - T_min) Π(p)^s_ref` [-].
"""
const s_ref = 7

"""
    RH_ref

Reference relative humidity used to build a smooth `q_tot_r(p)` profile for the
scalar hyperdiffusion split (`q_tot_r(p) = RH_ref · q_sat(T_r(p), ρ_r(p))`.
"""
const RH_ref = 0.5

"""
    p_q_tot_r_cutoff

Pressure cutoff above which `q_tot_r(p)` is clipped to zero [Pa]. Set at
250 hPa to keep the reference profile confined to the troposphere, avoiding
the unphysical stratospheric growth of `RH_ref · q_sat(T_r, ρ_r)`.
"""
const p_q_tot_r_cutoff = 25000  # Pa (250 hPa)

"""
    theta_v(thermo_params, T, p, q_tot, q_liq, q_ice)

Compute the virtual potential temperature `θ_v = T_v / Π` [K].

The virtual temperature `T_v = T R_m / R_d` uses the gas constant of moist air
`R_m`, so `θ_v = T R_m / (Π R_d)` with `Π` the Exner function of `p`.

# Arguments

  - `thermo_params`: `Thermodynamics` parameter set.
  - `T`: Air temperature [K].
  - `p`: Air pressure [Pa].
  - `q_tot`: Total specific humidity [kg/kg].
  - `q_liq`: Liquid water specific humidity [kg/kg].
  - `q_ice`: Ice specific humidity [kg/kg].
"""
function theta_v(thermo_params, T, p, q_tot, q_liq, q_ice)
    R_d = TD.TP.R_d(thermo_params)
    R_m = TD.gas_constant_air(thermo_params, q_tot, q_liq, q_ice)
    Π = TD.exner_given_pressure(thermo_params, p)
    return T * R_m / (Π * R_d)
end

"""
    air_temperature_reference(thermo_params, p)

Compute the reference-state air temperature at pressure `p` [K].

The profile interpolates between the reference surface temperature `T_surf_ref`
at `p = p_ref` and the reference minimum temperature `T_min_ref` aloft,

```math
T_r(p) = T_\\mathrm{min} + (T_\\mathrm{sfc} - T_\\mathrm{min})\\, Π(p)^{s_\\mathrm{ref}},
```

with `Π` the Exner function and `s_ref` the exponent defined in this file.
"""
function air_temperature_reference(thermo_params, p)
    T_min = TD.TP.T_min_ref(thermo_params)
    T_sfc = TD.TP.T_surf_ref(thermo_params)

    Π = TD.exner_given_pressure(thermo_params, p)

    return T_min + (T_sfc - T_min) * Π^s_ref
end

"""
    theta_vr(thermo_params, p)

Compute the reference-state virtual potential temperature `θ_vr = T_r / Π` [K].

The reference state is dry, so its virtual and actual temperatures coincide.
Subtracting `θ_vr` from `theta_v` gives the perturbation that enters the
split-form horizontal pressure gradient.
"""
function theta_vr(thermo_params, p)
    T_r = air_temperature_reference(thermo_params, p)
    Π = TD.exner_given_pressure(thermo_params, p)
    return T_r / Π
end

"""
    phi_r(thermo_params, p)

Compute the geopotential of the hydrostatic reference state at pressure `p`
[m²/s²].

Integrating hydrostatic balance in Exner coordinates, `dΦ_r = -cp_d θ_vr dΠ`,
with the reference profile of `air_temperature_reference` and the
normalization `Φ_r = 0` at `Π = 1`, gives

```math
Φ_r(p) = -c_{p,d}\\left[T_\\mathrm{min}\\ln Π
    + \\frac{T_\\mathrm{sfc} - T_\\mathrm{min}}{s_\\mathrm{ref}}
      \\left(Π^{s_\\mathrm{ref}} - 1\\right)\\right].
```

Only the gradient of `Φ - Φ_r` enters the momentum tendency, so the choice of
normalization constant is immaterial.
"""
function phi_r(thermo_params, p)
    cp_d = TD.TP.cp_d(thermo_params)
    T_min = TD.TP.T_min_ref(thermo_params)
    T_sfc = TD.TP.T_surf_ref(thermo_params)

    Π = TD.exner_given_pressure(thermo_params, p)

    return -cp_d * (T_min * log(Π) + (T_sfc - T_min) / s_ref * (Π^s_ref - 1))
end

"""
    q_tot_r(thermo_params, p)

Compute the reference-state total specific humidity
`q_tot_r(p) = RH_ref · q_sat(T_r, ρ_r)` for `p ≥ p_q_tot_r_cutoff` and zero
above [kg/kg], with the dry reference-state density `ρ_r = p / (R_d T_r)` and
saturation over liquid.
"""
function q_tot_r(thermo_params, p)
    R_d = TD.TP.R_d(thermo_params)
    T_r = air_temperature_reference(thermo_params, p)
    ρ_r = p / (R_d * T_r)
    return ifelse(
        p < oftype(p, p_q_tot_r_cutoff),
        zero(p),
        oftype(p, RH_ref) *
        TD.q_vap_saturation(thermo_params, T_r, ρ_r, TD.Liquid()),
    )
end

"""
    sd_r(thermo_params, p)

Compute the dry static energy of the reference state at pressure `p`,
`s_{d,r}(p) = cp_d (T_r(p) - T_0) + Φ_r(p)` [J/kg], where `T_r(p)` is the
reference temperature and `Φ_r(p)` is the reference geopotential (both pure
functions of `p`; see `air_temperature_reference` and `phi_r`).
"""
function sd_r(thermo_params, p)
    T_0 = TD.TP.T_0(thermo_params)
    cp_d = TD.TP.cp_d(thermo_params)

    T_r = air_temperature_reference(thermo_params, p)
    return cp_d * (T_r - T_0) + phi_r(thermo_params, p)
end

"""
    pref_from_phi(thermo_params, Φ)

Reference pressure at geopotential `Φ`: inverts [`phi_r`](@ref) by Newton
iteration (`dΦ_r/dp = −R_d T_r/p`), starting from the isothermal profile.
"""
function pref_from_phi(thermo_params, Φ)
    R_d = TD.TP.R_d(thermo_params)
    T_sfc = TD.TP.T_surf_ref(thermo_params)
    p = TD.TP.MSLP(thermo_params) * exp(-Φ / (R_d * T_sfc))
    for _ in 1:8
        T_r = air_temperature_reference(thermo_params, p)
        p += p * (phi_r(thermo_params, p) - Φ) / (R_d * T_r)
    end
    return p
end

