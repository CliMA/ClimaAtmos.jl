#####
##### Held-Suarez
#####

import Thermodynamics as TD
import Thermodynamics.Parameters as TDP
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

#####
##### Held-Suarez forcing
#####

"""
    held_suarez_ΔT_y_T_equator(params, microphysics_model)

Return the equator-to-pole temperature contrast `ΔT_y` [K] and the equatorial equilibrium
surface temperature `T_equator` [K] of the Held-Suarez forcing.

Dispatches on the microphysics model: `DryModel` uses the dry values `ΔT_y_dry` and
`T_equator_dry`, while any `MoistMicrophysics` model uses the moist values `ΔT_y_wet` and
`T_equator_wet`.
"""
function held_suarez_ΔT_y_T_equator(params, microphysics_model::DryModel)
    FT = eltype(params)
    ΔT_y = FT(CAP.ΔT_y_dry(params))
    T_equator = FT(CAP.T_equator_dry(params))
    return ΔT_y, T_equator
end

function held_suarez_ΔT_y_T_equator(
    params,
    microphysics_model::T,
) where {T <: MoistMicrophysics}
    FT = eltype(params)
    ΔT_y = FT(CAP.ΔT_y_wet(params))
    T_equator = FT(CAP.T_equator_wet(params))
    return ΔT_y, T_equator
end

"""
    HeldSuarezForcingParams{FT}

Bundle of scalar parameters needed to evaluate the Held-Suarez forcing pointwise.

Collecting the parameters in one broadcastable struct keeps the tendency broadcasts free of
repeated parameter lookups. `Base.Broadcast.broadcastable` wraps instances in a tuple, so
they are treated as scalars inside `@.` expressions.

# Fields

  - `ΔT_y`: Equator-to-pole temperature contrast of the equilibrium profile [K].
  - `day`: Length of a day, used to set the relaxation time scales [s].
  - `σ_b`: Top of the boundary layer, in σ coordinates [-].
  - `R_d`: Gas constant of dry air [J/kg/K].
  - `T_min`: Floor on the equilibrium temperature [K].
  - `T_equator`: Equatorial equilibrium surface temperature [K].
  - `Δθ_z`: Static-stability parameter of the equilibrium profile [K].
  - `p_ref_theta`: Reference pressure of the potential temperature [Pa].
  - `κ_d`: Ratio `R_d / cp_d` [-].
  - `grav`: Gravitational acceleration [m/s²].
  - `MSLP`: Mean sea-level pressure, used as the reference surface pressure [Pa].
"""
struct HeldSuarezForcingParams{FT}
    ΔT_y::FT
    day::FT
    σ_b::FT
    R_d::FT
    T_min::FT
    T_equator::FT
    Δθ_z::FT
    p_ref_theta::FT
    κ_d::FT
    grav::FT
    MSLP::FT
end
Base.Broadcast.broadcastable(x::HeldSuarezForcingParams) = tuple(x)

"""
    compute_ΔρT(T_sfc, ρ, p, lat, z_surface, s)

Return `ρ k_T (T - T_equil)`, the density-weighted Newtonian relaxation of temperature
toward the Held-Suarez equilibrium profile [kg K/m³/s].

The relaxation rate and equilibrium temperature follow Held and Suarez (1994):

```math
k_T = k_a + (k_s - k_a) \\, \\max\\left(0, \\frac{σ - σ_b}{1 - σ_b}\\right) \\cos^4φ,
```

```math
T_{equil} = \\max\\left[T_{min},
  \\left(T_{eq} - ΔT_y \\sin^2φ - Δθ_z \\log(p/p_0) \\cos^2φ\\right)
  (p/p_0)^{κ_d}\\right],
```

with `k_a = 1/(40 day)`, `k_s = 1/(4 day)`, and `σ` from `compute_σ`. Temperature is
diagnosed from the ideal gas law as `p / (ρ R_d)`, so the forcing is independent of the
moisture state. The sign is positive where the air is warmer than equilibrium; callers
negate it to obtain a tendency.

# Arguments

  - `T_sfc`: Surface temperature, used to reduce pressure to the surface [K].
  - `ρ`: Air density [kg/m³].
  - `p`: Air pressure [Pa].
  - `lat`: Latitude [degrees].
  - `z_surface`: Surface elevation [m].
  - `s`: The `HeldSuarezForcingParams` parameter bundle.
"""
function compute_ΔρT(
    T_sfc::FT,
    ρ::FT,
    p::FT,
    lat::FT,
    z_surface::FT,
    s::HeldSuarezForcingParams,
) where {FT}
    σ = compute_σ(z_surface, p, T_sfc, s)
    k_a = 1 / (40 * s.day)
    k_s = 1 / (4 * s.day)

    φ = deg2rad(lat)
    return (k_a + (k_s - k_a) * height_factor(σ, s.σ_b) * abs2(abs2(cos(φ)))) *
           ρ *
           ( # ᶜT - ᶜT_equil
               p / (ρ * s.R_d) - max(
                   s.T_min,
                   (
                       s.T_equator - s.ΔT_y * abs2(sin(φ)) -
                       s.Δθ_z * log(p / s.p_ref_theta) * abs2(cos(φ))
                   ) * fast_pow(p / s.p_ref_theta, s.κ_d),
               )
           )
end

"""
    compute_σ(z_surface, p, T_sfc, s)

Return the σ coordinate `p / p_surface` [-], where the surface pressure is obtained by
reducing the mean sea-level pressure `s.MSLP` to the surface elevation `z_surface` with the
hydrostatic relation `p_surface = MSLP exp(-g z_surface / (R_d T_sfc))`.

Called from `compute_ΔρT` and `height_factor`.
"""
function compute_σ(
    z_surface::FT,
    p::FT,
    T_sfc::FT,
    s::HeldSuarezForcingParams,
) where {FT}
    p / (s.MSLP * exp(-s.grav * z_surface / s.R_d / T_sfc))
end

"""
    height_factor(σ, σ_b)
    height_factor(z_surface, p, T_sfc, s)

Return the Held-Suarez boundary-layer weight `max(0, (σ - σ_b) / (1 - σ_b))` [-], which is
one at the surface and tapers to zero at and above the boundary-layer top `σ_b`.

The four-argument method computes `σ` from the pressure with `compute_σ` and takes `σ_b`
from the `HeldSuarezForcingParams` bundle `s`.
"""
height_factor(σ::FT, σ_b::FT) where {FT} = max(0, (σ - σ_b) / (1 - σ_b))
height_factor(z_surface::FT, p::FT, T_sfc::FT, s::HeldSuarezForcingParams) where {FT} =
    height_factor(compute_σ(z_surface, p, T_sfc, s), s.σ_b)

"""
    held_suarez_forcing_tendency_ρe_tot(ᶜρ, ᶜuₕ, ᶜp, params, T_sfc, microphysics_model,
                                        forcing)

Return a lazy broadcast of the Held-Suarez thermal relaxation tendency for `ρe_tot`,
`-ρ cv_d k_T (T - T_equil)` [J/m³/s], or a `NullBroadcasted` when `forcing isa Nothing`.

The relaxation rate `k_T` and the equilibrium temperature `T_equil` are given by
`compute_ΔρT`; the equator-to-pole contrast and equatorial temperature depend on whether
the run is dry or moist (`held_suarez_ΔT_y_T_equator`). Because energy is relaxed at
constant density, the temperature tendency is converted with the dry isochoric heat
capacity `cv_d`.

Held-Suarez is selected with `rad: held_suarez`. It performs no radiative transfer, so it
is applied in `remaining_tendency!` at every timestepper stage rather than on the `dt_rad`
radiation-callback cadence.

See also `held_suarez_forcing_tendency_uₕ` for the boundary-layer momentum drag.
"""
function held_suarez_forcing_tendency_ρe_tot(
    ᶜρ,
    ᶜuₕ,
    ᶜp,
    params,
    T_sfc,
    microphysics_model,
    forcing,
)
    forcing isa Nothing && return NullBroadcasted()
    ᶜspace = axes(ᶜρ)
    (; ᶜz, ᶠz) = z_coordinate_fields(ᶜspace)
    lat = Fields.coordinate_field(ᶜspace).lat

    # TODO: Don't need to enforce FT here, it should be done at param creation.
    FT = Spaces.undertype(ᶜspace)
    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    day = FT(CAP.day(params))
    MSLP = FT(CAP.MSLP(params))
    p_ref_theta = FT(CAP.p_ref_theta(params))
    grav = FT(CAP.grav(params))
    Δθ_z = FT(CAP.Δθ_z(params))
    T_min = FT(CAP.T_min_hs(params))
    σ_b = CAP.σ_b(params)
    k_f = 1 / day

    z_surface = Fields.level(ᶠz, Fields.half)

    ΔT_y, T_equator = held_suarez_ΔT_y_T_equator(params, microphysics_model)

    hs_params = HeldSuarezForcingParams{FT}(
        ΔT_y,
        day,
        σ_b,
        R_d,
        T_min,
        T_equator,
        Δθ_z,
        p_ref_theta,
        κ_d,
        grav,
        MSLP,
    )

    return @. lazy(
        -compute_ΔρT(T_sfc, ᶜρ, ᶜp, lat, z_surface, hs_params) * cv_d,
    )
end

"""
    held_suarez_forcing_tendency_uₕ(ᶜuₕ, ᶜp, params, T_sfc, microphysics_model, forcing)

Return a lazy broadcast of the Held-Suarez boundary-layer drag on the horizontal velocity,
`-k_f max(0, (σ - σ_b) / (1 - σ_b)) uₕ` [m/s²], or a `NullBroadcasted` when
`forcing isa Nothing`.

The Rayleigh drag coefficient is `k_f = 1/day`, and the drag is confined to the boundary
layer by the `height_factor` weight. Like the thermal relaxation, it is applied at every
timestepper stage from `remaining_tendency!`.

See also `held_suarez_forcing_tendency_ρe_tot`.
"""
function held_suarez_forcing_tendency_uₕ(
    ᶜuₕ,
    ᶜp,
    params,
    T_sfc,
    microphysics_model,
    forcing,
)
    forcing isa Nothing && return NullBroadcasted()
    ᶜspace = axes(ᶜp)
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜp))
    # TODO: Don't need to enforce FT here, it should be done at param creation.
    FT = Spaces.undertype(ᶜspace)
    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    day = FT(CAP.day(params))
    MSLP = FT(CAP.MSLP(params))
    p_ref_theta = FT(CAP.p_ref_theta(params))
    grav = FT(CAP.grav(params))
    Δθ_z = FT(CAP.Δθ_z(params))
    T_min = FT(CAP.T_min_hs(params))
    σ_b = CAP.σ_b(params)
    k_f = 1 / day

    z_surface = Fields.level(ᶠz, Fields.half)

    ΔT_y, T_equator = held_suarez_ΔT_y_T_equator(params, microphysics_model)

    hs_params = HeldSuarezForcingParams{FT}(
        ΔT_y,
        day,
        σ_b,
        R_d,
        T_min,
        T_equator,
        Δθ_z,
        p_ref_theta,
        κ_d,
        grav,
        MSLP,
    )

    return @. lazy(-(k_f * height_factor(z_surface, ᶜp, T_sfc, hs_params)) * ᶜuₕ)
end
