"""
    RCEMIPExperiment

Abstract supertype for the RCEMIP protocol variants. Concrete subtypes
([`RCEMIPI`], [`RCEMIPII`]) act as dispatch tags on
[`RCEMIPProfile`], selecting protocol-specific behavior such as the
surface-temperature model.
"""
abstract type RCEMIPExperiment end

"""
    RCEMIPI <: RCEMIPExperiment

Tag for the original RCEMIP protocol of [Wing2018](@cite). Selects a uniform,
prescribed surface temperature.
"""
struct RCEMIPI <: RCEMIPExperiment end

"""
    RCEMIPII <: RCEMIPExperiment

Tag for the revised RCEMIP-II protocol of [Wing2024](@cite). Selects the
mock-Walker surface-temperature distribution (a prescribed SST pattern in the
horizontal rather than a uniform value).
"""
struct RCEMIPII <: RCEMIPExperiment end

"""
    RCEMIPProfile(experiment, temperature, humidity)

Initial condition following an RCEMIP sounding. The `experiment` tag
([`RCEMIPI`] or [`RCEMIPII`]) selects the protocol variant and
drives dispatch (e.g. of the surface-temperature model); `temperature` and
`humidity` set the surface state of the analytic sounding, shared by both
protocols.

# Fields

  - `experiment`: Protocol-variant tag, an [`RCEMIPExperiment`] subtype.
  - `temperature`: Surface temperature of the sounding [K].
  - `humidity`: Surface specific humidity of the sounding [kg/kg].

The parameters `E` and `FT` are the experiment-tag type and the float type;
the single `FT` ties `temperature` and `humidity` to the same precision.

See [`RCEMIPIProfile_300`] and friends for convenience constructors at
the protocol's three prescribed SSTs.

!!! note

    The RCEMIP-II protocol prescribes its sounding for the small-domain
    experiment only; the large-domain experiment is instead initialized from
    the final state of the small-domain run. RCEMIP-I has no such staging — all
    domains are initialized directly from the sounding.
"""
struct RCEMIPProfile{E <: RCEMIPExperiment, FT}
    experiment::E
    temperature::FT
    humidity::FT
end

"""
    RCEMIPIIProfile_295()
    RCEMIPIIProfile_300()
    RCEMIPIIProfile_305()

Convenience constructors for the RCEMIP-II sounding ([Wing2024](@cite)) at the
protocol's three prescribed SSTs: 295 K, 300 K, and 305 K, with the
corresponding surface specific humidities (12, 18.65, 24 g/kg). Each returns an
[`RCEMIPProfile`] tagged [`RCEMIPII`].
"""
RCEMIPIIProfile_295() = RCEMIPProfile(RCEMIPII(), 295.0,   12e-3)
RCEMIPIIProfile_300() = RCEMIPProfile(RCEMIPII(), 300.0, 18.65e-3)
RCEMIPIIProfile_305() = RCEMIPProfile(RCEMIPII(), 305.0,   24e-3)

"""
    RCEMIPIProfile_295()
    RCEMIPIProfile_300()
    RCEMIPIProfile_305()

Convenience constructors for the RCEMIP-I sounding ([Wing2018](@cite)) at the
protocol's three prescribed SSTs: 295 K, 300 K, and 305 K, with the
corresponding surface specific humidities (12, 18.65, 24 g/kg). Each returns an
[`RCEMIPProfile`] tagged [`RCEMIPI`].
"""
RCEMIPIProfile_295()  = RCEMIPProfile(RCEMIPI(),  295.0,   12e-3)
RCEMIPIProfile_300()  = RCEMIPProfile(RCEMIPI(),  300.0, 18.65e-3)
RCEMIPIProfile_305()  = RCEMIPProfile(RCEMIPI(),  305.0,   24e-3)

function center_initial_condition(setup::RCEMIPProfile, local_geometry, params)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    grav = CAP.grav(params)

    T_0 = FT(setup.temperature)
    q_0 = FT(setup.humidity)

    q_t = FT(10^(-14))   # kg/kg
    z_q1 = FT(4000)      # m
    z_q2 = FT(7500)      # m
    z_t = FT(15000)      # m
    Γ = FT(0.0067)       # K/m
    p_0 = FT(101480)     # Pa

    T_v0 = T_0 * (1 + FT(0.608) * q_0)
    T_vt = T_v0 - Γ * z_t

    p_t = p_0 * (T_vt / T_v0)^(grav / (R_d * Γ))

    (; z) = local_geometry.coordinates
    if z ≤ z_t
        q = q_0 * exp(-z / z_q1) * exp(-(z / z_q2)^2)
        T_v = T_v0 - Γ * z
        p = p_0 * ((T_v0 - Γ * z) / T_v0)^(grav / (R_d * Γ))
    else
        q = q_t
        T_v = T_vt
        p = p_t * exp(-grav * (z - z_t) / (R_d * T_vt))
    end
    T = T_v / (1 + FT(0.608) * q)

    return physical_state(; T, p, q_tot = q)
end

insolation_model(::RCEMIPProfile) = RCEMIPIIInsolation()

# Sphere SST distribution from Wing et al. (2023) https://gmd.copernicus.org/preprints/gmd-2023-235/
function rcemipii_temperature(
    coordinates::Union{Geometry.LatLongZPoint, Geometry.LatLongPoint},
    surface_temp_params, _,
)
    (; lat) = coordinates
    (; SST_mean, SST_delta, SST_wavelength_latitude) = surface_temp_params
    return SST_mean + SST_delta / 2 * cosd(360 * lat / SST_wavelength_latitude)
end

# Box SST distribution from Wing et al. (2023)
function rcemipii_temperature(
    coordinates::Union{Geometry.XZPoint, Geometry.XYZPoint},
    surface_temp_params, _,
)
    (; x) = coordinates
    (; SST_mean, SST_delta, SST_wavelength) = surface_temp_params
    return SST_mean - SST_delta / 2 * cospi(2 * x / SST_wavelength)
end

import Random

# original RCEMIP setup -- random noise from Wing et. al. (2018)
function rcemipi_temperature(
    coordinates::Union{Geometry.XZPoint, Geometry.XYZPoint},
    surface_temp_params, _,
)
    (; x) = coordinates
    (; SST_mean) = surface_temp_params
     FT = eltype(surface_temp_params)
    rng = Random.Xoshiro(hash(x))
    return SST_mean + eps(FT) * randn(rng)
end

surface_temperature_model(::RCEMIPProfile{RCEMIPII}) =
    AnalyticTemperature(rcemipii_temperature)

surface_temperature_model(::RCEMIPProfile{RCEMIPI}) =
    AnalyticTemperature(rcemipi_temperature)
