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
RCEMIPIIProfile_295() = RCEMIPProfile(RCEMIPII(), 295.0, 12e-3)
RCEMIPIIProfile_300() = RCEMIPProfile(RCEMIPII(), 300.0, 18.65e-3)
RCEMIPIIProfile_305() = RCEMIPProfile(RCEMIPII(), 305.0, 24e-3)

"""
    RCEMIPIProfile_295()
    RCEMIPIProfile_300()
    RCEMIPIProfile_305()

Convenience constructors for the RCEMIP-I sounding ([Wing2018](@cite)) at the
protocol's three prescribed SSTs: 295 K, 300 K, and 305 K, with the
corresponding surface specific humidities (12, 18.65, 24 g/kg). Each returns an
[`RCEMIPProfile`] tagged [`RCEMIPI`].
"""
RCEMIPIProfile_295() = RCEMIPProfile(RCEMIPI(), 295.0, 12e-3)
RCEMIPIProfile_300() = RCEMIPProfile(RCEMIPI(), 300.0, 18.65e-3)
RCEMIPIProfile_305() = RCEMIPProfile(RCEMIPI(), 305.0, 24e-3)

# ── RCEMIP-I symmetry-breaking noise (Wing et al. 2018) ───────────────────────
# One-off thermal perturbation added to the *initial* temperature in the five
# lowest layers: 0.1 K in the lowest, decreasing linearly to 0.02 K in the
# fifth, drawn independently per column and per layer so convective symmetry is
# broken. Applied for RCEMIPI only. This is a simplified linear function
# based on the z-mesh given by Wing et. al. (2018), but allowing for
# other similar z-grids. Ideally, the noise should be implemented in the
# bottom five layers of the simulation.
function rcemipi_thermal_noise(coords)
    z = coords.z
    FT = typeof(z)

    if z >= 520.0 # sixth height level given by Wing et. al. (2018)
        noise = zero(FT)
    else
        amplitude = FT(0.1 * (FT(520.0) - z)) / FT(520.0 - 37.0)
        noise = amplitude * (2 * rand(FT) - 1) # random seed should be set in your runscript
    end

    return noise
end

function center_initial_condition(setup::RCEMIPProfile, local_geometry, params)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    grav = CAP.grav(params)

    T_0 = FT(setup.temperature)
    q_0 = FT(setup.humidity)

    # check that the surface mean value set in the TOML matches the RCEMIPProfile
    surface_temp_params = CAP.surface_temp_params(params)
    (; SST_mean) = surface_temp_params

    if SST_mean != T_0
        error("SST_mean set in TOML does not match RCEMIPProfile.")
    end

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

    if setup.experiment isa RCEMIPI
        T += rcemipi_thermal_noise(local_geometry.coordinates)
    end

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

# original RCEMIP setup -- Wing et. al. (2018)
function rcemipi_temperature(
    coordinates::Union{Geometry.XZPoint, Geometry.XYZPoint},
    surface_temp_params, _,
)
    (; SST_mean) = surface_temp_params
    return SST_mean
end

surface_temperature_model(::RCEMIPProfile{RCEMIPII}) =
    AnalyticTemperature(rcemipii_temperature)

surface_temperature_model(::RCEMIPProfile{RCEMIPI}) =
    AnalyticTemperature(rcemipi_temperature)
