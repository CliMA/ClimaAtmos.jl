"""
    DecayingProfile(; perturb = true, params)

A setup with a decaying temperature profile (from Thermodynamics.jl), with an
optional perturbation to the temperature field.

Uses `DecayingTemperatureProfile` with T_surface=290K, T_min=220K, z_scale=8km.
"""
struct DecayingProfile{TP}
    perturb::Bool
    thermo_params::TP
end

function DecayingProfile(;
    perturb::Bool = true,
    thermo_params = nothing,
    params = nothing,
)
    if !isnothing(params)
        return DecayingProfile(perturb, params.thermodynamics_params)
    end
    return DecayingProfile(perturb, thermo_params)
end

_temperature_perturbation(coord::Geometry.LatLongZPoint{FT}) where {FT} =
    FT(0.1) * sind(coord.long) * (coord.z < 5000)
_temperature_perturbation(coord::Geometry.XZPoint{FT}) where {FT} =
    FT(0.1) * sin(coord.x) * (coord.z < 5000)
_temperature_perturbation(coord::Geometry.XYZPoint{FT}) where {FT} =
    FT(0.1) * sin(coord.x) * (coord.z < 5000)
# TODO: Is this a good default?
_temperature_perturbation(::Geometry.AbstractPoint{FT}) where {FT} =
    FT(0)

function center_initial_condition(setup::DecayingProfile, local_geometry, params)
    FT = eltype(params)
    thermo_params = setup.thermo_params
    temp_profile = DecayingTemperatureProfile{FT}(
        thermo_params,
        FT(290),
        FT(220),
        FT(8e3),
    )

    (; z) = local_geometry.coordinates
    T, p = temp_profile(thermo_params, z)
    if setup.perturb
        T += _temperature_perturbation(local_geometry.coordinates)
    end

    return physical_state(; T, p, draft_area = FT(0.1))
end
