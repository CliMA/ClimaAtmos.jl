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

    # Gas-tracer initial concentrations for the test chemistry mechanisms. Each
    # mechanism only creates the tracers it defines (unused entries are ignored):
    #   * q_gas_AB (ABBA) / q_gas_ED (EDDE): parent molecules of the toy mechanisms
    #   * JPM (jpm.json): typical urban concentrations (see `_jpm_urban_gas_tracers`)
    return physical_state(;
        T, p,
        gas_tracers = (;
            q_gas_AB = FT(0.6),
            q_gas_ED = FT(0.6),
            _jpm_urban_gas_tracers(FT)...,
        ),
    )
end

"""
    _jpm_urban_gas_tracers(FT)

Typical-urban initial concentrations (mass mixing ratio, kg kg⁻¹) for the Julia
Photochemical Mechanism (Sturm and Silva, 2025). Converted from
volume mixing ratios via `q = ppb·1e-9·M_species/M_air` with `M_air = 0.029`
kg mol⁻¹; the commented ppb values are the source. H₂O is a product-only species
in the JPM, so it is left at its zero default.
"""
_jpm_urban_gas_tracers(::Type{FT}) where {FT} = (;
    q_gas_O3 = FT(50*1e-9*0.048/0.029),     # 50 ppb
    q_gas_NO = FT(20*1e-9*0.030/0.029),     # 20 ppb
    q_gas_NO2 = FT(15*1e-9*0.046/0.029),    # 15 ppb
    q_gas_HCHO = FT(30*1e-9*0.030/0.029),   # 30 ppb
    q_gas_HO2 = FT(0.001*1e-9*0.033/0.029),   # 0.001 ppb
    q_gas_H2O2 = FT(1*1e-9*0.034/0.029),   # 1 ppb
    q_gas_OH = FT(2e-4*1e-9*0.017/0.029),    # 2e-4 ppb
    q_gas_HNO3 = FT(2*1e-9*0.063/0.029),   # 2 ppb
    q_gas_CO = FT(150*1e-9*0.028/0.029),     # 150 ppb
    q_gas_H2 = FT(550*1e-9*0.002/0.029),     # 550 ppb
    q_gas_ALD2 = FT(2*1e-9*0.044/0.029),   # 2 ppb
    q_gas_MGLY = FT(1*1e-9*0.072/0.029),   # 1 ppb
    q_gas_MCO3 = FT(0.003*1e-9*0.075/0.029),  # 0.003 ppb
    q_gas_PAN = FT(1.5*1e-9*0.121/0.029),    # 1.5 ppb
)
