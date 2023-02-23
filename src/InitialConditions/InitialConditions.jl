module InitialConditions

import ClimaCore.Fields as Fields
import LinearAlgebra: norm_sqr
import Thermodynamics as TD
import ClimaCore.Geometry as Geometry
import ..TurbulenceConvection as TC
import ..Parameters as CAP

# Types
import ..AtmosModel
import ..DryModel
import ..PotentialTemperature
import ..TotalEnergy
import ..InternalEnergy
import ..EquilMoistModel
import ..NonEquilMoistModel
import ..NoPrecipitation
import ..AbstractPrecipitationModel
import ..AbstractPerformanceMode
import ..PerfStandard
import ..PerfExperimental

energy_vars(thermo_params, ts, K, Φ, atmos) =
    energy_vars(thermo_params, ts, K, Φ, atmos.energy_form)

energy_vars(thermo_params, ts, K, Φ, ::PotentialTemperature) = (;
    ρθ = TD.air_density(thermo_params, ts) *
         TD.liquid_ice_pottemp(thermo_params, ts)
)

energy_vars(thermo_params, ts, K, Φ, ::TotalEnergy) = (;
    ρe_tot = TD.air_density(thermo_params, ts) *
             (TD.internal_energy(thermo_params, ts) + K + Φ)
)

energy_vars(thermo_params, ts, K, Φ, ::InternalEnergy) = (;
    ρe_int = TD.air_density(thermo_params, ts) *
             TD.internal_energy(thermo_params, ts)
)

moisture_vars(thermo_params, ts, atmos::AtmosModel) =
    moisture_vars(thermo_params, ts, atmos.moisture_model)
moisture_vars(thermo_params, ts, ::DryModel) = NamedTuple()
function moisture_vars(thermo_params, ts, ::EquilMoistModel)
    ρ = TD.air_density(thermo_params, ts)
    return (; ρq_tot = ρ * TD.total_specific_humidity(thermo_params, ts))
end
function moisture_vars(thermo_params, ts, ::NonEquilMoistModel)
    ρ = TD.air_density(thermo_params, ts)
    return (;
        ρq_tot = ρ * TD.total_specific_humidity(thermo_params, ts),
        ρq_liq = ρ * TD.liquid_specific_humidity(thermo_params, ts),
        ρq_ice = ρ * TD.ice_specific_humidity(thermo_params, ts),
    )
end

turbconv_vars(FT, atmos::AtmosModel) = turbconv_vars(FT, atmos.turbconv_model)
turbconv_vars(FT, turbconv_model::Nothing) = NamedTuple()
turbconv_vars(FT, turbconv_model::TC.EDMFModel) = (;
    ρq_tot = FT(0), # TC needs this, for now.
    TC.cent_prognostic_vars_edmf(FT, turbconv_model)...,
)

# TODO: Remove dependence on perf mode
precipitation_vars(FT, atmos::AtmosModel) =
    precipitation_vars(FT, atmos, atmos.perf_mode)

# TODO: Remove. Currently, adding tracers hurts performance
precipitation_vars(FT, atmos::AtmosModel, ::PerfExperimental) =
    (; ρq_rai = FT(0), ρq_sno = FT(0))

precipitation_vars(FT, atmos::AtmosModel, ::PerfStandard) =
    precipitation_vars(FT, atmos.precip_model, atmos.turbconv_model)

precipitation_vars(FT, ::AbstractPrecipitationModel, turbconv_model::Nothing) =
    NamedTuple()

# TODO: have precip vars only with Microphysics1Moment
precipitation_vars(
    FT,
    ::AbstractPrecipitationModel,
    turbconv_model::TC.EDMFModel,
) = (; ρq_rai = FT(0), ρq_sno = FT(0))

function init_state(
    center_initial_condition,
    face_initial_condition,
    center_space,
    face_space,
    params,
    atmos::AtmosModel,
    perturb_initstate,
)
    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(face_space)
    c =
        center_initial_condition.(
            ᶜlocal_geometry,
            params,
            atmos,
            perturb_initstate,
        )
    f = face_initial_condition.(ᶠlocal_geometry, params, atmos)
    Y = Fields.FieldVector(; c, f)
    return Y
end

function face_initial_condition(local_geometry, params, atmos)
    z = local_geometry.coordinates.z
    FT = eltype(z)
    tc_kwargs = if atmos.turbconv_model isa Nothing
        NamedTuple()
    else
        TC.face_prognostic_vars_edmf(FT, local_geometry, atmos.turbconv_model)
    end
    (; w = Geometry.Covariant3Vector(FT(0)), tc_kwargs...)
end

include("baro_wave.jl")
include("box.jl")
include("sphere.jl")
include("single_column.jl")

end
