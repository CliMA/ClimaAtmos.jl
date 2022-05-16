import CLIMAParameters
using CLIMAParameters: AbstractEarthParameterSet, Planet, Atmos, astro_unit
const CP = CLIMAParameters

# TODO: combine/generalize these two structs
struct EarthParameterSet <: CP.AbstractEarthParameterSet end

struct BaroclinicWaveParameterSet{NT} <: CP.AbstractEarthParameterSet
    named_tuple::NT
end
Planet.R_d(::BaroclinicWaveParameterSet) = 287.0
Planet.MSLP(::BaroclinicWaveParameterSet) = 1.0e5
Planet.grav(::BaroclinicWaveParameterSet) = 9.80616
Planet.Omega(::BaroclinicWaveParameterSet) = 7.29212e-5
Planet.planet_radius(::BaroclinicWaveParameterSet) = 6.371229e6

# parameters for 0-Moment Microphysics
Atmos.Microphysics_0M.τ_precip(param_set::BaroclinicWaveParameterSet) =
    param_set.named_tuple.dt # timescale for precipitation removal
Atmos.Microphysics_0M.qc_0(::BaroclinicWaveParameterSet) = 5e-6 # criterion for removal after supersaturation

function create_parameter_set(::Type{FT}, parsed_args) where {FT}
    dt = FT(time_to_seconds(parsed_args["dt"]))
    return if is_column_radiative_equilibrium(parsed_args)
        EarthParameterSet()
    else
        BaroclinicWaveParameterSet((; dt))
    end
end
