import CLIMAParameters
using CLIMAParameters: AbstractEarthParameterSet, Planet, Atmos, astro_unit
const CP = CLIMAParameters

# TODO: combine/generalize these structs

#####
##### EarthParameterSet
#####

struct EarthParameterSet <: CP.AbstractEarthParameterSet end

#####
##### BaroclinicWaveParameterSet
#####

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

#####
##### TCEarthParameterSet
#####
struct TCEarthParameterSet{NT} <: CP.AbstractEarthParameterSet
    named_tuple::NT
end
CP.Planet.MSLP(ps::TCEarthParameterSet) = ps.named_tuple.MSLP
CP.Planet.cp_d(ps::TCEarthParameterSet) = ps.named_tuple.cp_d
CP.Planet.cp_v(ps::TCEarthParameterSet) = ps.named_tuple.cp_v
CP.Planet.R_d(ps::TCEarthParameterSet) = ps.named_tuple.R_d
CP.Planet.R_v(ps::TCEarthParameterSet) = ps.named_tuple.R_v
CP.Planet.molmass_ratio(ps::TCEarthParameterSet) = ps.named_tuple.molmass_ratio

const TCICP = TC.InternalClimaParams
# TODO: CLIMAParameter refactor should make
# this easier to encapsulate:
const TCECP = TC.ExperimentalClimaParams
TCECP.entrainment_massflux_div_factor(ps::TCEarthParameterSet) =
    ps.named_tuple.c_div
TCECP.area_limiter_scale(ps::TCEarthParameterSet) = ps.named_tuple.γ_lim
TCECP.area_limiter_power(ps::TCEarthParameterSet) = ps.named_tuple.β_lim
TCECP.static_stab_coeff(ps::TCEarthParameterSet) = ps.named_tuple.c_b
TCECP.l_max(ps::TCEarthParameterSet) = ps.named_tuple.l_max
TCECP.Π_norm(ps::TCEarthParameterSet) = ps.named_tuple.Π_norm
TCECP.c_nn_params(ps::TCEarthParameterSet) = ps.named_tuple.c_nn_params
TCECP.nn_arc(ps::TCEarthParameterSet) = ps.named_tuple.nn_arc
TCECP.w_fno(ps::TCEarthParameterSet) = ps.named_tuple.w_fno
TCECP.nm_fno(ps::TCEarthParameterSet) = ps.named_tuple.nm_fno
TCECP.c_fno(ps::TCEarthParameterSet) = ps.named_tuple.c_fno
TCECP.c_rf_fix(ps::TCEarthParameterSet) = ps.named_tuple.c_rf_fix
TCECP.c_rf_opt(ps::TCEarthParameterSet) = ps.named_tuple.c_rf_opt
TCECP.c_linear(ps::TCEarthParameterSet) = ps.named_tuple.c_linear
TCECP.c_gen_stoch(ps::TCEarthParameterSet) = ps.named_tuple.c_gen_stoch
TCECP.covar_lim(ps::TCEarthParameterSet) = ps.named_tuple.covar_lim


function create_parameter_set(::Type{FT}, parsed_args, namelist) where {FT}
    dt = FT(time_to_seconds(parsed_args["dt"]))
    return if is_column_edmf(parsed_args)
        TCEarthParameterSet((;
            MSLP = 100000.0, # or grab from, e.g., namelist[""][...]
            cp_d = 1004.0,
            cp_v = 1859.0,
            R_d = 287.1,
            R_v = 461.5,
            molmass_ratio = 461.5 / 287.1,
            γ_lim = TC.parse_namelist(
                namelist,
                "turbulence",
                "EDMF_PrognosticTKE",
                "area_limiter_scale",
            ),
            β_lim = TC.parse_namelist(
                namelist,
                "turbulence",
                "EDMF_PrognosticTKE",
                "area_limiter_power",
            ),
            c_div = TC.parse_namelist(
                namelist,
                "turbulence",
                "EDMF_PrognosticTKE",
                "entrainment_massflux_div_factor";
                default = 0.0,
            ),
            l_max = TC.parse_namelist(
                namelist,
                "turbulence",
                "EDMF_PrognosticTKE",
                "l_max";
                default = 1.0e6,
            ),
            c_b = TC.parse_namelist(
                namelist,
                "turbulence",
                "EDMF_PrognosticTKE",
                "static_stab_coeff";
                default = 0.4,
            ), # this is here due to a value error in CliMAParmameters.jl
            covar_lim = TC.parse_namelist(
                namelist,
                "thermodynamics",
                "diagnostic_covar_limiter",
            ),
        ))
    elseif is_column_radiative_equilibrium(parsed_args)
        EarthParameterSet()
    else
        BaroclinicWaveParameterSet((; dt))
    end
end

Base.broadcastable(x::CP.AbstractEarthParameterSet) = Ref(x)
