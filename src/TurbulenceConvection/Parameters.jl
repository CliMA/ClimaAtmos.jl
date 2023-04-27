"""
    Parameters

"""
module Parameters

import Thermodynamics as TD
import SurfaceFluxes as SF
import CloudMicrophysics as CM

abstract type AbstractTurbulenceConvectionParameters end
const ATCP = AbstractTurbulenceConvectionParameters

#####
##### TurbulenceConvection parameters
#####

Base.@kwdef struct TurbulenceConvectionParameters{FT, MP, SFP} <: ATCP
    Omega::FT
    planet_radius::FT
    surface_area::FT
    max_area::FT
    min_area::FT
    tke_ed_coeff::FT
    tke_diss_coeff::FT
    static_stab_coeff::FT
    Prandtl_number_scale::FT
    Prandtl_number_0::FT
    Ri_crit::FT
    smin_ub::FT
    smin_rm::FT
    l_max::FT
    min_updraft_top::FT
    pressure_normalmode_buoy_coeff1::FT
    pressure_normalmode_adv_coeff::FT
    pressure_normalmode_drag_coeff::FT
    updraft_number::Int
    microphys_params::MP
    surf_flux_params::SFP
end

thermodynamics_params(ps::ATCP) =
    CM.Parameters.thermodynamics_params(ps.microphys_params)
surface_fluxes_params(ps::ATCP) = ps.surf_flux_params
microphysics_params(ps::ATCP) = ps.microphys_params

#####
##### Forwarding parameters
#####

const TCPS = TurbulenceConvectionParameters
for fn in fieldnames(TCPS)
    @eval $(fn)(ps::ATCP) = ps.$(fn)
end

##### Forwarding Thermodynamics.jl

const TDPS = TD.Parameters.ThermodynamicsParameters
for var in fieldnames(TDPS)
    @eval $var(ps::ATCP) = TD.Parameters.$var(thermodynamics_params(ps))
end

# derived parameters
molmass_ratio(ps::ATCP) = TD.Parameters.molmass_ratio(thermodynamics_params(ps))
R_d(ps::ATCP) = TD.Parameters.R_d(thermodynamics_params(ps))
R_v(ps::ATCP) = TD.Parameters.R_v(thermodynamics_params(ps))
cp_d(ps::ATCP) = TD.Parameters.cp_d(thermodynamics_params(ps))
cv_v(ps::ATCP) = TD.Parameters.cv_v(thermodynamics_params(ps))
cv_l(ps::ATCP) = TD.Parameters.cv_l(thermodynamics_params(ps))

##### Forwarding SurfaceFluxes.jl

von_karman_const(ps::ATCP) =
    SF.Parameters.von_karman_const(surface_fluxes_params(ps))

##### Forwarding CloudMicrophysics.jl

ρ_cloud_liq(ps::ATCP) = CM.Parameters.ρ_cloud_liq(microphysics_params(ps))

end
