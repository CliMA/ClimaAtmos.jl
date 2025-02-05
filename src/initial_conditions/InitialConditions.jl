module InitialConditions

import ..AtmosModel
import ..DryModel
import ..EquilMoistModel
import ..NonEquilMoistModel
import ..NoPrecipitation
import ..Microphysics0Moment
import ..Microphysics1Moment
import ..PrescribedSurfaceTemperature
import ..PrognosticSurfaceTemperature
import ..ᶜinterp
import ..ᶠinterp
import ..C3
import ..C12
import ..compute_kinetic
import ..PrognosticEDMFX
import ..DiagnosticEDMFX
import ..EDOnlyEDMFX
import ..n_mass_flux_subdomains
# import ..gcm_driven_profile
import ..gcm_height
import ..era5_height
import ..gcm_driven_profile_tmean
import ..era5_driven_profile_tmean
import ..gcm_driven_rho_profile_tmean
import ..era5_driven_rho_profile_tmean

# import ..era5_driven_profile
# import ..era5_height

# need to take these out of initial conditions.jl and put them back in utils but have to settle circularity issues first
import ..external_height
import ..external_driven_profile_tmean
import ..external_driven_rho_profile_tmean

import Thermodynamics.TemperatureProfiles:
    DecayingTemperatureProfile, DryAdiabaticProfile
import ClimaCore: Fields, Geometry
import LinearAlgebra: norm_sqr

import ..Parameters as CAP
import Thermodynamics as TD
import AtmosphericProfilesLibrary as APL
import SciMLBase
import Interpolations as Intp
import NCDatasets as NC
import Statistics: mean
import ClimaUtilities.SpaceVaryingInputs

include("local_state.jl")
include("atmos_state.jl")
include("initial_conditions.jl")

end
