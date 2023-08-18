using LinearAlgebra: ×, norm, dot

import ClimaAtmos.Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

import ClimaCore.Fields: ColumnField

# Functions on which the model depends:
# CAP.R_d(params)         # dry specific gas constant
# CAP.kappa_d(params)     # dry adiabatic exponent
# CAP.T_triple(params)    # triple point temperature of water
# CAP.MSLP(params)        # reference pressure
# CAP.grav(params)        # gravitational acceleration
# CAP.Omega(params)       # rotation rate (only used if space is spherical)
# CAP.cv_d(params)        # dry isochoric specific heat capacity
# The value of cv_d is implied by the values of R_d and kappa_d

# The model also depends on f_plane_coriolis_frequency(params)
# This is a constant Coriolis frequency that is only used if space is flat

# Fields used to store variables that only need to be used in a single function
# but cannot be computed on the fly. Unlike the precomputed quantities, these
# can be modified at any point, so they should never be assumed to be unchanged
# between function calls.
function temporary_quantities(atmos, center_space, face_space)
    FT = Spaces.undertype(center_space)
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    return (;
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠp, ᶠρK_E
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜ1
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜtke_exch
        temp_data_level = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ), # ρaʲu³ʲ_data
        temp_data_level_2 = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ), # ρaʲu³ʲ_datau³ʲ_data
        temp_data_level_3 = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ), # ρaʲu³ʲ_datah_tot
        ᶜtemp_CT3 = Fields.Field(CT3{FT}, center_space), # ᶜω³, ᶜ∇Φ³
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶠtemp_CT12 = Fields.Field(CT12{FT}, face_space), # ᶠω¹²
        ᶠtemp_CT12ʲs = Fields.Field(NTuple{n, CT12{FT}}, face_space), # ᶠω¹²ʲs
        ᶜtemp_C123 = Fields.Field(C123{FT}, center_space), # χ₁₂₃
        ᶠtemp_C123 = Fields.Field(C123{FT}, face_space), # χ₁₂₃
        ᶜtemp_C3 = Fields.Field(C3{FT}, center_space),
        ᶜtemp_C3_2 = Fields.Field(C3{FT}, center_space),
        ᶠtemp_C3 = Fields.Field(C3{FT}, face_space),
        ᶠtemp_C3_2 = Fields.Field(C3{FT}, face_space),
        ᶜtemp_C12 = Fields.Field(C12{FT}, center_space),
        ᶠtemp_C12 = Fields.Field(C12{FT}, face_space),
        # TODO: Remove this hack
        sfc_temp_C3 = Fields.Field(C3{FT}, Spaces.level(face_space, half)), # ρ_flux_χ
    )
end
