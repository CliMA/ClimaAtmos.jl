using ClimaCore: Fields
using ClimaCore.Utilities: half

# Fields used to store variables that only need to be used in a single function
# but cannot be computed on the fly. Unlike the precomputed quantities, these
# can be modified at any point, so they should never be assumed to be unchanged
# between function calls.
function temporary_quantities(Y, atmos)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
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
        ᶠtemp_CT12ʲs = Fields.Field(
            NTuple{n_mass_flux_subdomains(atmos.turbconv_model), CT12{FT}},
            face_space,
        ), # ᶠω¹²ʲs
        ᶠtemp_C123 = Fields.Field(C123{FT}, face_space), # χ₁₂₃
        ᶜtemp_UVWxUVW = Fields.Field(
            typeof(UVW(FT(0), FT(0), FT(0)) * UVW(FT(0), FT(0), FT(0))'),
            center_space,
        ), # ᶜstrain_rate
        ᶠtemp_UVWxUVW = Fields.Field(
            typeof(UVW(FT(0), FT(0), FT(0)) * UVW(FT(0), FT(0), FT(0))'),
            face_space,
        ), # ᶠstrain_rate
        # TODO: Remove this hack
        sfc_temp_C3 = Fields.Field(C3{FT}, Spaces.level(face_space, half)), # ρ_flux_χ
    )
end
