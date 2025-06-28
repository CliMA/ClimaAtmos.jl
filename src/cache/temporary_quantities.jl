using ClimaCore: Fields
using ClimaCore.Utilities: half

# Fields used to store variables that only need to be used in a single function
# but cannot be computed on the fly. Unlike the precomputed quantities, these
# can be modified at any point, so they should never be assumed to be unchanged
# between function calls.
function temporary_quantities(Y, atmos)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
    CTh = CTh_vector_type(Y.c)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    return (;
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠp, ᶠρK_E
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜ1
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜtke_exch
        ᶜtemp_scalar_3 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_4 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_5 = Fields.Field(FT, center_space),
        ᶠtemp_field_level = Fields.level(Fields.Field(FT, face_space), half),
        temp_field_level = Fields.level(Fields.Field(FT, center_space), 1),
        temp_field_level_2 = Fields.level(Fields.Field(FT, center_space), 1),
        temp_field_level_3 = Fields.level(Fields.Field(FT, center_space), 1),
        temp_tuple_field_level = similar(
            Fields.level(Fields.Field(FT, center_space), 1),
            Tuple{FT, FT},
        ),
        temp_data = Fields.field_values(Fields.Field(FT, center_space)),
        temp_data_face_level = Fields.field_values(
            Fields.level(Fields.Field(FT, face_space), half),
        ), # ρaʲu³ʲ_data
        temp_data_level = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ), # ρaʲu³ʲ_data
        temp_data_level_2 = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ), # ρaʲu³ʲ_datau³ʲ_data
        temp_data_level_3 = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ), # ρaʲu³ʲ_datah_tot
        ᶜtemp_C12 = Fields.Field(C12{FT}, center_space), # ᶜuₕ_mean
        ᶜtemp_C3 = Fields.Field(C3{FT}, center_space), # ᶜ∇Φ₃
        ᶜtemp_CT3 = Fields.Field(CT3{FT}, center_space), # ᶜω³, ᶜ∇Φ³
        ᶜtemp_CT123 = Fields.Field(CT123{FT}, center_space),
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶠtemp_CT12 = Fields.Field(CT12{FT}, face_space), # ᶠω¹²
        ᶠtemp_CT12ʲs = Fields.Field(
            NTuple{n_mass_flux_subdomains(atmos.turbconv_model), CT12{FT}},
            face_space,
        ), # ᶠω¹²ʲs
        ᶠtemp_C123 = Fields.Field(C123{FT}, face_space), # χ₁₂₃
        ᶜtemp_UVW = Fields.Field(typeof(uvw_vec), center_space), # UVW(ᶜu)
        ᶠtemp_UVW = Fields.Field(typeof(uvw_vec), face_space), # UVW(ᶠu³)
        ᶜtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜstrain_rate
        ᶠtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠstrain_rate
        ᶜtemp_strain = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜstrain_rate
        ᶠtemp_strain = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠstrain_rate
        # TODO: Remove this hack
        sfc_temp_C3 = Fields.Field(C3{FT}, Spaces.level(face_space, half)), # ρ_flux_χ
        # Implicit solver cache:
        ∂ᶜK_∂ᶜuₕ = similar(Y.c, DiagonalMatrixRow{Adjoint{FT, CTh{FT}}}),
        ∂ᶜK_∂ᶠu₃ = similar(Y.c, BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}),
        ᶠp_grad_matrix = similar(Y.f, BidiagonalMatrixRow{C3{FT}}),
        ᶠbidiagonal_matrix_ct3 = similar(Y.f, BidiagonalMatrixRow{CT3{FT}}),
        ᶠbidiagonal_matrix_ct3_2 = similar(Y.f, BidiagonalMatrixRow{CT3{FT}}),
        ᶜadvection_matrix = similar(
            Y.c,
            BidiagonalMatrixRow{Adjoint{FT, C3{FT}}},
        ),
        ᶜdiffusion_h_matrix = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶜdiffusion_h_matrix_scaled = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶜdiffusion_u_matrix = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶠtridiagonal_matrix_c3 = similar(Y.f, TridiagonalMatrixRow{C3{FT}}),
    )
end
