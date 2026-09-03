using ClimaCore: Fields
using ClimaCore.Utilities: half
using ClimaCore.MatrixFields
using ClimaCore
import StaticArrays: SMatrix

"""
    implicit_temporary_quantities(Y, atmos)

Allocate the subset of scratch fields that the implicit tendency needs.

Called by `jacobian_cache` for the autodiff Jacobian algorithms, which convert
the result to a `ForwardDiff.Dual` element type; the plain-`FT` scratch in
`p.scratch` from `temporary_quantities` cannot be written to under automatic
differentiation. Like all scratch, the returned fields hold no persistent state
and may be overwritten by any function, so they must never be assumed unchanged
between calls.

# Returns

A `NamedTuple` of scratch fields keyed by name and type (`ᶜtemp_scalar`,
`ᶠtemp_CT3`, `temp_data_level`, ...). The trailing comment on each entry records
a representative use.
"""
function implicit_temporary_quantities(Y, atmos)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    return (;
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠρaK_h
        ᶠtemp_scalar_2 = Fields.Field(FT, face_space), # ᶠρaK_u
        ᶠtemp_scalar_3 = Fields.Field(FT, face_space), # ᶠwaχ in advection
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜρχₜ_diffusion, ᶜa_scalar
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜKᵥʲ
        ᶜtemp_scalar_3 = Fields.Field(FT, center_space), # ᶜK_h_scaled
        ᶜtemp_scalar_4 = Fields.Field(FT, center_space), # ᶜq_icl (in surface conditions)
        ᶜtemp_C3 = Fields.Field(C3{FT}, center_space), # ᶜu₃ʲ
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³, ᶠu³_diff
        ᶠtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠstrain_rate
        temp_data_level = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ),
        temp_data_level_2 = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ),
        temp_data_level_3 = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ),
    )
end
"""
    temporary_quantities(Y, atmos)

Allocate the scratch fields stored in `p.scratch`.

These hold values that are used within a single function but cannot be computed
on the fly. Unlike the precomputed quantities, they carry no persistent state:
any function may overwrite any scratch field, so a caller must never assume a
scratch field is unchanged across a call. Using scratch (or lazy broadcasting)
is how tendencies and cache setters avoid allocating `Field`s in the hot path.

Fields are named by location and type: a `ᶜ`/`ᶠ` prefix for cell centers/faces,
then the element type (`scalar`, `C3`, `CT3`, `UVWxUVW`, matrix rows for the
implicit solver). Entries whose names end in `_data` or `_level` are
`Fields.field_values` or single levels rather than full fields. The trailing
comment on each entry records a representative use.

# Returns

A `NamedTuple` of scratch fields keyed by name.
"""
function temporary_quantities(Y, atmos)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    return (;
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠp, ᶠρK_h; ᶠκ in set_face_diffusivities!
        ᶠtemp_scalar_2 = Fields.Field(FT, face_space), # ᶠρK_u; ᶠN²_eff in set_face_diffusivities!
        ᶠtemp_scalar_3 = Fields.Field(FT, face_space), # ᶠstrain in set_face_diffusivities!
        ᶠtemp_scalar_4 = Fields.Field(FT, face_space), # ᶠPr in set_face_diffusivities!
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜ1
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜtke_exch
        ᶜtemp_scalar_3 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_4 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_5 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_6 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_7 = Fields.Field(FT, center_space),
        cosp_temporary_quantities(Y, atmos.cosp)...,
        ᶠtemp_field_level = Fields.level(Fields.Field(FT, face_space), half),
        ᶠtemp_field_level_2 = Fields.level(Fields.Field(FT, face_space), half),
        temp_field_level = Fields.level(Fields.Field(FT, center_space), 1),
        temp_field_level_2 = Fields.level(Fields.Field(FT, center_space), 1),
        temp_field_level_3 = Fields.level(Fields.Field(FT, center_space), 1),
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
        temp_data_level_4 = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ),
        temp_data_level_5 = Fields.field_values(
            Fields.level(Fields.Field(FT, center_space), 1),
        ),
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
        ᶜbidiagonal_adjoint_matrix_c3 = Fields.Field(
            BidiagonalMatrixRow{typeof(C3(FT(0))')},
            center_space,
        ),
        ᶜtemp_bdmr = similar(Y.c, BidiagonalMatrixRow{FT}),
        ᶜtemp_bdmr_2 = similar(Y.c, BidiagonalMatrixRow{FT}),
        ᶜtemp_bdmr_3 = similar(Y.c, BidiagonalMatrixRow{FT}),
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
        ∂ᶜK_∂ᶜuₕ = similar(Y.c, DiagonalMatrixRow{typeof(CT12(FT(0), FT(0))')}),
        ∂ᶜK_∂ᶠu₃ = similar(Y.c, BidiagonalMatrixRow{typeof(CT3(FT(0))')}),
        ᶠp_grad_matrix = similar(Y.f, BidiagonalMatrixRow{C3{FT}}),
        ᶠbidiagonal_matrix_ct3 = similar(Y.f, BidiagonalMatrixRow{CT3{FT}}),
        ᶠband_matrix_wvec = similar(
            Y.f,
            ClimaCore.MatrixFields.BandMatrixRow{
                ClimaCore.Utilities.PlusHalf{Int64}(0),
                1,
                ClimaCore.Geometry.WVector{FT},
            },
        ),
        ᶠsed_tracer_advection = similar(
            Y.f,
            ClimaCore.MatrixFields.BandMatrixRow{
                ClimaCore.Utilities.PlusHalf{Int64}(0),
                1,
                ClimaCore.Geometry.WVector{FT},
            },
        ),
        ᶠbidiagonal_matrix_ct3_2 = similar(Y.f, BidiagonalMatrixRow{CT3{FT}}),
        ᶠbidiagonal_matrix_ct3xct12 = similar(
            Y.f,
            BidiagonalMatrixRow{
                ClimaCore.Geometry.AxisTensor{
                    FT,
                    2,
                    Tuple{
                        ClimaCore.Geometry.ContravariantAxis{(3,)},
                        ClimaCore.Geometry.ContravariantAxis{(1, 2)},
                    },
                    SMatrix{1, 2, FT, 2},
                },
            },
        ),
        ᶜadvection_matrix = similar(
            Y.c,
            BidiagonalMatrixRow{typeof(C3(FT(0))')},
        ),
        ᶜdiffusion_h_matrix = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶜdiffusion_u_matrix = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶜtridiagonal_matrix_scalar = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶠtridiagonal_matrix_c3 = similar(Y.f, TridiagonalMatrixRow{C3{FT}}),
        (!isnothing(atmos.prescribed_flow) ? (; temp_Yₜ_imp = similar(Y)) : (;))...,
    )
end
