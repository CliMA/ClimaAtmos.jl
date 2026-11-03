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
`ᶠtemp_scalar`, `ᶜtemp_UVWxUVW`, ...). The trailing comment on each entry records
a representative use.
"""
function implicit_temporary_quantities(Y, atmos)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    return (;
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠρaK_h, ᶠρK_h of the EDOnly diffusion
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜρχₜ_diffusion, ᶜa_scalar
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜKᵥʲ
        ᶜtemp_scalar_3 = Fields.Field(FT, center_space), # ᶜK_h_scaled
        ᶜtemp_scalar_4 = Fields.Field(FT, center_space), # ᶜq_icl (in surface conditions)
        ᶜtemp_C3 = Fields.Field(C3{FT}, center_space), # ᶜu₃ʲ
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶜtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜτ_smag in the implicit Smagorinsky tendency
        ᶠtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠstrain_rate
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
then the element type (`scalar`, `C3`, `CT3`, `UVWxUVW`), with a trailing `s`
for a tuple of values per point. Entries whose names end in `_data` or `_level`
are `Fields.field_values` or single levels rather than full fields. The trailing
comment on each entry records a representative use.

# Returns

A `NamedTuple` of scratch fields keyed by name.
"""
function temporary_quantities(Y, atmos)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    return (;
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠp in the orographic gravity wave drag
        ᶠtemp_scalar_2 = Fields.Field(FT, face_space), # ᶠK of the implicit-stage u₃ solve
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜ1
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜtke_exch
        ᶜtemp_scalar_3 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_4 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_5 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_6 = Fields.Field(FT, center_space),
        cosp_temporary_quantities(Y, atmos.cosp)...,
        ᶠtemp_field_level = Fields.level(Fields.Field(FT, face_space), half),
        temp_field_level = Fields.level(Fields.Field(FT, center_space), 1),
        temp_data = Fields.field_values(Fields.Field(FT, center_space)),
        # DSS buffer for the element-filtered gradient invariant of the geometric SGS
        # variance (`hgrad_invariant!`); `nothing` on spaces that need no DSS (single columns).
        ᶜscalar_dss_buffer = (uses_covariances(atmos) && do_dss(center_space)) ?
                             Spaces.create_dss_buffer(Fields.Field(FT, center_space)) :
                             nothing,
        ᶜtemp_C3 = Fields.Field(C3{FT}, center_space), # ᶜq_precip of the environment, in its component
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶠtemp_CT12s = Fields.Field(
            NTuple{n_mass_flux_subdomains(atmos.turbconv_model) + 1, CT12{FT}},
            face_space,
        ), # ᶠω¹²ₕs, the curls of the grid-mean and draft u₃
        ᶜtemp_UVW = Fields.Field(typeof(uvw_vec), center_space), # ᶜΔu, vertical velocity differences
        ᶠtemp_UVW = Fields.Field(typeof(uvw_vec), face_space), # ᶠΔu, vertical velocity differences
        ᶜtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜstrain_rate
        ᶠtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠτ_smag
    )
end
