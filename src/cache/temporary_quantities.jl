using ClimaCore: Fields
using ClimaCore.Utilities: half
using ClimaCore.MatrixFields
using ClimaCore
import StaticArrays: SVector, SMatrix

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
        ᶜtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜτ_smag in the implicit Smagorinsky tendency
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
    cloud_species_diffusion_temporary_quantities(Y, atmos)

Allocate the scratch fields of the blended cloud-species flux
(`add_cloud_species_diffusionᵥ!`): the face fluxes of total water and
liquid–ice static energy, and the center blend weight and condensate
sensitivities. The 1-moment microphysics is the only scheme that blends, so
every other scheme gets an empty `NamedTuple`.
"""
function cloud_species_diffusion_temporary_quantities(Y, atmos)
    atmos.microphysics_model isa NonEquilibriumMicrophysics1M || return (;)
    center_space, face_space = axes(Y.c), axes(Y.f)
    FT = Spaces.undertype(center_space)
    return (;
        ᶠtemp_C3 = Fields.Field(C3{FT}, face_space), # ᶠF_qt
        ᶠtemp_C3_2 = Fields.Field(C3{FT}, face_space), # ᶠF_sl
        ᶜtemp_scalar_8 = Fields.Field(FT, center_space), # ᶜλ
        ᶜtemp_scalar_9 = Fields.Field(FT, center_space), # ᶜc_qt
        ᶜtemp_scalar_10 = Fields.Field(FT, center_space), # ᶜc_sl
    )
end

@inline function pad_b_imp_stages(b_imp, ::Type{FT}) where {FT}
    n = length(b_imp)
    n <= 8 ||
        error("Algorithm has $n implicit stages; maximum supported is 8")
    return SVector{8, FT}(ntuple(i -> i <= n ? FT(b_imp[i]) : zero(FT), Val(8)))
end

"""
    precipitation_surface_flux_accumulator_quantities(Y, atmos, b_imp_stages, n_imp_stages)

Allocate the scratch fields and stage counter that accumulate the stage-weighted
bottom-face sedimentation fluxes (`surface_rain_flux`, `surface_snow_flux`, and
`col_integrated_precip_energy_tendency`) across the implicit stages of each
IMEX timestep for the non-equilibrium microphysics schemes. Returns `(;)` for
dry and 0-moment schemes.
"""
function precipitation_surface_flux_accumulator_quantities(
    Y,
    atmos,
    b_imp_stages::SVector{8, FT},
    n_imp_stages::Int,
) where {FT}
    atmos.microphysics_model isa Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
        NonEquilibriumMicrophysics2MP3,
    } || return (;)
    sfc_space = axes(Fields.level(Y.f, half))
    PrecipLinNT = @NamedTuple{
        w_lcl_base::FT,
        w_icl_base::FT,
        w_rai_jac::FT,
        w_sno_jac::FT,
        f_guess_lcl::FT,
        f_guess_icl::FT,
        f_guess_rai::FT,
        f_guess_sno::FT,
        ρq_lcl_guess::FT,
        ρq_icl_guess::FT,
        ρq_rai_guess::FT,
        ρq_sno_guess::FT,
        e_lcl::FT,
        e_icl::FT,
        e_rai::FT,
        e_sno::FT,
        f_e_sgs::FT,
    }
    return (;
        b_imp_stages,
        n_imp_stages,
        precip_stage_idx = Base.RefValue(0),
        sfc_precip_lin_state = Fields.Field(PrecipLinNT, sfc_space),
        surface_rain_flux_accum = zeros(sfc_space),
        surface_snow_flux_accum = zeros(sfc_space),
        col_integrated_precip_energy_tendency_accum = zeros(sfc_space),
    )
end
function precipitation_surface_flux_accumulator_quantities(
    Y,
    atmos,
    b_imp_stages = (),
    n_imp_stages = 0,
)
    FT = Spaces.undertype(axes(Y.c))
    b_imp_svec =
        b_imp_stages isa SVector{8, FT} ? b_imp_stages :
        pad_b_imp_stages(b_imp_stages, FT)
    n_imp =
        b_imp_stages isa SVector{8, FT} ? Int(n_imp_stages) :
        (
            isempty(b_imp_stages) ? 0 :
            (n_imp_stages == 0 ? length(b_imp_stages) : Int(n_imp_stages))
        )
    return precipitation_surface_flux_accumulator_quantities(
        Y,
        atmos,
        b_imp_svec,
        n_imp,
    )
end

"""
    temporary_quantities(Y, atmos; b_imp_stages = (), n_imp_stages = 0)

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
function temporary_quantities(Y, atmos; b_imp_stages = (), n_imp_stages = 0)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    return (;
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠp, ᶠρK_h; ᶠκ in set_face_diffusivities!
        ᶠtemp_scalar_2 = Fields.Field(FT, face_space), # ᶠρK_u; ᶠstrain in set_face_diffusivities!
        ᶠtemp_scalar_3 = Fields.Field(FT, face_space), # ᶠPr in set_face_diffusivities!
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜ1
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜtke_exch
        ᶜtemp_scalar_3 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_4 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_5 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_6 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_7 = Fields.Field(FT, center_space),
        cloud_species_diffusion_temporary_quantities(Y, atmos)...,
        precipitation_surface_flux_accumulator_quantities(
            Y,
            atmos,
            b_imp_stages,
            n_imp_stages,
        )...,
        cosp_temporary_quantities(Y, atmos.cosp)...,
        ᶠtemp_field_level = Fields.level(Fields.Field(FT, face_space), half),
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
        # DSS buffer for the element-filtered gradient invariant of the geometric SGS
        # variance (`hgrad_invariant!`); `nothing` on spaces that need no DSS (single columns).
        ᶜscalar_dss_buffer = (uses_covariances(atmos) && do_dss(center_space)) ?
                             Spaces.create_dss_buffer(Fields.Field(FT, center_space)) :
                             nothing,
        ᶜtemp_C3 = Fields.Field(C3{FT}, center_space), # ᶜ∇Φ₃
        ᶜtemp_CT3 = Fields.Field(CT3{FT}, center_space), # ᶜω³, ᶜ∇Φ³
        ᶜtemp_CT123 = Fields.Field(CT123{FT}, center_space),
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶠtemp_CT12 = Fields.Field(CT12{FT}, face_space), # ᶠω¹²
        ᶠtemp_CT12ʲs = Fields.Field(
            NTuple{n_mass_flux_subdomains(atmos.turbconv_model), CT12{FT}},
            face_space,
        ), # ᶠω¹²ʲs
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
        # Implicit solver cache:
        ᶠp_grad_matrix = similar(Y.f, BidiagonalMatrixRow{C3{FT}}),
        ᶠsed_tracer_advection = similar(
            Y.f,
            ClimaCore.MatrixFields.BandMatrixRow{
                ClimaCore.Utilities.PlusHalf{Int64}(0),
                1,
                FT,
            },
        ),
        ᶠbidiagonal_matrix_ct3_2 = similar(Y.f, BidiagonalMatrixRow{CT3{FT}}),
        ᶜdiffusion_h_matrix = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶜdiffusion_u_matrix = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶜtridiagonal_matrix_scalar = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶠtridiagonal_matrix_c3 = similar(Y.f, TridiagonalMatrixRow{C3{FT}}),
        (!isnothing(atmos.prescribed_flow) ? (; temp_Yₜ_imp = similar(Y)) : (;))...,
    )
end
