using ClimaCore: Fields
using ClimaCore.Utilities: half
using ClimaCore.MatrixFields
using ClimaCore

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
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶜtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜτ_smag in the implicit Smagorinsky tendency
        ᶠtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠstrain_rate
    )
end

"""
    temporary_quantities(
        Y, atmos;
        n_tableau_stages = 8,
        post_step_hook! = Returns(nothing),
    )

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

# Keyword Arguments

  - `n_tableau_stages = 8`: Number of stages in the IMEX tableau.
  - `post_step_hook! = Returns(nothing)`: Optional callable invoked inside
    `step_post_ark_physics!`.

# Returns

A `NamedTuple` of scratch fields keyed by name.
"""
function temporary_quantities(
    Y,
    atmos;
    n_tableau_stages::Int = 8,
    post_step_hook! = Returns(nothing),
)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    n_stage_bufs = max(1, n_tableau_stages)
    return (;
        post_step = PostStepPhysicsContext(;
            hook! = post_step_hook!,
            state_type = typeof(Y),
        ),
        ᶠF̃ = Fields.Field(CT3{FT}, face_space),
        temp_sfc_uₕ_level = atmos.turbconv_model isa PrognosticEDMFX ?
                            Fields.level(
            Fields.Field(typeof(C3(FT(0)) ⊗ C12(FT(0), FT(0))), center_space),
            1,
        ) : (),
        coupled_1m_compact =
        let ρ_data = Fields.field_values(Y.c.ρ)
            if (ClimaComms.device(Y.c) isa ClimaComms.CUDADevice) &&
               (atmos.microphysics_model isa NonEquilibriumMicrophysics1M) &&
               !(atmos.turbconv_model isa PrognosticEDMFX) &&
               (ρ_data isa DataLayouts.VIJHWithF) &&
               (DataLayouts.nelems(ρ_data) > 0)
                _build_coupled_1m_compact_workspace(ρ_data)
            else
                nothing
            end
        end,
        ᶠstage_entry_χ_e = ntuple(
            _ -> Fields.Field(FT, face_space),
            Val(n_stage_bufs),
        ),
        stage_entry_χ_e_mask = Base.RefValue(UInt16(0)),
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠp in the orographic gravity wave drag
        ᶠtemp_scalar_2 = Fields.Field(FT, face_space), # ᶠK of the implicit-stage u₃ solve
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜ1
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜtke_exch
        ᶜtemp_scalar_3 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_4 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_5 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_6 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_7 = Fields.Field(FT, center_space), # ᶜq_precip of the environment in the cloud fraction
        cosp_temporary_quantities(Y, atmos.cosp)...,
        ᶠtemp_field_level = Fields.level(Fields.Field(FT, face_space), half),
        temp_field_level = Fields.level(Fields.Field(FT, center_space), 1),
        temp_field_level_2 = Fields.level(Fields.Field(FT, center_space), 1),
        temp_field_level_3 = Fields.level(Fields.Field(FT, center_space), 1),
        temp_data = Fields.field_values(Fields.Field(FT, center_space)),
        ᶜtemp_C12 = Fields.Field(C12{FT}, center_space), # ᶜuₕ_mean
        # DSS buffer for scalar center fields (used by `hgrad_invariant!` and
        # `vertical_transport_step.jl`), plus two extra buffers so the three
        # horizontal-gradient invariants in
        # `set_covariance_cache_and_cloud_fraction!` batch into one
        # `Spaces.weighted_dss!`; `nothing` when no DSS is needed.
        ᶜscalar_dss_buffer = do_dss(center_space) ?
                             Spaces.create_dss_buffer(Fields.Field(FT, center_space)) :
                             nothing,
        ᶜscalar_dss_buffer_2 = (uses_covariances(atmos) && do_dss(center_space)) ?
                               Spaces.create_dss_buffer(Fields.Field(FT, center_space)) :
                               nothing,
        ᶜscalar_dss_buffer_3 = (uses_covariances(atmos) && do_dss(center_space)) ?
                               Spaces.create_dss_buffer(Fields.Field(FT, center_space)) :
                               nothing,
        ᶜtemp_C3 = Fields.Field(C3{FT}, center_space), # ᶜ∇Φ₃
        ᶜtemp_CT123 = Fields.Field(CT123{FT}, center_space),
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶠtemp_CT12 = Fields.Field(CT12{FT}, face_space), # ᶠω¹²
        ᶠtemp_CT12ʲs = Fields.Field(
            NTuple{n_mass_flux_subdomains(atmos.turbconv_model), CT12{FT}},
            face_space,
        ), # ᶠω¹²ʲs
        ᶠtemp_C123 = Fields.Field(C123{FT}, face_space), # χ₁₂₃
        ᶜtemp_UVW = Fields.Field(typeof(uvw_vec), center_space), # ᶜΔu, vertical velocity differences
        ᶠtemp_UVW = Fields.Field(typeof(uvw_vec), face_space), # ᶠΔu, vertical velocity differences
        ᶜtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜstrain_rate
        ᶠtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠτ_smag
        ᶜtemp_strain = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜstrain_rate
        ᶠtemp_strain = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠstrain_rate
    )
end
