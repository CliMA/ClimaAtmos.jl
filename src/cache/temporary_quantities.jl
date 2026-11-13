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
    c_lev1_space = axes(Fields.level(Y.c, 1))
    PrecipLinNT = @NamedTuple{
        w_lcl_jac::FT,
        w_icl_jac::FT,
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
        c_lev1_uvw = Fields.Field(UVW{FT}, c_lev1_space),
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
    temporary_quantities(
        Y, atmos;
        b_imp_stages = (), n_imp_stages = 0,
        compensate_explicit_scalar_advection = true,
    )

Allocate the scratch fields stored in `p.scratch`.

These hold values that are used within a single function but cannot be computed
on the fly. Unlike the precomputed quantities, they carry no persistent state:
any function may overwrite any scratch field, so a caller must never assume a
scratch field is unchanged across a call. Using scratch (or lazy broadcasting)
is how tendencies and cache setters avoid allocating `Field`s in the hot path.
The exception is the stage-persistent precipitation surface-flux accumulator
(`precipitation_surface_flux_accumulator_quantities`).

Fields are named by location and type: a `ᶜ`/`ᶠ` prefix for cell centers/faces,
then the element type (`scalar`, `C3`, `CT3`, `UVWxUVW`), with a trailing `s`
for a tuple of values per point. Entries whose names end in `_data` or `_level`
are `Fields.field_values` or single levels rather than full fields. The trailing
comment on each entry records a representative use.

# Keyword Arguments

  - `b_imp_stages = ()`: Stage weights of the IMEX implicit tableau, forwarded to
    `precipitation_surface_flux_accumulator_quantities`.
  - `n_imp_stages = 0`: Number of implicit tableau stages (`0` disables the
    precipitation accumulator).
  - `compensate_explicit_scalar_advection = true`: Whether post-Newton implicit
    dissipation weights by `1 + 2 |u³| dtγ` (`_dissipation_weight`).

# Returns

A `NamedTuple` of scratch fields keyed by name.
"""
function temporary_quantities(
    Y,
    atmos;
    b_imp_stages = (),
    n_imp_stages = 0,
    compensate_explicit_scalar_advection::Bool = true,
)
    center_space, face_space = axes(Y.c), axes(Y.f)

    FT = Spaces.undertype(center_space)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    return (;
        compensate_explicit_scalar_advection,
        ᶠtemp_scalar = Fields.Field(FT, face_space), # ᶠp in the orographic gravity wave drag
        ᶠtemp_scalar_2 = Fields.Field(FT, face_space), # ᶠK of the implicit-stage u₃ solve
        ᶜtemp_scalar = Fields.Field(FT, center_space), # ᶜ1
        ᶜtemp_scalar_2 = Fields.Field(FT, center_space), # ᶜtke_exch
        ᶜtemp_scalar_3 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_4 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_5 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_6 = Fields.Field(FT, center_space),
        ᶜtemp_scalar_7 = Fields.Field(FT, center_space), # ᶜq_precip of the environment in the cloud fraction
        precipitation_surface_flux_accumulator_quantities(
            Y,
            atmos,
            b_imp_stages,
            n_imp_stages,
        )...,
        cosp_temporary_quantities(Y, atmos.cosp)...,
        ᶠtemp_field_level = Fields.level(Fields.Field(FT, face_space), half),
        temp_field_level = Fields.level(Fields.Field(FT, center_space), 1),
        temp_data = Fields.field_values(Fields.Field(FT, center_space)),
        # DSS buffer for the element-filtered gradient invariant of the geometric SGS
        # variance (`hgrad_invariant!`); `nothing` on spaces that need no DSS (single columns).
        ᶜscalar_dss_buffer = (uses_covariances(atmos) && do_dss(center_space)) ?
                             Spaces.create_dss_buffer(Fields.Field(FT, center_space)) :
                             nothing,
        ᶠtemp_CT3 = Fields.Field(CT3{FT}, face_space), # ᶠuₕ³
        ᶠtemp_CT12 = Fields.Field(CT12{FT}, face_space), # ᶠω¹²ₕ, the curl of u₃
        ᶠtemp_CT12ʲs = Fields.Field(
            NTuple{n_mass_flux_subdomains(atmos.turbconv_model), CT12{FT}},
            face_space,
        ), # ᶠω¹²ₕʲs, the curls of the draft u₃
        ᶜtemp_UVW = Fields.Field(typeof(uvw_vec), center_space), # ᶜΔu, vertical velocity differences
        ᶠtemp_UVW = Fields.Field(typeof(uvw_vec), face_space), # ᶠΔu, vertical velocity differences
        ᶜtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), center_space), # ᶜstrain_rate
        ᶠtemp_UVWxUVW = Fields.Field(typeof(uvw_vec * uvw_vec'), face_space), # ᶠτ_smag
        # Post-Newton implicit upwind dissipation
        # (`add_vertical_advection_corrections!`): the system `I - dtγ 𝒟` and
        # the tridiagonal solver cache.
        ᶜupwind_dissipation_system = similar(Y.c, TridiagonalMatrixRow{FT}),
        ᶜupwind_dissipation_solver_cache = MatrixFields.single_field_solver_cache(
            similar(Y.c, TridiagonalMatrixRow{FT}),
            Fields.Field(FT, center_space),
        ),
    )
end
