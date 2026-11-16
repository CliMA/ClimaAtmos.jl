#####
##### Precomputed quantities
#####
import Thermodynamics as TD
import ClimaCore: DataLayouts, Spaces, Fields

"""
    smagorinsky_lilly_fields(Y, model)

Allocate the precomputed fields of the Smagorinsky-Lilly closure: the center and
face strain-rate tensors `ᶜS`/`ᶠS`, the horizontal and vertical strain-rate norms,
mixing lengths, eddy viscosities, and eddy diffusivities. Return an empty
`NamedTuple` when `model` is `nothing`. The fields live with the explicit
precomputed quantities, since the wave-core implicit stage does not read them.
"""
smagorinsky_lilly_fields(Y, ::Nothing) = (;)
function smagorinsky_lilly_fields(Y, ::SmagorinskyLilly)
    FT = eltype(Y)
    uvw_vec = UVW(FT(0), FT(0), FT(0))
    return (;
        ᶜS = similar(Y.c, typeof(uvw_vec * uvw_vec')),
        ᶠS = similar(Y.f, typeof(uvw_vec * uvw_vec')),
        ᶜS_norm_h = similar(Y.c, FT), ᶜS_norm_v = similar(Y.c, FT),
        ᶜL_h = similar(Y.c, FT), ᶜL_v = similar(Y.c, FT),
        ᶜνₜ_h = similar(Y.c, FT), ᶜνₜ_v = similar(Y.c, FT),
        ᶜD_h = similar(Y.c, FT), ᶜD_v = similar(Y.c, FT),
    )
end

"""
    implicit_precomputed_quantities(Y, atmos)

Allocate the precomputed quantities that are treated implicitly, i.e., updated
on each iteration of the implicit solver. This includes all velocity and
thermodynamic quantities used in the implicit tendency.

# Returns

A `NamedTuple` of fields containing, on the grid scale:

  - `ᶜu`: Velocity on cell centers, in the local covariant basis
    (`Covariant123Vector`).
  - `ᶠu³`: Third contravariant component of the velocity on cell faces [1/s].
  - `ᶠu`: Velocity on cell faces, in the local contravariant basis
    (`Contravariant123Vector`).
  - `ᶜK`: Specific kinetic energy on cell centers [m²/s²].
  - `ᶜT`: Air temperature on cell centers [K].
  - `ᶜh_tot`: Total specific enthalpy on cell centers [J/kg].
  - `ᶜp`: Air pressure on cell centers [Pa].
  - `ᶜq_tot_nonneg`: Total water specific humidity, clipped to ≥ 0 [kg/kg].
  - `ᶜq_liq`: Liquid water specific humidity (cloud liquid + rain), clipped to
    ≥ 0 [kg/kg].
  - `ᶜq_ice`: Ice water specific humidity (cloud ice + snow), clipped to ≥ 0 [kg/kg].
  - `ᶜsa_result`: Saturation-adjustment result `(; T, q_liq, q_ice)`; only for
    `EquilibriumMicrophysics0M`.

If the `turbconv_model` is `PrognosticEDMFX`, there are also SGS versions of the
velocity and thermodynamic quantities (`ᶜp` is shared across all subdomains):

  - `_⁰` suffix: value for the environment (`ᶠu₃⁰`, `ᶜu⁰`, `ᶠu³⁰`, `ᶜK⁰`, `ᶜT⁰`,
    `ᶜq_tot_nonneg⁰`, `ᶜq_liq⁰`, `ᶜq_ice⁰`).
  - `ʲs` suffix: tuples of values for the mass-flux subdomains (`ᶜuʲs`, `ᶠu³ʲs`,
    `ᶜKʲs`, `ᶠKᵥʲs`, `ᶜTʲs`, `ᶜq_tot_nonnegʲs`, `ᶜq_liqʲs`, `ᶜq_iceʲs`), plus the
    subdomain air densities `ᶜρʲs` [kg/m³].

For `NonEquilibriumMicrophysics1M` with covariance quadrature, `ᶜsgs_moments` is
also allocated here so autodiff gets Dual-typed copies.
"""
function implicit_precomputed_quantities(Y, atmos)
    (; microphysics_model, turbconv_model) = atmos
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)
    gs_quantities = (;
        ᶜu = similar(Y.c, C123{FT}),
        ᶠu³ = similar(Y.f, CT3{FT}),
        ᶠu = similar(Y.f, CT123{FT}),
        ᶜK = similar(Y.c, FT),
        ᶜT = similar(Y.c, FT),
        ᶜh_tot = similar(Y.c, FT),
        ᶜp = similar(Y.c, FT),
    )
    # Moisture-related quantities depend on microphysics model:
    # - EquilibriumMicrophysics0M: allocate fields + thermo_state cache for saturation adjustment
    # - Others: allocate fields only
    sa_result_type = @NamedTuple{T::FT, q_liq::FT, q_ice::FT}
    moist_gs_quantities =
        if microphysics_model isa EquilibriumMicrophysics0M
            (;
                ᶜq_tot_nonneg = similar(Y.c, FT),
                ᶜq_liq = similar(Y.c, FT),
                ᶜq_ice = similar(Y.c, FT),
                ᶜsa_result = similar(Y.c, sa_result_type),
            )
        else  # DryModel or NonEquilibriumMicrophysics
            (;
                ᶜq_tot_nonneg = similar(Y.c, FT),
                ᶜq_liq = similar(Y.c, FT),
                ᶜq_ice = similar(Y.c, FT),
            )
        end
    sgs_quantities = (;)
    prognostic_sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶠu₃⁰ = similar(Y.f, C3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜK⁰ = similar(Y.c, FT),
            ᶜT⁰ = similar(Y.c, FT),
            ᶜq_tot_nonneg⁰ = similar(Y.c, FT),
            ᶜq_liq⁰ = similar(Y.c, FT),
            ᶜq_ice⁰ = similar(Y.c, FT),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶠKᵥʲs = similar(Y.f, NTuple{n, FT}),
            ᶜTʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_tot_nonnegʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_liqʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_iceʲs = similar(Y.c, NTuple{n, FT}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
        ) : (;)

    # 1M SGS moments: `ᶜsgs_moments` is updated from the current state during
    # covariance setup, so allocate it here for Dual-typed copies.
    SGSMomentsNT = @NamedTuple{sigma_S::FT, λ_lagrange::FT, λ_eq::FT}
    implicit_1m_moments =
        uses_covariances(atmos) &&
        microphysics_model isa NonEquilibriumMicrophysics1M ?
        (; ᶜsgs_moments = similar(Y.c, SGSMomentsNT)) : (;)

    return (;
        gs_quantities...,
        moist_gs_quantities...,
        sgs_quantities...,
        prognostic_sgs_quantities...,
        implicit_1m_moments...,
    )
end

"""
    precomputed_quantities(Y, atmos)

Allocate all precomputed quantities, i.e., the contents of `p.precomputed`.

This includes the quantities treated implicitly (updated before each tendency
evaluation; see `implicit_precomputed_quantities`) and the quantities treated
explicitly (updated only before explicit tendency evaluations). Which fields
are allocated depends on the model configuration: microphysics scheme,
turbulence-convection model, SGS quadrature, COSP, and LES closures.

# Returns

A `NamedTuple` of fields keyed by quantity name.

TODO: Reduce the number of cached values by computing them on the fly.
"""
function precomputed_quantities(Y, atmos)
    FT = eltype(Y)
    @assert !(atmos.microphysics_model isa DryModel) ||
            !(atmos.turbconv_model isa PrognosticEDMFX)
    @assert isnothing(atmos.turbconv_model) ||
            isnothing(atmos.vertical_diffusion)
    SCT = SurfaceConditions.surface_conditions_type(atmos, FT)
    cspace = axes(Y.c)
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    n_prog = n_prognostic_mass_flux_subdomains(atmos.turbconv_model)
    @assert !(atmos.turbconv_model isa PrognosticEDMFX) || n_prog == 1
    gs_quantities = (;
        ᶜwₕhₜ = Geometry.WVector.(zeros(cspace)),
        # Moist buoyancy gradient N² at centers; same physical quantity as the
        # face-native `ᶠK`-pipeline `ᶠbuoygrad`, built from the centered
        # (cloud-fraction-blended) vertical gradient instead of the two-point
        # face gradient.
        ᶜbuoygrad = similar(Y.c, FT),
        # Pointwise chain-rule coefficients of the moist buoyancy gradient
        # and exact two-point face gradients of (θ_li, q_tot); filled once
        # per update by `set_buoyancy_gradient_inputs!` and shared by the
        # centered, one-sided, and face-native buoyancy-gradient stencils.
        ᶜbg_coeffs = similar(
            Y.c,
            @NamedTuple{Cθ_unsat::FT, ΔCθ::FT, Cq_unsat::FT, ΔCq::FT}
        ),
        ᶠ∂θli∂z = similar(Y.f, FT),
        ᶠ∂qt∂z = similar(Y.f, FT),
        # Face-native moist buoyancy gradient, face-native eddy diffusivity/
        # viscosity, and the master mixing length at centers. Every consumer is an
        # AbstractEDMF path, so they are allocated only for AbstractEDMF;
        # other closures use the center ᶜK_h/ᶜK_u instead.
        # Evaluating the stability closure at the faces, where the fluxes
        # live, keeps the collapse of K at an unresolved inversion from
        # leaking to the adjacent interior face.
        #
        # All three face fields are written by `set_face_diffusivities!` on
        # every explicit update, and ᶜl_mix by `set_covariance_cache!` or the
        # explicit stage, before any read, so `similar` is safe.
        (
            atmos.turbconv_model isa AbstractEDMF ?
            (;
                ᶠbuoygrad = similar(Y.f, FT),
                ᶠK_h = similar(Y.f, FT),
                ᶠK_u = similar(Y.f, FT),
                ᶜl_mix = similar(Y.c, FT),
                # Horizontal eddy viscosity/diffusivity of the TKE-based
                # closure, with the mixing length limited by the horizontal
                # node spacing; written by `set_horizontal_diffusivities!`.
                (
                    atmos.edmfx_model.sgs_diffusive_flux_horizontal isa
                    Val{true} ||
                    atmos.edmfx_model.horizontal_diffusion isa Val{true} ?
                    (;
                        ᶜK_u_h = similar(Y.c, FT),
                        ᶜK_h_h = similar(Y.c, FT),
                    ) : (;)
                )...,
            ) : (;)
        )...,
        ᶜstrain_rate_norm = similar(Y.c, FT),
        sfc_conditions = similar(Spaces.level(Y.f, half), SCT),
    )
    # Cloud fraction is used to calculate buoyancy gradient, so we initialize it to 0 here.
    ᶜcloud_fraction = similar(Y.c, FT)
    @. ᶜcloud_fraction = FT(0)

    cosp_quantities = cosp_precomputed_quantities(Y, atmos.cosp)

    # SGS covariances for hybrid cloud fraction and microphysics quadrature.
    # NonEquilibriumMicrophysics1M/2M always route through the quadrature API
    # internally (with GridMeanSGS), so they also need covariance fields
    # allocated. This allocation guard must match the write guard in
    # `set_covariance_cache!` and the ᶜl_mix-caching guard in
    # `set_explicit_precomputed_quantities!`, so all three share the one
    # `uses_covariances` predicate.
    uses_sgs_quadrature = uses_covariances(atmos)
    uses_microphysics_quadrature_moments =
        atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
    # `ᶜsgs_moments` caches `(sigma_S, λ_lagrange, λ_eq)` — the SGS standard
    # deviation and the actual and equilibrium Lagrange multipliers used by
    # `Microphysics1MEvaluator`. Allocated only for 1M/2M schemes, together
    # with `ᶜprecip_frac`, the overlap precipitation fraction
    # (`set_precip_fraction!`).
    SGSMomentsNT = @NamedTuple{sigma_S::FT, λ_lagrange::FT, λ_eq::FT}
    covariance_quantities = if uses_sgs_quadrature
        base = (;
            # σ²_T and σ²_q of the subgrid displacement acting on the
            # resolved gradients (`set_covariance_cache!`).
            ᶜT′T′ = zeros(axes(Y.c)),
            ᶜq′q′ = zeros(axes(Y.c)),
            # T–q correlation the quadrature actually samples (filled by
            # `set_covariance_cache!`): `T′q′ / √(T′T′ q′q′)` for
            # `tq_correlation_model: diagnosed`, or the prescribed
            # `Tq_correlation_coefficient` for `tq_correlation_model: constant`.
            # The displacement covariance `ᶜT′q′` is only allocated for the
            # diagnosed model.
            ᶜcorr_Tq = zeros(axes(Y.c)),
            # Local SGS variance fidelity (see `sgs_variance_fidelity`),
            # refreshed inside `set_covariance_cache!`.
            ᶜα = zeros(axes(Y.c)),
            (
                atmos.tq_correlation_model isa DiagnosedTqCorrelation ?
                (; ᶜT′q′ = zeros(axes(Y.c))) : (;)
            )...,
            # Linear and curvature coefficients `(lin, curv)` of the saturation
            # excess under the vertical displacement, `S = μ_S + lin x +
            # ½ curv (x² − 1) + h` with `x` standard normal
            # (`displacement_curvature_coefficients`, set in
            # `set_covariance_cache!`); only for the curvature-aware
            # equilibrium condensate (`sgs_variance_curvature`).
            (
                atmos.sgs_variance_curvature ?
                (;
                    ᶜdisp_S = fill!(
                        similar(Y.c, @NamedTuple{lin::FT, curv::FT}),
                        (; lin = zero(FT), curv = zero(FT)),
                    ),
                ) : (;)
            )...,
        )
        uses_microphysics_quadrature_moments ?
        (;
            base...,
            # `ᶜsgs_moments` for 1M lives in `implicit_precomputed_quantities`
            # so autodiff gets Dual copies. Allocate here only for 2M.
            (
                atmos.microphysics_model isa NonEquilibriumMicrophysics2M ?
                (; ᶜsgs_moments = similar(Y.c, SGSMomentsNT)) : (;)
            )...,
            ᶜprecip_frac = zeros(axes(Y.c)),
            # The cloud fraction of the PrognosticEDMFX environment before the
            # updraft weighting, for the clear-air buoyancy reference
            # (`set_updraft_reference_density_excess!`).
            (
                atmos.turbconv_model isa PrognosticEDMFX &&
                atmos.microphysics_model isa NonEquilibriumMicrophysics1M ?
                (; ᶜcloud_fraction⁰ = zeros(axes(Y.c))) : (;)
            )...,
        ) : base
    else
        (;)
    end
    surface_precip_fluxes = (;
        surface_rain_flux = zeros(axes(Fields.level(Y.f, half))),
        surface_snow_flux = zeros(axes(Fields.level(Y.f, half))),
        (
            !(atmos.microphysics_model isa DryModel) ?
            (;
                col_integrated_precip_energy_tendency = zeros(
                    axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
                ),
            ) : (;)
        )...,
    )
    sedimentation_quantities =
        atmos.microphysics_model isa NonEquilibriumMicrophysics ?
        (; ᶜwₗ = similar(Y.c, FT), ᶜwᵢ = similar(Y.c, FT)) : (;)

    if atmos.microphysics_model isa EquilibriumMicrophysics0M
        # Step means of the 0M precipitation mass and energy tendencies
        # [kg m⁻³ s⁻¹] and [W m⁻³], below-cloud evaporation of the precipitation
        # flux [kg m⁻³ s⁻¹], its column integrals, and the column-integrated
        # updraft precipitation [kg m⁻² s⁻¹], written by
        # `step_coupled_microphysics_sedimentation!`. The evaporation stays
        # zero when it is off.
        ᶜprecip_evap = similar(Y.c, @NamedTuple{ρ_evap_rai::FT, ρ_evap_sno::FT})
        fill!(parent(ᶜprecip_evap), 0)
        precipitation_quantities = (;
            ᶜρ_dq_tot_dt = zeros(axes(Y.c)),
            ᶜρ_de_tot_dt = zeros(axes(Y.c)),
            ᶜprecip_evap,
            col_integrated_precip_evap_rai = zeros(axes(Fields.level(Y.f, half))),
            col_integrated_precip_evap_sno = zeros(axes(Fields.level(Y.f, half))),
            col_integrated_updraft_precip = zeros(axes(Fields.level(Y.f, half))),
        )
    elseif atmos.microphysics_model isa NonEquilibriumMicrophysics1M
        precipitation_quantities = (;
            ᶜwₗ = similar(Y.c, FT),
            ᶜwᵢ = similar(Y.c, FT),
            ᶜwᵣ = similar(Y.c, FT),
            ᶜwₛ = similar(Y.c, FT),
        )
    elseif atmos.microphysics_model isa
           Union{NonEquilibriumMicrophysics2M, NonEquilibriumMicrophysics2MP3}
        precipitation_quantities = (;
            ᶜwₗ = similar(Y.c, FT),
            ᶜwᵢ = similar(Y.c, FT),
            ᶜwᵣ = similar(Y.c, FT),
            ᶜwₛ = similar(Y.c, FT),
            ᶜwₙₗ = similar(Y.c, FT),
            ᶜwₙᵣ = similar(Y.c, FT),
        )
        if atmos.microphysics_model isa NonEquilibriumMicrophysics2MP3
            precipitation_quantities = (;
                precipitation_quantities...,
                ᶜwnᵢ = similar(Y.c, FT),
            )
        end
    else
        precipitation_quantities = (;)
    end
    if atmos.microphysics_model isa NonEquilibriumMicrophysics1M
        precipitation_sgs_quantities = (;
            ᶜwₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵢʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵣʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₛʲs = similar(Y.c, NTuple{n, FT}),
        )
    elseif atmos.microphysics_model isa NonEquilibriumMicrophysics2M
        precipitation_sgs_quantities = (;
            ᶜwₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵢʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵣʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₛʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₙₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₙᵣʲs = similar(Y.c, NTuple{n, FT}),
        )
    else
        precipitation_sgs_quantities = (;)
    end

    advective_sgs_quantities =
        atmos.turbconv_model isa PrognosticEDMFX ?
        (;
            ρtke_flux = similar(Fields.level(Y.f, half), C3{FT}),
            ᶜentr_vel_scaleʲs = similar(Y.c, NTuple{n, FT}),
            ᶜentr_nonvel_rateʲs = similar(Y.c, NTuple{n, FT}),
            ᶜturb_entrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜarea_bounding_entr_detrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜρ_diffʲs = similar(Y.c, NTuple{n, FT}),
            # Excess of the updraft reference density over the grid mean
            # (`set_updraft_reference_density_excess!`).
            ᶜρ_ref_excess = similar(Y.c, FT),
            # Per-updraft surface conditions at level 1.
            sfc_mass_flux_sourceʲs = similar(Fields.level(Y.c, 1), NTuple{n, FT}),
            sfc_mse_buoyantʲs = similar(Fields.level(Y.c, 1), NTuple{n, FT}),
            sfc_q_tot_buoyantʲs = similar(Fields.level(Y.c, 1), NTuple{n, FT}),
            precipitation_sgs_quantities...,
        ) : (;)

    edonly_quantities =
        atmos.turbconv_model isa EDOnlyEDMFX ?
        (; ρtke_flux = similar(Fields.level(Y.f, half), C3{FT}),) : (;)

    # Gradient fields for covariance computation (used in cloud fraction/microphysics)
    sgs_quantities = (;
        ᶜgradᵥ_q_tot = Fields.Field(C3{FT}, cspace),
        ᶜgradᵥ_θ_liq_ice = Fields.Field(C3{FT}, cspace),
    )

    smagorinsky_lilly_quantities =
        smagorinsky_lilly_fields(Y, atmos.smagorinsky_lilly)
    return (;
        implicit_precomputed_quantities(Y, atmos)...,
        gs_quantities...,
        sgs_quantities...,
        advective_sgs_quantities...,
        edonly_quantities...,
        sedimentation_quantities...,
        precipitation_quantities...,
        surface_precip_fluxes...,
        ᶜcloud_fraction,
        cosp_quantities...,
        covariance_quantities...,
        smagorinsky_lilly_quantities...)
end

"""
    compute_ᶠuₕ³(ᶜuₕ, ᶜρ)

Return a lazy broadcast of the third contravariant component of the horizontal
velocity `ᶜuₕ`, interpolated from centers to faces with the mass-weighted
interpolation `ᶠwinterp(ρ J, ·)` [1/s].

This component is nonzero wherever the coordinate surfaces are sloped, and so
enters the impenetrability conditions applied by `set_velocity_at_surface!` and
`set_velocity_at_top!`.
"""
function compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    return @. lazy(ᶠwinterp(ᶜρ * ᶜJ, CT3(ᶜuₕ)))
end

"""
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)

Modify `Y.f.u₃` so that `ᶠu³` is 0 at the surface. Specifically, since
`u³ = uₕ³ + u³ = uₕ³ + u₃ * g³³`, setting `u³` to 0 gives `u₃ = -uₕ³ / g³³`. If
the `turbconv_model` is `PrognosticEDMFX`, the `Y.f.sgsʲs` are also modified so
that each `u₃ʲ` is equal to `u₃` at the surface. Returns `nothing`.
"""
function set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
    sfc_u₃ = Fields.level(Y.f.u₃.components.data.:1, half)
    sfc_u₃ .= surface_velocity(Y.f.u₃, ᶠuₕ³)
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            sfc_u₃ʲ = Fields.level(Y.f.sgsʲs.:($j).u₃.components.data.:1, half)
            @. sfc_u₃ʲ = sfc_u₃
        end
    end
    return nothing
end

"""
    surface_velocity(ᶠu₃, ᶠuₕ³)

Return a lazy broadcast of the covariant vertical velocity component at the
surface that makes the contravariant vertical velocity vanish there,
`u₃ = -uₕ³ / g³³`. Called from `set_velocity_at_surface!`.
"""
function surface_velocity(ᶠu₃, ᶠuₕ³)
    sfc_u₃ = Fields.level(ᶠu₃.components.data.:1, half)
    sfc_uₕ³ = Fields.level(ᶠuₕ³.components.data.:1, half)
    sfc_g³³ = g³³_field(axes(sfc_u₃))
    return @. lazy(-sfc_uₕ³ / sfc_g³³) # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
end

"""
    top_velocity(ᶠu₃, ᶠuₕ³)

Return a lazy broadcast of the covariant vertical velocity component at the model
top that makes the contravariant vertical velocity vanish there,
`u₃ = -uₕ³ / g³³`. Called from `set_velocity_at_top!`.
"""
function top_velocity(ᶠu₃, ᶠuₕ³)
    top_level = Spaces.nlevels(axes(ᶠu₃)) - half
    top_u₃ = Fields.level(ᶠu₃.components.data.:1, top_level)
    top_uₕ³ = Fields.level(ᶠuₕ³.components.data.:1, top_level)
    top_g³³ = g³³_field(axes(top_u₃))
    return @. lazy(-top_uₕ³ / top_g³³) # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
end

"""
    set_velocity_at_top!(Y, ᶠuₕ³, turbconv_model)

Modify `Y.f.u₃` so that `ᶠu³` is 0 at the model top. As at the surface,
since `u³ = uₕ³ + u₃ * g³³`, setting `u³` to 0 gives `u₃ = -uₕ³ / g³³`. This
makes the total contravariant flux through the top boundary vanish even where
terrain-following coordinate surfaces are still sloped at the model top
(`g³ʰ ≠ 0`, so `uₕ³ ≠ 0`). If the `turbconv_model` is `PrognosticEDMFX`, the
`Y.f.sgsʲs` are also modified so that each `u₃ʲ` is equal to `u₃` at the model
top. Returns `nothing`.
"""
function set_velocity_at_top!(Y, ᶠuₕ³, turbconv_model)
    top_u₃ = Fields.level(
        Y.f.u₃.components.data.:1,
        Spaces.nlevels(axes(Y.c)) + half,
    )
    top_u₃ .= top_velocity(Y.f.u₃, ᶠuₕ³)
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            top_u₃ʲ = Fields.level(
                Y.f.sgsʲs.:($j).u₃.components.data.:1,
                Spaces.nlevels(axes(Y.c)) + half,
            )
            @. top_u₃ʲ = top_u₃
        end
    end
    return nothing
end

"""
    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠu₃, ᶜuₕ, ᶠuₕ³)

Set the velocity `ᶜu` at centers, the contravariant vertical velocity `ᶠu³` at
faces, and the specific kinetic energy `ᶜK` at centers, from the vertical
velocity `ᶠu₃` and the horizontal velocity `ᶜuₕ`. Mutates its first three
arguments and returns `nothing`.

Used both for the grid-scale quantities (from `Y.f.u₃`) and for the EDMFX
subdomain quantities (from `ᶠu₃⁰` and each `ᶠu₃ʲ`).
"""
@inline function set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠu₃, ᶜuₕ, ᶠuₕ³)
    @. ᶜu = C123(ᶜuₕ) + ᶜinterp(C123(ᶠu₃))
    @. ᶠu³ = ᶠuₕ³ + CT3(ᶠu₃)
    ᶜK .= compute_kinetic(ᶜuₕ, ᶠu₃)
    return nothing
end

"""
    set_enthalpy_and_pressure!(ᶜh_tot, ᶜp, Yc, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, thermo_params)

Set the total specific enthalpy `ᶜh_tot` and the pressure `ᶜp` from the state
`Yc` (`ρ`, `ρe_tot`), the temperature, and the water species. Also runs inside
the fused column loop of `set_implicit_precomputed_quantities!`, so it is
inlined.
"""
@inline function set_enthalpy_and_pressure!(
    ᶜh_tot,
    ᶜp,
    Yc,
    ᶜT,
    ᶜq_tot_nonneg,
    ᶜq_liq,
    ᶜq_ice,
    thermo_params,
)
    ᶜe_tot = @inline @. lazy(specific(Yc.ρe_tot, Yc.ρ))
    @. ᶜh_tot =
        TD.total_enthalpy(
            thermo_params,
            ᶜe_tot,
            ᶜT,
            ᶜq_tot_nonneg,
            ᶜq_liq,
            ᶜq_ice,
        )
    @. ᶜp = TD.air_pressure(
        thermo_params,
        ᶜT,
        Yc.ρ,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    )
    return nothing
end

"""
    set_sgs_ᶠu₃!(w_function, ᶠu₃, Y, turbconv_model)

Set the SGS vertical velocity `ᶠu₃` at faces by applying `w_function` to the
face-interpolated subdomain area-weighted densities, the subdomain vertical
velocities, the face-interpolated grid-mean density, and the grid-mean vertical
velocity. Mutates `ᶠu₃` and returns `nothing`.

Passing `u₃⁰` as `w_function` recovers the environment vertical velocity from the
grid mean and the drafts.
"""
set_sgs_ᶠu₃!(w_function, ᶠu₃, Y, turbconv_model) =
    set_sgs_ᶠu₃!(w_function, ᶠu₃, Y.c, Y.f, turbconv_model)
# The (Yc, Yf) method also runs inside fused column loops on column slices of Y.c and Y.f.
@inline function set_sgs_ᶠu₃!(w_function, ᶠu₃, Yc, Yf, turbconv_model)
    ρaʲs(sgsʲs) = map(sgsʲ -> sgsʲ.ρa, sgsʲs)
    u₃ʲs(sgsʲs) = map(sgsʲ -> sgsʲ.u₃, sgsʲs)
    @. ᶠu₃ = w_function(
        ᶠinterp(ρaʲs(Yc.sgsʲs)),
        u₃ʲs(Yf.sgsʲs),
        ᶠinterp(Yc.ρ),
        Yf.u₃,
        turbconv_model,
    )
    return nothing
end

"""
    add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠu₃⁰, turbconv_model)

Add the SGS contributions to the grid-mean specific kinetic energy `ᶜK`, i.e. the
area-weighted variance of the environment and draft vertical velocities about the
grid mean. Mutates `ᶜK` and returns `nothing`.

!!! note

    Currently unused: the call in `set_implicit_precomputed_quantities!` is
    commented out because the exact increment depends circularly on `ᶜK` itself
    through the subdomain densities. See the comment at that call site.
"""
function add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠu₃⁰, turbconv_model)
    DataLayouts.foreach_column(Y.c, Y.f, ᶜK, ᶜρa⁰, ᶠu₃⁰) do Yc, Yf, ᶜK, ᶜρa⁰, ᶠu₃⁰
        @. ᶜK += ᶜρa⁰ * ᶜinterp(dot(ᶠu₃⁰ - Yf.u₃, CT3(ᶠu₃⁰ - Yf.u₃))) / 2 / Yc.ρ
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            ᶜρaʲ = Yc.sgsʲs.:($j).ρa
            ᶠu₃ʲ = Yf.sgsʲs.:($j).u₃
            @. ᶜK +=
                ᶜρaʲ * ᶜinterp(dot(ᶠu₃ʲ - Yf.u₃, CT3(ᶠu₃ʲ - Yf.u₃))) / 2 / Yc.ρ
        end
    end
    return nothing
end

"""
    saturation_adjustment_tuple(thermo_params, ::TD.ρe, ρ, e_int, q_tot)

Return `(; T, q_liq, q_ice)` from a density-internal-energy saturation
adjustment.

Bundling the three outputs into one `NamedTuple` keeps `EquilibriumMicrophysics0M`
to a single `TD.saturation_adjustment` call per grid point. The `TD.ph`
counterpart used for the EDMFX subdomains lives in
`prognostic_edmf_precomputed_quantities.jl`.

# Arguments

  - `ρ`: Air density [kg/m³].
  - `e_int`: Specific internal energy [J/kg].
  - `q_tot`: Total water specific humidity [kg/kg].
"""
function saturation_adjustment_tuple(thermo_params, ::TD.ρe, ρ, e_int, q_tot)
    sa_result = TD.saturation_adjustment(thermo_params, TD.ρe(), ρ, e_int, q_tot)
    return (; T = sa_result.T, q_liq = sa_result.q_liq, q_ice = sa_result.q_ice)
end

"""
    eddy_diffusivity_coefficient_H(D₀, H, z_sfc, z)

Return the eddy diffusivity `D₀ exp(-(z - z_sfc) / H)` [m²/s], decaying
exponentially with height above the surface. Called from
`ᶜcompute_eddy_diffusivity_coefficient`.
"""
function eddy_diffusivity_coefficient_H(D₀, H, z_sfc, z)
    return D₀ * exp(-(z - z_sfc) / H)
end

"""
    eddy_diffusivity_coefficient(C_E, norm_v_a, z_a, p)

Return the eddy diffusivity `K_E = C_E |v_a| z_a` [m²/s] inside the planetary
boundary layer, tapered by a Gaussian in pressure above it.

The boundary layer top is fixed at 850 hPa and the taper scale is 100 hPa.
Called from `ᶜcompute_eddy_diffusivity_coefficient`.

# Arguments

  - `C_E`: Bulk transfer coefficient for the vertical diffusion scheme [-].
  - `norm_v_a`: Wind speed at the lowest model level [m/s].
  - `z_a`: Height scale of the lowest model level [m].
  - `p`: Air pressure [Pa].
"""
function eddy_diffusivity_coefficient(C_E, norm_v_a, z_a, p)
    p_pbl = 85000
    p_strato = 10000
    K_E = C_E * norm_v_a * z_a
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end

"""
    _set_implicit_grid_mean_precomputed_quantities!(Y, p, t)

Update the grid-mean implicit precomputed quantities (`ᶜu`, `ᶠu³`, `ᶠu`, `ᶜK`,
`ᶜT`, `ᶜq_tot_nonneg`, `ᶜq_liq`, `ᶜq_ice`, `ᶜh_tot`, `ᶜp`) from state `Y` and
enforce impenetrable top and bottom velocity boundary conditions. Returns `ᶠuₕ³`
stored in `p.scratch.ᶠtemp_CT3`.
"""
NVTX.@annotate function _set_implicit_grid_mean_precomputed_quantities!(Y, p, t)
    (; turbconv_model, microphysics_model) = p.atmos
    (; ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶠu, ᶜK, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜh_tot, ᶜp) =
        p.precomputed
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_params = CAP.thermodynamics_params(p.params)

    @. ᶠuₕ³ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)

    # TODO: We might want to move this to constrain_state!
    if !(p.atmos.prescribed_flow isa PrescribedFlow)
        set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
        set_velocity_at_top!(Y, ᶠuₕ³, turbconv_model)
    end

    ᶜJ = Fields.local_geometry_field(Y.c).J
    # Grid-mean velocities and kinetic energy in one fused column loop.
    DataLayouts.foreach_column(
        ᶜu, ᶠu³, ᶜK, ᶠu, Y.c, Y.f.u₃, ᶠuₕ³, ᶜJ,
    ) do ᶜu, ᶠu³, ᶜK, ᶠu, Yc, ᶠu₃, ᶠuₕ³, ᶜJ
        set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠu₃, Yc.uₕ, ᶠuₕ³)
        @. ᶠu = CT123(ᶠwinterp(Yc.ρ * ᶜJ, CT12(ᶜu))) + CT123(ᶠu³)
    end
    if n > 0
        # TODO: In the following increments to ᶜK, we actually need to add
        # quantities of the form ᶜρaχ⁰ / ᶜρ⁰ and ᶜρaχʲ / ᶜρʲ to ᶜK, rather than
        # quantities of the form ᶜρaχ⁰ / ᶜρ and ᶜρaχʲ / ᶜρ. However, we cannot
        # compute ᶜρ⁰ and ᶜρʲ without first computing ᶜT⁰ and ᶜTʲ, both of
        # which depend on the value of ᶜp, which in turn depends on ᶜT. Since
        # ᶜT depends on ᶜK, this
        # means that the amount by which ᶜK needs to be incremented is a
        # function of ᶜK itself. So, unless we run a nonlinear solver here, this
        # circular dependency will prevent us from computing the exact value of
        # ᶜK. For now, we will make the anelastic approximation ᶜρ⁰ ≈ ᶜρʲ ≈ ᶜρ.
        # add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠu₃⁰, turbconv_model)
        # @. ᶜK += Y.c.ρtke / Y.c.ρ
        # TODO: We should think more about these increments before we use them.
    end
    T_min_sgs = CAP.T_min_sgs(p.params)
    if microphysics_model isa NonEquilibriumMicrophysics
        # Thermodynamic state from the prognostic water species, total specific
        # enthalpy, and pressure in one fused column loop.
        DataLayouts.foreach_column(
            ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜh_tot, ᶜp, Y.c, ᶜK, ᶜΦ,
        ) do ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜh_tot, ᶜp, Yc, ᶜK, ᶜΦ
            # (Named differently from the enclosing function's ᶜe_int: assigning
            # a captured name inside the loop body would box it.)
            ᶜe_int_col = @. lazy(specific(Yc.ρe_tot, Yc.ρ) - ᶜK - ᶜΦ)
            @. ᶜq_liq =
                max(0, specific(Yc.ρq_lcl, Yc.ρ)) +
                max(0, specific(Yc.ρq_rai, Yc.ρ))
            @. ᶜq_ice =
                max(0, specific(Yc.ρq_icl, Yc.ρ)) +
                max(0, specific(Yc.ρq_sno, Yc.ρ))
            @. ᶜq_tot_nonneg =
                max(ᶜq_liq + ᶜq_ice, specific(Yc.ρq_tot, Yc.ρ))
            @. ᶜT = max(
                T_min_sgs,
                TD.air_temperature(
                    thermo_params,
                    ᶜe_int_col,
                    ᶜq_tot_nonneg,
                    ᶜq_liq,
                    ᶜq_ice,
                ),
            )
            set_enthalpy_and_pressure!(
                ᶜh_tot, ᶜp, Yc, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, thermo_params,
            )
        end
    else
        ᶜe_int = @. lazy(specific(Y.c.ρe_tot, Y.c.ρ) - ᶜK - ᶜΦ)
        if microphysics_model isa EquilibriumMicrophysics0M
            # Compute thermodynamic state variables using combined getter function.
            # This avoids redundant saturation_adjustment calls for EquilibriumMicrophysics0M.
            @. ᶜq_tot_nonneg = max(0, specific(Y.c.ρq_tot, Y.c.ρ))
            (; ᶜsa_result) = p.precomputed
            @. ᶜsa_result =
                saturation_adjustment_tuple(
                    thermo_params,
                    TD.ρe(),
                    Y.c.ρ,
                    ᶜe_int,
                    ᶜq_tot_nonneg,
                )
            @. ᶜT = ᶜsa_result.T
            @. ᶜq_liq = ᶜsa_result.q_liq
            @. ᶜq_ice = ᶜsa_result.q_ice

            # Two-pass SGS: recompute condensate using SGS quadrature over (T, q_tot)
            sgs_quad = p.atmos.sgs_quadrature
            if !isnothing(sgs_quad)
                (; ᶜT′T′, ᶜq′q′, ᶜcorr_Tq) = p.precomputed
                # Displacement coefficients of the curvature-aware condensate,
                # zero where it is not configured.
                ᶜlin, ᶜcurv = if p.atmos.sgs_variance_curvature
                    (p.precomputed.ᶜdisp_S.lin, p.precomputed.ᶜdisp_S.curv)
                else
                    (zero(eltype(ᶜT)), zero(eltype(ᶜT)))
                end
                c_trunc = equilibrium_condensate_truncation(p)
                @. ᶜsa_result = compute_sgs_saturation_adjustment(
                    thermo_params,
                    $(sgs_quad),
                    Y.c.ρ,
                    ᶜT,
                    ᶜq_tot_nonneg,
                    ᶜT′T′,
                    ᶜq′q′,
                    ᶜcorr_Tq,
                    ᶜlin,
                    ᶜcurv,
                    c_trunc,
                )
                @. ᶜq_liq = ᶜsa_result.q_liq
                @. ᶜq_ice = ᶜsa_result.q_ice
            end
        else  # DryModel: q values are set to zero
            @. ᶜq_tot_nonneg = zero(eltype(ᶜT))
            @. ᶜq_liq = zero(eltype(ᶜT))
            @. ᶜq_ice = zero(eltype(ᶜT))
            @. ᶜT = max(
                T_min_sgs,
                TD.air_temperature(
                    thermo_params,
                    ᶜe_int,
                    ᶜq_tot_nonneg,
                    ᶜq_liq,
                    ᶜq_ice,
                ),
            )
        end
        set_enthalpy_and_pressure!(
            ᶜh_tot, ᶜp, Y.c, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, thermo_params,
        )
    end

    return ᶠuₕ³
end

"""
    set_implicit_precomputed_quantities_core!(Y, p, t)

Update the grid-mean implicit precomputed quantities and, for
`PrognosticEDMFX`, the draft and environment thermodynamic and velocity fields
from state `Y` at time `t`.

Applies the surface and model-top impenetrable velocity conditions on `ᶠu³` (and
each `ᶠu³ʲ` under `PrognosticEDMFX`). Called from post-step column physics and
full cache refreshes (`set_precomputed_quantities!`,
`set_end_of_step_precomputed_quantities!`).
"""
NVTX.@annotate function set_implicit_precomputed_quantities_core!(Y, p, t)
    ᶠuₕ³ = _set_implicit_grid_mean_precomputed_quantities!(Y, p, t)
    (; turbconv_model) = p.atmos
    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_draft!(Y, p, ᶠuₕ³, t)
        set_prognostic_edmf_precomputed_quantities_environment!(Y, p, ᶠuₕ³, t)
    end
    return nothing
end

"""
    set_implicit_precomputed_quantities!(Y, p, t)

Update only the grid-mean wave-core precomputed quantities for `Y` at time `t`
during Newton inner stages (`cache_imp!`) and wave-core Jacobian evaluation,
skipping EDMF draft and environment thermodynamics.
"""
function set_implicit_precomputed_quantities!(Y, p, t)
    _set_implicit_grid_mean_precomputed_quantities!(Y, p, t)
    return nothing
end

"""
    set_explicit_precomputed_quantities!(Y, p, t; update_sfc = true)

Update the precomputed quantities that are handled explicitly based on the
current state `Y`.

Updates surface conditions, EDMFX explicit closures, SGS covariances and
cloud fraction, EDMFX draft closures (`set_prognostic_edmf_draft_closures!`),
face diffusivities, mixing length, and Smagorinsky-Lilly quantities as active
on the current configuration.

When `is_stage_cache_update(Y, p)` is `true` (intermediate ARK stages), skips
surface conditions, vertical EDMF closures/diffusivities/mixing length, and
covariances, updating only the horizontal EDMF and Smagorinsky-Lilly
diffusivities read by `remaining_tendency!`.

When `update_sfc` is `false` (end-of-step finalization in `post_step_callback!`),
skips `update_surface_conditions!` so realized turbulent surface fluxes stay
intact while refreshing `sfc_conditions.T_sfc` from `Y.sfc.T` for
`SlabOceanTemperature`.
"""
NVTX.@annotate function set_explicit_precomputed_quantities!(
    Y,
    p,
    t;
    update_sfc::Bool = true,
)
    (; turbconv_model) = p.atmos

    FT = eltype(p.params)

    if is_stage_cache_update(Y, p)
        if turbconv_model isa AbstractEDMF &&
           (
               p.atmos.edmfx_model.sgs_diffusive_flux_horizontal isa Val{true} ||
               p.atmos.edmfx_model.horizontal_diffusion isa Val{true}
           ) &&
           !iscolumn(axes(Y.c))
            thermo_params = CAP.thermodynamics_params(p.params)
            set_buoyancy_gradient_inputs!(Y, p, thermo_params, false)
            ᶜlg = Fields.local_geometry_field(Y.c)
            @. p.precomputed.ᶜbuoygrad = blended_N²(
                p.precomputed.ᶜbg_coeffs,
                p.precomputed.ᶜcloud_fraction,
                projected_vector_data(C3, p.precomputed.ᶜgradᵥ_θ_liq_ice, ᶜlg),
                projected_vector_data(C3, p.precomputed.ᶜgradᵥ_q_tot, ᶜlg),
            )
            set_strain_rate_norm!(Y, p)
            set_horizontal_diffusivities!(Y, p)
        end
        sl = p.atmos.smagorinsky_lilly
        if !isnothing(sl) && is_smagorinsky_horizontal(sl)
            set_smagorinsky_lilly_precomputed_quantities!(Y, p, sl)
        end
        return nothing
    end

    if update_sfc
        if !isnothing(p.sfc_setup)
            SurfaceConditions.update_surface_conditions!(Y, p, FT(t))
        end
    elseif p.atmos.surface.temperature isa
           SurfaceConditions.SlabOceanTemperature &&
           !isnothing(p.atmos.surface.flux_scheme)
        Fields.field_values(p.precomputed.sfc_conditions.T_sfc) .=
            Fields.field_values(Y.sfc.T)
    end

    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_explicit_closures!(Y, p, t)
    end
    if turbconv_model isa EDOnlyEDMFX
        set_edonly_precomputed_quantities_env_closures!(Y, p, t)
    end

    set_covariance_cache_and_cloud_fraction!(Y, p)

    # The draft closures, whose buoyancy reference reads the environment cloud
    # fraction from the update above.
    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_draft_closures!(Y, p)
    end

    # Face-native diffusivities. Need the final cloud fraction from the
    # covariance/cloud-fraction update above.
    set_face_diffusivities!(Y, p)

    if turbconv_model isa AbstractEDMF && (
        p.atmos.edmfx_model.sgs_diffusive_flux_horizontal isa Val{true} ||
        p.atmos.edmfx_model.horizontal_diffusion isa Val{true}
    )
        set_horizontal_diffusivities!(Y, p)
    end

    # Master mixing length at centers for consumers that live at centers
    # (TKE dissipation, covariance closure, updraft internal diffusion,
    # diagnostics). When the configuration uses (co)variances, ᶜl_mix is
    # materialized inside the covariance/cloud-fraction iteration (see
    # set_covariance_cache!), so it would be redundant to recompute it
    # here; `uses_covariances` is the shared predicate that keeps the two
    # paths from disagreeing.
    if !uses_covariances(p.atmos) && turbconv_model isa AbstractEDMF
        p.precomputed.ᶜl_mix .= ᶜmixing_length(Y, p)
    end

    set_smagorinsky_lilly_precomputed_quantities!(
        Y,
        p,
        p.atmos.smagorinsky_lilly,
    )

    return nothing
end

"""
    set_end_of_step_precomputed_quantities!(Y, p, t)

Refresh end-of-step precomputed quantities after post-step `dss!` and
`constrain_state!` without overwriting realized turbulent surface fluxes
(`sfc_conditions.ρ_flux_uₕ`, `ρ_flux_h_tot`, `ρ_flux_q_tot`) or bottom
precipitation fluxes (`surface_rain_flux`, `surface_snow_flux`,
`col_integrated_precip_energy_tendency`).
"""
function set_end_of_step_precomputed_quantities!(Y, p, t)
    set_implicit_precomputed_quantities_core!(Y, p, t)
    set_explicit_precomputed_quantities!(Y, p, t; update_sfc = false)
    return nothing
end

"""
    set_precomputed_quantities!(Y, p, t)

Update all precomputed quantities in `p.precomputed` based on the current
state `Y` via `set_implicit_precomputed_quantities_core!` and
`set_explicit_precomputed_quantities!`.
"""
function set_precomputed_quantities!(Y, p, t)
    set_implicit_precomputed_quantities_core!(Y, p, t)
    set_explicit_precomputed_quantities!(Y, p, t)
    return nothing
end
