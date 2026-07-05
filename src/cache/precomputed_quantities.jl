#####
##### Precomputed quantities
#####
import Thermodynamics as TD
import ClimaCore: Spaces, Fields

"""
    implicit_precomputed_quantities(Y, atmos)

Allocates precomputed quantities that are treated implicitly (i.e., updated
on each iteration of the implicit solver). This includes all quantities related
to velocity and thermodynamics that are used in the implicit tendency.

The following grid-scale quantities are treated implicitly and are precomputed:

  - `ᶜu`: covariant velocity on cell centers
  - `ᶠu`: contravariant velocity on cell faces
  - `ᶜK`: kinetic energy on cell centers
  - `ᶜT`: air temperature on cell centers
  - `ᶜq_tot_nonneg`: total water specific humidity, clipped to ≥ 0
  - `ᶜq_liq`: total liquid water (cloud liquid + rain), clipped to ≥ 0
  - `ᶜq_ice`: total ice water (cloud ice + snow), clipped to ≥ 0
  - `ᶜp`: air pressure on cell centers
    If the `turbconv_model` is `PrognosticEDMFX`, there also two SGS versions of
    every quantity except for `ᶜp` (which is shared across all subdomains):
  - `_⁰`: value for the environment
  - `_ʲs`: a tuple of values for the mass-flux subdomains
    In addition, there are several other SGS quantities for `PrognosticEDMFX`:
  - `ᶜρʲs`: a tuple of the air densities of the mass-flux subdomains on cell
    centers

TODO: Rename `ᶜK` to `ᶜκ`.
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
    # Base prognostic EDMFX quantities (for all moisture models)
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
    # Microphysics quantities that are written during set_implicit_precomputed_quantities!
    # and depend on Y (through ρa⁰), so they need Dual-typed copies for autodiff.
    # TODO - are they not needed?
    implicit_mp_quantities = (;)

    # Surface precipitation fluxes need Dual-typed copies so that
    # set_precipitation_surface_fluxes! can be called during the implicit
    # stage (AD writes Dual values into these fields).
    implicit_sfc_precip_quantities =
        if !(microphysics_model isa DryModel)
            (;
                surface_rain_flux = zeros(axes(Fields.level(Y.f, half))),
                surface_snow_flux = zeros(axes(Fields.level(Y.f, half))),
                col_integrated_precip_energy_tendency = zeros(
                    axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
                ),
            )
        else
            (;)
        end
    return (;
        gs_quantities...,
        moist_gs_quantities...,
        sgs_quantities...,
        prognostic_sgs_quantities...,
        implicit_mp_quantities...,
        implicit_sfc_precip_quantities...,
    )
end

"""
    precomputed_quantities(Y, atmos)

Allocates all precomputed quantities. This includes the quantities treated
implicitly (updated before each tendency evaluation), and also the quantities
treated explicitly (updated only before explicit tendency evaluations).

TODO: Reduce the number of cached values by computing them on the fly.
"""
function precomputed_quantities(Y, atmos)
    FT = eltype(Y)
    @assert !(atmos.microphysics_model isa DryModel) ||
            !(atmos.turbconv_model isa PrognosticEDMFX)
    @assert isnothing(atmos.turbconv_model) ||
            isnothing(atmos.vertical_diffusion)
    sa_result_type = @NamedTuple{T::FT, q_liq::FT, q_ice::FT}
    SCT = SurfaceConditions.surface_conditions_type(atmos, FT)
    cspace = axes(Y.c)
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    n_prog = n_prognostic_mass_flux_subdomains(atmos.turbconv_model)
    @assert !(atmos.turbconv_model isa PrognosticEDMFX) || n_prog == 1
    gs_quantities = (;
        ᶜwₜqₜ = similar(Y.c, Geometry.WVector{FT}),
        ᶜwₕhₜ = similar(Y.c, Geometry.WVector{FT}),
        ᶜlinear_buoygrad = similar(Y.c, FT),
        # Stability-biased buoyancy gradient (max of one-sided estimates);
        # feeds the mixing-length and Pr_t(Ri) closures near sharp inversions.
        ᶜbuoygrad_stab = similar(Y.c, FT),
        ᶜstrain_rate_norm = similar(Y.c, FT),
        sfc_conditions = similar(Spaces.level(Y.f, half), SCT),
    )
    # Cloud fraction is used to calculate buoyancy gradient, so we initialize it to 0 here.
    ᶜcloud_fraction = similar(Y.c, FT)
    @. ᶜcloud_fraction = FT(0)

    # SGS covariances for hybrid cloud fraction and microphysics quadrature.
    # NonEquilibriumMicrophysics1M/2M always route through the quadrature API
    # internally (with GridMeanSGS), so they also need covariance fields allocated.
    uses_sgs_quadrature =
        !isnothing(atmos.sgs_quadrature) ||
        atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ||
        atmos.cloud_model isa Union{QuadratureCloud, MLCloud}
    uses_microphysics_quadrature_moments =
        atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
    # `ᶜsgs_moments` caches `(sigma_S, λ_lagrange)` — the SGS standard
    # deviation and the Lagrange multiplier used by `Microphysics1MEvaluator`.
    #  Allocated only for 1M/2M schemes.
    SGSMomentsNT = @NamedTuple{
        sigma_S::FT,
        λ_lagrange::FT,
    }
    covariance_quantities = if uses_sgs_quadrature
        base = (;
            ᶜT′T′ = zeros(axes(Y.c)),
            ᶜq′q′ = zeros(axes(Y.c)),
        )
        uses_microphysics_quadrature_moments ?
        (; base..., ᶜsgs_moments = similar(Y.c, SGSMomentsNT)) :
        base
    else
        (;)
    end
    surface_precip_fluxes = (;
        surface_rain_flux = zeros(axes(Fields.level(Y.f, half))),
        surface_snow_flux = zeros(axes(Fields.level(Y.f, half))),
    )
    sedimentation_quantities =
        atmos.microphysics_model isa NonEquilibriumMicrophysics ?
        (; ᶜwₗ = similar(Y.c, FT), ᶜwᵢ = similar(Y.c, FT)) : (;)

    # Helper named tuples for microphysics cache
    MP0_NT = @NamedTuple{dq_tot_dt::FT, e_tot_hlpr::FT}
    MP1_NT = @NamedTuple{
        dq_lcl_dt::FT, dq_icl_dt::FT, dq_rai_dt::FT, dq_sno_dt::FT,
    }
    MP23_NT = @NamedTuple{
        dq_lcl_dt::FT, dn_lcl_dt::FT, dq_rai_dt::FT, dn_rai_dt::FT,
        dq_ice_dt::FT, dq_rim_dt::FT, db_rim_dt::FT,
    }

    if atmos.microphysics_model isa EquilibriumMicrophysics0M
        precipitation_quantities = (;
            ᶜmp_tendency = similar(Y.c, MP0_NT),
            ᶜρ_dq_tot_dt = similar(Y.c, FT), # Used in implicit tendency and surface fluxes
            ᶜρ_de_tot_dt = similar(Y.c, FT),
        )
    elseif atmos.microphysics_model isa NonEquilibriumMicrophysics1M
        precipitation_quantities = (;
            ᶜwₗ = similar(Y.c, FT),
            ᶜwᵢ = similar(Y.c, FT),
            ᶜwᵣ = similar(Y.c, FT),
            ᶜwₛ = similar(Y.c, FT),
            ᶜmp_tendency = similar(Y.c, MP1_NT),
        )
    elseif atmos.microphysics_model isa
           Union{NonEquilibriumMicrophysics2M, NonEquilibriumMicrophysics2MP3}
        # 2-moment microphysics
        precipitation_quantities = (;
            ᶜwₗ = similar(Y.c, FT),
            ᶜwᵢ = similar(Y.c, FT),
            ᶜwᵣ = similar(Y.c, FT),
            ᶜwₛ = similar(Y.c, FT),
            ᶜwₙₗ = similar(Y.c, FT),
            ᶜwₙᵣ = similar(Y.c, FT),
            ᶜmp_tendency = similar(Y.c, MP23_NT),
        )
        # Add additional quantities for 2M + P3
        if atmos.microphysics_model isa NonEquilibriumMicrophysics2MP3
            precipitation_quantities = (;
                # liquid quantities (2M warm rain)
                precipitation_quantities...,
                # ice quantities (P3)
                ᶜwnᵢ = similar(Y.c, FT),
                ᶜlogλ = similar(Y.c, FT),
                ᶜScoll = similar(Y.c,
                    @NamedTuple{
                        ∂ₜq_c::FT, ∂ₜq_r::FT, ∂ₜN_c::FT, ∂ₜN_r::FT,
                        ∂ₜL_rim::FT, ∂ₜL_ice::FT, ∂ₜB_rim::FT,
                    }
                ),
            )
        end
    else
        precipitation_quantities = (;)
    end
    if atmos.microphysics_model isa EquilibriumMicrophysics0M
        precipitation_sgs_quantities = (;
            ᶜmp_tendencyʲs = similar(Y.c, NTuple{n, MP0_NT}),
        )
        if atmos.turbconv_model isa PrognosticEDMFX
            precipitation_sgs_quantities = (;
                precipitation_sgs_quantities...,
                ᶜmp_tendency⁰ = similar(Y.c, MP0_NT),
            )
        end
    elseif atmos.microphysics_model isa NonEquilibriumMicrophysics1M
        precipitation_sgs_quantities = (;
            ᶜmp_tendencyʲs = similar(Y.c, NTuple{n, MP1_NT}),
            ᶜwₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵢʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵣʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₛʲs = similar(Y.c, NTuple{n, FT}),
        )
        if atmos.turbconv_model isa PrognosticEDMFX
            precipitation_sgs_quantities = (;
                precipitation_sgs_quantities...,
                ᶜmp_tendency⁰ = similar(Y.c, MP1_NT),
            )
        end
    elseif atmos.microphysics_model isa NonEquilibriumMicrophysics2M
        precipitation_sgs_quantities = (;
            ᶜmp_tendencyʲs = similar(Y.c, NTuple{n, MP23_NT}),
            ᶜwₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵢʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵣʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₛʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₙₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₙᵣʲs = similar(Y.c, NTuple{n, FT}),
        )
        if atmos.turbconv_model isa PrognosticEDMFX
            precipitation_sgs_quantities = (;
                precipitation_sgs_quantities...,
                ᶜmp_tendency⁰ = similar(Y.c, MP23_NT),
            )
        end
    else
        precipitation_sgs_quantities = (;)
    end

    advective_sgs_quantities =
        atmos.turbconv_model isa PrognosticEDMFX ?
        (;
            ρtke_flux = similar(Fields.level(Y.f, half), C3{FT}),
            ᶜentr_vel_scaleʲs = similar(Y.c, NTuple{n, FT}),
            ᶜturb_entrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜarea_bounding_entr_detrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜρ_diffʲs = similar(Y.c, NTuple{n, FT}),
            ᶠu₃_tendencyʲs = similar(Y.f, NTuple{n, C3{FT}}),
            ᶜρa_tendencyʲs = similar(Y.c, NTuple{n, FT}),
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

    diagnostic_precipitation_sgs_quantities =
        atmos.microphysics_model isa NonEquilibriumMicrophysics1M ?
        (;
            ᶜq_lclʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_iclʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_raiʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_snoʲs = similar(Y.c, NTuple{n, FT}),
        ) : (;)

    smagorinsky_lilly_quantities =
        if atmos.smagorinsky_lilly isa SmagorinskyLilly
            uvw_vec = UVW(FT(0), FT(0), FT(0))
            (;
                ᶜS = similar(Y.c, typeof(uvw_vec * uvw_vec')),
                ᶠS = similar(Y.f, typeof(uvw_vec * uvw_vec')),
                ᶜS_norm_h = similar(Y.c, FT), ᶜS_norm_v = similar(Y.c, FT),
                ᶜL_h = similar(Y.c, FT), ᶜL_v = similar(Y.c, FT),
                ᶜνₜ_h = similar(Y.c, FT), ᶜνₜ_v = similar(Y.c, FT),
                ᶜD_h = similar(Y.c, FT), ᶜD_v = similar(Y.c, FT),
            )
        else
            (;)
        end
    amd_les_quantities =
        if atmos.amd_les isa AnisotropicMinimumDissipation
            uvw_vec = UVW(FT(0), FT(0), FT(0))
            (;
                ᶜτ_amd = similar(Y.c, typeof(uvw_vec * uvw_vec')),
                ᶠτ_amd = similar(Y.f, typeof(uvw_vec * uvw_vec')),
                ᶜD_amd = similar(Y.c, FT),
                ᶠD_amd = similar(Y.f, FT),
            )
        else
            (;)
        end

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
        covariance_quantities...,
        smagorinsky_lilly_quantities...,
        amd_les_quantities...)
end

# Interpolates the third contravariant component of Y.c.uₕ to cell faces.
function compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    return @. lazy(ᶠwinterp(ᶜρ * ᶜJ, CT3(ᶜuₕ)))
end

"""
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)

Modifies `Y.f.u₃` so that `ᶠu³` is 0 at the surface. Specifically, since
`u³ = uₕ³ + u³ = uₕ³ + u₃ * g³³`, setting `u³` to 0 gives `u₃ = -uₕ³ / g³³`. If
the `turbconv_model` is EDMFX, the `Y.f.sgsʲs` are also modified so that each
`u₃ʲ` is equal to `u₃` at the surface.
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

function surface_velocity(ᶠu₃, ᶠuₕ³)
    sfc_u₃ = Fields.level(ᶠu₃.components.data.:1, half)
    sfc_uₕ³ = Fields.level(ᶠuₕ³.components.data.:1, half)
    sfc_g³³ = g³³_field(axes(sfc_u₃))
    return @. lazy(-sfc_uₕ³ / sfc_g³³) # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
end

function top_velocity(ᶠu₃, ᶠuₕ³)
    top_level = Spaces.nlevels(axes(ᶠu₃)) - half
    top_u₃ = Fields.level(ᶠu₃.components.data.:1, top_level)
    top_uₕ³ = Fields.level(ᶠuₕ³.components.data.:1, top_level)
    top_g³³ = g³³_field(axes(top_u₃))
    return @. lazy(-top_uₕ³ / top_g³³) # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
end

"""
    set_velocity_at_top!(Y, ᶠuₕ³, turbconv_model)

Modifies `Y.f.u₃` so that `ᶠu³` is 0 at the model top. As at the surface,
since `u³ = uₕ³ + u₃ * g³³`, setting `u³` to 0 gives `u₃ = -uₕ³ / g³³`. This
makes the total contravariant flux through the top boundary vanish even where
terrain-following coordinate surfaces are still sloped at the model top
(`g³ʰ ≠ 0`, so `uₕ³ ≠ 0`). If the `turbconv_model` is EDMFX, the `Y.f.sgsʲs`
are also modified so that each `u₃ʲ` is equal to `u₃` at the model top.
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

# This is used to set the grid-scale velocity quantities ᶜu, ᶠu³, ᶜK based on
# ᶠu₃, and it is also used to set the SGS quantities based on ᶠu₃⁰ and ᶠu₃ʲ.
function set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠu₃, ᶜuₕ, ᶠuₕ³)
    @. ᶜu = C123(ᶜuₕ) + ᶜinterp(C123(ᶠu₃))
    @. ᶠu³ = ᶠuₕ³ + CT3(ᶠu₃)
    ᶜK .= compute_kinetic(ᶜuₕ, ᶠu₃)
    return nothing
end

function set_sgs_ᶠu₃!(w_function, ᶠu₃, Y, turbconv_model)
    ρaʲs(sgsʲs) = map(sgsʲ -> sgsʲ.ρa, sgsʲs)
    u₃ʲs(sgsʲs) = map(sgsʲ -> sgsʲ.u₃, sgsʲs)
    @. ᶠu₃ = w_function(
        ᶠinterp(ρaʲs(Y.c.sgsʲs)),
        u₃ʲs(Y.f.sgsʲs),
        ᶠinterp(Y.c.ρ),
        Y.f.u₃,
        turbconv_model,
    )
    return nothing
end

function add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠu₃⁰, turbconv_model)
    @. ᶜK += ᶜρa⁰ * ᶜinterp(dot(ᶠu₃⁰ - Y.f.u₃, CT3(ᶠu₃⁰ - Y.f.u₃))) / 2 / Y.c.ρ
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
        @. ᶜK +=
            ᶜρaʲ * ᶜinterp(dot(ᶠu₃ʲ - Y.f.u₃, CT3(ᶠu₃ʲ - Y.f.u₃))) / 2 / Y.c.ρ
    end
    return nothing
end

# Combined getter function for thermodynamic state variables from saturation adjustment.
# Returns a NamedTuple with T, q_liq, q_ice.
# This avoids redundant saturation_adjustment calls for EquilibriumMicrophysics0M.
function saturation_adjustment_tuple(thermo_params, ::TD.ρe, ρ, e_int, q_tot)
    sa_result = TD.saturation_adjustment(thermo_params, TD.ρe(), ρ, e_int, q_tot)
    return (; T = sa_result.T, q_liq = sa_result.q_liq, q_ice = sa_result.q_ice)
end

function eddy_diffusivity_coefficient_H(D₀, H, z_sfc, z)
    return D₀ * exp(-(z - z_sfc) / H)
end
function eddy_diffusivity_coefficient(C_E, norm_v_a, z_a, p)
    p_pbl = 85000
    p_strato = 10000
    K_E = C_E * norm_v_a * z_a
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end

"""
    set_implicit_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities that are handled implicitly based on the
current state `Y`. This is called before each evaluation of either
`implicit_tendency!` or `remaining_tendency!`, and it includes quantities used
in both tedencies.

This function also applies a "filter" to `Y` in order to ensure that `ᶠu³` is 0
at the surface (i.e., to enforce the impenetrable boundary condition). If the
`turbconv_model` is EDMFX, the filter also ensures that `ᶠu³⁰` and `ᶠu³ʲs` are 0
at the surface. In the future, we will probably want to move this filtering
elsewhere, but doing it here ensures that it occurs whenever the precomputed
quantities are updated.
"""
NVTX.@annotate function set_implicit_precomputed_quantities!(Y, p, t)
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

    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y.f.u₃, Y.c.uₕ, ᶠuₕ³)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    @. ᶠu = CT123(ᶠwinterp(Y.c.ρ * ᶜJ, CT12(ᶜu))) + CT123(ᶠu³)
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
            (; ᶜT′T′, ᶜq′q′) = p.precomputed
            corr_Tq = correlation_Tq(p.params)
            @. ᶜsa_result = compute_sgs_saturation_adjustment(
                thermo_params,
                $(sgs_quad),
                Y.c.ρ,
                ᶜT,
                ᶜq_tot_nonneg,
                ᶜT′T′,
                ᶜq′q′,
                corr_Tq,
            )
            @. ᶜq_liq = ᶜsa_result.q_liq
            @. ᶜq_ice = ᶜsa_result.q_ice
        end
    else  # DryModel or NonEquilibriumMicrophysics
        # For DryModel: q values are set to zero
        # For NonEquilibriumMicrophysics: q values are computed from state variables
        if microphysics_model isa DryModel
            @. ᶜq_tot_nonneg = zero(eltype(ᶜT))
            @. ᶜq_liq = zero(eltype(ᶜT))
            @. ᶜq_ice = zero(eltype(ᶜT))
        else  # NonEquilibriumMicrophysics
            @. ᶜq_liq =
                max(0, specific(Y.c.ρq_lcl, Y.c.ρ) + specific(Y.c.ρq_rai, Y.c.ρ))
            @. ᶜq_ice =
                max(0, specific(Y.c.ρq_icl, Y.c.ρ) + specific(Y.c.ρq_sno, Y.c.ρ))
            # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
            @. ᶜq_tot_nonneg =
                max(ᶜq_liq + ᶜq_ice, specific(Y.c.ρq_tot, Y.c.ρ))
        end
        # Floor T to prevent negative pressure during implicit Newton iterations
        T_min_sgs = CAP.T_min_sgs(p.params)
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
    ᶜe_tot = @. lazy(specific(Y.c.ρe_tot, Y.c.ρ))
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
        Y.c.ρ,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    )

    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_draft!(Y, p, ᶠuₕ³, t)
        set_prognostic_edmf_precomputed_quantities_environment!(Y, p, ᶠuₕ³, t)
    elseif !(isnothing(turbconv_model))
        # Do nothing for other turbconv models for now
    end

    # When microphysics is implicit, refresh ᶜS_ρq_tot / ᶜS_ρe_tot and the
    # surface precipitation fluxes so that they reflect the current Y.
    # The surface flux fields have Dual-typed copies in
    # implicit_precomputed_quantities, so AD can write into them safely.
    if p.atmos.microphysics_tendency_timestepping == Implicit()
        update_implicit_microphysics_cache!(Y, p, microphysics_model, turbconv_model)
    end
end

"""
    set_explicit_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities that are handled explicitly based on the
current state `Y`. This is only called before each evaluation of
`remaining_tendency!`, though it includes quantities used in both
`implicit_tendency!` and `remaining_tendency!`.
"""
NVTX.@annotate function set_explicit_precomputed_quantities!(Y, p, t)
    (; turbconv_model) = p.atmos

    FT = eltype(p.params)

    if !isnothing(p.sfc_setup)
        SurfaceConditions.update_surface_conditions!(Y, p, FT(t))
    end

    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_explicit_closures!(Y, p, t)
    end
    if turbconv_model isa EDOnlyEDMFX
        set_edonly_precomputed_quantities_env_closures!(Y, p, t)
        # TODO do I need env precipitation/cloud formation here?
    end

    set_covariance_cache_and_cloud_fraction!(Y, p)

    # Cache precipitation terminal velocities for grid mean and prognostic EDMF updrafts.
    set_precipitation_velocities!(
        Y,
        p,
        p.atmos.microphysics_model,
        p.atmos.turbconv_model,
    )
    # Compute microphysics sources from grid mean and sub-domains.
    # Always compute ᶜmp_tendency and ᶜS_ρq_tot here so both are fresh.
    # When microphysics is implicit, the implicit stage will additionally
    # refresh ᶜS_ρq_tot / ᶜS_ρe_tot from the (now-fresh) ᶜmp_tendency
    # using the current Newton-iterate Y, avoiding the allocating BMT broadcast.
    set_microphysics_tendency_cache!(
        Y,
        p,
        p.atmos.microphysics_model,
        p.atmos.turbconv_model,
    )
    # Compute surface precipitation fluxes (has to be after microphysics_sources_cache
    # because for the 0 moment microphysics it's an integral of the q_tot sink).
    set_precipitation_surface_fluxes!(Y, p, p.atmos.microphysics_model)

    set_smagorinsky_lilly_precomputed_quantities!(Y, p, p.atmos.smagorinsky_lilly)

    if p.atmos.amd_les isa AnisotropicMinimumDissipation
        set_amd_precomputed_quantities!(Y, p)
    end

    return nothing
end

"""
    set_precomputed_quantities!(Y, p, t)

Updates all precomputed quantities based on the current state `Y`.
"""
function set_precomputed_quantities!(Y, p, t)
    set_implicit_precomputed_quantities!(Y, p, t)
    set_explicit_precomputed_quantities!(Y, p, t)
end
