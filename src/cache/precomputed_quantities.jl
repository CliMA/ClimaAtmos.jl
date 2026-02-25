#####
##### Precomputed quantities
#####
import Thermodynamics as TD
import ClimaCore: Spaces, Fields
using Base.Broadcast: materialize

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
    - `ᶜq_tot_safe`: total water specific humidity on cell centers
    - `ᶜq_liq_rai`: liquid water specific humidity on cell centers
    - `ᶜq_ice_sno`: ice specific humidity on cell centers
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
    (; moisture_model, turbconv_model, microphysics_model) = atmos
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
    # Moisture-related quantities depend on moisture model:
    # - EquilMoistModel: allocate fields + thermo_state cache for saturation adjustment
    # - Ohters: allocate fields only
    sa_result_type = @NamedTuple{T::FT, q_liq::FT, q_ice::FT}
    moist_gs_quantities =
        if moisture_model isa EquilMoistModel
            (;
                ᶜq_tot_safe = similar(Y.c, FT),
                ᶜq_liq_rai = similar(Y.c, FT),
                ᶜq_ice_sno = similar(Y.c, FT),
                ᶜsa_result = similar(Y.c, sa_result_type),
            )
        else  # DryModel or NonEquilMoistModel
            (;
                ᶜq_tot_safe = similar(Y.c, FT),
                ᶜq_liq_rai = similar(Y.c, FT),
                ᶜq_ice_sno = similar(Y.c, FT),
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
            ᶜq_tot_safe⁰ = similar(Y.c, FT),
            ᶜq_liq_rai⁰ = similar(Y.c, FT),
            ᶜq_ice_sno⁰ = similar(Y.c, FT),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶠKᵥʲs = similar(Y.f, NTuple{n, FT}),
            ᶜTʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_tot_safeʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_liq_raiʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_ice_snoʲs = similar(Y.c, NTuple{n, FT}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
            ᶠnh_pressure₃_dragʲs = similar(Y.f, NTuple{n, C3{FT}}),
        ) : (;)
    return (;
        gs_quantities...,
        moist_gs_quantities...,
        sgs_quantities...,
        prognostic_sgs_quantities...,
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
    @assert !(atmos.moisture_model isa DryModel) ||
            !(atmos.turbconv_model isa DiagnosticEDMFX)
    @assert !(atmos.moisture_model isa DryModel) ||
            !(atmos.turbconv_model isa PrognosticEDMFX)
    @assert isnothing(atmos.turbconv_model) ||
            isnothing(atmos.vertical_diffusion)
    @assert !(atmos.moisture_model isa NonEquilMoistModel) ||
            !(atmos.microphysics_model isa Microphysics0Moment)
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
        ᶜstrain_rate_norm = similar(Y.c, FT),
        sfc_conditions = similar(Spaces.level(Y.f, half), SCT),
    )
    cloud_diagnostics_tuple =
        similar(Y.c, @NamedTuple{cf::FT, q_liq::FT, q_ice::FT})
    # Cloud fraction is used to calculate buoyancy gradient, so we initialize it to 0 here.
    @. cloud_diagnostics_tuple.cf = FT(0)
    surface_precip_fluxes = (;
        surface_rain_flux = zeros(axes(Fields.level(Y.f, half))),
        surface_snow_flux = zeros(axes(Fields.level(Y.f, half))),
    )
    sedimentation_quantities =
        atmos.moisture_model isa NonEquilMoistModel ?
        (; ᶜwₗ = similar(Y.c, FT), ᶜwᵢ = similar(Y.c, FT)) : (;)
    if atmos.microphysics_model isa Microphysics0Moment
        precipitation_quantities =
            (; ᶜS_ρq_tot = similar(Y.c, FT), ᶜS_ρe_tot = similar(Y.c, FT))
    elseif atmos.microphysics_model isa Microphysics1Moment
        precipitation_quantities = (;
            ᶜwᵣ = similar(Y.c, FT),
            ᶜwₛ = similar(Y.c, FT),
            ᶜSqₗᵖ = similar(Y.c, FT),
            ᶜSqᵢᵖ = similar(Y.c, FT),
            ᶜSqᵣᵖ = similar(Y.c, FT),
            ᶜSqₛᵖ = similar(Y.c, FT),
        )
    elseif atmos.microphysics_model isa Union{Microphysics2Moment, Microphysics2MomentP3}
        # 2-moment microphysics
        precipitation_quantities = (;
            ᶜwᵣ = similar(Y.c, FT),
            ᶜwₛ = similar(Y.c, FT),
            ᶜSqₗᵖ = similar(Y.c, FT),
            ᶜSqᵢᵖ = similar(Y.c, FT),
            ᶜSqᵣᵖ = similar(Y.c, FT),
            ᶜSqₛᵖ = similar(Y.c, FT),
            ᶜwₙₗ = similar(Y.c, FT),
            ᶜwₙᵣ = similar(Y.c, FT),
            ᶜSnₗᵖ = similar(Y.c, FT),
            ᶜSnᵣᵖ = similar(Y.c, FT),
        )
        # Add additional quantities for 2M + P3
        if atmos.microphysics_model isa Microphysics2MomentP3
            precipitation_quantities = (;
                # liquid quantities (2M warm rain)
                precipitation_quantities...,
                # ice quantities (P3)
                ᶜwᵢ = similar(Y.c, FT),
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
    precipitation_sgs_quantities =
        atmos.microphysics_model isa Microphysics0Moment ?
        (; ᶜSqₜᵖʲs = similar(Y.c, NTuple{n, FT}), ᶜSqₜᵖ⁰ = similar(Y.c, FT)) :
        atmos.microphysics_model isa Microphysics1Moment ?
        (;
            ᶜSqₗᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqᵢᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqᵣᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqₛᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵢʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵣʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₛʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqₗᵖ⁰ = similar(Y.c, FT),
            ᶜSqᵢᵖ⁰ = similar(Y.c, FT),
            ᶜSqᵣᵖ⁰ = similar(Y.c, FT),
            ᶜSqₛᵖ⁰ = similar(Y.c, FT),
        ) :
        atmos.microphysics_model isa Microphysics2Moment ?
        (;
            ᶜSqₗᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqᵢᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqᵣᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqₛᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSnₗᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSnᵣᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵢʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwᵣʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₛʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₙₗʲs = similar(Y.c, NTuple{n, FT}),
            ᶜwₙᵣʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqₗᵖ⁰ = similar(Y.c, FT),
            ᶜSqᵢᵖ⁰ = similar(Y.c, FT),
            ᶜSqᵣᵖ⁰ = similar(Y.c, FT),
            ᶜSqₛᵖ⁰ = similar(Y.c, FT),
            ᶜSnₗᵖ⁰ = similar(Y.c, FT),
            ᶜSnᵣᵖ⁰ = similar(Y.c, FT),
        ) : (;)
    advective_sgs_quantities =
        atmos.turbconv_model isa PrognosticEDMFX ?
        (;
            ρtke_flux = similar(Fields.level(Y.f, half), C3{FT}),
            bdmr_l = similar(Y.c, BidiagonalMatrixRow{FT}),
            bdmr_r = similar(Y.c, BidiagonalMatrixRow{FT}),
            bdmr = similar(Y.c, BidiagonalMatrixRow{FT}),
            ᶜentrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜdetrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜturb_entrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜgradᵥ_q_tot = Fields.Field(C3{FT}, cspace),
            ᶜgradᵥ_θ_liq_ice = Fields.Field(C3{FT}, cspace),
            ᶠnh_pressure₃_buoyʲs = similar(Y.f, NTuple{n, C3{FT}}),
            precipitation_sgs_quantities...,
        ) : (;)

    edonly_quantities =
        atmos.turbconv_model isa EDOnlyEDMFX ?
        (; ρtke_flux = similar(Fields.level(Y.f, half), C3{FT}),) : (;)

    sgs_quantities = (;
        ᶜgradᵥ_q_tot = Fields.Field(C3{FT}, cspace),
        ᶜgradᵥ_θ_liq_ice = Fields.Field(C3{FT}, cspace),
    )

    diagnostic_precipitation_sgs_quantities =
        atmos.microphysics_model isa Microphysics1Moment ?
        (;
            ᶜq_liqʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_iceʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_raiʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_snoʲs = similar(Y.c, NTuple{n, FT}),
        ) : (;)

    diagnostic_sgs_quantities =
        atmos.turbconv_model isa DiagnosticEDMFX ?
        (;
            ᶜρaʲs = similar(Y.c, NTuple{n, FT}),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶜTʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_tot_safeʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_liq_raiʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_ice_snoʲs = similar(Y.c, NTuple{n, FT}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
            ᶜmseʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_totʲs = similar(Y.c, NTuple{n, FT}),
            ᶜentrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜdetrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜturb_entrʲs = similar(Y.c, NTuple{n, FT}),
            ᶠnh_pressure³_buoyʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶠnh_pressure³_dragʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶜK⁰ = similar(Y.c, FT),
            ρtke_flux = similar(Fields.level(Y.f, half), C3{FT}),
            precipitation_sgs_quantities...,
            diagnostic_precipitation_sgs_quantities...,
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
        diagnostic_sgs_quantities...,
        sedimentation_quantities...,
        precipitation_quantities...,
        surface_precip_fluxes...,
        cloud_diagnostics_tuple,
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

"""
    set_velocity_at_top!(Y, turbconv_model)

Modifies `Y.f.u₃` so that `u₃` is 0 at the model top.
"""
function set_velocity_at_top!(Y, turbconv_model)
    top_u₃ = Fields.level(
        Y.f.u₃.components.data.:1,
        Spaces.nlevels(axes(Y.c)) + half,
    )
    @. top_u₃ = 0
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
# This avoids redundant saturation_adjustment calls for EquilMoistModel.
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

# Physical bounds used when clamping Newton iterates (so implicit residual/Jacobian
# are never evaluated at unphysical state; allows stable implicit solve without reducing dt).
const _implicit_ρ_min = 1e-6
const _implicit_ρ_max = 1e3
const _implicit_p_min = 1.0

"""
    clamp_implicit_state!(Y)

Clamp the state `Y` to physical bounds in-place so that the implicit residual
and Jacobian are never evaluated at unphysical values (e.g. negative ρ or p).
Called at the start of `set_implicit_precomputed_quantities!` so every Newton
iterate is kept in a valid range; allows the direct Newton solve to remain
stable without reducing dt.
"""
function clamp_implicit_state!(Y)
    @. Y.c.ρ = max(_implicit_ρ_min, min(_implicit_ρ_max, Y.c.ρ))
    for name in (:ρq_tot, :ρq_liq, :ρq_ice, :ρq_rai, :ρq_sno, :ρn_liq, :ρn_rai, :ρtke)
        hasproperty(Y.c, name) || continue
        f = getproperty(Y.c, name)
        parent(f) .= max.(0, parent(f))
    end
    if hasproperty(Y.c, :sgsʲs)
        for j in eachindex(Y.c.sgsʲs)
            hasproperty(Y.c.sgsʲs[j], :ρa) || continue
            ρa = Y.c.sgsʲs[j].ρa
            parent(ρa) .= max.(0, parent(ρa))
        end
    end
    return nothing
end

"""
    set_implicit_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities that are handled implicitly based on the
current state `Y`. This is called before each evaluation of either
`implicit_tendency!` or `remaining_tendency!`, and it includes quantities used
in both tedencies.

Before updating, the state is clamped to physical bounds via `clamp_implicit_state!`
so that Newton iterates never become unphysical (negative ρ, etc.). After pressure
is computed, `ᶜp` is clamped to a minimum so refstate thermodynamics never see
non-positive pressure.

This function also applies a "filter" to `Y` in order to ensure that `ᶠu³` is 0
at the surface (i.e., to enforce the impenetrable boundary condition). If the
`turbconv_model` is EDMFX, the filter also ensures that `ᶠu³⁰` and `ᶠu³ʲs` are 0
at the surface. In the future, we will probably want to move this filtering
elsewhere, but doing it here ensures that it occurs whenever the precomputed
quantities are updated.
"""
NVTX.@annotate function set_implicit_precomputed_quantities!(Y, p, t)
    clamp_implicit_state!(Y)
    _data = parent(Y)
    _has_nan = any(isnan, _data)
    _has_inf = any(isinf, _data)
    if _has_nan || _has_inf
        # Field-level diagnosis: identify which component carries the bad value.
        _bad_fields = String[]
        for name in propertynames(Y.c)
            d = parent(getproperty(Y.c, name))
            (any(isnan, d) || any(isinf, d)) && push!(_bad_fields, "Y.c.$name")
        end
        # Y.f may be a NamedTuple of face fields; check each property.
        for name in propertynames(Y.f)
            d = parent(getproperty(Y.f, name))
            (any(isnan, d) || any(isinf, d)) && push!(_bad_fields, "Y.f.$name")
        end
        kind = _has_nan ? "NaN" : "Inf"
        error(
            "State Y contains $kind at entry to set_implicit_precomputed_quantities! " *
            "(Newton trial state). " *
            "Bad fields: $(join(_bad_fields, ", ")). " *
            "Typical causes: (NaN) LU solve on ill-conditioned implicit system; " *
            "(Inf) Float32 overflow from super-CFL u₃ or kinetic energy.",
        )
    end
    (; turbconv_model, moisture_model) = p.atmos
    (; ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶠu, ᶜK, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno, ᶜh_tot, ᶜp) = p.precomputed
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_params = CAP.thermodynamics_params(p.params)

    @. ᶠuₕ³ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)

    # TODO: We might want to move this to constrain_state!
    if !(p.atmos.prescribed_flow isa PrescribedFlow)
        set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
        set_velocity_at_top!(Y, turbconv_model)
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
    if moisture_model isa EquilMoistModel
        # Compute thermodynamic state variables using combined getter function.
        # This avoids redundant saturation_adjustment calls for EquilMoistModel.
        @. ᶜq_tot_safe = max(0, specific(Y.c.ρq_tot, Y.c.ρ))
        (; ᶜsa_result) = p.precomputed
        @. ᶜsa_result =
            saturation_adjustment_tuple(thermo_params, TD.ρe(), Y.c.ρ, ᶜe_int, ᶜq_tot_safe)
        @. ᶜT = ᶜsa_result.T
        @. ᶜq_liq_rai = ᶜsa_result.q_liq
        @. ᶜq_ice_sno = ᶜsa_result.q_ice
    else  # DryModel or NonEquilMoistModel
        # For DryModel: q values are set to zero
        # For NonEquilMoistModel: q values are computed from state variables
        if moisture_model isa DryModel
            @. ᶜq_tot_safe = zero(eltype(ᶜT))
            @. ᶜq_liq_rai = zero(eltype(ᶜT))
            @. ᶜq_ice_sno = zero(eltype(ᶜT))
        else  # NonEquilMoistModel
            @. ᶜq_tot_safe = max(0, specific(Y.c.ρq_tot, Y.c.ρ))
            @. ᶜq_liq_rai =
                max(0, specific(Y.c.ρq_liq, Y.c.ρ) + specific(Y.c.ρq_rai, Y.c.ρ))
            @. ᶜq_ice_sno =
                max(0, specific(Y.c.ρq_ice, Y.c.ρ) + specific(Y.c.ρq_sno, Y.c.ρ))
        end
        @. ᶜT =
            TD.air_temperature(thermo_params, ᶜe_int, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)
    end
    ᶜe_tot = @. lazy(specific(Y.c.ρe_tot, Y.c.ρ))
    @. ᶜh_tot =
        TD.total_enthalpy(thermo_params, ᶜe_tot, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜT, Y.c.ρ, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)
    @. ᶜp = max(_implicit_p_min, ᶜp)

    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_draft!(Y, p, ᶠuₕ³, t)
        set_prognostic_edmf_precomputed_quantities_environment!(Y, p, ᶠuₕ³, t)
        set_prognostic_edmf_precomputed_quantities_implicit_closures!(Y, p, t)
    elseif !(isnothing(turbconv_model))
        # Do nothing for other turbconv models for now
    end

    # Horizontal acoustic reference state is updated only in set_explicit_precomputed_quantities!
    # so that it is never overwritten with unphysical Newton trial state (see docstring there).
end

"""
    set_explicit_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities that are handled explicitly based on the
current state `Y`. This is only called before each evaluation of
`remaining_tendency!`, though it includes quantities used in both
`implicit_tendency!` and `remaining_tendency!`.
"""
NVTX.@annotate function set_explicit_precomputed_quantities!(Y, p, t)
    (; turbconv_model, moisture_model, cloud_model) = p.atmos
    (; call_cloud_diagnostics_per_stage) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)

    # Update horizontal acoustic reference state only from physical state (step-start Y).
    # Never updated in set_implicit_precomputed_quantities!, which runs on Newton trial states.
    if hasproperty(p, :horizontal_acoustic_cache) &&
       !isnothing(p.horizontal_acoustic_cache)
        update_horizontal_acoustic_reference_state!(
            p.horizontal_acoustic_cache,
            Y,
            p,
        )
    end

    if !isnothing(p.sfc_setup)
        SurfaceConditions.update_surface_conditions!(Y, p, FT(t))
    end

    if turbconv_model isa AbstractEDMF
        (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
        @. p.precomputed.ᶜgradᵥ_q_tot = ᶜgradᵥ(ᶠinterp(ᶜq_tot_safe))
        @. p.precomputed.ᶜgradᵥ_θ_liq_ice = ᶜgradᵥ(
            ᶠinterp(
                TD.liquid_ice_pottemp(
                    thermo_params,
                    ᶜT,
                    Y.c.ρ,
                    ᶜq_tot_safe,
                    ᶜq_liq_rai,
                    ᶜq_ice_sno,
                ),
            ),
        )
    end

    # The buoyancy gradient depends on the cloud fraction, and the cloud fraction
    # depends on the mixing length, which depends on the buoyancy gradient.
    # We break this circular dependency by using cloud fraction from the previous time step in the
    # buoyancy gradient calculation. This breaks reproducible restart in general,
    # but we support reproducible restart by recalculating the cloud fraction with GridScaleCloud here.
    if p.atmos.numerics.reproducible_restart isa ReproducibleRestart
        set_cloud_fraction!(Y, p, moisture_model, GridScaleCloud())
    end

    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_explicit_closures!(Y, p, t)
        set_prognostic_edmf_precomputed_quantities_precipitation!(
            Y,
            p,
            p.atmos.microphysics_model,
        )
    end
    if turbconv_model isa DiagnosticEDMFX
        set_diagnostic_edmf_precomputed_quantities_bottom_bc!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_do_integral!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_top_bc!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_env_closures!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_env_precipitation!(
            Y,
            p,
            t,
            p.atmos.microphysics_model,
        )
    end
    if turbconv_model isa EDOnlyEDMFX
        set_diagnostic_edmf_precomputed_quantities_env_closures!(Y, p, t)
        # TODO do I need env precipitation/cloud formation here?
    end

    set_precipitation_velocities!(
        Y,
        p,
        p.atmos.moisture_model,
        p.atmos.microphysics_model,
        p.atmos.turbconv_model,
    )
    # Needs to be done after edmf precipitation is computed in sub-domains
    set_precipitation_cache!(
        Y,
        p,
        p.atmos.microphysics_model,
        p.atmos.turbconv_model,
    )
    set_precipitation_surface_fluxes!(Y, p, p.atmos.microphysics_model)

    # TODO
    if call_cloud_diagnostics_per_stage isa CallCloudDiagnosticsPerStage
        set_cloud_fraction!(Y, p, moisture_model, cloud_model)
    end

    set_smagorinsky_lilly_precomputed_quantities!(Y, p, p.atmos.smagorinsky_lilly)

    if p.atmos.amd_les isa AnisotropicMinimumDissipation
        set_amd_precomputed_quantities!(Y, p)
    end

    return nothing
end

"""
    set_precomputed_quantities!(Y, p, t)

Updates all precomputed quantities based on the current state `Y`.

When `horizontal_acoustic_mode == Implicit()` (Option B predictor-corrector),
this is where the Helmholtz correction is applied — after Newton convergence and
at end-of-timestep. This function is mapped to `cache!` in the IMEX ARK solver,
which is called only at those two points, never on stage initial guesses. Moving
the correction here prevents double-application to the initial guess.
"""
function set_precomputed_quantities!(Y, p, t)
    # Option B predictor-corrector: apply horizontal Helmholtz correction and re-DSS.
    # cache! is called only after Newton convergence and at end-of-timestep (never on
    # initial guesses, which use cache_imp! = set_implicit_precomputed_quantities!).
    if !isnothing(p.horizontal_acoustic_cache)
        horizontal_helmholtz_correction!(Y, p, t)
        do_dss(axes(Y.c)) &&
            Spaces.weighted_dss!(Y.c => p.ghost_buffer.c, Y.f => p.ghost_buffer.f)
    end
    if any(isnan, parent(Y))
        error(
            "State Y contains NaN in set_precomputed_quantities! " *
            "(post-Newton or end-of-timestep state). " *
            "This usually means Newton produced a NaN solution or the Helmholtz " *
            "correction introduced NaN. Inspect min(ρ), min(p); try smaller dt.",
        )
    end
    set_implicit_precomputed_quantities!(Y, p, t)
    set_explicit_precomputed_quantities!(Y, p, t)
end
