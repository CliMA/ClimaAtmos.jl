#####
##### Precomputed quantities for precipitation processes
#####

import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2

import Thermodynamics as TD
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields

const Iₗ = TD.internal_energy_liquid
const Iᵢ = TD.internal_energy_ice

"""
   Kin(ᶜw_precip, ᶜu_air)

    - ᶜw_precip - teminal velocity of cloud consensate or precipitation
    - ᶜu_air - air velocity

Helper function to compute the kinetic energy of cloud condensate and
precipitation.
"""
function Kin(ᶜw_precip, ᶜu_air)
    return @. lazy(
        norm_sqr(
            Geometry.UVWVector(0, 0, -(ᶜw_precip)) + Geometry.UVWVector(ᶜu_air),
        ) / 2,
    )
end

"""
    set_precipitation_velocities!(Y, p, moisture_model, microphysics_model)

Updates the precipitation terminal velocity, cloud sedimentation velocity,
and their contribution to total water and energy advection.
"""
function set_precipitation_velocities!(Y, p, _, _)
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed
    @. ᶜwₜqₜ = Geometry.WVector(0)
    @. ᶜwₕhₜ = Geometry.WVector(0)
    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    moisture_model::NonEquilMoistModel,
    microphysics_model::Microphysics1Moment,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    cmc = CAP.microphysics_cloud_params(p.params)
    cmp = CAP.microphysics_1m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwᵣ = CM1.terminal_velocity(
        cmp.pr,
        cmp.tv.rain,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_rai / Y.c.ρ),
    )
    @. ᶜwₛ = CM1.terminal_velocity(
        cmp.ps,
        cmp.tv.snow,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_sno / Y.c.ρ),
    )
    # compute sedimentation velocity for cloud condensate [m/s]
    @. ᶜwₗ = CMNe.terminal_velocity(
        cmc.liquid,
        cmc.Ch2022.rain,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_liq / Y.c.ρ),
    )
    @. ᶜwᵢ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        Y.c.ρ,
        max(zero(Y.c.ρ), Y.c.ρq_ice / Y.c.ρ),
    )

    # compute their contributions to energy and total water advection
    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq +
            ᶜwᵢ * Y.c.ρq_ice +
            ᶜwᵣ * Y.c.ρq_rai +
            ᶜwₛ * Y.c.ρq_sno,
        ) / Y.c.ρ
    @. ᶜwₕhₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * Y.c.ρq_ice * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * Y.c.ρq_rai * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))) +
            ᶜwₛ * Y.c.ρq_sno * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₛ, ᶜu))),
        ) / Y.c.ρ
    return nothing
end
function set_precipitation_velocities!(
    Y,
    p,
    moisture_model::NonEquilMoistModel,
    microphysics_model::Microphysics2Moment,
)
    (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₙₗ, ᶜwₙᵣ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core

    cmc = CAP.microphysics_cloud_params(p.params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    thp = CAP.thermodynamics_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    # TODO sedimentation of snow is based on the 1M scheme
    @. ᶜwₙᵣ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.sb,
            cm2p.rtv,
            max(zero(Y.c.ρ), specific(Y.c.ρq_rai, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_rai),
        ),
        1,
    )
    @. ᶜwᵣ = getindex(
        CM2.rain_terminal_velocity(
            cm2p.sb,
            cm2p.rtv,
            max(zero(Y.c.ρ), specific(Y.c.ρq_rai, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_rai),
        ),
        2,
    )
    @. ᶜwₛ = CM1.terminal_velocity(
        cm1p.ps,
        cm1p.tv.snow,
        Y.c.ρ,
        max(zero(Y.c.ρ), specific(Y.c.ρq_sno, Y.c.ρ)),
    )
    # compute sedimentation velocity for cloud condensate [m/s]
    # TODO sedimentation of ice is based on the 1M scheme
    @. ᶜwₙₗ = getindex(
        CM2.cloud_terminal_velocity(
            cm2p.sb.pdf_c,
            cm2p.ctv,
            max(zero(Y.c.ρ), specific(Y.c.ρq_liq, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_liq),
        ),
        1,
    )
    @. ᶜwₗ = getindex(
        CM2.cloud_terminal_velocity(
            cm2p.sb.pdf_c,
            cm2p.ctv,
            max(zero(Y.c.ρ), specific(Y.c.ρq_liq, Y.c.ρ)),
            Y.c.ρ,
            max(zero(Y.c.ρ), Y.c.ρn_liq),
        ),
        2,
    )
    @. ᶜwᵢ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        Y.c.ρ,
        max(zero(Y.c.ρ), specific(Y.c.ρq_ice, Y.c.ρ)),
    )

    # compute their contributions to energy and total water advection
    @. ᶜwₜqₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq +
            ᶜwᵢ * Y.c.ρq_ice +
            ᶜwᵣ * Y.c.ρq_rai +
            ᶜwₛ * Y.c.ρq_sno,
        ) / Y.c.ρ
    @. ᶜwₕhₜ =
        Geometry.WVector(
            ᶜwₗ * Y.c.ρq_liq * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
            ᶜwᵢ * Y.c.ρq_ice * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
            ᶜwᵣ * Y.c.ρq_rai * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))) +
            ᶜwₛ * Y.c.ρq_sno * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₛ, ᶜu))),
        ) / Y.c.ρ
    return nothing
end

"""
    set_precipitation_cache!(Y, p, microphysics_model, turbconv_model)

Computes the cache needed for precipitation tendencies. When run without edmf
model this involves computing precipitation sources based on the grid mean
properties. When running with edmf model this means summing the precipitation
sources from the sub-domains.
"""
set_precipitation_cache!(Y, p, _, _) = nothing
function set_precipitation_cache!(Y, p, ::Microphysics0Moment, _)
    (; params, dt) = p
    dt = float(dt)
    (; ᶜts) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; ᶜΦ) = p.core
    cm_params = CAP.microphysics_0m_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜS_ρq_tot =
        Y.c.ρ * q_tot_0M_precipitation_sources(
            thermo_params,
            cm_params,
            dt,
            Y.c.ρq_tot / Y.c.ρ,
            ᶜts,
        )
    @. ᶜS_ρe_tot =
        ᶜS_ρq_tot *
        e_tot_0M_precipitation_sources_helper(thermo_params, ᶜts, ᶜΦ)
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics0Moment,
    ::DiagnosticEDMFX,
)
    # For environment we multiply by grid mean ρ and not byᶜρa⁰
    # assuming a⁰=1
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ⁰, ᶜSqₜᵖʲs, ᶜρaʲs) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; ᶜts, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    ρ = Y.c.ρ

    @. ᶜS_ρq_tot = ᶜSqₜᵖ⁰ * ρ
    @. ᶜS_ρe_tot =
        ᶜSqₜᵖ⁰ *
        ρ *
        e_tot_0M_precipitation_sources_helper(thermo_params, ᶜts, ᶜΦ)
    for j in 1:n
        @. ᶜS_ρq_tot += ᶜSqₜᵖʲs.:($$j) * ᶜρaʲs.:($$j)
        @. ᶜS_ρe_tot +=
            ᶜSqₜᵖʲs.:($$j) *
            ᶜρaʲs.:($$j) *
            e_tot_0M_precipitation_sources_helper(
                thermo_params,
                ᶜtsʲs.:($$j),
                ᶜΦ,
            )
    end
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics0Moment,
    ::PrognosticEDMFX,
)
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ⁰, ᶜSqₜᵖʲs) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; ᶜts⁰, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model))

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    @. ᶜS_ρq_tot = ᶜSqₜᵖ⁰ * ᶜρa⁰
    @. ᶜS_ρe_tot =
        ᶜSqₜᵖ⁰ *
        ᶜρa⁰ *
        e_tot_0M_precipitation_sources_helper(thermo_params, ᶜts⁰, ᶜΦ)
    for j in 1:n
        @. ᶜS_ρq_tot += ᶜSqₜᵖʲs.:($$j) * Y.c.sgsʲs.:($$j).ρa
        @. ᶜS_ρe_tot +=
            ᶜSqₜᵖʲs.:($$j) *
            Y.c.sgsʲs.:($$j).ρa *
            e_tot_0M_precipitation_sources_helper(
                thermo_params,
                ᶜtsʲs.:($$j),
                ᶜΦ,
            )
    end
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics1Moment, _)
    (; dt) = p
    (; ᶜts, ᶜwᵣ, ᶜwₛ, ᶜu) = p.precomputed
    (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
    ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
    ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))

    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2
    ᶜ∇T = p.scratch.ᶜtemp_CT123

    # get thermodynamics and 1-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_1m_params(params)
    thp = CAP.thermodynamics_params(params)

    # compute precipitation source terms on the grid mean
    compute_precipitation_sources!(
        ᶜSᵖ,
        ᶜSᵖ_snow,
        ᶜSqₗᵖ,
        ᶜSqᵢᵖ,
        ᶜSqᵣᵖ,
        ᶜSqₛᵖ,
        Y.c.ρ,
        ᶜq_tot,
        ᶜq_liq,
        ᶜq_ice,
        ᶜq_rai,
        ᶜq_sno,
        ᶜts,
        dt,
        cmp,
        thp,
    )

    # compute precipitation sinks on the grid mean
    compute_precipitation_sinks!(
        ᶜSᵖ,
        ᶜSqᵣᵖ,
        ᶜSqₛᵖ,
        Y.c.ρ,
        ᶜq_tot,
        ᶜq_liq,
        ᶜq_ice,
        ᶜq_rai,
        ᶜq_sno,
        ᶜts,
        dt,
        cmp,
        thp,
    )
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics1Moment,
    ::DiagnosticEDMFX,
)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics1Moment,
    ::PrognosticEDMFX,
)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end
function set_precipitation_cache!(Y, p, ::Microphysics2Moment, _)
    (; dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed
    (; ᶜSnₗᵖ, ᶜSnᵣᵖ) = p.precomputed

    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜS₂ᵖ = p.scratch.ᶜtemp_scalar_2

    # get thermodynamics and microphysics params
    (; params) = p
    cmp = CAP.microphysics_2m_params(params)
    thp = CAP.thermodynamics_params(params)

    # compute warm precipitation sources on the grid mean (based on SB2006 2M scheme)
    compute_warm_precipitation_sources_2M!(
        ᶜSᵖ,
        ᶜS₂ᵖ,
        ᶜSnₗᵖ,
        ᶜSnᵣᵖ,
        ᶜSqₗᵖ,
        ᶜSqᵣᵖ,
        Y.c.ρ,
        lazy.(specific.(Y.c.ρn_liq, Y.c.ρ)),
        lazy.(specific.(Y.c.ρn_rai, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_tot, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_liq, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_ice, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_rai, Y.c.ρ)),
        lazy.(specific.(Y.c.ρq_sno, Y.c.ρ)),
        ᶜts,
        dt,
        cmp,
        thp,
    )

    #TODO - implement 2M cold processes!
    @. ᶜSqᵢᵖ = 0
    @. ᶜSqₛᵖ = 0

    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics2Moment,
    ::DiagnosticEDMFX,
)
    error("Not implemented yet")
    return nothing
end
function set_precipitation_cache!(
    Y,
    p,
    ::Microphysics2Moment,
    ::PrognosticEDMFX,
)
    # Nothing needs to be done on the grid mean. The Sources are computed
    # in edmf sub-domains.
    return nothing
end

"""
    set_precipitation_surface_fluxes!(Y, p, precipitation model)

Computes the flux of rain and snow at the surface. For the 0-moment microphysics
it is an integral of the source terms in the column. For 1-moment microphysics
it is the flux through the bottom cell face.
"""
set_precipitation_surface_fluxes!(Y, p, _) = nothing
function set_precipitation_surface_fluxes!(
    Y,
    p,
    microphysics_model::Microphysics0Moment,
)
    ᶜT = p.scratch.ᶜtemp_scalar
    (; ᶜts) = p.precomputed  # assume ᶜts has been updated
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.conservation_check

    # update total column energy source for surface energy balance
    Operators.column_integral_definite!(
        col_integrated_precip_energy_tendency,
        ᶜS_ρe_tot,
    )
    # update surface precipitation fluxes in cache for coupler's use
    thermo_params = CAP.thermodynamics_params(p.params)
    T_freeze = TD.Parameters.T_freeze(thermo_params)
    FT = eltype(p.params)
    @. ᶜT = TD.air_temperature(thermo_params, ᶜts)
    ᶜ3d_rain = @. lazy(ifelse(ᶜT >= T_freeze, ᶜS_ρq_tot, FT(0)))
    ᶜ3d_snow = @. lazy(ifelse(ᶜT < T_freeze, ᶜS_ρq_tot, FT(0)))
    Operators.column_integral_definite!(surface_rain_flux, ᶜ3d_rain)
    Operators.column_integral_definite!(surface_snow_flux, ᶜ3d_snow)
    return nothing
end
function set_precipitation_surface_fluxes!(
    Y,
    p,
    microphysics_model::Union{Microphysics1Moment, Microphysics2Moment},
)
    (; surface_rain_flux, surface_snow_flux) = p.precomputed
    (; col_integrated_precip_energy_tendency) = p.conservation_check
    (; ᶜwᵣ, ᶜwₛ, ᶜwₗ, ᶜwᵢ, ᶜwₕhₜ) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    sfc_J = Fields.level(ᶠJ, Fields.half)
    sfc_space = axes(sfc_J)

    # Jacobian-weighted extrapolation from interior to surface, consistent with
    # the reconstruction of density on cell faces, ᶠρ = ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    sfc_lev(x) =
        Fields.Field(Fields.field_values(Fields.level(x, 1)), sfc_space)
    int_J = sfc_lev(ᶜJ)
    int_ρ = sfc_lev(Y.c.ρ)
    sfc_ρ = @. lazy(int_ρ * int_J / sfc_J)

    # Constant extrapolation to surface, consistent with simple downwinding
    # Temporary scratch variables are used here until CC.field_values supports <lazy> fields
    ᶜq_rai = p.scratch.ᶜtemp_scalar
    ᶜq_sno = p.scratch.ᶜtemp_scalar_2
    ᶜq_liq = p.scratch.ᶜtemp_scalar_3
    ᶜq_ice = p.scratch.ᶜtemp_scalar_4
    @. ᶜq_rai = specific(Y.c.ρq_rai, Y.c.ρ)
    @. ᶜq_sno = specific(Y.c.ρq_sno, Y.c.ρ)
    @. ᶜq_liq = specific(Y.c.ρq_liq, Y.c.ρ)
    @. ᶜq_ice = specific(Y.c.ρq_ice, Y.c.ρ)
    sfc_qᵣ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_rai, 1)), sfc_space)
    sfc_qₛ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_sno, 1)), sfc_space)
    sfc_qₗ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_liq, 1)), sfc_space)
    sfc_qᵢ =
        Fields.Field(Fields.field_values(Fields.level(ᶜq_ice, 1)), sfc_space)
    sfc_wᵣ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵣ, 1)), sfc_space)
    sfc_wₛ = Fields.Field(Fields.field_values(Fields.level(ᶜwₛ, 1)), sfc_space)
    sfc_wₗ = Fields.Field(Fields.field_values(Fields.level(ᶜwₗ, 1)), sfc_space)
    sfc_wᵢ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵢ, 1)), sfc_space)
    sfc_wₕhₜ = Fields.Field(
        Fields.field_values(Fields.level(ᶜwₕhₜ.components.data.:1, 1)),
        sfc_space,
    )

    @. surface_rain_flux = sfc_ρ * (sfc_qᵣ * (-sfc_wᵣ) + sfc_qₗ * (-sfc_wₗ))
    @. surface_snow_flux = sfc_ρ * (sfc_qₛ * (-sfc_wₛ) + sfc_qᵢ * (-sfc_wᵢ))
    @. col_integrated_precip_energy_tendency = sfc_ρ * (-sfc_wₕhₜ)

    return nothing
end
