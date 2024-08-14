#####
##### Precipitation models
#####

import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics as CM
import Cloudy as CL
import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields
import ClimaCore.Utilities: half

precipitation_cache(Y, atmos::AtmosModel) =
    precipitation_cache(Y, atmos.precip_model)

#####
##### No Precipitation
#####

precipitation_cache(Y, precip_model::NoPrecipitation) = (;)
precipitation_tendency!(Yₜ, Y, p, t, colidx, ::NoPrecipitation, _) = nothing

#####
##### 0-Moment without sgs scheme or with diagnostic/prognostic edmf
#####

function precipitation_cache(Y, precip_model::Microphysics0Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜS_ρq_tot = similar(Y.c, FT),
        ᶜS_ρe_tot = similar(Y.c, FT),
        ᶜ3d_rain = similar(Y.c, FT),
        ᶜ3d_snow = similar(Y.c, FT),
        col_integrated_rain = zeros(
            axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
        ),
        col_integrated_snow = zeros(
            axes(Fields.level(Geometry.WVector.(Y.f.u₃), half)),
        ),
    )
end

function compute_precipitation_cache!(Y, p, colidx, ::Microphysics0Moment, _)
    (; params, dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precipitation
    (; ᶜΦ) = p.core
    cm_params = CAP.microphysics_precipitation_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜS_ρq_tot[colidx] =
        Y.c.ρ[colidx] * q_tot_precipitation_sources(
            Microphysics0Moment(),
            thermo_params,
            cm_params,
            dt,
            Y.c.ρq_tot[colidx] / Y.c.ρ[colidx],
            ᶜts[colidx],
        )
    @. ᶜS_ρe_tot[colidx] =
        ᶜS_ρq_tot[colidx] * e_tot_0M_precipitation_sources_helper(
            thermo_params,
            ᶜts[colidx],
            ᶜΦ[colidx],
        )
end
function compute_precipitation_cache!(
    Y,
    p,
    colidx,
    ::Microphysics0Moment,
    ::DiagnosticEDMFX,
)
    # For environment we multiply by grid mean ρ and not byᶜρa⁰
    # assuming a⁰=1
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ⁰, ᶜSqₜᵖʲs, ᶜρaʲs) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precipitation
    (; ᶜts, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    ρ = Y.c.ρ

    @. ᶜS_ρq_tot[colidx] = ᶜSqₜᵖ⁰[colidx] * ρ[colidx]
    @. ᶜS_ρe_tot[colidx] =
        ᶜSqₜᵖ⁰[colidx] *
        ρ[colidx] *
        e_tot_0M_precipitation_sources_helper(
            thermo_params,
            ᶜts[colidx],
            ᶜΦ[colidx],
        )
    for j in 1:n
        @. ᶜS_ρq_tot[colidx] += ᶜSqₜᵖʲs.:($$j)[colidx] * ᶜρaʲs.:($$j)[colidx]
        @. ᶜS_ρe_tot[colidx] +=
            ᶜSqₜᵖʲs.:($$j)[colidx] *
            ᶜρaʲs.:($$j)[colidx] *
            e_tot_0M_precipitation_sources_helper(
                thermo_params,
                ᶜtsʲs.:($$j)[colidx],
                ᶜΦ[colidx],
            )
    end
end
function compute_precipitation_cache!(
    Y,
    p,
    colidx,
    ::Microphysics0Moment,
    ::PrognosticEDMFX,
)
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ⁰, ᶜSqₜᵖʲs, ᶜρa⁰) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precipitation
    (; ᶜts⁰, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    @. ᶜS_ρq_tot[colidx] = ᶜSqₜᵖ⁰[colidx] * ᶜρa⁰[colidx]
    @. ᶜS_ρe_tot[colidx] =
        ᶜSqₜᵖ⁰[colidx] *
        ᶜρa⁰[colidx] *
        e_tot_0M_precipitation_sources_helper(
            thermo_params,
            ᶜts⁰[colidx],
            ᶜΦ[colidx],
        )
    for j in 1:n
        @. ᶜS_ρq_tot[colidx] +=
            ᶜSqₜᵖʲs.:($$j)[colidx] * Y.c.sgsʲs.:($$j).ρa[colidx]
        @. ᶜS_ρe_tot[colidx] +=
            ᶜSqₜᵖʲs.:($$j)[colidx] *
            Y.c.sgsʲs.:($$j).ρa[colidx] *
            e_tot_0M_precipitation_sources_helper(
                thermo_params,
                ᶜtsʲs.:($$j)[colidx],
                ᶜΦ[colidx],
            )
    end
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics0Moment,
    _,
)
    (; ᶜT, ᶜΦ) = p.core
    (; ᶜts) = p.precomputed  # assume ᶜts has been updated
    (; params) = p
    (; turbconv_model) = p.atmos
    (; ᶜ3d_rain, ᶜ3d_snow, ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precipitation
    (; col_integrated_rain, col_integrated_snow) = p.precipitation
    (; col_integrated_precip_energy_tendency,) = p.conservation_check
    thermo_params = CAP.thermodynamics_params(params)

    # Compute the ρq_tot and ρe_tot precipitation source terms
    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)

    # Add the source terms to the tendencies
    @. Yₜ.c.ρq_tot[colidx] += ᶜS_ρq_tot[colidx]
    @. Yₜ.c.ρ[colidx] += ᶜS_ρq_tot[colidx]
    @. Yₜ.c.ρe_tot[colidx] += ᶜS_ρe_tot[colidx]

    # update total column energy source for surface energy balance
    Operators.column_integral_definite!(
        col_integrated_precip_energy_tendency[colidx],
        ᶜS_ρe_tot[colidx],
    )
    # update precip in cache for coupler's use
    # 3d rain and snow
    T_freeze = TD.Parameters.T_freeze(thermo_params)
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])
    @. ᶜ3d_rain[colidx] = ifelse(ᶜT[colidx] >= T_freeze, ᶜS_ρq_tot[colidx], 0)
    @. ᶜ3d_snow[colidx] = ifelse(ᶜT[colidx] < T_freeze, ᶜS_ρq_tot[colidx], 0)
    Operators.column_integral_definite!(
        col_integrated_rain[colidx],
        ᶜ3d_rain[colidx],
    )
    Operators.column_integral_definite!(
        col_integrated_snow[colidx],
        ᶜ3d_snow[colidx],
    )
    return nothing
end

#####
##### 1-Moment without sgs scheme
#####

function precipitation_cache(Y, precip_model::Microphysics1Moment)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜSqₜᵖ = similar(Y.c, FT),
        ᶜSqᵣᵖ = similar(Y.c, FT),
        ᶜSqₛᵖ = similar(Y.c, FT),
        ᶜSeₜᵖ = similar(Y.c, FT),
    )
end

function compute_precipitation_cache!(Y, p, colidx, ::Microphysics1Moment, _)
    FT = Spaces.undertype(axes(Y.c))
    (; dt) = p
    (; ᶜts, ᶜqᵣ, ᶜqₛ) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation
    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2

    # get thermodynamics and 1-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_precipitation_params(params)
    thp = CAP.thermodynamics_params(params)

    # compute precipitation source terms on the grid mean
    compute_precipitation_sources!(
        ᶜSᵖ[colidx],
        ᶜSᵖ_snow[colidx],
        ᶜSqₜᵖ[colidx],
        ᶜSqᵣᵖ[colidx],
        ᶜSqₛᵖ[colidx],
        ᶜSeₜᵖ[colidx],
        Y.c.ρ[colidx],
        ᶜqᵣ[colidx],
        ᶜqₛ[colidx],
        ᶜts[colidx],
        ᶜΦ[colidx],
        dt,
        cmp,
        thp,
    )

    # compute precipitation sinks
    # (For now only done on the grid mean)
    compute_precipitation_sinks!(
        ᶜSᵖ[colidx],
        ᶜSqₜᵖ[colidx],
        ᶜSqᵣᵖ[colidx],
        ᶜSqₛᵖ[colidx],
        ᶜSeₜᵖ[colidx],
        Y.c.ρ[colidx],
        ᶜqᵣ[colidx],
        ᶜqₛ[colidx],
        ᶜts[colidx],
        ᶜΦ[colidx],
        dt,
        cmp,
        thp,
    )
end
function compute_precipitation_cache!(
    Y,
    p,
    colidx,
    ::Microphysics1Moment,
    ::Union{DiagnosticEDMFX, PrognosticEDMFX},
)
    FT = Spaces.undertype(axes(Y.c))
    (; dt) = p
    (; ᶜts, ᶜqᵣ, ᶜqₛ) = p.precomputed
    (; ᶜΦ) = p.core
    # Grid mean precipitation sinks
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation
    # additional scratch storage
    ᶜSᵖ = p.scratch.ᶜtemp_scalar

    # get thermodynamics and 1-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_precipitation_params(params)
    thp = CAP.thermodynamics_params(params)

    # zero out the helper source terms
    @. ᶜSqₜᵖ[colidx] = FT(0)
    @. ᶜSqᵣᵖ[colidx] = FT(0)
    @. ᶜSqₛᵖ[colidx] = FT(0)
    @. ᶜSeₜᵖ[colidx] = FT(0)
    # compute precipitation sinks
    # (For now only done on the grid mean)
    compute_precipitation_sinks!(
        ᶜSᵖ[colidx],
        ᶜSqₜᵖ[colidx],
        ᶜSqᵣᵖ[colidx],
        ᶜSqₛᵖ[colidx],
        ᶜSeₜᵖ[colidx],
        Y.c.ρ[colidx],
        ᶜqᵣ[colidx],
        ᶜqₛ[colidx],
        ᶜts[colidx],
        ᶜΦ[colidx],
        dt,
        cmp,
        thp,
    )
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics1Moment,
    _,
)
    (; turbconv_model) = p.atmos
    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)

    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    # Update grid mean tendencies
    @. Yₜ.c.ρ[colidx] += Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρe_tot[colidx] += Y.c.ρ[colidx] * ᶜSeₜᵖ[colidx]
    @. Yₜ.c.ρq_rai[colidx] += Y.c.ρ[colidx] * ᶜSqᵣᵖ[colidx]
    @. Yₜ.c.ρq_sno[colidx] += Y.c.ρ[colidx] * ᶜSqₛᵖ[colidx]

    return nothing
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics1Moment,
    turbconv_model::DiagnosticEDMFX,
)
    # Source terms from EDMFX environment
    (; ᶜSeₜᵖ⁰, ᶜSqₜᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰) = p.precomputed
    # Source terms from EDMFX updrafts
    (; ᶜSeₜᵖʲs, ᶜSqₜᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs) = p.precomputed
    # Grid mean precipitation sinks
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    (; ᶜρaʲs) = p.precomputed

    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)

    # Update from environment precipitation sources
    # and the grid mean precipitation sinks
    @. Yₜ.c.ρ[colidx] += Y.c.ρ[colidx] * (ᶜSqₜᵖ⁰[colidx] + ᶜSqₜᵖ[colidx])
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * (ᶜSqₜᵖ⁰[colidx] + ᶜSqₜᵖ[colidx])
    @. Yₜ.c.ρe_tot[colidx] += Y.c.ρ[colidx] * (ᶜSeₜᵖ⁰[colidx] + ᶜSeₜᵖ[colidx])
    @. Yₜ.c.ρq_rai[colidx] += Y.c.ρ[colidx] * (ᶜSqᵣᵖ⁰[colidx] + ᶜSqᵣᵖ[colidx])
    @. Yₜ.c.ρq_sno[colidx] += Y.c.ρ[colidx] * (ᶜSqₛᵖ⁰[colidx] + ᶜSqₛᵖ[colidx])

    # Update from the updraft precipitation sources
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρ[colidx] += ᶜρaʲs.:($$j)[colidx] * ᶜSqₜᵖʲs.:($$j)[colidx]
        @. Yₜ.c.ρq_tot[colidx] += ᶜρaʲs.:($$j)[colidx] * ᶜSqₜᵖʲs.:($$j)[colidx]
        @. Yₜ.c.ρe_tot[colidx] += ᶜρaʲs.:($$j)[colidx] * ᶜSeₜᵖʲs.:($$j)[colidx]
        @. Yₜ.c.ρq_rai[colidx] += ᶜρaʲs.:($$j)[colidx] * ᶜSqᵣᵖʲs.:($$j)[colidx]
        @. Yₜ.c.ρq_sno[colidx] += ᶜρaʲs.:($$j)[colidx] * ᶜSqₛᵖʲs.:($$j)[colidx]
    end
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    precip_model::Microphysics1Moment,
    turbconv_model::PrognosticEDMFX,
)
    # Source terms from EDMFX environment
    (; ᶜSeₜᵖ⁰, ᶜSqₜᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰, ᶜρa⁰) = p.precomputed
    # Source terms from EDMFX updrafts
    (; ᶜSeₜᵖʲs, ᶜSqₜᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs) = p.precomputed
    # Grid mean precipitation sinks
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    compute_precipitation_cache!(Y, p, colidx, precip_model, turbconv_model)

    # Update from environment precipitation sources
    # and the grid mean precipitation sinks
    @. Yₜ.c.ρ[colidx] +=
        ᶜρa⁰[colidx] * ᶜSqₜᵖ⁰[colidx] + Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρq_tot[colidx] +=
        ᶜρa⁰[colidx] * ᶜSqₜᵖ⁰[colidx] + Y.c.ρ[colidx] * ᶜSqₜᵖ[colidx]
    @. Yₜ.c.ρe_tot[colidx] +=
        ᶜρa⁰[colidx] * ᶜSeₜᵖ⁰[colidx] + Y.c.ρ[colidx] * ᶜSeₜᵖ[colidx]
    @. Yₜ.c.ρq_rai[colidx] +=
        ᶜρa⁰[colidx] * ᶜSqᵣᵖ⁰[colidx] + Y.c.ρ[colidx] * ᶜSqᵣᵖ[colidx]
    @. Yₜ.c.ρq_sno[colidx] +=
        ᶜρa⁰[colidx] * ᶜSqₛᵖ⁰[colidx] + Y.c.ρ[colidx] * ᶜSqₛᵖ[colidx]

    # Update from the updraft precipitation sources
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρ[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * ᶜSqₜᵖʲs.:($$j)[colidx]
        @. Yₜ.c.ρq_tot[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * ᶜSqₜᵖʲs.:($$j)[colidx]
        @. Yₜ.c.ρe_tot[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * ᶜSeₜᵖʲs.:($$j)[colidx]
        @. Yₜ.c.ρq_rai[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * ᶜSqᵣᵖʲs.:($$j)[colidx]
        @. Yₜ.c.ρq_sno[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * ᶜSqₛᵖʲs.:($$j)[colidx]
    end
end


#####
##### Cloudy without sgs scheme
#####
function separate_liq_rai(FT, moments, pdists, cloudy_params, ρd)
    tmp = CL.ParticleDistributions.get_standard_N_q(pdists, cloudy_params.size_threshold / cloudy_params.norms[2])
    moments_like = ntuple(length(moments)) do k
        if k == 1
            max(tmp.N_liq * cloudy_params.mom_norms[1], FT(0))
        elseif k == 2
            max(tmp.N_rai * cloudy_params.mom_norms[1], FT(0))
        elseif k == 3
            max(tmp.M_liq / ρd * cloudy_params.mom_norms[2], FT(0))
        elseif k == 4
            max(tmp.M_rai / ρd * cloudy_params.mom_norms[2], FT(0))
        else
            FT(0)
        end
    end
    return moments_like
end