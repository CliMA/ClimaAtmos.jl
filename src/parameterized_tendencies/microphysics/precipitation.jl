#####
##### Precipitation models
#####

import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics as CM
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

function precipitation_cache(Y, precip_model::NoPrecipitation)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        surface_rain_flux = zeros(axes(Fields.level(Y.f, half))),
        surface_snow_flux = zeros(axes(Fields.level(Y.f, half))),
    )
end
precipitation_tendency!(Yₜ, Y, p, t, ::NoPrecipitation, _) = nothing

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
        surface_rain_flux = zeros(axes(Fields.level(Y.f, half))),
        surface_snow_flux = zeros(axes(Fields.level(Y.f, half))),
    )
end

function compute_precipitation_cache!(Y, p, ::Microphysics0Moment, _)
    (; params, dt) = p
    (; ᶜts) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precipitation
    (; ᶜΦ) = p.core
    cm_params = CAP.microphysics_precipitation_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜS_ρq_tot =
        Y.c.ρ * q_tot_precipitation_sources(
            Microphysics0Moment(),
            thermo_params,
            cm_params,
            dt,
            Y.c.ρq_tot / Y.c.ρ,
            ᶜts,
        )
    @. ᶜS_ρe_tot =
        ᶜS_ρq_tot *
        e_tot_0M_precipitation_sources_helper(thermo_params, ᶜts, ᶜΦ)
end
function compute_precipitation_cache!(
    Y,
    p,
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
end
function compute_precipitation_cache!(
    Y,
    p,
    ::Microphysics0Moment,
    ::PrognosticEDMFX,
)
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ⁰, ᶜSqₜᵖʲs, ᶜρa⁰) = p.precomputed
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precipitation
    (; ᶜts⁰, ᶜtsʲs) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

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
end

function compute_precipitation_surface_fluxes!(
    Y,
    p,
    precip_model::Microphysics0Moment,
)
    ᶜT = p.scratch.ᶜtemp_scalar
    (; ᶜts) = p.precomputed  # assume ᶜts has been updated
    (; ᶜ3d_rain, ᶜ3d_snow, ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precipitation
    (; surface_rain_flux, surface_snow_flux) = p.precipitation
    (; col_integrated_precip_energy_tendency,) = p.conservation_check

    # update total column energy source for surface energy balance
    Operators.column_integral_definite!(
        col_integrated_precip_energy_tendency,
        ᶜS_ρe_tot,
    )
    # update surface precipitation fluxes in cache for coupler's use
    thermo_params = CAP.thermodynamics_params(p.params)
    T_freeze = TD.Parameters.T_freeze(thermo_params)
    @. ᶜ3d_rain = ifelse(ᶜT >= T_freeze, ᶜS_ρq_tot, 0)
    @. ᶜ3d_snow = ifelse(ᶜT < T_freeze, ᶜS_ρq_tot, 0)
    Operators.column_integral_definite!(surface_rain_flux, ᶜ3d_rain)
    Operators.column_integral_definite!(surface_snow_flux, ᶜ3d_snow)
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    precip_model::Microphysics0Moment,
    _,
)
    (; turbconv_model) = p.atmos
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precipitation

    # Compute the ρq_tot and ρe_tot precipitation source terms
    compute_precipitation_cache!(Y, p, precip_model, turbconv_model)
    # Compute surface precipitation flux
    compute_precipitation_surface_fluxes!(Y, p, precip_model)

    # Add the source terms to the tendencies
    @fused_direct begin
        @. Yₜ.c.ρq_tot += ᶜS_ρq_tot
        @. Yₜ.c.ρ += ᶜS_ρq_tot
    end
    @. Yₜ.c.ρe_tot += ᶜS_ρe_tot

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
        surface_rain_flux = zeros(axes(Fields.level(Y.f, half))),
        surface_snow_flux = zeros(axes(Fields.level(Y.f, half))),
    )
end

function compute_precipitation_cache!(Y, p, ::Microphysics1Moment, _)
    FT = Spaces.undertype(axes(Y.c))
    (; dt) = p
    (; ᶜts, ᶜqᵣ, ᶜqₛ, ᶜwᵣ, ᶜwₛ, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2
    ᶜ∇T = p.scratch.ᶜtemp_CT123

    # get thermodynamics and 1-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_precipitation_params(params)
    thp = CAP.thermodynamics_params(params)

    # compute precipitation source terms on the grid mean
    compute_precipitation_sources!(
        ᶜSᵖ,
        ᶜSᵖ_snow,
        ᶜSqₜᵖ,
        ᶜSqᵣᵖ,
        ᶜSqₛᵖ,
        ᶜSeₜᵖ,
        Y.c.ρ,
        ᶜqᵣ,
        ᶜqₛ,
        ᶜts,
        ᶜΦ,
        dt,
        cmp,
        thp,
    )

    # compute precipitation sinks
    # (For now only done on the grid mean)
    compute_precipitation_sinks!(
        ᶜSᵖ,
        ᶜSqₜᵖ,
        ᶜSqᵣᵖ,
        ᶜSqₛᵖ,
        ᶜSeₜᵖ,
        Y.c.ρ,
        ᶜqᵣ,
        ᶜqₛ,
        ᶜts,
        ᶜΦ,
        dt,
        cmp,
        thp,
    )
    # first term of eq 36 from Raymond 2013
    compute_precipitation_heating!(ᶜSeₜᵖ, ᶜwᵣ, ᶜwₛ, ᶜu, ᶜqᵣ, ᶜqₛ, ᶜts, ᶜ∇T, thp)
end
function compute_precipitation_cache!(
    Y,
    p,
    ::Microphysics1Moment,
    ::Union{DiagnosticEDMFX, PrognosticEDMFX},
)
    FT = Spaces.undertype(axes(Y.c))
    (; dt) = p
    (; ᶜts, ᶜqᵣ, ᶜqₛ, ᶜwᵣ, ᶜwₛ, ᶜu) = p.precomputed
    (; ᶜΦ) = p.core
    # Grid mean precipitation sinks
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation
    # additional scratch storage
    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜ∇T = p.scratch.ᶜtemp_CT123

    # get thermodynamics and 1-moment microphysics params
    (; params) = p
    cmp = CAP.microphysics_precipitation_params(params)
    thp = CAP.thermodynamics_params(params)

    # zero out the helper source terms
    @. ᶜSqₜᵖ = FT(0)
    @. ᶜSqᵣᵖ = FT(0)
    @. ᶜSqₛᵖ = FT(0)
    @. ᶜSeₜᵖ = FT(0)
    # compute precipitation sinks
    # (For now only done on the grid mean)
    compute_precipitation_sinks!(
        ᶜSᵖ,
        ᶜSqₜᵖ,
        ᶜSqᵣᵖ,
        ᶜSqₛᵖ,
        ᶜSeₜᵖ,
        Y.c.ρ,
        ᶜqᵣ,
        ᶜqₛ,
        ᶜts,
        ᶜΦ,
        dt,
        cmp,
        thp,
    )
    # first term of eq 36 from Raymond 2013
    compute_precipitation_heating!(ᶜSeₜᵖ, ᶜwᵣ, ᶜwₛ, ᶜu, ᶜqᵣ, ᶜqₛ, ᶜts, ᶜ∇T, thp)
end

function compute_precipitation_surface_fluxes!(
    Y,
    p,
    precip_model::Microphysics1Moment,
)
    (; surface_rain_flux, surface_snow_flux) = p.precipitation
    (; col_integrated_precip_energy_tendency,) = p.conservation_check
    (; ᶜwᵣ, ᶜwₛ, ᶜspecific) = p.precomputed

    (; ᶠtemp_scalar) = p.scratch
    slg = Fields.level(Fields.local_geometry_field(ᶠtemp_scalar), Fields.half)

    # Constant extrapolation: - put values from bottom cell center to bottom cell face
    ˢρ = Fields.Field(Fields.field_values(Fields.level(Y.c.ρ, 1)), axes(slg))
    # For density this is equivalent with ᶠwinterp(ᶜJ, Y.c.ρ) and therefore
    # consistent with the way we do vertical advection
    ˢqᵣ = Fields.Field(
        Fields.field_values(Fields.level(ᶜspecific.q_rai, 1)),
        axes(slg),
    )
    ˢqₛ = Fields.Field(
        Fields.field_values(Fields.level(ᶜspecific.q_sno, 1)),
        axes(slg),
    )
    ˢwᵣ = Fields.Field(Fields.field_values(Fields.level(ᶜwᵣ, 1)), axes(slg))
    ˢwₛ = Fields.Field(Fields.field_values(Fields.level(ᶜwₛ, 1)), axes(slg))

    # Project the flux to CT3 vector and convert to physical units.
    @. surface_rain_flux =
        -projected_vector_data(CT3, ˢρ * ˢqᵣ * Geometry.WVector(ˢwᵣ), slg)
    @. surface_snow_flux =
        -projected_vector_data(CT3, ˢρ * ˢqₛ * Geometry.WVector(ˢwₛ), slg)
end

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    precip_model::Microphysics1Moment,
    _,
)
    (; turbconv_model) = p.atmos
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    # Populate the cache and precipitation surface fluxes
    compute_precipitation_cache!(Y, p, precip_model, turbconv_model)
    compute_precipitation_surface_fluxes!(Y, p, precip_model)

    # Update grid mean tendencies
    @fused_direct begin
        @. Yₜ.c.ρ += Y.c.ρ * ᶜSqₜᵖ
        @. Yₜ.c.ρq_tot += Y.c.ρ * ᶜSqₜᵖ
        @. Yₜ.c.ρe_tot += Y.c.ρ * ᶜSeₜᵖ
        @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜSqᵣᵖ
        @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜSqₛᵖ
    end

    return nothing
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
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

    # Populate the cache and precipitation surface fluxes
    compute_precipitation_cache!(Y, p, precip_model, turbconv_model)
    compute_precipitation_surface_fluxes!(Y, p, precip_model)

    # Update from environment precipitation sources
    # and the grid mean precipitation sinks
    @fused_direct begin
        @. Yₜ.c.ρ += Y.c.ρ * (ᶜSqₜᵖ⁰ + ᶜSqₜᵖ)
        @. Yₜ.c.ρq_tot += Y.c.ρ * (ᶜSqₜᵖ⁰ + ᶜSqₜᵖ)
        @. Yₜ.c.ρe_tot += Y.c.ρ * (ᶜSeₜᵖ⁰ + ᶜSeₜᵖ)
        @. Yₜ.c.ρq_rai += Y.c.ρ * (ᶜSqᵣᵖ⁰ + ᶜSqᵣᵖ)
        @. Yₜ.c.ρq_sno += Y.c.ρ * (ᶜSqₛᵖ⁰ + ᶜSqₛᵖ)
    end

    # Update from the updraft precipitation sources
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @fused_direct begin
            @. Yₜ.c.ρ += ᶜρaʲs.:($$j) * ᶜSqₜᵖʲs.:($$j)
            @. Yₜ.c.ρq_tot += ᶜρaʲs.:($$j) * ᶜSqₜᵖʲs.:($$j)
            @. Yₜ.c.ρe_tot += ᶜρaʲs.:($$j) * ᶜSeₜᵖʲs.:($$j)
            @. Yₜ.c.ρq_rai += ᶜρaʲs.:($$j) * ᶜSqᵣᵖʲs.:($$j)
            @. Yₜ.c.ρq_sno += ᶜρaʲs.:($$j) * ᶜSqₛᵖʲs.:($$j)
        end
    end
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    precip_model::Microphysics1Moment,
    turbconv_model::PrognosticEDMFX,
)
    # Source terms from EDMFX environment
    (; ᶜSeₜᵖ⁰, ᶜSqₜᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰, ᶜρa⁰) = p.precomputed
    # Source terms from EDMFX updrafts
    (; ᶜSeₜᵖʲs, ᶜSqₜᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs) = p.precomputed
    # Grid mean precipitation sinks
    (; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    # Populate the cache and precipitation surface fluxes
    compute_precipitation_cache!(Y, p, precip_model, turbconv_model)
    compute_precipitation_surface_fluxes!(Y, p, precip_model)

    # Update from environment precipitation sources
    # and the grid mean precipitation sinks
    @fused_direct begin
        @. Yₜ.c.ρ += ᶜρa⁰ * ᶜSqₜᵖ⁰ + Y.c.ρ * ᶜSqₜᵖ
        @. Yₜ.c.ρq_tot += ᶜρa⁰ * ᶜSqₜᵖ⁰ + Y.c.ρ * ᶜSqₜᵖ
        @. Yₜ.c.ρe_tot += ᶜρa⁰ * ᶜSeₜᵖ⁰ + Y.c.ρ * ᶜSeₜᵖ
        @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜSqᵣᵖ⁰ + Y.c.ρ * ᶜSqᵣᵖ
        @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜSqₛᵖ⁰ + Y.c.ρ * ᶜSqₛᵖ
    end

    # Update from the updraft precipitation sources
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @fused_direct begin
            @. Yₜ.c.ρ += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵖʲs.:($$j)
            @. Yₜ.c.ρq_tot += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵖʲs.:($$j)
            @. Yₜ.c.ρe_tot += Y.c.sgsʲs.:($$j).ρa * ᶜSeₜᵖʲs.:($$j)
            @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜSqᵣᵖʲs.:($$j)
            @. Yₜ.c.ρq_sno += Y.c.sgsʲs.:($$j).ρa * ᶜSqₛᵖʲs.:($$j)
        end
    end
end
