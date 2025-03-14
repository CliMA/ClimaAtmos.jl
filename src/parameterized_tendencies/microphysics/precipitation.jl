#####
##### No Precipitation
#####

precipitation_tendency!(Yₜ, Y, p, t, _, ::NoPrecipitation, _) = nothing

#####
##### 0-moment microphysics w/wo sgs model
#####

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::DryModel,
    ::Microphysics0Moment,
    _,
)
    error("Microphysics0Moment precipitation should not be run with DryModel.")
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::EquilMoistModel,
    precip_model::Microphysics0Moment,
    turbconv_model,
)
    (; ᶜS_ρq_tot, ᶜS_ρe_tot) = p.precomputed

    # Add the source terms to the tendencies
    @. Yₜ.c.ρq_tot += ᶜS_ρq_tot
    @. Yₜ.c.ρ += ᶜS_ρq_tot
    @. Yₜ.c.ρe_tot += ᶜS_ρe_tot

    return nothing
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonEquilMoistModel,
    ::Microphysics0Moment,
    _,
)
    error(
        "Microphysics0Moment precipitation and NonEquilibriumMost model precipitation_tendency has not been implemented.",
    )
end

#####
##### 1-moment microphysics without sgs scheme
#####

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::DryModel,
    precip_model::Microphysics1Moment,
    _,
)
    error("Microphysics1Moment precipitation should not be used with DryModel")
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::EquilMoistModel,
    precip_model::Microphysics1Moment,
    _,
)
    error(
        "Microphysics1Moment precipitation and EquilMoistModel precipitation_tendency is not implemented",
    )
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonEquilMoistModel,
    precip_model::Microphysics1Moment,
    _,
)
    (; turbconv_model) = p.atmos
    (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed

    # Update grid mean tendencies
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜSqₗᵖ
    @. Yₜ.c.ρq_ice += Y.c.ρ * ᶜSqᵢᵖ
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜSqᵣᵖ
    @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜSqₛᵖ

    return nothing
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonEquilMoistModel,
    precip_model::Microphysics1Moment,
    turbconv_model::DiagnosticEDMFX,
)
    error("Not implemented yet")
    ## Source terms from EDMFX environment
    #(; ᶜSeₜᵖ⁰, ᶜSqₜᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰) = p.precomputed
    ## Source terms from EDMFX updrafts
    #(; ᶜSeₜᵖʲs, ᶜSqₜᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs) = p.precomputed
    ## Grid mean precipitation sinks
    #(; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    #(; ᶜρaʲs) = p.precomputed

    ## Populate the cache and precipitation surface fluxes
    #compute_precipitation_cache!(Y, p, precip_model, turbconv_model)
    #compute_precipitation_surface_fluxes!(Y, p, precip_model)

    ## Update from environment precipitation sources
    ## and the grid mean precipitation sinks
    #@. Yₜ.c.ρ += Y.c.ρ * (ᶜSqₜᵖ⁰ + ᶜSqₜᵖ)
    #@. Yₜ.c.ρq_tot += Y.c.ρ * (ᶜSqₜᵖ⁰ + ᶜSqₜᵖ)
    #@. Yₜ.c.ρe_tot += Y.c.ρ * (ᶜSeₜᵖ⁰ + ᶜSeₜᵖ)
    #@. Yₜ.c.ρq_rai += Y.c.ρ * (ᶜSqᵣᵖ⁰ + ᶜSqᵣᵖ)
    #@. Yₜ.c.ρq_sno += Y.c.ρ * (ᶜSqₛᵖ⁰ + ᶜSqₛᵖ)

    ## Update from the updraft precipitation sources
    #n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    #for j in 1:n
    #    @. Yₜ.c.ρ += ᶜρaʲs.:($$j) * ᶜSqₜᵖʲs.:($$j)
    #    @. Yₜ.c.ρq_tot += ᶜρaʲs.:($$j) * ᶜSqₜᵖʲs.:($$j)
    #    @. Yₜ.c.ρe_tot += ᶜρaʲs.:($$j) * ᶜSeₜᵖʲs.:($$j)
    #    @. Yₜ.c.ρq_rai += ᶜρaʲs.:($$j) * ᶜSqᵣᵖʲs.:($$j)
    #    @. Yₜ.c.ρq_sno += ᶜρaʲs.:($$j) * ᶜSqₛᵖʲs.:($$j)
    #end
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonEquilMoistModel,
    precip_model::Microphysics1Moment,
    turbconv_model::PrognosticEDMFX,
)
    error("Not implemented yet")
    ## Source terms from EDMFX environment
    #(; ᶜSeₜᵖ⁰, ᶜSqₜᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰, ᶜρa⁰) = p.precomputed
    ## Source terms from EDMFX updrafts
    #(; ᶜSeₜᵖʲs, ᶜSqₜᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs) = p.precomputed
    ## Grid mean precipitation sinks
    #(; ᶜSqₜᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ, ᶜSeₜᵖ) = p.precipitation

    ## Populate the cache and precipitation surface fluxes
    #compute_precipitation_cache!(Y, p, precip_model, turbconv_model)
    #compute_precipitation_surface_fluxes!(Y, p, precip_model)

    ## Update from environment precipitation sources
    ## and the grid mean precipitation sinks
    #@. Yₜ.c.ρ += ᶜρa⁰ * ᶜSqₜᵖ⁰ + Y.c.ρ * ᶜSqₜᵖ
    #@. Yₜ.c.ρq_tot += ᶜρa⁰ * ᶜSqₜᵖ⁰ + Y.c.ρ * ᶜSqₜᵖ
    #@. Yₜ.c.ρe_tot += ᶜρa⁰ * ᶜSeₜᵖ⁰ + Y.c.ρ * ᶜSeₜᵖ
    #@. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜSqᵣᵖ⁰ + Y.c.ρ * ᶜSqᵣᵖ
    #@. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜSqₛᵖ⁰ + Y.c.ρ * ᶜSqₛᵖ

    ## Update from the updraft precipitation sources
    #n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    #for j in 1:n
    #    @. Yₜ.c.ρ += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵖʲs.:($$j)
    #    @. Yₜ.c.ρq_tot += Y.c.sgsʲs.:($$j).ρa * ᶜSqₜᵖʲs.:($$j)
    #    @. Yₜ.c.ρe_tot += Y.c.sgsʲs.:($$j).ρa * ᶜSeₜᵖʲs.:($$j)
    #    @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜSqᵣᵖʲs.:($$j)
    #    @. Yₜ.c.ρq_sno += Y.c.sgsʲs.:($$j).ρa * ᶜSqₛᵖʲs.:($$j)
    #end
end
