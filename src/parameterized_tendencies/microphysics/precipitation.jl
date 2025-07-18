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
    microphysics_model::Microphysics0Moment,
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
    microphysics_model::Microphysics1Moment,
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
    microphysics_model::Microphysics1Moment,
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
    microphysics_model::Microphysics1Moment,
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
    microphysics_model::Microphysics1Moment,
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
    microphysics_model::Microphysics1Moment,
    turbconv_model::PrognosticEDMFX,
)
    # Source terms from EDMFX updrafts
    (; ᶜSqₗᵖʲs, ᶜSqᵢᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs) = p.precomputed
    # Source terms from EDMFX environment
    (; ᶜSqₗᵖ⁰, ᶜSqᵢᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰, ᶜρa⁰) = p.precomputed

    # Update from environment precipitation and cloud formation sources/sinks
    @. Yₜ.c.ρq_liq += ᶜρa⁰ * ᶜSqₗᵖ⁰
    @. Yₜ.c.ρq_ice += ᶜρa⁰ * ᶜSqᵢᵖ⁰
    @. Yₜ.c.ρq_rai += ᶜρa⁰ * ᶜSqᵣᵖ⁰
    @. Yₜ.c.ρq_sno += ᶜρa⁰ * ᶜSqₛᵖ⁰

    # Update from the updraft precipitation sources
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.ρq_liq += Y.c.sgsʲs.:($$j).ρa * ᶜSqₗᵖʲs.:($$j)
        @. Yₜ.c.ρq_ice += Y.c.sgsʲs.:($$j).ρa * ᶜSqᵢᵖʲs.:($$j)
        @. Yₜ.c.ρq_rai += Y.c.sgsʲs.:($$j).ρa * ᶜSqᵣᵖʲs.:($$j)
        @. Yₜ.c.ρq_sno += Y.c.sgsʲs.:($$j).ρa * ᶜSqₛᵖʲs.:($$j)
    end
end

#####
##### 2-moment microphysics without sgs scheme
#####

function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::DryModel,
    microphysics_model::Microphysics2Moment,
    _,
)
    error("Microphysics2Moment precipitation should not be used with DryModel")
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::EquilMoistModel,
    microphysics_model::Microphysics2Moment,
    _,
)
    error(
        "Microphysics2Moment precipitation and EquilMoistModel precipitation_tendency is not implemented",
    )
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonEquilMoistModel,
    microphysics_model::Microphysics2Moment,
    _,
)
    (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed
    (; ᶜSnₗᵖ, ᶜSnᵣᵖ) = p.precomputed

    # Update grid mean tendencies
    @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜSqₗᵖ
    @. Yₜ.c.ρq_ice += Y.c.ρ * ᶜSqᵢᵖ
    @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜSqᵣᵖ
    @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜSqₛᵖ

    @. Yₜ.c.ρn_liq += Y.c.ρ * ᶜSnₗᵖ
    @. Yₜ.c.ρn_rai += Y.c.ρ * ᶜSnᵣᵖ

    return nothing
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonEquilMoistModel,
    microphysics_model::Microphysics2Moment,
    turbconv_model::DiagnosticEDMFX,
)
    error("Not implemented yet")
end
function precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonEquilMoistModel,
    microphysics_model::Microphysics2Moment,
    turbconv_model::PrognosticEDMFX,
)
    error("Not implemented yet")
end
