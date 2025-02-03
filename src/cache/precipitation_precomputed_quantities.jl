#####
##### Precomputed quantities
#####
import CloudMicrophysics.Microphysics1M as CM1

# helper function to safely get precipitation from state
function qₚ(ρqₚ, ρ)
    return max(zero(ρ), ρqₚ / ρ)
end

"""
    set_precipitation_precomputed_quantities!(Y, p, t)

Updates the precipitation terminal velocity stored in `p`
for the 1-moment microphysics scheme
"""
function set_precipitation_precomputed_quantities!(Y, p, t)
    @assert (p.atmos.precip_model isa Microphysics1Moment)

    (; ᶜwᵣ, ᶜwₛ, ᶜqᵣ, ᶜqₛ) = p.precomputed

    cmp = CAP.microphysics_1m_params(p.params)

    # compute the precipitation specific humidities
    @. ᶜqᵣ = qₚ(Y.c.ρq_rai, Y.c.ρ)
    @. ᶜqₛ = qₚ(Y.c.ρq_sno, Y.c.ρ)

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwᵣ = CM1.terminal_velocity(
        cmp.pr,
        cmp.tv.rain,
        Y.c.ρ,
        abs(Y.c.ρq_rai / Y.c.ρ),
    )
    @. ᶜwₛ = CM1.terminal_velocity(
        cmp.ps,
        cmp.tv.snow,
        Y.c.ρ,
        abs(Y.c.ρq_sno / Y.c.ρ),
    )
    return nothing
end


"""
    set_sedimentation_precomputed_quantities!(Y, p, t)
Updates the sedimentation terminal velocity stored in `p`
for the non-equilibrium microphysics scheme
"""
function set_sedimentation_precomputed_quantities!(Y, p, t)
    @assert (p.atmos.moisture_model isa NonEquilMoistModel)

    (; ᶜwₗ, ᶜwᵢ) = p.precomputed

    FT = eltype(Y)

    cmc = CAP.microphysics_cloud_params(p.params)

    # compute the precipitation terminal velocity [m/s]
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
    return nothing
end
