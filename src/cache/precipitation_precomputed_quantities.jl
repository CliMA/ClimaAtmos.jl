#####
##### Precomputed quantities
#####
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.MicrophysicsNonEq as CMNe

"""
    set_precipitation_precomputed_quantities!(Y, p, t)

Updates the precipitation terminal velocity stored in `p`
for the 1-moment microphysics scheme
"""
function set_precipitation_precomputed_quantities!(Y, p, t)
    @assert (p.atmos.precip_model isa Microphysics1Moment)

    (; ᶜwᵣ, ᶜwₛ) = p.precomputed
    cmp = CAP.microphysics_1m_params(p.params)

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
    cmc = CAP.microphysics_cloud_params(p.params)
    FT = eltype(p.params)

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwₗ = FT(0) #CMNe.terminal_velocity(
    #    cmc.liquid,
    #    cmc.Ch2022.rain,
    #    Y.c.ρ,
    #    max(zero(Y.c.ρ), Y.c.ρq_liq / Y.c.ρ),
    #)
    @. ᶜwᵢ = FT(0) #CMNe.terminal_velocity(
    #    cmc.ice,
    #    cmc.Ch2022.small_ice,
    #    Y.c.ρ,
    #    max(zero(Y.c.ρ), Y.c.ρq_ice / Y.c.ρ),
    #)
    return nothing
end
