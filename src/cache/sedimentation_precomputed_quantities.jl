#####
##### Precomputed quantities
#####
import CloudMicrophysics.MicrophysicsNonEq as CMNe

"""
    set_sedimentation_precomputed_quantities!(Y, p, t)

Updates the sedimentation terminal velocity stored in `p`
for the non-equilibrium microphysics scheme
"""
function set_sedimentation_precomputed_quantities!(Y, p, t)
    @assert (p.atmos.moisture_model isa NonEquilMoistModel)

    (; ᶜwₗ, ᶜwᵢ) = p.precomputed
    cmc = CAP.microphysics_cloud_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwₗ = CMNe.terminal_velocity(
        $(Ref(cmc.liquid)),
        $(Ref(cmc.Ch2022.rain)),
        Y.c.ρ,
        max(0, Y.c.ρq_liq / Y.c.ρ),
    )
    @. ᶜwᵢ = CMNe.terminal_velocity(
        $(Ref(cmc.ice)),
        $(Ref(cmc.Ch2022.small_ice)),
        Y.c.ρ,
        max(0, Y.c.ρq_ice / Y.c.ρ),
    )
    return nothing
end
