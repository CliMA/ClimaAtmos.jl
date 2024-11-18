#####
##### Precomputed quantities
#####
import CloudMicrophysics.MicrophysicsNonEq as CMNe

# TODO - duplicated with precip, should be moved to some common helper module
# helper function to safely get precipitation from state
function q_cc(ρq::FT, ρ::FT) where {FT}
    return max(FT(0), ρq / ρ)
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

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwₗ = CMNe.terminal_velocity(
        cmc.liquid,
        cmc.Ch2022.rain,
        Y.c.ρ,
        q_cc(Y.c.ρq_liq, Y.c.ρ),
    )
    @. ᶜwᵢ = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        Y.c.ρ,
        q_cc(Y.c.ρq_ice, Y.c.ρ),
    )
    return nothing
end
