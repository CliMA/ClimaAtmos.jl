#####
##### Precomputed quantities
#####
import CloudMicrophysics.Microphysics1M as CM1

"""
    set_precipitation_precomputed_quantities!(Y, p, t)

Updates the precipitation terminal velocity stored in `p`
for the 1-moment microphysics scheme
"""
function set_precipitation_precomputed_quantities!(Y, p, t)
    @assert (p.atmos.precip_model isa Microphysics1Moment)

    (; ᶜwᵣ, ᶜwₛ) = p.precomputed

    cmp = CAP.microphysics_params(p.params)

    # compute the precipitation terminal velocity [m/s]
    @. ᶜwᵣ =
        CM1.terminal_velocity(cmp.pr, cmp.tv.rain, Y.c.ρ, Y.c.ρq_rai / Y.c.ρ)
    @. ᶜwₛ =
        CM1.terminal_velocity(cmp.ps, cmp.tv.snow, Y.c.ρ, Y.c.ρq_sno / Y.c.ρ)
    return nothing
end
