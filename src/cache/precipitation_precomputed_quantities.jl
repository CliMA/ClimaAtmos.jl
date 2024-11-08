#####
##### Precomputed quantities
#####
import CloudMicrophysics.Microphysics1M as CM1
import Cloudy as CL

# helper function to safely get precipitation from state
function qₚ(ρqₚ::FT, ρ::FT) where {FT}
    return max(FT(0), ρqₚ / ρ)
end

"""
    set_precipitation_precomputed_quantities!(Y, p, t)

Updates the precipitation terminal velocity stored in `p`
for the 1-moment or N-moment microphysics scheme
"""
function set_precipitation_precomputed_quantities!(Y, p, t, _)
    return nothing
end

function set_precipitation_precomputed_quantities!(Y, p, t, precip_model::Microphysics1Moment)
    (; ᶜwᵣ, ᶜwₛ, ᶜqᵣ, ᶜqₛ) = p.precomputed

    cmp = CAP.microphysics_precipitation_params(p.params)

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