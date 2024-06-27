#####
##### Precomputed quantities
#####
import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Microphysics2M as CM2

# helper function to safely get precipitation from state
function qₚ(ρqₚ::FT, ρ::FT) where {FT}
    return max(FT(0), ρqₚ / ρ)
end

"""
    set_precipitation_precomputed_quantities!(Y, p, t, precip_model)

Updates the precipitation terminal velocity and tracers stored in cache
"""
function set_precipitation_precomputed_quantities!(Y, p, t, _)
    return nothing
end
function set_precipitation_precomputed_quantities!(Y, p, t, ::Microphysics1Moment)

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

function set_precipitation_precomputed_quantities!(Y, p, t, ::Microphysics2Moment)

    (; ᶜwᵣ, ᶜw_nᵣ, ᶜqᵣ) = p.precomputed

    cmp = CAP.microphysics_precipitation_params(p.params)

    # compute the precipitation specific humidities
    @. ᶜqᵣ = qₚ(Y.c.ρq_rai, Y.c.ρ)
    @. ᶜwᵣ = getindex(
        CM2.rain_terminal_velocity(
                cmp.SB2006,
                cmp.SB2006Vel,
                ᶜqᵣ,
                Y.c.ρ,
                Y.c.ρn_rai,
        ),
        2,
    )
    @. ᶜw_nᵣ = getindex(
        CM2.rain_terminal_velocity(
                cmp.SB2006,
                cmp.SB2006Vel,
                ᶜqᵣ,
                Y.c.ρ,
                Y.c.ρn_rai,
        ),
        1,
    )
    return nothing
end
