include("../radiation_utilities.jl")

jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(Y, params, dt) = merge(
    hyperdiffusion_cache(Y; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(Y, dt) : NamedTuple(),
    zero_moment_microphysics_cache(Y),
    rrtmgp_model_cache(Y, params),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    zero_moment_microphysics_tendency!(Yₜ, Y, p, t)
    rrtmgp_model_tendency!(Yₜ, Y, p, t)
end
additional_callbacks = (PeriodicCallback(
    rrtmgp_model_callback!,
    FT(6 * 60 * 60); # update RRTMGPModel every 6 hours
    initial_affect = true,
),)
