jacobian_flags = (; ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode = :exact, ∂ᶠ𝕄ₜ∂ᶜρ_mode = :exact)

additional_cache(Y, params, dt) = merge(
    hyperdiffusion_cache(Y; κ₄ = FT(2e17)),
    sponge ? rayleigh_sponge_cache(Y, dt) : NamedTuple(),
    held_suarez_cache(Y),
    vertical_diffusion_boundary_layer_cache(Y),
    zero_moment_microphysics_cache(Y),
)
function additional_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Y, p, t)
    sponge && rayleigh_sponge_tendency!(Yₜ, Y, p, t)
    held_suarez_tendency!(Yₜ, Y, p, t)
    vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    zero_moment_microphysics_tendency!(Yₜ, Y, p, t)
end

center_initial_condition(local_geometry, params) = center_initial_condition(
    local_geometry,
    params,
    Val(:ρe_int);
    moisture_mode = Val(:equil),
)
