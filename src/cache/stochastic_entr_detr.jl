
function set_stochastic_entrainment_with_exponential_solver!(integrator)
    (; params, precomputed) = integrator.p
    sde_params = params.stochastic_params.entr
    (; ᶜentrʲs) = precomputed
    AR1_step!(ᶜentrʲs, sde_params, integrator)
end

function set_stochastic_detrainment_with_exponential_solver!(integrator)
    (; params, precomputed) = integrator.p
    sde_params = params.stochastic_params.detr
    (; ᶜdetrʲs) = precomputed
    AR1_step!(ᶜdetrʲs, sde_params, integrator)
end

"""
AR1_step!(u, sde_params, integrator)

Given the continuous-time OU process, 
    `` du = -τ( u - μ )dt + σ dW_t, ``
solves the corresponding discrete-time AR(1) process, with the "exponential" solve,
    `` u_{n+1} = u_n e^{-τ dt} + μ (1 - e^{-τ dt}) + σ √{ (1 - e^{-2τ dt} ) / (2τ) ) z_n, ``
where `z_n` is a random number drawn from a standard normal distribution.

# Arguments
- `u`: The state to in-place evolve the AR(1) process, e.g. entrainment or detrainment.
- `sde_params`: The AR(1) parameters, a struct/NamedTuple with fields `μ`, `τ`, and `σ`.
- `integrator`: The integrator

Notes:
(;μ, θ, σ) = p
exp_minus_θh = @. exp(-θ * h)
umean = uprev * exp_minus_θh + μ * (1 - exp_minus_θh)
unoise = √(σ^2 / 2θ * (1 - exp_minus_θh^2)) * z
unext = umean + unoise
unext
"""
function AR1_step!(u, sde_params, integrator)
    (; atmos, dt) = integrator.p
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    nlevels = Spaces.nlevels(axes(integrator.u.c))
    # closure
    (; μ, τ, σ) = sde_params
    σ_z = σ√( (1 - exp(-2τ * dt)) / 2τ )
    z = integrator.p.scratch.ᶜtemp_scalar
    for colⱼ in 1:n
        uₙ₊₁ = uₙ = u.:($colⱼ)
        # Set z to a random number drawn from a standard normal distribution, for each column
        for level in 1:nlevels  # Note: only works in SCM
            z_level = Fields.field_values(Fields.level(z, level))
            z_level .= randn()
        end
        @. uₙ₊₁ = uₙ * exp(-τ * dt) + μ * (1 - exp(-τ * dt)) + √uₙ * σ_z * z
        @. uₙ₊₁ = max(uₙ₊₁, 0)
    end
end