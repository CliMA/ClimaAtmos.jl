#####
##### Shock Capturing for Split Divergence Advection
#####
##### Provides artificial viscosity to prevent oscillations and negative values
##### in scalar advection using split_divₕ
#####

"""
    compute_shock_indicator(∇χ_mag, threshold)

Computes a shock indicator function that activates in regions with sharp gradients.

The indicator is based on the gradient magnitude and provides smooth activation
in regions where oscillations might occur.

# Arguments
- `∇χ_mag`: Magnitude of the gradient of the scalar field
- `threshold`: Threshold value for gradient magnitude

# Returns
- Shock indicator value (0 to 1, where 1 indicates strong shock/gradient)
"""
function compute_shock_indicator(∇χ_mag, threshold)
    FT = eltype(∇χ_mag)
    # Smooth activation function: indicator = tanh((|∇χ| - threshold) / threshold)
    # This provides smooth transition from 0 (no shock) to 1 (strong shock)
    normalized_grad = @. (∇χ_mag - threshold) / max(threshold, FT(1e-10))
    indicator = @. max(FT(0), tanh(normalized_grad))
    return indicator
end

"""
    split_divₕ_with_shock_capturing(ρu, χ, shock_capturing)

Computes split divergence advection with shock capturing to prevent oscillations
and negative values.

This function wraps `split_divₕ` and adds an artificial viscosity term that
activates in regions with sharp gradients.

# Arguments
- `ρu`: Density-weighted velocity field (flux) or velocity field
- `χ`: Scalar field to advect
- `shock_capturing`: ShockCapturing configuration (or nothing)

# Returns
- Advection tendency with optional shock capturing: `-split_divₕ(ρu, χ) + ∇·(ν_shock ∇χ)`
"""
function split_divₕ_with_shock_capturing(ρu, χ, shock_capturing)
    # Base advection using split divergence
    advection = @. -split_divₕ(ρu, χ)
    
    # If shock capturing is disabled or not configured, return base advection
    if isnothing(shock_capturing) || !shock_capturing.enabled
        return advection
    end
    
    # Compute gradient of scalar field for shock indicator
    ∇χ = @. gradₕ(χ)
    ∇χ_mag = @. sqrt(∇χ · ∇χ)
    
    # Compute shock indicator based on gradient magnitude
    shock_indicator = compute_shock_indicator(∇χ_mag, shock_capturing.threshold)
    
    # Artificial viscosity: ν_shock = coeff * indicator
    # Materialize to ensure it's a concrete field for multiplication
    ν_shock = Base.materialize(@. shock_capturing.coeff * shock_indicator)
    
    # Add diffusion term: ∇·(ν_shock ∇χ) using weak divergence
    # This provides stabilization in regions with sharp gradients
    # Compute gradient inside wdivₕ and multiply by materialized scalar field
    # (matching pattern from hyperdiffusion.jl line 169: wdivₕ(ᶜρ * gradₕ(...)))
    diffusion = @. wdivₕ(ν_shock * gradₕ(χ))
    
    return @. advection + diffusion
end
