
"""
    ᶜcompute_eddy_diffusivity_coefficient(ᶜY, vertical_diffusion::DecayWithHeightDiffusion)

Return lazy representation of the vertical profile of eddy diffusivity 
for the `DecayWithHeightDiffusion` model.

The profile is given by:
```
K(z) = D₀ ⋅ exp(-(z - z_sfc) / H)
```

# Arguments
- `ᶜY`: Cell-centered prognostic state vector whose axes provide vertical coordinates.
- `vertical_diffusion`: Instance of `DecayWithHeightDiffusion` model, with fields:
    - `D₀`: Surface eddy diffusivity magnitude.
    - `H`: E-folding height for the exponential decay.

# See also
- [`DecayWithHeightDiffusion`] for the model specification
- [`vertical_diffusion_boundary_layer_tendency!`] where this coefficient is applied
"""
function ᶜcompute_eddy_diffusivity_coefficient(
    ᶜY, vertical_diffusion::DecayWithHeightDiffusion,
)
    (; D₀, H) = vertical_diffusion
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜY))
    ᶠz_sfc = Fields.level(ᶠz, Fields.half)
    return @. lazy(eddy_diffusivity_coefficient_H(D₀, H, ᶠz_sfc, ᶜz))
end
eddy_diffusivity_coefficient_H(D₀, H, z_sfc, z) = D₀ * exp(-(z - z_sfc) / H)

"""
    ᶜcompute_eddy_diffusivity_coefficient(ᶜY, vertical_diffusion::VerticalDiffusion)

Return lazy representation of the vertical profile of eddy diffusivity 
for the `VerticalDiffusion` model.

The profile is given by:
```
K(z) = K_E, if p > p_pbl
     = K_E * exp(-((p_pbl - p) / p_strato)^2), otherwise
```
where `K_E` is given by:
```
K_E = C_E ⋅ norm(ᶜuₕ(z_bot)) ⋅ Δz_bot / 2
```
where `z_bot` is the first interior center level of the model,
and `Δz_bot` is the thickness of the surface layer.

# Arguments
- `ᶜY`: Cell-centered prognostic state vector from which we extract:
    - `ᶜY.uₕ`: Cell-centered horizontal velocity field; its first interior level is used.
    - `ᶜY.p`: Cell-centered thermodynamic pressure field (or proxy) used by the closure.
- `vertical_diffusion`: Instance of `VerticalDiffusion` model, with field `C_E`:
    - `C_E`: Dimensionless eddy-coefficient factor.

# See also
- [`VerticalDiffusion`] for the model specification
"""
function ᶜcompute_eddy_diffusivity_coefficient(ᶜY, vertical_diffusion::VerticalDiffusion)
    (; C_E) = vertical_diffusion
    interior_uₕ = Fields.level(ᶜY.uₕ, 1)
    ᶜΔz_surface = Fields.Δz_field(interior_uₕ)
    return @. lazy(
        eddy_diffusivity_coefficient(C_E, norm(interior_uₕ), ᶜΔz_surface / 2, ᶜY.p),
    )
end

function eddy_diffusivity_coefficient(C_E, norm_uₕ_bottom, Δz_bottom, p)
    p_pbl = 85000
    p_strato = 10000
    K_E = C_E * norm_uₕ_bottom * Δz_bottom
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end
