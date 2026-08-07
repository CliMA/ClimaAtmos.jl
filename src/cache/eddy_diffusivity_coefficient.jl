"""
    ᶜcompute_eddy_diffusivity_coefficient(ᶜρ, vert_diff::DecayWithHeightDiffusion)
    ᶜcompute_eddy_diffusivity_coefficient(ᶜuₕ, ᶜp, vert_diff::VerticalDiffusion)

Return a lazy broadcast of the eddy diffusivity at cell centers [m²/s] for the
given vertical diffusion model.

For `DecayWithHeightDiffusion`, the diffusivity decays exponentially with height
above the surface with scale height `H` from its surface value `D₀`. For
`VerticalDiffusion`, it is built from the bulk transfer coefficient `C_E`, the
wind speed at the lowest level, and half the depth of the lowest cell, and is
tapered above the planetary boundary layer as a function of pressure; see
`eddy_diffusivity_coefficient_H` and `eddy_diffusivity_coefficient` in
`precomputed_quantities.jl`.
"""
function ᶜcompute_eddy_diffusivity_coefficient(
    ᶜρ,
    vert_diff::DecayWithHeightDiffusion,
)
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    ᶠz_sfc = Fields.level(ᶠz, Fields.half)
    return @. lazy(
        eddy_diffusivity_coefficient_H(vert_diff.D₀, vert_diff.H, ᶠz_sfc, ᶜz),
    )
end

function ᶜcompute_eddy_diffusivity_coefficient(
    ᶜuₕ,
    ᶜp,
    vert_diff::VerticalDiffusion,
)
    interior_uₕ = Fields.level(ᶜuₕ, 1)
    ᶜΔz_surface = Fields.Δz_field(interior_uₕ)
    return @. lazy(
        eddy_diffusivity_coefficient(
            vert_diff.C_E,
            norm(interior_uₕ),
            ᶜΔz_surface / 2,
            ᶜp,
        ),
    )
end
