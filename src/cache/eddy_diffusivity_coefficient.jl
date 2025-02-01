function compute_eddy_diffusivity_coefficient(
    ᶜρ,
    vert_diff::DecayWithHeightDiffusion,
)
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    ᶠz_sfc = Fields.level(ᶠz, Fields.half)
    return @lazy @. eddy_diffusivity_coefficient_H(
        vert_diff.D₀,
        vert_diff.H,
        ᶠz_sfc,
        ᶜz,
    )
end

function compute_eddy_diffusivity_coefficient(
    ᶜuₕ,
    ᶜp,
    vert_diff::VerticalDiffusion,
)
    interior_uₕ = Fields.level(ᶜuₕ, 1)
    ᶜΔz_surface = Fields.Δz_field(interior_uₕ)
    return @lazy @. eddy_diffusivity_coefficient(
        vert_diff.C_E,
        norm(interior_uₕ),
        ᶜΔz_surface / 2,
        ᶜp,
    )
end
