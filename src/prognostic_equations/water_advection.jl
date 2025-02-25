import ClimaCore: Fields

function vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed

    if !(p.atmos.moisture_model isa DryModel)
        @. Yₜ.c.ρ -=
            ᶜprecipdivᵥ(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(-(ᶜwₜqₜ)))
        @. Yₜ.c.ρe_tot -=
            ᶜprecipdivᵥ(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(-(ᶜwₕhₜ)))
        @. Yₜ.c.ρq_tot -=
            ᶜprecipdivᵥ(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(-(ᶜwₜqₜ)))
    end
    return nothing
end
