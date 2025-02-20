import ClimaCore: Fields

function vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed

    if !(p.atmos.moisture_model isa DryModel)
        @. Yₜ.c.ρ -= ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwₜqₜ)))
        @. Yₜ.c.ρe_tot -=
            ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwₕhₜ)))
        @. Yₜ.c.ρq_tot -=
            ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwₜqₜ)))

        if p.atmos.moisture_model isa NonEquilMoistModel
            (; ᶜwₗ, ᶜwᵢ) = p.precomputed
            @. Yₜ.c.ρq_liq -=
            ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwₗqₗ))) 
            @. Yₜ.c.ρq_ice -=
            ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwᵢqᵢ)))
        end
    end
    return nothing
end
