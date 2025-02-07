import ClimaCore: Fields

function vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed

    if !(p.atmos.moisture_model isa DryModel)
        NVTX.@range "ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwₜqₜ)))" begin
            @. Yₜ.c.ρ -= ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwₜqₜ)))
        end
        @. Yₜ.c.ρe_tot -=
            ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwₕhₜ)))
        @. Yₜ.c.ρq_tot -=
            ᶜprecipdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠright_bias(-(ᶜwₜqₜ)))
    end
    return nothing
end
