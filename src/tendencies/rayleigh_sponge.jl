#####
##### Rayleigh sponge
#####

import ClimaCore.Fields as Fields

function rayleigh_sponge_cache(
    Y,
    dt::FT;
    zd_rayleigh = FT(15e3),
    α_rayleigh_uₕ = FT(1e-4),
    α_rayleigh_w = FT(1),
) where {FT}
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜαₘ_uₕ = @. ifelse(ᶜz > zd_rayleigh, α_rayleigh_uₕ, FT(0))
    ᶠαₘ_w = @. ifelse(ᶠz > zd_rayleigh, α_rayleigh_w, FT(0))
    zmax = maximum(ᶠz)
    ᶜβ_rayleigh_uₕ =
        @. ᶜαₘ_uₕ * sin(FT(π) / 2 * (ᶜz - zd_rayleigh) / (zmax - zd_rayleigh))^2
    ᶠβ_rayleigh_w =
        @. ᶠαₘ_w * sin(FT(π) / 2 * (ᶠz - zd_rayleigh) / (zmax - zd_rayleigh))^2
    return (; ᶜβ_rayleigh_uₕ, ᶠβ_rayleigh_w)
end

function rayleigh_sponge_tendency!(Yₜ, Y, p, t, colidx)
    (; ᶜβ_rayleigh_uₕ) = p
    @. Yₜ.c.uₕ[colidx] -= ᶜβ_rayleigh_uₕ[colidx] * Y.c.uₕ[colidx]
end
