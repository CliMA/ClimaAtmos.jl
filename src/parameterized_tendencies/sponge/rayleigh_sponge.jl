#####
##### Rayleigh sponge
#####

import ClimaCore.Fields as Fields

rayleigh_sponge_cache(Y, atmos::AtmosModel) =
    rayleigh_sponge_cache(Y, atmos.rayleigh_sponge)

rayleigh_sponge_cache(Y, ::Nothing) = (;)
rayleigh_sponge_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

rayleigh_sponge_cache(Y, rs::RayleighSponge) = nothing

αₘ(s::RayleighSponge{FT}, z, α) where {FT} = ifelse(z > s.zd, α, FT(0))
ζ_rayleigh(s::RayleighSponge{FT}, z) where {FT} =
    sin(FT(π) / 2 * (z - s.zd) / (s.zmax - s.zd))^2
β_rayleigh_uₕ(s::RayleighSponge{FT}, z) where {FT} =
    αₘ(s, z, s.α_uₕ) * ζ_rayleigh(s, z)
β_rayleigh_w(s::RayleighSponge{FT}, z) where {FT} =
    αₘ(s, z, s.α_w) * ζ_rayleigh(s, z)

function rayleigh_sponge_tendency!(Yₜ, Y, p, t, s::RayleighSponge)
    ᶜz = Fields.coordinate_field(Y.c).z
    @. Yₜ.c.uₕ -= β_rayleigh_uₕ(s, ᶜz) * Y.c.uₕ
end
