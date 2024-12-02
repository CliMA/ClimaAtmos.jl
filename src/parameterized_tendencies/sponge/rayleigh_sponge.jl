#####
##### Rayleigh sponge
#####

import ClimaCore.Fields as Fields

rayleigh_sponge_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

αₘ(s::RayleighSponge{FT}, z, α) where {FT} = ifelse(z > s.zd, α, FT(0))
ζ_rayleigh(s::RayleighSponge{FT}, z, zmax) where {FT} =
    sin(FT(π) / 2 * (z - s.zd) / (zmax - s.zd))^2
β_rayleigh_uₕ(s::RayleighSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z, s.α_uₕ) * ζ_rayleigh(s, z, zmax)
β_rayleigh_w(s::RayleighSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z, s.α_w) * ζ_rayleigh(s, z, zmax)

function rayleigh_sponge_tendency!(Yₜ, Y, p, t, s::RayleighSponge)
    ᶜz = Fields.coordinate_field(Y.c).z
    zmax = z_max(axes(Y.f))
    @. Yₜ.c.uₕ -= β_rayleigh_uₕ(s, ᶜz, zmax) * Y.c.uₕ
end
