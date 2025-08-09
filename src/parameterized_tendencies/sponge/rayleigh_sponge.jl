#####
##### Rayleigh sponge
#####

import ClimaCore.Fields as Fields

αₘ(s::RayleighSponge{FT}, z, α) where {FT} = ifelse(z > s.zd, α, FT(0))
ζ_rayleigh(s::RayleighSponge{FT}, z, zmax) where {FT} =
    sin(FT(π) / 2 * (z - s.zd) / (zmax - s.zd))^2
β_rayleigh_uₕ(s::RayleighSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z, s.α_uₕ) * ζ_rayleigh(s, z, zmax)
β_rayleigh_w(s::RayleighSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z, s.α_w) * ζ_rayleigh(s, z, zmax)
β_rayleigh_sgs_tracer(s::RayleighSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z, s.α_sgs_tracer) * ζ_rayleigh(s, z, zmax)

function rayleigh_sponge_tendency_uₕ(ᶜuₕ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_uₕ(s, ᶜz, zmax) * ᶜuₕ)
end

function rayleigh_sponge_tendency_sgs_tracer(ᶜχ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜχ))
    zmax = z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_sgs_tracer(s, ᶜz, zmax) * ᶜχ)
end

function rayleigh_sponge_tendency_sgs_tracer(ᶜχʲ, ᶜχ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜχ))
    zmax = z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_sgs_tracer(s, ᶜz, zmax) * (ᶜχʲ - ᶜχ))
end
