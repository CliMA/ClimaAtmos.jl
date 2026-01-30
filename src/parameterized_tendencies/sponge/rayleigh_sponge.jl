#####
##### Rayleigh sponge
#####

import ClimaCore.Fields as Fields

αₘ(s::RayleighSponge, z, α) = ifelse(z > s.zd, α, zero(α))
ζ_rayleigh(s::RayleighSponge, z, zmax) = sinpi((z - s.zd) / (zmax - s.zd) / 2)^2
β_rayleigh_uₕ(s::RayleighSponge, z, zmax) = αₘ(s, z, s.α_uₕ) * ζ_rayleigh(s, z, zmax)
β_rayleigh_u₃(s::RayleighSponge, z, zmax) = αₘ(s, z, s.α_w) * ζ_rayleigh(s, z, zmax)
β_rayleigh_sgs_tracer(s::RayleighSponge, z, zmax) =
    αₘ(s, z, s.α_sgs_tracer) * ζ_rayleigh(s, z, zmax)

function rayleigh_sponge_tendency_uₕ(ᶜuₕ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_uₕ(s, ᶜz, zmax) * ᶜuₕ)
end

function rayleigh_sponge_tendency_u₃(ᶠu₃, s)
    s isa Nothing && return NullBroadcasted()
    ᶠz = Fields.coordinate_field(ᶠu₃).z
    zmax = z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_u₃(s, ᶠz, zmax) * ᶠu₃)
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
