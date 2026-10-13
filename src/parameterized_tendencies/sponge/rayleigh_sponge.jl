#####
##### Rayleigh sponge
#####

import ClimaCore: Fields, Spaces

αₘ(s::RayleighSponge, z, α) = ifelse(z > s.zd, α, zero(α))
ζ_rayleigh(s::RayleighSponge, z, zmax) = sinpi((z - s.zd) / (zmax - s.zd) / 2)^2

"""
    β_rayleigh_uₕ(s::RayleighSponge, z, zmax)
    β_rayleigh_u₃(s::RayleighSponge, z, zmax)
    β_rayleigh_tracer(s::RayleighSponge, z, zmax)

Return the Rayleigh damping rate ``β(z) = α · sin²(π (z - zd) / (2 (zmax - zd)))`` for
`z > zd` and zero below, where `α` is the variable-specific damping coefficient
(`s.α_uₕ`, `s.α_w`, or `s.α_tracer`) [1/s].

# Arguments

  - `s`: The `RayleighSponge` model, with lower damping height `zd` [m].
  - `z`: Altitude [m].
  - `zmax`: Domain top height [m].
"""
β_rayleigh_uₕ(s::RayleighSponge, z, zmax) = αₘ(s, z, s.α_uₕ) * ζ_rayleigh(s, z, zmax)
β_rayleigh_u₃(s::RayleighSponge, z, zmax) = αₘ(s, z, s.α_w) * ζ_rayleigh(s, z, zmax)
β_rayleigh_tracer(s::RayleighSponge, z, zmax) =
    αₘ(s, z, s.α_tracer) * ζ_rayleigh(s, z, zmax)

"""
    rayleigh_sponge_tendency_uₕ(ᶜuₕ, s)

Return a lazy broadcast of the Rayleigh sponge tendency `-β_rayleigh_uₕ(s, z, zmax) * ᶜuₕ`
for the cell-center horizontal velocity, or a `NullBroadcasted` when `s isa Nothing`.

The sponge relaxes `uₕ` toward zero above the damping height `s.zd`; the returned
broadcast is materialized into `Yₜ.c.uₕ` by the caller.
"""
function rayleigh_sponge_tendency_uₕ(ᶜuₕ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_uₕ(s, ᶜz, zmax) * ᶜuₕ)
end

"""
    rayleigh_sponge_tendency_u₃(ᶠu₃, s)

Return a lazy broadcast of the Rayleigh sponge tendency `-β_rayleigh_u₃(s, z, zmax) * ᶠu₃`
for the cell-face vertical velocity, or a `NullBroadcasted` when `s isa Nothing`.
"""
function rayleigh_sponge_tendency_u₃(ᶠu₃, s)
    s isa Nothing && return NullBroadcasted()
    ᶠz = Fields.coordinate_field(ᶠu₃).z
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_u₃(s, ᶠz, zmax) * ᶠu₃)
end

"""
    rayleigh_sponge_tendency_tracer(ᶜχ, s)

Return a lazy broadcast of the Rayleigh sponge tendency
`-β_rayleigh_tracer(s, z, zmax) * ᶜχ` for a cell-center tracer (e.g. `tke` or a
condensate specific content), or a `NullBroadcasted` when `s isa Nothing`.
"""
function rayleigh_sponge_tendency_tracer(ᶜχ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜχ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_tracer(s, ᶜz, zmax) * ᶜχ)
end

"""
    rayleigh_sponge_tendency_sgs_tracer(ᶜχʲ, ᶜχ, s)

Return a lazy broadcast of the Rayleigh sponge tendency
`-β_rayleigh_tracer(s, z, zmax) * (ᶜχʲ - ᶜχ)` for an EDMFX subdomain tracer `ᶜχʲ`, or a
`NullBroadcasted` when `s isa Nothing`.

The subdomain value is relaxed toward the grid-mean value `ᶜχ` rather than toward zero, so
the sponge damps only the subgrid-scale departure.
"""
function rayleigh_sponge_tendency_sgs_tracer(ᶜχʲ, ᶜχ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜχ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(-β_rayleigh_tracer(s, ᶜz, zmax) * (ᶜχʲ - ᶜχ))
end
