#####
##### Viscous sponge
#####

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

αₘ(s::ViscousSponge, z) = ifelse(z > s.zd, s.κ₂, zero(s.κ₂))
ζ_viscous(s::ViscousSponge, z, zmax) = sinpi((z - s.zd) / (zmax - s.zd) / 2)^2

"""
    β_viscous(s::ViscousSponge, z, zmax)

Return the viscous sponge coefficient
``β(z) = κ₂ · sin²(π (z - zd) / (2 (zmax - zd)))`` for `z > zd` and zero below [m²/s].

# Arguments

  - `s`: The `ViscousSponge` model, with lower damping height `zd` [m] and damping
    coefficient `κ₂` [m²/s].
  - `z`: Altitude [m].
  - `zmax`: Domain top height [m].
"""
β_viscous(s::ViscousSponge, z, zmax) = αₘ(s, z) * ζ_viscous(s, z, zmax)

"""
    viscous_sponge_tendency_uₕ(ᶜuₕ, s)

Return a lazy broadcast of the viscous sponge tendency `β_viscous(s, z, zmax) * ∇ₕ²uₕ` for
the cell-center horizontal velocity, or a `NullBroadcasted` when `s isa Nothing` or the
space is a single column (horizontal operators are undefined there).

The horizontal vector Laplacian is computed as `∇ₕ²u = ∇ₕ(∇ₕ·u) - ∇ₕ×(∇ₕ×u)`, projected
onto the horizontal (Covariant12) axis.
"""
function viscous_sponge_tendency_uₕ(ᶜuₕ, s)
    if s isa Nothing || axes(ᶜuₕ) isa Spaces.FiniteDifferenceSpace
        return NullBroadcasted()
    end
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = Spaces.z_max(axes(ᶠz))
    axis_C12 = (Geometry.Covariant12Axis(),)
    axis_C3 = (Geometry.Covariant3Axis(),)
    # vector Laplacian: ∇²u = ∇·∇u - ∇×(∇×u)
    ᶜ∇²uₕ = @. lazy(
        wgradₕ(divₕ(ᶜuₕ)) -
        Geometry.project(axis_C12, wcurlₕ(Geometry.project(axis_C3, curlₕ(ᶜuₕ)))),
    )
    return @. lazy(β_viscous(s, ᶜz, zmax) * ᶜ∇²uₕ)
end

"""
    viscous_sponge_tendency_u₃(u₃, s)

Return a lazy broadcast of the viscous sponge tendency `β_viscous(s, z, zmax) * ∇ₕ²u₃` for
the physical component of the cell-face vertical velocity, or a `NullBroadcasted` when
`s isa Nothing`.
"""
function viscous_sponge_tendency_u₃(u₃, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶠz) = z_coordinate_fields(axes(u₃))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶠz, zmax) * wdivₕ(gradₕ(u₃.components.data.:1)))
end

"""
    viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, s)

Return a lazy broadcast of the viscous sponge tendency
`β_viscous(s, z, zmax) * ∇ₕ·(ρ ∇ₕh_tot)` for total energy density, or a `NullBroadcasted`
when `s isa Nothing`.

Energy is diffused via the total specific enthalpy `ᶜh_tot` rather than `ρe_tot` itself.
"""
function viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜh_tot)))
end

"""
    viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, s)

Return a lazy broadcast of the viscous sponge tendency
`β_viscous(s, z, zmax) * ∇ₕ·(ρ ∇ₕχ)` for a cell-center specific tracer `ᶜχ`, or a
`NullBroadcasted` when `s isa Nothing`.
"""
function viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜχ)))
end
