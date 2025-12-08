#####
##### Viscous sponge
#####

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

αₘ(s::ViscousSponge{FT}, z) where {FT} = ifelse(z > s.zd, s.κ₂, FT(0))
ζ_viscous(s::ViscousSponge{FT}, z, zmax) where {FT} =
    sin(FT(π) / 2 * (z - s.zd) / (zmax - s.zd))^2
β_viscous(s::ViscousSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z) * ζ_viscous(s, z, zmax)

function viscous_sponge_tendency_uₕ(ᶜuₕ, s)
    if s isa Nothing || axes(ᶜuₕ) isa Spaces.FiniteDifferenceSpace
        return NullBroadcasted()
    end
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = z_max(axes(ᶠz))
    return @. lazy(
        β_viscous(s, ᶜz, zmax) * (
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
                Geometry.Covariant12Axis(),
                wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
            )
        ),
    )
end

function viscous_sponge_tendency_u₃(u₃, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶠz) = z_coordinate_fields(axes(u₃))
    zmax = z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶠz, zmax) * wdivₕ(gradₕ(u₃.components.data.:1)))
end

function viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    zmax = z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜh_tot)))
end

function viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    zmax = z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜχ)))
end
