#####
##### Viscous sponge
#####

import LazyBroadcast: @lazy
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

αₘ(s::ViscousSponge{FT}, z) where {FT} = ifelse(z > s.zd, s.κ₂, FT(0))
ζ_viscous(s::ViscousSponge{FT}, z, zmax) where {FT} =
    sin(FT(π) / 2 * (z - s.zd) / (zmax - s.zd))^2
β_viscous(s::ViscousSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z) * ζ_viscous(s, z, zmax)

viscous_sponge_tendency_uₕ(ᶜuₕ, ::Nothing) = (zero(eltype(ᶜuₕ)),)
function viscous_sponge_tendency_uₕ(ᶜuₕ, s::ViscousSponge)
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = z_max(axes(ᶠz))
    return @lazy @. β_viscous(s, ᶜz, zmax) * (
        wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
            Geometry.Covariant12Axis(),
            wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
        )
    )
end

viscous_sponge_tendency_u₃(u₃, ::Nothing) =
    (zero(eltype(u₃.components.data.:1)),)
function viscous_sponge_tendency_u₃(u₃, s::ViscousSponge)
    (; ᶠz) = z_coordinate_fields(axes(u₃))
    zmax = z_max(axes(ᶠz))
    return @lazy @. β_viscous(s, ᶠz, zmax) * wdivₕ(gradₕ(u₃.components.data.:1))
end

viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, ::Nothing) = (zero(eltype(ᶜρ)),)
function viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, s::ViscousSponge)
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    zmax = z_max(axes(ᶠz))
    return @lazy @. β_viscous(s, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜh_tot))
end
