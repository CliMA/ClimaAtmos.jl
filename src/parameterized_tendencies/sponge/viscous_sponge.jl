#####
##### Viscous sponge
#####

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces


viscous_sponge_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

αₘ(s::ViscousSponge{FT}, z) where {FT} = ifelse(z > s.zd, s.κ₂, FT(0))
ζ_viscous(s::ViscousSponge{FT}, z, zmax) where {FT} =
    sin(FT(π) / 2 * (z - s.zd) / (zmax - s.zd))^2
β_viscous(s::ViscousSponge{FT}, z, zmax) where {FT} =
    αₘ(s, z) * ζ_viscous(s, z, zmax)

function viscous_sponge_tendency!(Yₜ, Y, p, t, s::ViscousSponge)
    (; ᶜh_tot, ᶜspecific) = p.precomputed
    ᶜuₕ = Y.c.uₕ
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = z_max(axes(ᶠz))
    @. Yₜ.c.uₕ +=
        β_viscous(s, ᶜz, zmax) * (
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.project(
                Geometry.Covariant12Axis(),
                wcurlₕ(Geometry.project(Geometry.Covariant3Axis(), curlₕ(ᶜuₕ))),
            )
        )
    @. Yₜ.f.u₃.components.data.:1 +=
        β_viscous(s, ᶠz, zmax) * wdivₕ(gradₕ(Y.f.u₃.components.data.:1))

    @. Yₜ.c.ρe_tot += β_viscous(s, ᶜz, zmax) * wdivₕ(Y.c.ρ * gradₕ(ᶜh_tot))
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        @. ᶜρχₜ += β_viscous(s, ᶜz, zmax) * wdivₕ(Y.c.ρ * gradₕ(ᶜχ))
        @. Yₜ.c.ρ += β_viscous(s, ᶜz, zmax) * wdivₕ(Y.c.ρ * gradₕ(ᶜχ))
    end
end
