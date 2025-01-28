#####
##### Sponge tendencies
#####

import LazyBroadcast: @lazy
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

function z_coordinate_fields(space::Spaces.AbstractSpace)
    ᶜz = Fields.coordinate_field(Spaces.center_space(space)).z
    ᶠz = Fields.coordinate_field(Spaces.face_space(space)).z
    return (; ᶜz, ᶠz)
end


function sponge_tendencies!(Yₜ, Y, p, t)
    rs, vs = p.atmos.rayleigh_sponge, p.atmos.viscous_sponge
    (; ᶜh_tot, ᶜspecific) = p.precomputed
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Yₜ.f.u₃
    ᶜρ = Y.c.ρ
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = z_max(axes(ᶠz))
    vst_uₕ = viscous_sponge_tendency_uₕ(ᶜuₕ, vs)
    vst_u₃ = viscous_sponge_tendency_u₃(ᶠu₃, vs)
    vst_ρe_tot = viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, vs)
    rst_uₕ = rayleigh_sponge_tendency_uₕ(ᶜuₕ, rs)

    @. Yₜ.c.uₕ += vst_uₕ + rst_uₕ
    @. Yₜ.f.u₃.components.data.:1 += vst_u₃
    @. Yₜ.c.ρe_tot += vst_ρe_tot

    # TODO: can we write this out explicitly?
    if vs isa ViscousSponge
        for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
            χ_name == :e_tot && continue
            @. ᶜρχₜ += β_viscous(vs, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜχ))
            @. Yₜ.c.ρ += β_viscous(vs, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        end
    end
end
