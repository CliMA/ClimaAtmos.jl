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
    vst_uₕ = viscous_sponge_tendency_uₕ(ᶜuₕ, vs)
    vst_u₃ = viscous_sponge_tendency_u₃(ᶠu₃, vs)
    vst_ρe_tot = viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, vs)
    rst_uₕ = rayleigh_sponge_tendency_uₕ(ᶜuₕ, rs)

    # TODO: fuse, once we fix
    #       https://github.com/CliMA/ClimaCore.jl/issues/2165
    @. Yₜ.c.uₕ += vst_uₕ
    @. Yₜ.c.uₕ += rst_uₕ
    @. Yₜ.f.u₃.components.data.:1 += vst_u₃
    @. Yₜ.c.ρe_tot += vst_ρe_tot

    # TODO: can we write this out explicitly?
    if vs isa ViscousSponge
        for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
            χ_name == :e_tot && continue
            vst_tracer = viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, vs)
            @. ᶜρχₜ += vst_tracer
            @. Yₜ.c.ρ += vst_tracer
        end
    end
end
