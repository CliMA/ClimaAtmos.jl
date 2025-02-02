#####
##### Large-scale advection (for single column experiments)
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

function edmf_coriolis_tendency_uₕ(ᶜuₕ, edmf_coriolis)
    edmf_coriolis isa EDMFCoriolis || return NullBroadcasted()
    (; prof_ug, prof_vg) = edmf_coriolis
    ᶜspace = axes(ᶜuₕ)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    (; coriolis_param) = edmf_coriolis
    coords = Fields.coordinate_field(ᶜspace)
    coriolis_fn(coord) = Geometry.WVector(coriolis_param)
    ᶜf_coriolis = @lazy @. Geometry.Contravariant3Vector(coriolis_fn(coords))
    ᶜuₕ_g = @lazy @. Geometry.Covariant12Vector(
        Geometry.UVVector(prof_ug(ᶜz), prof_vg(ᶜz)),
    )

    # Coriolis
    C123 = Geometry.Covariant123Vector
    C12 = Geometry.Contravariant12Vector
    return @lazy @. - ᶜf_coriolis × (C12(C123(ᶜuₕ)) - C12(C123(ᶜuₕ_g)))
end
