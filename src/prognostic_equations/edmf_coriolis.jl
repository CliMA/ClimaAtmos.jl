#####
##### Large-scale advection (for single column experiments)
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

edmf_coriolis_cache(Y, atmos::AtmosModel) =
    edmf_coriolis_cache(Y, atmos.edmf_coriolis)

edmf_coriolis_cache(Y, edmf_coriolis::Nothing) = (;)
function edmf_coriolis_cache(Y, edmf_coriolis::EDMFCoriolis)
    (; coriolis_param) = edmf_coriolis
    coords = Fields.coordinate_field(axes(Y.c.ρ))
    coriolis_fn(coord) = Geometry.WVector(coriolis_param)
    ᶜf_coriolis = @. Geometry.Contravariant3Vector(coriolis_fn(coords))
    return (; ᶜuₕ_g = similar(Y.c.uₕ), ᶜf_coriolis)
end

edmf_coriolis_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing
function edmf_coriolis_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    edmf_coriolis::EDMFCoriolis,
)
    (; prof_ug, prof_vg) = edmf_coriolis
    ᶜf_coriolis = p.edmf_coriolis.ᶜf_coriolis[colidx]
    ᶜuₕ_g = p.edmf_coriolis.ᶜuₕ_g[colidx]
    z = Fields.coordinate_field(axes(ᶜuₕ_g)).z
    @. ᶜuₕ_g =
        Geometry.Covariant12Vector(Geometry.UVVector(prof_ug(z), prof_vg(z)))

    # Coriolis
    C123 = Geometry.Covariant123Vector
    C12 = Geometry.Contravariant12Vector
    @. Yₜ.c.uₕ[colidx] -=
        ᶜf_coriolis × (C12(C123(Y.c.uₕ[colidx])) - C12(C123(ᶜuₕ_g)))

    return nothing
end
