#####
##### Coriolis Acceleration for Single-Column Configurations
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    scm_coriolis_tendency_uₕ(ᶜuₕ, scm_coriolis)

Return the Coriolis tendency of the horizontal velocity in a single-column model.

Computes `-f k̂ × (uₕ - uₕ_g)`, the Coriolis acceleration relative to the
geostrophic wind, and projects the result back onto the horizontal components. The
geostrophic wind is read from the prescribed profiles `prof_ug` and `prof_vg`,
which represent the large-scale forcing of a single-column configuration.

# Arguments

  - `ᶜuₕ`: Cell-center horizontal velocity [m/s].
  - `scm_coriolis`: An object with fields `prof_ug`, `prof_vg`, and
    `coriolis_param`, or `nothing` for no Coriolis forcing.

# Returns

A lazy broadcast with the Coriolis tendency of `ᶜuₕ` [m/s²], or a
`NullBroadcasted()` when `scm_coriolis` is `nothing`. The caller
(`additional_tendency!`) adds it to `Yₜ.c.uₕ`.
"""
scm_coriolis_tendency_uₕ(ᶜuₕ, ::Nothing) = NullBroadcasted()
function scm_coriolis_tendency_uₕ(ᶜuₕ, scm_coriolis)
    (; prof_ug, prof_vg) = scm_coriolis
    (; coriolis_param) = scm_coriolis

    ᶜspace = axes(ᶜuₕ)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    coords = Fields.coordinate_field(ᶜspace)

    # Coriolis vector f k̂, where k̂ is the vertical unit vector and f is the Coriolis parameter.
    # Here, f is represented as a WVector (vertical component) and then cast to Contravariant3Vector.
    coriolis_fn(coord) = Geometry.WVector(coriolis_param)
    ᶜf_coriolis = @. lazy(CT3(coriolis_fn(coords)))

    # Geostrophic wind uₕ_g = (u_g, v_g) from prescribed profiles.
    ᶜuₕ_g = @. lazy(C12(Geometry.UVVector(prof_ug(ᶜz), prof_vg(ᶜz))))

    # Coriolis tendency: - f k̂ × (uₕ - uₕ_g)
    # ᶜuₕ and ᶜuₕ_g are Covariant12Vector, their difference is also Covariant12Vector.
    # Contravariant3Vector × Covariant12Vector -> Covariant123Vector.
    ᶜcoriolis_3d = @. lazy(- ᶜf_coriolis × (ᶜuₕ - ᶜuₕ_g))
    # Project the 3D result back to horizontal components, as it's applied to uₕ
    return @. lazy(C12(ᶜcoriolis_3d))
end
