#####
##### Coriolis Acceleration for Single-Column Configurations
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    scm_coriolis_tendency_uₕ(ᶜuₕ, scm_coriolis)

Calculates the Coriolis tendency for the horizontal velocity (`ᶜuₕ`) within a
single-column model (SCM) framework.

This function computes the Coriolis force acting on the difference between the
current horizontal velocity (`ᶜuₕ`) and the geostrophic wind (`ᶜuₕ_g`). The
geostrophic wind is determined from prescribed profiles (`prof_ug`, `prof_vg`)
which are typically used in single-column model (SCM) configurations to represent
large-scale forcing.

Arguments:
- `ᶜuₕ`: The cell-centered horizontal velocity field [m/s].
- `scm_coriolis`: An object containing Coriolis parameters and geostrophic wind
                   profiles. If not of type `SCMCoriolis`, a `NullBroadcasted`
                   is returned, indicating no Coriolis forcing.

Returns:
- A `ClimaCore.Fields.Field` representing the Coriolis tendency for `ᶜuₕ` [m/s²],
  or `NullBroadcasted()` if `scm_coriolis` is not an `SCMCoriolis` type.
"""
function scm_coriolis_tendency_uₕ(ᶜuₕ, scm_coriolis)
    scm_coriolis isa SCMCoriolis || return NullBroadcasted()
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
