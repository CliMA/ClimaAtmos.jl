#####
##### EDMF Coriolis for Single-Column Configurations
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    edmf_coriolis_tendency_uₕ(ᶜuₕ, edmf_coriolis)

Calculates the Coriolis tendency for the horizontal velocity (`ᶜuₕ`) within the
EDMF framework.

This function computes the Coriolis force acting on the difference between the
current horizontal velocity (`ᶜuₕ`) and the geostrophic wind (`ᶜuₕ_g`). The
geostrophic wind is determined from prescribed profiles (`prof_ug`, `prof_vg`)
which are typically used in single-column model (SCM) configurations to represent
large-scale forcing.

Arguments:
- `ᶜuₕ`: The cell-centered horizontal velocity field [m/s].
- `edmf_coriolis`: An object containing Coriolis parameters and geostrophic wind
                   profiles. If not of type `EDMFCoriolis`, a `NullBroadcasted`
                   is returned, indicating no Coriolis forcing.

Returns:
- A `ClimaCore.Fields.Field` representing the Coriolis tendency for `ᶜuₕ` [m/s²],
  or `NullBroadcasted()` if `edmf_coriolis` is not an `EDMFCoriolis` type.
"""
function edmf_coriolis_tendency_uₕ(ᶜuₕ, edmf_coriolis)
    edmf_coriolis isa EDMFCoriolis || return NullBroadcasted()
    (; prof_ug, prof_vg) = edmf_coriolis
    (; coriolis_param) = edmf_coriolis

    ᶜspace = axes(ᶜuₕ)
    ᶜz = Fields.coordinate_field(ᶜspace).z
    coords = Fields.coordinate_field(ᶜspace)

    # Coriolis vector f k̂, where k̂ is the vertical unit vector and f is the Coriolis parameter.
    # Here, f is represented as a WVector (vertical component) and then cast to Contravariant3Vector.
    coriolis_fn(coord) = Geometry.WVector(coriolis_param)
    ᶜf_coriolis = @. lazy(Geometry.Contravariant3Vector(coriolis_fn(coords)))

    # Geostrophic wind uₕ_g = (u_g, v_g) from prescribed profiles.
    ᶜuₕ_g = @. lazy(
        Geometry.Covariant12Vector(Geometry.UVVector(prof_ug(ᶜz), prof_vg(ᶜz))),
    )

    # Coriolis tendency: - f k̂ × (uₕ - uₕ_g)
    # ᶜuₕ and ᶜuₕ_g are Covariant12Vector, their difference is also Covariant12Vector.
    # The cross product Contravariant3Vector × Covariant12Vector is defined in ClimaCore.
    return @. lazy(-ᶜf_coriolis × (ᶜuₕ - ᶜuₕ_g))
end
