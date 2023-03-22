#####
##### Precomputed quantities
#####

import Thermodynamics as TD
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces:
    column, ExtrudedFiniteDifferenceSpace, FiniteDifferenceSpace

Base.@propagate_inbounds function column(
    space::ExtrudedFiniteDifferenceSpace,
    i,
    h,
)
    FiniteDifferenceSpace(
        space.staggering,
        space.vertical_topology,
        Geometry.CartesianGlobalGeometry(),
        column(space.center_local_geometry, i, h),
        column(space.face_local_geometry, i, h),
    )
end


function precomputed_quantities!(Y, p, t)
    Fields.bycolumn(axes(Y.c)) do colidx
        precomputed_quantities!(Y, p, t, colidx)
    end
end

function set_boundary_velocity!(uₕ, uᵥ)
    uₕ_surface_data = Fields.level(Fields.field_values(uₕ), 1)
    uₕ_surface_geom = Fields.level(Spaces.local_geometry_data(axes(uₕ)), 1)
    uᵥ_surface_data = Fields.level(Fields.field_values(uᵥ), 1)
    uᵥ_surface_geom = Fields.level(Spaces.local_geometry_data(axes(uᵥ)), 1)

    @. uᵥ_surface_data = Geometry.Covariant3Vector(
        -Geometry.contravariant3(uₕ_surface_data, uₕ_surface_geom) /
        Geometry.contravariant3(one(uᵥ_surface_data), uᵥ_surface_geom),
    )
end

function precomputed_quantities!(Y, p, t, colidx)
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w

    (; ᶜK, ᶠu_tilde, ᶜu_bar, ᶠu³, ᶜts, ᶜp, params) = p
    (; ᶠwinterp, ᶜinterp) = p.operators
    ᶜJ = Fields.local_geometry_field(axes(Y.c.ρ)).J
    C123 = Geometry.Covariant123Vector

    set_boundary_velocity!(ᶜuₕ[colidx], ᶠw[colidx])

    @. ᶜu_bar[colidx] = C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))
    @. ᶠu_tilde[colidx] =
        ᶠwinterp(
            Y.c.ρ[colidx] * ᶜJ[colidx],
            Geometry.Contravariant123Vector(ᶜuₕ[colidx]),
        ) + Geometry.Contravariant123Vector(ᶠw[colidx])
    @. ᶠu³[colidx] =
        Geometry.project(Geometry.Contravariant3Axis(), ᶠu_tilde[colidx])
    compute_kinetic!(ᶜK[colidx], ᶜuₕ[colidx], ᶠw[colidx])
    thermo_params = CAP.thermodynamics_params(params)
    thermo_state!(Y, p, ᶜinterp, colidx; time = t)
    @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])
    return nothing
end
