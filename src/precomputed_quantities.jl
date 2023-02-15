#####
##### Precomputed quantities
#####

import Thermodynamics as TD
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

function precomputed_quantities!(Y, p, t)
    Fields.bycolumn(axes(Y.c)) do colidx
        precomputed_quantities!(Y, p, t, colidx)
    end
end

function precomputed_quantities!(Y, p, t, colidx)
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶠu_tilde, ᶜu_bar, ᶠu³, ᶜts, ᶜp, params) = p
    (; ᶠwinterp, ᶜinterp) = p.operators
    ᶜJ = Fields.local_geometry_field(axes(Y.c.ρ)).J
    C123 = Geometry.Covariant123Vector
    @. ᶜu_bar[colidx] = C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))
    @. ᶠu_tilde[colidx] =
        ᶠwinterp(
            Y.c.ρ[colidx] * ᶜJ[colidx],
            Geometry.Contravariant123Vector(ᶜuₕ[colidx]),
        ) + Geometry.Contravariant123Vector(Y.f.w[colidx])
    @. ᶠu³[colidx] =
        Geometry.project(Geometry.Contravariant3Axis(), ᶠu_tilde[colidx])
    compute_kinetic!(ᶜK[colidx], Y.c.uₕ[colidx], Y.f.w[colidx])
    thermo_params = CAP.thermodynamics_params(params)
    thermo_state!(Y, p, ᶜinterp, colidx; time = t)
    @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])
    return nothing
end
