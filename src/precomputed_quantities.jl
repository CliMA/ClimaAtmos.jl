#####
##### Precomputed quantities
#####

import Thermodynamics as TD
import LinearAlgebra: norm_sqr
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
    (; ᶜuvw, ᶜK, ᶜts, ᶜp, params) = p
    (; ᶜinterp) = p.operators
    C123 = Geometry.Covariant123Vector
    @. ᶜuvw[colidx] = C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))
    @. ᶜK[colidx] = norm_sqr(ᶜuvw[colidx]) / 2
    thermo_params = CAP.thermodynamics_params(params)
    thermo_state!(Y, p, ᶜinterp, colidx; time = t)
    @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])
    return nothing
end
