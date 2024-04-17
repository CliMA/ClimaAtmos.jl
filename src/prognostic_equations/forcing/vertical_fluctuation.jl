#####
##### Vertical fluctuation (for single column experiments)
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

vertical_fluctuation_cache(Y, atmos::AtmosModel) =
    vertical_fluctuation_cache(Y, atmos.vert_fluc)

vertical_fluctuation_cache(Y, ::Nothing) = (;)
function vertical_fluctuation_cache(Y, ::VerticalFluctuation)
    FT = Spaces.undertype(axes(Y.c))
    ᶜdTdt_fluc = similar(Y.c, FT)
    ᶜdqtdt_fluc = similar(Y.c, FT)
    # TODO: read profiles from LES
    @. ᶜdTdt_fluc = 0
    @. ᶜdqtdt_fluc = 0
    return (; ᶜdTdt_fluc, ᶜdqtdt_fluc)
end

vertical_fluctuation_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing
function vertical_fluctuation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    ::VerticalFluctuation,
)
    (; params) = p
    (; ᶜts) = p.precomputed
    (; ᶜdTdt_fluc, ᶜdqtdt_fluc) = p.vertical_fluctuation
    thermo_params = CAP.thermodynamics_params(params)
    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)

    @. Yₜ.c.ρe_tot[colidx] +=
        Y.c.ρ[colidx] * (
            TD.cv_m(thermo_params, ᶜts[colidx]) * ᶜdTdt_fluc[colidx] +
            (
                cv_v * (TD.air_temperature(thermo_params, ᶜts[colidx]) - T_0) +
                Lv_0 - R_v * T_0
            ) * ᶜdqtdt_fluc[colidx]
        )
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * ᶜdqtdt_fluc[colidx]

    return nothing
end
