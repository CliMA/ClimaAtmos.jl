#####
##### Large-scale advection (for single column experiments)
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

function large_scale_advection_tendency_ρq_tot(
    ᶜρ,
    thermo_params,
    ᶜts,
    t,
    ls_adv,
)
    ls_adv isa LargeScaleAdvection || return NullBroadcasted()
    (; prof_dTdt, prof_dqtdt) = ls_adv
    ᶜz = Fields.coordinate_field(axes(ᶜρ)).z
    ᶜdqtdt_hadv = @lazy @. prof_dqtdt(thermo_params, ᶜts, t, ᶜz)
    return @lazy @. ᶜρ * ᶜdqtdt_hadv
end

function large_scale_advection_tendency_ρe_tot(
    ᶜρ,
    thermo_params,
    ᶜts,
    t,
    ls_adv,
)
    ls_adv isa LargeScaleAdvection || return NullBroadcasted()
    (; prof_dTdt, prof_dqtdt) = ls_adv
    z = Fields.coordinate_field(axes(ᶜρ)).z
    ᶜdTdt_hadv = @lazy @. prof_dTdt(thermo_params, ᶜts, t, z)
    ᶜdqtdt_hadv = @lazy @. prof_dqtdt(thermo_params, ᶜts, t, z)
    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)
    # TODO: should `hv` be a thermo function?
    #     (hv = cv_v * (ᶜT - T_0) + Lv_0 - R_v * T_0)
    return @lazy @. ᶜρ * (
        TD.cv_m(thermo_params, ᶜts) * ᶜdTdt_hadv +
        (
            cv_v * (TD.air_temperature(thermo_params, ᶜts) - T_0) + Lv_0 -
            R_v * T_0
        ) * ᶜdqtdt_hadv
    )
    return nothing
end
