#####
##### Large-scale advection (for single column experiments)
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

large_scale_advection_cache(Y, atmos::AtmosModel) =
    large_scale_advection_cache(Y, atmos.ls_adv)

large_scale_advection_cache(Y, ls_adv::Nothing) = (;)
function large_scale_advection_cache(Y, ls_adv::LargeScaleAdvection)
    FT = Spaces.undertype(axes(Y.c))
    return (; ᶜdqtdt_hadv = similar(Y.c, FT), ᶜdTdt_hadv = similar(Y.c, FT))
end

large_scale_advection_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
function large_scale_advection_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ls_adv::LargeScaleAdvection,
)
    (; prof_dTdt, prof_dqtdt) = ls_adv

    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜts = p.precomputed.ᶜts
    ᶜdqtdt_hadv = p.large_scale_advection.ᶜdqtdt_hadv
    ᶜdTdt_hadv = p.large_scale_advection.ᶜdTdt_hadv
    z = Fields.coordinate_field(axes(ᶜdqtdt_hadv)).z

    @. ᶜdTdt_hadv = prof_dTdt(thermo_params, ᶜts, t, z)
    @. ᶜdqtdt_hadv = prof_dqtdt(thermo_params, ᶜts, t, z)

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)

    @. Yₜ.c.ρq_tot += Y.c.ρ * ᶜdqtdt_hadv
    # TODO: should `hv` be a thermo function?
    #     (hv = cv_v * (ᶜT - T_0) + Lv_0 - R_v * T_0)
    @. Yₜ.c.ρe_tot +=
        Y.c.ρ * (
            TD.cv_m(thermo_params, ᶜts) * ᶜdTdt_hadv +
            (
                cv_v * (TD.air_temperature(thermo_params, ᶜts) - T_0) + Lv_0 -
                R_v * T_0
            ) * ᶜdqtdt_hadv
        )
    return nothing
end
