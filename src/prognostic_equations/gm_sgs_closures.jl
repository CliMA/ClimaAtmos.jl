#####
##### Grid-mean SGS closures (mixing length)
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    Compute the Smagorinsky length scale from
    - c_smag coefficient
    - N_eff - buoyancy frequency = sqrt(max(ᶜlinear_buoygrad, 0))
    - dz - vertical grid scale
    - Pr - Prandtl number
    - ϵ_st - strain rate norm
"""
function smagorinsky_lilly_length(c_smag, N_eff, dz, Pr, ϵ_st)
    FT = eltype(c_smag)
    return N_eff > FT(0) ?
           c_smag *
           dz *
           max(0, 1 - N_eff^2 / Pr / 2 / max(ϵ_st, eps(FT)))^(1 / 4) :
           c_smag * dz
end

function compute_gm_mixing_length!(ᶜmixing_length, Y, p)
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)

    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶜlg = Fields.local_geometry_field(Y.c)
    (; ᶜts, ᶜp, ᶠu³) = p.precomputed
    (; obukhov_length) = p.precomputed.sfc_conditions

    ᶜlinear_buoygrad = p.scratch.ᶜtemp_scalar
    @. ᶜlinear_buoygrad = buoyancy_gradients(
        BuoyGradMean(),
        thermo_params,
        p.atmos.moisture_model,
        EnvBuoyGradVars(
            thermo_params,
            ᶜts,
            projected_vector_data(
                C3,
                ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ᶜts))),
                ᶜlg,
            ),                                                                 # ∂θv∂z_unsat
            projected_vector_data(
                C3,
                ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ᶜts))),
                ᶜlg,
            ),            # ∂qt∂z_sat
            projected_vector_data(
                C3,
                ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts))),
                ᶜlg,
            ),                                                                 # ∂θl∂z_sat
        ),
    )

    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ᶜstrain_rate = p.scratch.ᶜtemp_UVWxUVW
    compute_strain_rate_center!(ᶜstrain_rate, ᶠu)

    ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar_2
    @. ᶜprandtl_nvec = turbulent_prandtl_number(
        params,
        obukhov_length,
        ᶜlinear_buoygrad,
        norm_sqr(ᶜstrain_rate),
    )

    @. ᶜmixing_length = smagorinsky_lilly_length(
        CAP.c_smag(params),
        sqrt(max(ᶜlinear_buoygrad, 0)),   #N_eff
        ᶜdz,
        ᶜprandtl_nvec,
        norm_sqr(ᶜstrain_rate),
    )
end
