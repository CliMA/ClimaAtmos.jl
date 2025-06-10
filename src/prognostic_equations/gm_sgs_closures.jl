#####
##### Grid-mean SGS closures (mixing length)
#####

import NVTX
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    smagorinsky_lilly_length(c_smag, N_eff, dz, Pr, ϵ_st)

Compute the Smagorinsky-Lilly length scale.

This scale is used for the subgrid mixing length in turbulent flows when no EDMFX 
model (with prognostic TKE) is available. It starts with the Smagorinsky scale
(proportional to the grid size `dz`) and incorporates the Lilly modification
to account for the effects of stable stratification (buoyancy).

Arguments:
- `c_smag`: The Smagorinsky coefficient (dimensionless).
- `N_eff`: Effective buoyancy frequency [s⁻¹] (`N_eff = sqrt(max(linear_buoygrad, 0))`).
- `dz`: Vertical grid scale [m].
- `Pr`: Turbulent Prandtl number (dimensionless).
- `ϵ_st`: Squared Frobenius norm of the strain rate tensor, `S_{ij}S_{ij}` [s⁻²].

Returns:
- The Smagorinsky-Lilly length scale [m].
"""
function smagorinsky_lilly_length(c_smag, N_eff, dz, Pr, ϵ_st)
    FT = eltype(c_smag)
    return N_eff > FT(0) ?
           c_smag *
           dz *
           max(0, 1 - N_eff^2 / Pr / 2 / max(ϵ_st, eps(FT)))^(FT(1) / 4) :
           c_smag * dz
end

"""
    compute_gm_mixing_length!(Y, p)

Computes the grid-mean subgrid-scale (SGS) mixing length using the
Smagorinsky-Lilly formulation and stores it in `ᶜmixing_length`.

This function performs several steps:
1. Calculates the linear buoyancy gradient (`ᶜlinear_buoygrad`).
2. Calculates the squared Frobenius norm of the strain rate tensor (`ᶜstrain_rate_norm`)
   from the resolved velocity fields.
3. Calculates the turbulent Prandtl number (`ᶜprandtl_nvec`) based on the buoyancy
   gradient and strain rate norm.
4. Uses these quantities, along with the Smagorinsky coefficient (`c_smag`) and
   vertical grid scale (`ᶜdz`), to compute the Smagorinsky-Lilly length scale,
   which is then assigned to the output field `ᶜmixing_length`.

Arguments:
- `ᶜmixing_length`: Output `ClimaCore.Field` where the computed mixing length will be stored.
- `Y`: The current state vector (containing `Y.c.uₕ`).
- `p`: Cache containing parameters (`p.params`), precomputed fields (e.g., `ᶜts`,
       `ᶠu³`, vertical gradients of thermodynamic variables), and scratch space.

Modifies `ᶜmixing_length` in place. Also modifies fields in `p.precomputed`
(like `ᶜlinear_buoygrad`, `ᶜstrain_rate_norm`) and uses `p.scratch` for
intermediate calculations.
"""
NVTX.@annotate function compute_gm_mixing_length!(Y, p)
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)

    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶜlg = Fields.local_geometry_field(Y.c)
    (; ᶜts, ᶠu³, ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed

    @. ᶜlinear_buoygrad = buoyancy_gradients(
        BuoyGradMean(),
        thermo_params,
        p.atmos.moisture_model,
        ᶜts,
        C3,
        p.precomputed.ᶜgradᵥ_θ_virt,    # ∂θv∂z_unsat
        p.precomputed.ᶜgradᵥ_q_tot,     # ∂qt∂z_sat
        p.precomputed.ᶜgradᵥ_θ_liq_ice, # ∂θl∂z_sat
        ᶜlg,
    )

    # TODO: move strain rate calculation to separate function
    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ᶜstrain_rate = p.scratch.ᶜtemp_UVWxUVW
    ᶜstrain_rate .= compute_strain_rate_center(ᶠu)
    @. ᶜstrain_rate_norm = norm_sqr(ᶜstrain_rate)

    ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar_2
    @. ᶜprandtl_nvec =
        turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm)

    return @. lazy(
        smagorinsky_lilly_length(
            CAP.c_smag(params),
            sqrt(max(ᶜlinear_buoygrad, 0)),   # N_eff
            ᶜdz,
            ᶜprandtl_nvec,
            ᶜstrain_rate_norm,
        ),
    )
end
