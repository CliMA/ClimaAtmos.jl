#####
##### Grid-mean SGS closures (mixing length)
#####

import NVTX
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    smagorinsky_lilly_length(c_smag, N_eff, dz, Pr, ϵ_st)

Compute the Smagorinsky-Lilly length scale.

This scale is used for the subgrid mixing length in turbulent flows when no
EDMFX model (with prognostic TKE) is available. It starts with the Smagorinsky
scale `c_smag * dz` and applies the Lilly reduction factor
`max(0, 1 - N_eff² / (2 Pr ϵ_st))^(1/4)` under stable stratification
(`N_eff > 0`), which drives the length to zero once stratification suppresses
the shear production.

# Arguments

  - `c_smag`: The Smagorinsky coefficient [-].
  - `N_eff`: Effective buoyancy frequency [s⁻¹], equal to `sqrt(max(ᶜbuoygrad, 0))`.
  - `dz`: Vertical grid scale [m].
  - `Pr`: Turbulent Prandtl number [-].
  - `ϵ_st`: Squared norm of the vertical shear of the horizontal wind,
    `|∂uₕ/∂z|² / 2` (`vertical_shear_norm_sqr`) [s⁻²], so that
    `N_eff² / (2 Pr ϵ_st)` is the flux Richardson number of the vertical
    momentum flux, which acts on the horizontal wind only.

# Returns

The Smagorinsky-Lilly length scale [m].
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
    compute_gm_mixing_length(Y, p)

Compute the grid-mean subgrid-scale (SGS) mixing length from the
Smagorinsky-Lilly closure and return it as a cell-center field (materialized
in `p.scratch.ᶜtemp_scalar`).

Used when no EDMFX model with prognostic TKE is active. Steps:

 1. Evaluate the cloud-fraction-blended moist buoyancy gradient
    (`ᶜbuoyancy_gradient`, from the chain-rule coefficients and face gradients
    materialized by `set_buoyancy_gradient_inputs!`).
 2. Fill `p.precomputed.ᶜstrain_rate_norm` with the squared norm of the
    vertical shear of the horizontal wind, `|∂uₕ/∂z|² / 2`
    (`set_strain_rate_norm!`).
 3. Evaluate the turbulent Prandtl number and
    `smagorinsky_lilly_length` with the vertical grid scale `ᶜdz`.

Mutates `ᶜstrain_rate_norm` in `p.precomputed`,
and uses `p.scratch` fields (including the returned `ᶜtemp_scalar`) for
intermediates.
"""
NVTX.@annotate function compute_gm_mixing_length(Y, p)
    (; params) = p

    ᶜdz = Fields.Δz_field(axes(Y.c))
    (; ᶜstrain_rate_norm) = p.precomputed
    # (chain-rule coefficients and face gradients materialized once per update
    # by `set_buoyancy_gradient_inputs!`, before the cloud-fraction Picard
    # iteration)
    ᶜbuoygrad = ᶜbuoyancy_gradient(p)

    set_strain_rate_norm!(Y, p)

    ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar_2
    @. ᶜprandtl_nvec =
        turbulent_prandtl_number(params, ᶜbuoygrad, ᶜstrain_rate_norm)

    # Materialize directly into scratch field to avoid lazy heap allocations
    ᶜmixing_length = p.scratch.ᶜtemp_scalar
    @. ᶜmixing_length = smagorinsky_lilly_length(
        CAP.c_smag(params),
        sqrt(max(ᶜbuoygrad, 0)),   # N_eff
        ᶜdz,
        ᶜprandtl_nvec,
        ᶜstrain_rate_norm,
    )
    return ᶜmixing_length
end
