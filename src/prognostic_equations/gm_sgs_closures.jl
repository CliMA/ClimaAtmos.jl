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
  - `N_eff`: Effective buoyancy frequency [s⁻¹], equal to `sqrt(max(ᶜN²_eff, 0))`,
    with `ᶜN²_eff` the interface-aware effective stability).
  - `dz`: Vertical grid scale [m].
  - `Pr`: Turbulent Prandtl number [-].
  - `ϵ_st`: Squared Frobenius norm of the strain-rate tensor, `SᵢⱼSᵢⱼ` [s⁻²].

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

 1. Fill `p.precomputed.ᶜbuoygrad` with the cloud-fraction-blended moist
    buoyancy gradient (`blended_N²`, using the chain-rule coefficients
    and face gradients materialized by `set_buoyancy_gradient_inputs!`).
 2. Fill `p.precomputed.ᶜN²_eff` with the stability-biased buoyancy gradient
    (`set_stability_buoyancy_gradient!`).
 3. Fill `p.precomputed.ᶜstrain_rate_norm` with the squared strain-rate norm
    of the resolved velocity.
 4. Evaluate the turbulent Prandtl number and
    `smagorinsky_lilly_length` with the vertical grid scale `ᶜdz`.

Mutates `ᶜbuoygrad`, `ᶜN²_eff`, and `ᶜstrain_rate_norm` in `p.precomputed`,
and uses `p.scratch` fields (including the returned `ᶜtemp_scalar`) for
intermediates.
"""
NVTX.@annotate function compute_gm_mixing_length(Y, p)
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)

    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶜlg = Fields.local_geometry_field(Y.c)
    (;
        ᶜT,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
        ᶠu³,
        ᶜbuoygrad,
        ᶜstrain_rate_norm,
        ᶜcloud_fraction,
    ) =
        p.precomputed

    # Chain-rule coefficients and face gradients are materialized once per
    # update by `set_buoyancy_gradient_inputs!` (called before the
    # cloud-fraction Picard iteration); see `blended_N²`.
    (; ᶜbg_coeffs) = p.precomputed
    @. ᶜbuoygrad = blended_N²(
        ᶜbg_coeffs,
        ᶜcloud_fraction,
        projected_vector_data(C3, p.precomputed.ᶜgradᵥ_θ_liq_ice, ᶜlg),
        projected_vector_data(C3, p.precomputed.ᶜgradᵥ_q_tot, ᶜlg),
    )
    # Stability-biased buoyancy gradient (max of one-sided estimates) for
    # the mixing-length and Pr_t(Ri) closures; see
    # set_stability_buoyancy_gradient! for rationale.
    set_stability_buoyancy_gradient!(Y, p, thermo_params)
    (; ᶜN²_eff) = p.precomputed

    # TODO: move strain rate calculation to separate function
    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ᶜstrain_rate = compute_strain_rate_center_vertical(ᶠu)
    @. ᶜstrain_rate_norm = norm_sqr(ᶜstrain_rate)

    ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar_2
    @. ᶜprandtl_nvec =
        turbulent_prandtl_number(params, ᶜN²_eff, ᶜstrain_rate_norm)

    # Materialize directly into scratch field to avoid lazy heap allocations
    ᶜmixing_length = p.scratch.ᶜtemp_scalar
    @. ᶜmixing_length = smagorinsky_lilly_length(
        CAP.c_smag(params),
        sqrt(max(ᶜN²_eff, 0)),   # N_eff
        ᶜdz,
        ᶜprandtl_nvec,
        ᶜstrain_rate_norm,
    )
    return ᶜmixing_length
end

"""
    compute_gm_horizontal_mixing_length(Y, p)

Horizontal grid-mean Smagorinsky-Lilly mixing length, the counterpart of
`compute_gm_mixing_length` capped by the horizontal node scale `Δx_h`
(`horizontal_filter_scale`) instead of `Δz`. Used for the horizontal turbulent
term of the 3D SGS variance closure when no EDMFX model is active.

Assumes `compute_gm_mixing_length` has already run this update, so `ᶜN²_eff` and
`ᶜstrain_rate_norm` in `p.precomputed` are current. Materializes into
`p.scratch.ᶜtemp_scalar_3` (its Prandtl number into `ᶜtemp_scalar_6`) and returns
it.
"""
function compute_gm_horizontal_mixing_length(Y, p)
    (; params) = p
    Δx_h = horizontal_filter_scale(axes(Y.c))
    (; ᶜN²_eff, ᶜstrain_rate_norm) = p.precomputed

    ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar_6
    @. ᶜprandtl_nvec =
        turbulent_prandtl_number(params, ᶜN²_eff, ᶜstrain_rate_norm)

    ᶜmixing_length_h = p.scratch.ᶜtemp_scalar_3
    @. ᶜmixing_length_h = smagorinsky_lilly_length(
        CAP.c_smag(params),
        sqrt(max(ᶜN²_eff, 0)),   # N_eff
        Δx_h,
        ᶜprandtl_nvec,
        ᶜstrain_rate_norm,
    )
    return ᶜmixing_length_h
end
