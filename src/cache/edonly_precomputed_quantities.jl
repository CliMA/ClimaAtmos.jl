#####
##### Precomputed quantities
#####
import NVTX
import ClimaCore: Fields

"""
    set_edonly_precomputed_quantities_env_closures!(Y, p, t)

Updates the environment closures in precomputed quantities stored in `p` for diagnostic edmfx.
"""
NVTX.@annotate function set_edonly_precomputed_quantities_env_closures!(
    Y,
    p,
    t,
)
    (; params) = p
    (; ᶠu³) = p.precomputed
    (; ustar) = p.precomputed.sfc_conditions
    (; ᶜstrain_rate_norm) = p.precomputed
    (; ρtke_flux) = p.precomputed
    turbconv_params = CAP.turbconv_params(params)

    # TODO: Currently the shear production only includes vertical gradients
    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ᶜstrain_rate = ᶜcompute_strain_rate_center_vertical(ᶠu)
    @. ᶜstrain_rate_norm = norm_sqr(ᶜstrain_rate)

    ρtke_flux_values = Fields.field_values(ρtke_flux)
    ρ_sfc_values = Fields.field_values(Fields.level(Y.c.ρ, 1)) # TODO: replace by surface value
    ustar_values = Fields.field_values(ustar)
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), half),
    )
    @. ρtke_flux_values = surface_flux_tke(
        turbconv_params,
        ρ_sfc_values,
        ustar_values,
        sfc_local_geometry_values,
    )
    return nothing
end
