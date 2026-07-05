#####
##### TKE Tendency for Eddy Diffusion Closure Used in EDMFX
#####

"""
 edmfx_tke_tendency!(Yₜ, Y, p, t, turbconv_model)

 Applies the tendency from the EDMFX subgrid scale turbulent kinetic energy (TKE)
 to the prognostic variables.

 This function calculates and applies the changes in the environment TKE
 (`Yₜ.c.ρtke`) due to various processes within the EDMFX framework,
 including shear production, buoyancy production, entrainment, detrainment,
 turbulent entrainment, pressure work, and dissipation.

 Arguments:
 - `Yₜ`: The tendency state vector.
 - `Y`: The current state vector.
 - `p`: The cache, containing precomputed quantities and parameters.
 - `t`: The current simulation time.
 - `turbconv_model`: The turbulence convection model (e.g., `EDOnlyEDMFX`, `PrognosticEDMFX`).

 Returns: `nothing`, modifies `Yₜ` in place.
"""

edmfx_tke_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_tke_tendency!(Yₜ, Y, p, t, turbconv_model::EDOnlyEDMFX)
    (; params) = p
    (; ᶜstrain_rate_norm, ᶜlinear_buoygrad) = p.precomputed
    turbconv_params = CAP.turbconv_params(p.params)
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= ᶜmixing_length(Y, p)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field))
    ᶜprandtl_nvec = @. lazy(
        turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
    )
    ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

    # shear production
    @. Yₜ.c.ρtke += 2 * Y.c.ρ * ᶜK_u * ᶜstrain_rate_norm
    # buoyancy production
    @. Yₜ.c.ρtke -= Y.c.ρ * ᶜK_h * ᶜlinear_buoygrad
end

function edmfx_tke_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    (; ᶜstrain_rate_norm, ᶜlinear_buoygrad) = p.precomputed
    turbconv_params = CAP.turbconv_params(p.params)

    if use_prognostic_tke(turbconv_model)
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_2
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
        ᶜK_u = @. lazy(
            eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field),
        )
        ᶜprandtl_nvec = @. lazy(
            turbulent_prandtl_number(
                p.params,
                ᶜlinear_buoygrad,
                ᶜstrain_rate_norm,
            ),
        )
        ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

        # shear production
        @. Yₜ.c.ρtke += 2 * Y.c.ρ * ᶜK_u * ᶜstrain_rate_norm
        # Buoyancy production: only the diffusive (intra-subdomain) piece,
        # -ρ K_h ∂b/∂z, of the Favre-averaged buoyancy flux enters the
        # isotropic-TKE budget. The coherent (mass-flux) piece
        # Σ_m ρa^m (w^m - w) b^m powers the inter-subdomain (coherent)
        # kinetic energy through the buoyancy term of the subdomain momentum
        # equations, which the prognostic subdomain velocities already carry;
        # adding it here double-counts buoyancy production and spuriously
        # inflates K near cloud tops with active drafts.
        @. Yₜ.c.ρtke -= Y.c.ρ * ᶜK_h * ᶜlinear_buoygrad
    end
    return nothing
end

"""
    tke_dissipation(turbconv_params, ρtke, tke, mixing_length)

Returns a scalar value representing the TKE dissipation rate
per unit volume, ρ * ε_d [kg m^-1 s^-3].

The physical dissipation is calculated as:
ρ * ε_d = c_d * ρtke * sqrt(abs(tke)) / mixing_length
where `c_d` is a parameter.

Arguments:

  - `turbconv_params`: Turbulence and convection model parameters.
  - `ρtke`: ρ_area_weighted * tke [kg m^-2 s^-2].
  - `tke`: Turbulent kinetic energy [m^2 s^-2].
  - `mixing_length`: Turbulent mixing length [m].
"""
function tke_dissipation(turbconv_params, ρtke, tke, mixing_length)
    FT = typeof(tke)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    dissipation_rate_vol = c_d * ρtke * sqrt(abs(tke)) / mixing_length
    return dissipation_rate_vol
end
