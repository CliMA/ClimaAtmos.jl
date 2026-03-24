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
 - `turbconv_model`: The turbulence convection model (e.g., `EDOnlyEDMFX`, `PrognosticEDMFX`, `DiagnosticEDMFX`).

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
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶠu³, ᶠu³ʲs, ᶠu³⁰, ᶜstrain_rate_norm, ᶜlinear_buoygrad) = p.precomputed
    turbconv_params = CAP.turbconv_params(p.params)
    FT = eltype(p.params)
    thermo_params = CAP.thermodynamics_params(p.params)

    if use_prognostic_tke(turbconv_model)
        (; ᶜρʲs) = p.precomputed
        ᶠz = Fields.coordinate_field(Y.f).z
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
        # buoyancy production
        @. Yₜ.c.ρtke -= Y.c.ρ * ᶜK_h * ᶜlinear_buoygrad
        for j in 1:n
            ᶜρaʲ =
                turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa :
                p.precomputed.ᶜρaʲs.:($j)
            @. Yₜ.c.ρtke -=
                ᶜρaʲ * adjoint(CT3(ᶜinterp(ᶠu³ʲs.:($$j) - ᶠu³))) *
                (ᶜρʲs.:($$j) - Y.c.ρ) *
                ᶜgradᵥ(CAP.grav(p.params) * ᶠz) / ᶜρʲs.:($$j)
        end
        # Note: Adding the following tendency breaks bm_aquaplanet_progedmf_dense_autodiff
        if turbconv_model isa PrognosticEDMFX
            ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
            (; ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
            ᶜρ⁰ = @. lazy(
                TD.air_density(
                    thermo_params,
                    ᶜT⁰,
                    ᶜp,
                    ᶜq_tot_safe⁰,
                    ᶜq_liq⁰,
                    ᶜq_ice⁰,
                ),
            )
            @. Yₜ.c.ρtke -=
                ᶜρa⁰ * adjoint(CT3(ᶜinterp(ᶠu³⁰ - ᶠu³))) *
                (ᶜρ⁰ - Y.c.ρ) *
                ᶜgradᵥ(CAP.grav(p.params) * ᶠz) / ᶜρ⁰
        end
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
