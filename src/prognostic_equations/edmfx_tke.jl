#####
##### TKE Tendency for Eddy Diffusion Closure Used in EDMFX
#####

"""
 edmfx_tke_tendency!(Yₜ, Y, p, t, turbconv_model)

 Applies the tendency from the EDMFX subgrid scale turbulent kinetic energy (TKE)
 to the prognostic variables.

 This function calculates and applies the changes in the environment TKE
 (`Yₜ.c.sgs⁰.ρatke`) due to various processes within the EDMFX framework,
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
    ᶜρa⁰ = Y.c.ρ # EDOnly
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= ᶜmixing_length(Y, p)
    ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field))
    ᶜprandtl_nvec = @. lazy(
        turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
    )
    ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

    # shear production
    @. Yₜ.c.sgs⁰.ρatke += 2 * Y.c.ρ * ᶜK_u * ᶜstrain_rate_norm
    # buoyancy production
    @. Yₜ.c.sgs⁰.ρatke -= Y.c.ρ * ᶜK_h * ᶜlinear_buoygrad
end

function edmfx_tke_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜturb_entrʲs, ᶜentrʲs, ᶜdetrʲs) = p.precomputed
    (; ᶜstrain_rate_norm, ᶜlinear_buoygrad) = p.precomputed
    turbconv_params = CAP.turbconv_params(p.params)
    FT = eltype(p.params)
    
    # Compute velocity quantities on demand
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    @. ᶠuₕ³ = compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)
    ᶜu = compute_ᶜu(Y, ᶠuₕ³)
    ᶠu³ = compute_ᶠu³(Y, ᶠuₕ³)
    ᶜu⁰, ᶠu³⁰, _ = compute_environment_velocity_quantities(Y, ᶠuₕ³, turbconv_model)


    ᶜρa⁰ =
        turbconv_model isa PrognosticEDMFX ?
        (@. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))) : Y.c.ρ
    nh_pressure3_buoyʲs =
        turbconv_model isa PrognosticEDMFX ?
        p.precomputed.ᶠnh_pressure₃_buoyʲs : p.precomputed.ᶠnh_pressure³_buoyʲs
    nh_pressure3_dragʲs =
        turbconv_model isa PrognosticEDMFX ?
        p.precomputed.ᶠnh_pressure₃_dragʲs : p.precomputed.ᶠnh_pressure³_dragʲs
    ᶜtke_press = p.scratch.ᶜtemp_scalar
    @. ᶜtke_press = 0
    for j in 1:n
        ᶜρaʲ =
            turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa :
            p.precomputed.ᶜρaʲs.:($j)
        # Compute SGS velocity on demand
        _, ᶠu³ʲ, _ = compute_sgs_velocity_quantities(Y, ᶠuₕ³, turbconv_model, j)
        
        @. ᶜtke_press +=
            ᶜρaʲ *
            adjoint(ᶜinterp.(ᶠu³ʲ - ᶠu³⁰)) *
            ᶜinterp(
                C3((nh_pressure3_buoyʲs.:($$j)) + nh_pressure3_dragʲs.:($$j)),
            )
    end

    if use_prognostic_tke(turbconv_model)

        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_2
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜρa⁰ =
            turbconv_model isa PrognosticEDMFX ?
            (@. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))) : Y.c.ρ
        ᶜtke⁰ =
            @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))
        ᶜK_u = @. lazy(
            eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length_field),
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
        @. Yₜ.c.sgs⁰.ρatke += 2 * ᶜρa⁰ * ᶜK_u * ᶜstrain_rate_norm
        # buoyancy production
        @. Yₜ.c.sgs⁰.ρatke -= ᶜρa⁰ * ᶜK_h * ᶜlinear_buoygrad

        ᶜtke⁰ =
            @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))

        # entrainment and detraiment
        # using ᶜu⁰ and local geometry results in allocation
        for j in 1:n
            ᶜρaʲ =
                turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa :
                p.precomputed.ᶜρaʲs.:($j)
            # Compute SGS velocity on demand for entrainment/detrainment
            _, ᶠu³ʲ, _ = compute_sgs_velocity_quantities(Y, ᶠuₕ³, turbconv_model, j)
            
            # dynamical entrainment/detrainment
            @. Yₜ.c.sgs⁰.ρatke +=
                ᶜρaʲ * (
                    ᶜdetrʲs.:($$j) * 1 / 2 *
                    norm_sqr(ᶜinterp(ᶠu³⁰) - ᶜinterp(ᶠu³ʲ)) -
                    ᶜentrʲs.:($$j) * ᶜtke⁰
                )
            # turbulent entrainment
            @. Yₜ.c.sgs⁰.ρatke +=
                ᶜρaʲ *
                ᶜturb_entrʲs.:($$j) *
                (
                    norm(ᶜinterp(ᶠu³⁰) - ᶜinterp(ᶠu³ʲ)) *
                    norm(ᶜinterp(ᶠu³⁰) - ᶜinterp(ᶠu³)) - ᶜtke⁰
                )
        end

        # pressure work
        @. Yₜ.c.sgs⁰.ρatke += ᶜtke_press
    end
    return nothing
end

"""
    tke_dissipation(turbconv_params, ρatke, tke, mixing_length)

Returns a scalar value representing the TKE dissipation rate
per unit volume, ρ * ε_d [kg m^-1 s^-3].

The physical dissipation is calculated as:
  ρ * ε_d = c_d * ρatke * sqrt(abs(tke)) / mixing_length
where `c_d` is a parameter.

Arguments:
- `turbconv_params`: Turbulence and convection model parameters.
- `ρatke`: ρ_area_weighted * tke [kg m^-2 s^-2].
- `tke`: Turbulent kinetic energy [m^2 s^-2].
- `mixing_length`: Turbulent mixing length [m].
"""
function tke_dissipation(turbconv_params, ρatke, tke, mixing_length)
    FT = typeof(tke)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    dissipation_rate_vol = c_d * ρatke * sqrt(abs(tke)) / mixing_length
    return dissipation_rate_vol
end
