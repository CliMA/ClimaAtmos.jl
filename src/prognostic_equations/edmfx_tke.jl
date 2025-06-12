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
    (; ᶜstrain_rate_norm, ᶜlinear_buoygrad) = p.precomputed
    (; ᶜK_u, ᶜK_h) = p.precomputed

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
    (; ᶜturb_entrʲs, ᶜentrʲs, ᶜdetrʲs, ᶠu³ʲs) = p.precomputed
    (;
        ᶠu³⁰,
        ᶠu³,
        ᶜstrain_rate_norm,
        ᶜlinear_buoygrad
    ) = p.precomputed
    turbconv_params = CAP.turbconv_params(p.params)
    FT = eltype(p.params)

    
    ᶜρa⁰ = turbconv_model isa PrognosticEDMFX ? (@.lazy(ρa⁰(Y.c))) : Y.c.ρ
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
        @. ᶜtke_press +=
            ᶜρaʲ *
            adjoint(ᶜinterp.(ᶠu³ʲs.:($$j) - ᶠu³⁰)) *
            ᶜinterp(
                C3((nh_pressure3_buoyʲs.:($$j)) + nh_pressure3_dragʲs.:($$j)),
            )
    end

    if use_prognostic_tke(turbconv_model)
        (; ustar, obukhov_length) = p.precomputed.sfc_conditions
        (; params) = p

        ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar
        @. ᶜprandtl_nvec =
            turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm)

        ᶜtke⁰ = @.lazy(specific_tke(Y.c.sgs⁰, Y.c, turbconv_model))
        sfc_tke = Fields.level(ᶜtke⁰, 1)
        z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
        ᶜz = Fields.coordinate_field(Y.c).z
        ᶜdz = Fields.Δz_field(axes(Y.c))

        ᶜtke_exch = p.scratch.ᶜtemp_scalar_2
        @. ᶜtke_exch = 0
        for j in 1:n
            ᶠu³ʲ = ᶠu³ʲs.:($j)
            @. ᶜtke_exch +=
                Y.c.sgsʲs.:($$j).ρa * ᶜdetrʲs.:($$j) / ᶜρa⁰ *
                (1 / 2 * norm_sqr(ᶜinterp(ᶠu³⁰) - ᶜinterp(ᶠu³ʲs.:($$j))) - ᶜtke⁰)
        end

        ᶜmixing_length = @. lazy(master_mixing_length(
            p.params,
            ustar,
            ᶜz,
            z_sfc,
            ᶜdz,
            max(sfc_tke, eps(FT)),
            ᶜlinear_buoygrad,
            max(ᶜtke⁰, 0),
            obukhov_length,
            ᶜstrain_rate_norm,
            ᶜprandtl_nvec,
            ᶜtke_exch,
            p.atmos.edmfx_model.scale_blending_method,
        ))

        ᶜK_u = @. lazy(eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length))
        ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))
        # shear production
        @. Yₜ.c.sgs⁰.ρatke += 2 * ᶜρa⁰ * ᶜK_u * ᶜstrain_rate_norm
        # buoyancy production
        @. Yₜ.c.sgs⁰.ρatke -= ᶜρa⁰ * ᶜK_h * ᶜlinear_buoygrad

        # entrainment and detraiment
        # using ᶜu⁰ and local geometry results in allocation
        for j in 1:n
            ᶜρaʲ =
                turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa :
                p.precomputed.ᶜρaʲs.:($j)
            # dynamical entrainment/detrainment
            @. Yₜ.c.sgs⁰.ρatke +=
                ᶜρaʲ * (
                    ᶜdetrʲs.:($$j) * 1 / 2 *
                    norm_sqr(ᶜinterp(ᶠu³⁰) - ᶜinterp(ᶠu³ʲs.:($$j))) -
                    ᶜentrʲs.:($$j) * ᶜtke⁰
                )
            # turbulent entrainment
            @. Yₜ.c.sgs⁰.ρatke +=
                ᶜρaʲ *
                ᶜturb_entrʲs.:($$j) *
                (
                    norm(ᶜinterp(ᶠu³⁰) - ᶜinterp(ᶠu³ʲs.:($$j))) *
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
