#####
##### EDMFX TKE Tendency
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
    (; ᶠu³⁰, ᶠu³, ᶜstrain_rate_norm, ᶜlinear_buoygrad, ᶜtke⁰, ᶜmixing_length) = p.precomputed
    (; ᶜK_u, ᶜK_h) = p.precomputed
    ᶜρa⁰ = turbconv_model isa PrognosticEDMFX ? p.precomputed.ᶜρa⁰ : Y.c.ρ
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
            # dynamical entr/detr
            @. Yₜ.c.sgs⁰.ρatke +=
                ᶜρaʲ * (
                    ᶜdetrʲs.:($$j) * 1 / 2 *
                    norm_sqr(ᶜinterp(ᶠu³⁰) - ᶜinterp(ᶠu³ʲs.:($$j))) -
                    ᶜentrʲs.:($$j) * ᶜtke⁰
                )
            # turbulent entr
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
