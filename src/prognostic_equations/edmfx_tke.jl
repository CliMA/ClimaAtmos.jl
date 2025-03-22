#####
##### EDMF SGS flux
#####

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
    (; ᶠu³⁰, ᶠu³, ᶜstrain_rate_norm, ᶜlinear_buoygrad, ᶜtke⁰) = p.precomputed
    (; ᶜK_u, ᶜK_h) = p.precomputed
    ᶜρa⁰ = turbconv_model isa PrognosticEDMFX ? p.precomputed.ᶜρa⁰ : Y.c.ρ
    nh_pressure3ʲs =
        turbconv_model isa PrognosticEDMFX ? p.precomputed.ᶠnh_pressure₃ʲs :
        p.precomputed.ᶠnh_pressure³ʲs
        
    if turbconv_model isa PrognosticEDMFX
        (; params) = p
        (; ᶠgradᵥ_ᶜΦ) = p.core
        (; ᶜρʲs, ᶠnh_pressure₃ʲs, ᶠu₃⁰) = p.precomputed
        ᶠlg = Fields.local_geometry_field(Y.f)
    
        scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    
        for j in 1:n
            if p.atmos.edmfx_model.nh_pressure isa Val{true}
                @. ᶠnh_pressure₃ʲs.:($$j) = ᶠupdraft_nh_pressure(
                    params,
                    ᶠlg,
                    ᶠbuoyancy(ᶠinterp(Y.c.ρ), ᶠinterp(ᶜρʲs.:($$j)), ᶠgradᵥ_ᶜΦ),
                    Y.f.sgsʲs.:($$j).u₃,
                    ᶠu₃⁰,
                    scale_height,
                )
            else
                @. ᶠnh_pressure₃ʲs.:($$j) = C3(0)
            end
        end
        nh_pressure3ʲs = p.precomputed.ᶠnh_pressure₃ʲs
    end

    ᶜtke_press = p.scratch.ᶜtemp_scalar
    @. ᶜtke_press = 0
    for j in 1:n
        ᶜρaʲ =
            turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa :
            p.precomputed.ᶜρaʲs.:($j)
        @. ᶜtke_press +=
            ᶜρaʲ *
            adjoint(ᶜinterp.(ᶠu³ʲs.:($$j) - ᶠu³⁰)) *
            ᶜinterp(C3(nh_pressure3ʲs.:($$j)))
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
