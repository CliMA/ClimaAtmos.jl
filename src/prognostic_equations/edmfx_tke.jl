#####
##### EDMF SGS flux
#####

edmfx_tke_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing

function edmfx_tke_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::Union{PrognosticEDMFX, DiagnosticEDMFX},
)

    FT = Spaces.undertype(axes(Y.c))
    n = n_mass_flux_subdomains(turbconv_model)
    (; params) = p
    turbconv_params = CAP.turbconv_params(params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    (; ᶜentrʲs, ᶜdetrʲs, ᶠu³ʲs) = p.precomputed
    (; ᶠu³⁰, ᶜstrain_rate_norm, ᶜlinear_buoygrad, ᶜtke⁰, ᶜmixing_length) =
        p.precomputed
    (; ᶜK_u, ᶜK_h) = p.precomputed
    (; dt) = p
    ᶜρa⁰ = turbconv_model isa PrognosticEDMFX ? p.precomputed.ᶜρa⁰ : Y.c.ρ
    nh_pressure3ʲs =
        turbconv_model isa PrognosticEDMFX ? p.precomputed.ᶠnh_pressure₃ʲs :
        p.precomputed.ᶠnh_pressure³ʲs
    ᶜtke_press = p.scratch.ᶜtemp_scalar[colidx]
    @. ᶜtke_press = 0
    for j in 1:n
        ᶜρaʲ_colidx =
            turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa[colidx] :
            p.precomputed.ᶜρaʲs.:($j)[colidx]
        @. ᶜtke_press +=
            ᶜρaʲ_colidx *
            adjoint(ᶜinterp.(ᶠu³ʲs.:($$j)[colidx] - ᶠu³⁰[colidx])) *
            ᶜinterp(C3(nh_pressure3ʲs.:($$j)[colidx]))
    end


    if use_prognostic_tke(turbconv_model)
        # shear production
        @. Yₜ.c.sgs⁰.ρatke[colidx] +=
            2 * ᶜρa⁰[colidx] * ᶜK_u[colidx] * ᶜstrain_rate_norm[colidx]
        # buoyancy production
        @. Yₜ.c.sgs⁰.ρatke[colidx] -=
            ᶜρa⁰[colidx] * ᶜK_h[colidx] * ᶜlinear_buoygrad[colidx]
        # entrainment and detraiment
        # using ᶜu⁰ and local geometry results in allocation
        for j in 1:n
            @. Yₜ.c.sgs⁰.ρatke[colidx] +=
                ᶜρa⁰[colidx] * (
                    ᶜentrʲs.:($$j)[colidx] * 1 / 2 * norm_sqr(
                        ᶜinterp(ᶠu³⁰[colidx]) - ᶜinterp(ᶠu³ʲs.:($$j)[colidx]),
                    ) - ᶜdetrʲs.:($$j)[colidx] * ᶜtke⁰[colidx]
                )
        end
        # pressure work
        @. Yₜ.c.sgs⁰.ρatke[colidx] += ᶜtke_press[colidx]
        # dissipation
        @. Yₜ.c.sgs⁰.ρatke[colidx] -=
            ᶜρa⁰[colidx] * c_d * max(ᶜtke⁰[colidx], 0)^(FT(3) / 2) / max(
                ᶜmixing_length[colidx],
                c_d * dt * sqrt(max(ᶜtke⁰[colidx], eps(FT))),
            )
    end

    return nothing
end
