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
        # dissipation
        @. Yₜ.c.sgs⁰.ρatke[colidx] -=
            ᶜρa⁰[colidx] * c_d * max(ᶜtke⁰[colidx], 0)^(FT(3) / 2) / max(
                ᶜmixing_length[colidx],
                c_d * dt * sqrt(max(ᶜtke⁰[colidx], eps(FT))),
            )
    end

    return nothing
end
