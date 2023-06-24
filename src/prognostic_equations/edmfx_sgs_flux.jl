#####
##### EDMF SGS flux
#####

edmfx_sgs_flux_cache(Y, turbconv_model) = (;)
function edmfx_sgs_flux_cache(Y, turbconv_model::Union{EDMFX, DiagnosticEDMFX})
    FT = Spaces.undertype(axes(Y.c))
    ᶜK_u = similar(Y.c, FT)
    ᶜK_h = similar(Y.c, FT)
    @. ᶜK_u = FT(1)
    @. ᶜK_h = FT(1)
    return (; ᶜK_u, ᶜK_h)
end

edmfx_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing

function edmfx_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, turbconv_model::EDMFX)

    FT = Spaces.undertype(axes(Y.c))
    n = n_mass_flux_subdomains(turbconv_model)
    (; params, edmfx_upwinding, sfc_conditions) = p
    (; ᶠu³, ᶜh_tot, ᶜspecific) = p
    (; ᶠu³ʲs, ᶜh_totʲs, ᶜspecificʲs) = p
    (; ᶜρa⁰, ᶠu³⁰, ᶜh_tot⁰, ᶜspecific⁰, ᶜmixing_length) = p
    (; ᶜK_u, ᶜK_h) = p
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠgradᵥ = Operators.GradientC2F()

    if p.atmos.edmfx_sgs_flux

        # diffusivity
        turbconv_params = CAP.turbconv_params(params)
        c_m = TCP.tke_ed_coeff(turbconv_params)
        @. ᶜK_u[colidx] =
            c_m * ᶜmixing_length[colidx] * sqrt(max(ᶜspecific⁰.tke[colidx], 0))
        # TODO: add Prantdl number
        @. ᶜK_h[colidx] = ᶜK_u[colidx]

        # mass flux
        ᶠu³_diff_colidx = p.ᶠtemp_CT3[colidx]
        ᶜh_tot_diff_colidx = ᶜq_tot_diff_colidx = p.ᶜtemp_scalar[colidx]
        for j in 1:n
            @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
            @. ᶜh_tot_diff_colidx = ᶜh_totʲs.:($$j)[colidx] - ᶜh_tot[colidx]
            vertical_transport!(
                Yₜ.c.ρe_tot[colidx],
                ᶜJ[colidx],
                Y.c.sgsʲs.:($j).ρa[colidx],
                ᶠu³_diff_colidx,
                ᶜh_tot_diff_colidx,
                dt,
                edmfx_upwinding,
            )
        end
        @. ᶠu³_diff_colidx = ᶠu³⁰[colidx] - ᶠu³[colidx]
        @. ᶜh_tot_diff_colidx = ᶜh_tot⁰[colidx] - ᶜh_tot[colidx]
        vertical_transport!(
            Yₜ.c.ρe_tot[colidx],
            ᶜJ[colidx],
            ᶜρa⁰[colidx],
            ᶠu³_diff_colidx,
            ᶜh_tot_diff_colidx,
            dt,
            edmfx_upwinding,
        )

        # diffusive flux
        ᶠρaK_h = p.ᶠtemp_scalar
        @. ᶠρaK_h[colidx] = ᶠinterp(ᶜρa⁰[colidx]) * ᶠinterp(ᶜK_h[colidx])

        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]),
        )
        @. Yₜ.c.ρe_tot[colidx] -=
            ᶜdivᵥ_ρe_tot(-(ᶠρaK_h[colidx] * ᶠgradᵥ(ᶜh_tot⁰[colidx])))

        if !(p.atmos.moisture_model isa DryModel)
            # mass flux
            for j in 1:n
                @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
                @. ᶜq_tot_diff_colidx =
                    ᶜspecificʲs.:($$j).q_tot[colidx] - ᶜspecific.q_tot[colidx]
                vertical_transport!(
                    Yₜ.c.ρq_tot[colidx],
                    ᶜJ[colidx],
                    Y.c.sgsʲs.:($j).ρa[colidx],
                    ᶠu³_diff_colidx,
                    ᶜq_tot_diff_colidx,
                    dt,
                    edmfx_upwinding,
                )
            end
            @. ᶠu³_diff_colidx = ᶠu³⁰[colidx] - ᶠu³[colidx]
            @. ᶜq_tot_diff_colidx =
                ᶜspecific⁰.q_tot[colidx] - ᶜspecific.q_tot[colidx]
            vertical_transport!(
                Yₜ.c.ρq_tot[colidx],
                ᶜJ[colidx],
                ᶜρa⁰[colidx],
                ᶠu³_diff_colidx,
                ᶜq_tot_diff_colidx,
                dt,
                edmfx_upwinding,
            )

            # diffusive flux
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(
                    sfc_conditions.ρ_flux_q_tot[colidx],
                ),
            )
            @. Yₜ.c.ρq_tot[colidx] -= ᶜdivᵥ_ρq_tot(
                -(ᶠρaK_h[colidx] * ᶠgradᵥ(ᶜspecific⁰.q_tot[colidx])),
            )
        end

        # diffusive flux
        ᶠρaK_u = p.ᶠtemp_scalar
        @. ᶠρaK_u[colidx] = ᶠinterp(ᶜρa⁰[colidx]) * ᶠinterp(ᶜK_u[colidx])
        ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
        )
        @. Yₜ.c.uₕ[colidx] -=
            ᶜdivᵥ_uₕ(-(ᶠρaK_u[colidx] * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]
    end

    # TODO: Add momentum mass flux

    # TODO: Add tracer flux

    return nothing
end

function edmfx_sgs_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::DiagnosticEDMFX,
)

    FT = Spaces.undertype(axes(Y.c))
    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_upwinding, sfc_conditions) = p
    (; ᶠu³, ᶜh_tot, ᶜspecific) = p
    (; ᶜρaʲs, ᶠu³ʲs, ᶜh_totʲs, ᶜq_totʲs) = p
    (; ᶜK_u, ᶜK_h) = p
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠgradᵥ = Operators.GradientC2F()

    if p.atmos.edmfx_sgs_flux
        # mass flux
        ᶠu³_diff_colidx = p.ᶠtemp_CT3[colidx]
        ᶜh_tot_diff_colidx = ᶜq_tot_diff_colidx = p.ᶜtemp_scalar[colidx]
        for j in 1:n
            @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
            @. ᶜh_tot_diff_colidx = ᶜh_totʲs.:($$j)[colidx] - ᶜh_tot[colidx]
            vertical_transport!(
                Yₜ.c.ρe_tot[colidx],
                ᶜJ[colidx],
                ᶜρaʲs.:($j)[colidx],
                ᶠu³_diff_colidx,
                ᶜh_tot_diff_colidx,
                dt,
                edmfx_upwinding,
            )
        end

        # diffusive flux
        ᶠρaK_h = p.ᶠtemp_scalar
        @. ᶠρaK_h[colidx] = ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(ᶜK_h[colidx])

        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]),
        )
        @. Yₜ.c.ρe_tot[colidx] -=
            ᶜdivᵥ_ρe_tot(-(ᶠρaK_h[colidx] * ᶠgradᵥ(ᶜh_tot[colidx])))

        if !(p.atmos.moisture_model isa DryModel)
            # mass flux
            for j in 1:n
                @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
                @. ᶜq_tot_diff_colidx =
                    ᶜq_totʲs.:($$j)[colidx] - ᶜspecific.q_tot[colidx]
                vertical_transport!(
                    Yₜ.c.ρq_tot[colidx],
                    ᶜJ[colidx],
                    ᶜρaʲs.:($j)[colidx],
                    ᶠu³_diff_colidx,
                    ᶜq_tot_diff_colidx,
                    dt,
                    edmfx_upwinding,
                )
            end

            # diffusive flux
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(
                    sfc_conditions.ρ_flux_q_tot[colidx],
                ),
            )
            @. Yₜ.c.ρq_tot[colidx] -= ᶜdivᵥ_ρq_tot(
                -(ᶠρaK_h[colidx] * ᶠgradᵥ(ᶜspecific.q_tot[colidx])),
            )
        end

        # diffusive flux
        ᶠρaK_u = p.ᶠtemp_scalar
        @. ᶠρaK_u[colidx] = ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(ᶜK_u[colidx])
        ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
        )
        @. Yₜ.c.uₕ[colidx] -=
            ᶜdivᵥ_uₕ(-(ᶠρaK_u[colidx] * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]
    end

    # TODO: Add momentum mass flux

    # TODO: Add tracer flux

    return nothing
end
