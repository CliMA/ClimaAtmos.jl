#####
##### EDMF SGS flux
#####

edmfx_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing

function edmfx_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, turbconv_model::EDMFX)

    FT = Spaces.undertype(axes(Y.c))
    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_upwinding, sfc_conditions) = p
    (; ᶠu³, ᶜh_tot, ᶜspecific) = p
    (; ᶠu³ʲs, ᶜh_totʲs, ᶜspecificʲs) = p
    (; ᶜρa⁰, ᶠu³⁰, ᶜh_tot⁰, ᶜspecific⁰) = p
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
    (; ᶜh_tot⁰, ᶜK_u, ᶜK_h) = p
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠgradᵥ = Operators.GradientC2F()

    if p.atmos.edmfx_sgs_flux
        # mass flux
        # TODO: check if there is contribution from the environment
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
            ᶜdivᵥ_ρe_tot(-(ᶠρaK_h[colidx] * ᶠgradᵥ(ᶜh_tot⁰[colidx])))

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
