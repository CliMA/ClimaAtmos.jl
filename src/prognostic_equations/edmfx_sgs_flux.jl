#####
##### EDMF SGS flux
#####

edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::PrognosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding) = p.atmos.numerics
    (; ᶠu³, ᶜh_tot, ᶜspecific) = p.precomputed
    (; ᶠu³ʲs, ᶜKʲs, ᶜρʲs) = p.precomputed
    (; ᶜρa⁰, ᶜρ⁰, ᶠu³⁰, ᶜK⁰, ᶜmse⁰, ᶜq_tot⁰) = p.precomputed
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J

    if p.atmos.edmfx_sgs_mass_flux
        # energy
        ᶠu³_diff_colidx = p.scratch.ᶠtemp_CT3[colidx]
        ᶜa_scalar_colidx = p.scratch.ᶜtemp_scalar[colidx]
        for j in 1:n
            @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
            @. ᶜa_scalar_colidx =
                (
                    Y.c.sgsʲs.:($$j).mse[colidx] + ᶜKʲs.:($$j)[colidx] -
                    ᶜh_tot[colidx]
                ) * draft_area(Y.c.sgsʲs.:($$j).ρa[colidx], ᶜρʲs.:($$j)[colidx])
            vertical_transport!(
                Yₜ.c.ρe_tot[colidx],
                ᶜJ[colidx],
                ᶜρʲs.:($j)[colidx],
                ᶠu³_diff_colidx,
                ᶜa_scalar_colidx,
                dt,
                edmfx_sgsflux_upwinding,
            )
        end
        @. ᶠu³_diff_colidx = ᶠu³⁰[colidx] - ᶠu³[colidx]
        @. ᶜa_scalar_colidx =
            (ᶜmse⁰[colidx] + ᶜK⁰[colidx] - ᶜh_tot[colidx]) *
            draft_area(ᶜρa⁰[colidx], ᶜρ⁰[colidx])
        vertical_transport!(
            Yₜ.c.ρe_tot[colidx],
            ᶜJ[colidx],
            ᶜρ⁰[colidx],
            ᶠu³_diff_colidx,
            ᶜa_scalar_colidx,
            dt,
            edmfx_sgsflux_upwinding,
        )

        if !(p.atmos.moisture_model isa DryModel)
            # specific humidity
            for j in 1:n
                @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
                @. ᶜa_scalar_colidx =
                    (Y.c.sgsʲs.:($$j).q_tot[colidx] - ᶜspecific.q_tot[colidx]) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa[colidx], ᶜρʲs.:($$j)[colidx])
                vertical_transport!(
                    Yₜ.c.ρq_tot[colidx],
                    ᶜJ[colidx],
                    ᶜρʲs.:($j)[colidx],
                    ᶠu³_diff_colidx,
                    ᶜa_scalar_colidx,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
            end
            @. ᶠu³_diff_colidx = ᶠu³⁰[colidx] - ᶠu³[colidx]
            @. ᶜa_scalar_colidx =
                (ᶜq_tot⁰[colidx] - ᶜspecific.q_tot[colidx]) *
                draft_area(ᶜρa⁰[colidx], ᶜρ⁰[colidx])
            vertical_transport!(
                Yₜ.c.ρq_tot[colidx],
                ᶜJ[colidx],
                ᶜρ⁰[colidx],
                ᶠu³_diff_colidx,
                ᶜa_scalar_colidx,
                dt,
                edmfx_sgsflux_upwinding,
            )
        end
    end

    # TODO: Add tracer flux

    return nothing
end

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::DiagnosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding) = p.atmos.numerics
    (; ᶠu³, ᶜh_tot, ᶜspecific) = p.precomputed
    (; ᶜρaʲs, ᶜρʲs, ᶠu³ʲs, ᶜh_totʲs, ᶜq_totʲs) = p.precomputed
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J

    if p.atmos.edmfx_sgs_mass_flux
        # energy
        ᶠu³_diff_colidx = p.scratch.ᶠtemp_CT3[colidx]
        ᶜa_scalar_colidx = p.scratch.ᶜtemp_scalar[colidx]
        for j in 1:n
            @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
            @. ᶜa_scalar_colidx =
                (ᶜh_totʲs.:($$j)[colidx] - ᶜh_tot[colidx]) *
                draft_area(ᶜρaʲs.:($$j)[colidx], ᶜρʲs.:($$j)[colidx])
            vertical_transport!(
                Yₜ.c.ρe_tot[colidx],
                ᶜJ[colidx],
                ᶜρʲs.:($j)[colidx],
                ᶠu³_diff_colidx,
                ᶜa_scalar_colidx,
                dt,
                edmfx_sgsflux_upwinding,
            )
        end

        if !(p.atmos.moisture_model isa DryModel)
            # specific humidity
            for j in 1:n
                @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
                @. ᶜa_scalar_colidx =
                    (ᶜq_totʲs.:($$j)[colidx] - ᶜspecific.q_tot[colidx]) *
                    draft_area(ᶜρaʲs.:($$j)[colidx], ᶜρʲs.:($$j)[colidx])
                vertical_transport!(
                    Yₜ.c.ρq_tot[colidx],
                    ᶜJ[colidx],
                    ᶜρʲs.:($j)[colidx],
                    ᶠu³_diff_colidx,
                    ᶜa_scalar_colidx,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
            end
        end
    end

    # TODO: Add tracer flux

    return nothing
end

edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) =
    nothing

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::PrognosticEDMFX,
)

    FT = Spaces.undertype(axes(Y.c))
    (; sfc_conditions) = p.precomputed
    (; ᶜρa⁰, ᶜu⁰, ᶜK⁰, ᶜmse⁰, ᶜq_tot⁰) = p.precomputed
    (; ᶜK_u, ᶜK_h) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()

    if p.atmos.edmfx_sgs_diffusive_flux
        # energy
        ᶠρaK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_h[colidx] = ᶠinterp(ᶜρa⁰[colidx]) * ᶠinterp(ᶜK_h[colidx])

        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]),
        )
        @. Yₜ.c.ρe_tot[colidx] -= ᶜdivᵥ_ρe_tot(
            -(ᶠρaK_h[colidx] * ᶠgradᵥ(ᶜmse⁰[colidx] + ᶜK⁰[colidx])),
        )

        if !(p.atmos.moisture_model isa DryModel)
            # specific humidity
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(
                    sfc_conditions.ρ_flux_q_tot[colidx],
                ),
            )
            @. Yₜ.c.ρq_tot[colidx] -=
                ᶜdivᵥ_ρq_tot(-(ᶠρaK_h[colidx] * ᶠgradᵥ(ᶜq_tot⁰[colidx])))
        end

        # momentum
        ᶠρaK_u = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_u[colidx] = ᶠinterp(ᶜρa⁰[colidx]) * ᶠinterp(ᶜK_u[colidx])
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        compute_strain_rate_face!(ᶠstrain_rate[colidx], ᶜu⁰[colidx])
        @. Yₜ.c.uₕ[colidx] -= C12(
            ᶜdivᵥ(-(2 * ᶠρaK_u[colidx] * ᶠstrain_rate[colidx])) / Y.c.ρ[colidx],
        )
        # apply boundary condition for momentum flux
        ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
        )
        @. Yₜ.c.uₕ[colidx] -=
            ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]
    end

    # TODO: Add tracer flux

    return nothing
end

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    turbconv_model::DiagnosticEDMFX,
)

    FT = Spaces.undertype(axes(Y.c))
    (; sfc_conditions) = p.precomputed
    (; ᶜu, ᶜh_tot, ᶜspecific) = p.precomputed
    (; ᶜK_u, ᶜK_h) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()

    if p.atmos.edmfx_sgs_diffusive_flux
        # energy
        ᶠρaK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_h[colidx] = ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(ᶜK_h[colidx])

        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]),
        )
        @. Yₜ.c.ρe_tot[colidx] -=
            ᶜdivᵥ_ρe_tot(-(ᶠρaK_h[colidx] * ᶠgradᵥ(ᶜh_tot[colidx])))

        if !(p.atmos.moisture_model isa DryModel)
            # specific humidity
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

        # momentum
        ᶠρaK_u = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_u[colidx] = ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(ᶜK_u[colidx])
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        compute_strain_rate_face!(ᶠstrain_rate[colidx], ᶜu[colidx])
        @. Yₜ.c.uₕ[colidx] -= C12(
            ᶜdivᵥ(-(2 * ᶠρaK_u[colidx] * ᶠstrain_rate[colidx])) / Y.c.ρ[colidx],
        )
        # apply boundary condition for momentum flux
        ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
        )
        @. Yₜ.c.uₕ[colidx] -=
            ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]
    end

    # TODO: Add tracer flux

    return nothing
end
