#####
##### EDMF SGS flux
#####

edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding) = p.atmos.numerics
    (; ᶠu³, ᶜh_tot) = p.precomputed
    (; ᶠu³ʲs, ᶜKʲs, ᶜρʲs) = p.precomputed
    (; ᶜρa⁰, ᶜρ⁰, ᶠu³⁰, ᶜK⁰, ᶜmse⁰, ᶜq_tot⁰) = p.precomputed
    if (
        p.atmos.moisture_model isa NonEquilMoistModel &&
        p.atmos.precip_model isa Microphysics1Moment
    )
        (; ᶜq_liq⁰, ᶜq_ice⁰, ᶜq_rai⁰, ᶜq_sno⁰) = p.precomputed
    end
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J

    if p.atmos.edmfx_model.sgs_mass_flux isa Val{true}
        # energy
        ᶠu³_diff = p.scratch.ᶠtemp_CT3
        ᶜa_scalar = p.scratch.ᶜtemp_scalar
        for j in 1:n
            @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
            @. ᶜa_scalar =
                (Y.c.sgsʲs.:($$j).mse + ᶜKʲs.:($$j) - ᶜh_tot) *
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
            vtt = vertical_transport(
                ᶜρʲs.:($j),
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρe_tot += vtt
        end
        @. ᶠu³_diff = ᶠu³⁰ - ᶠu³
        @. ᶜa_scalar = (ᶜmse⁰ + ᶜK⁰ - ᶜh_tot) * draft_area(ᶜρa⁰, ᶜρ⁰)
        vtt = vertical_transport(
            ᶜρ⁰,
            ᶠu³_diff,
            ᶜa_scalar,
            dt,
            edmfx_sgsflux_upwinding,
        )
        @. Yₜ.c.ρe_tot += vtt

        if !(p.atmos.moisture_model isa DryModel)
            # specific humidity
            for j in 1:n
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
                @. ᶜa_scalar =
                    (Y.c.sgsʲs.:($$j).q_tot - specific(Y.c.ρq_tot, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_tot += vtt
            end
            @. ᶠu³_diff = ᶠu³⁰ - ᶠu³
            @. ᶜa_scalar = (ᶜq_tot⁰ - specific(Y.c.ρq_tot, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρq_tot += vtt
        end

        if (
            p.atmos.moisture_model isa NonEquilMoistModel &&
            p.atmos.precip_model isa Microphysics1Moment
        )
            for j in 1:n
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³

                @. ᶜa_scalar =
                    (Y.c.sgsʲs.:($$j).q_liq - specific(Y.c.ρq_liq, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_liq += vtt

                @. ᶜa_scalar =
                    (Y.c.sgsʲs.:($$j).q_ice - specific(Y.c.ρq_ice, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_ice += vtt

                @. ᶜa_scalar =
                    (Y.c.sgsʲs.:($$j).q_rai - specific(Y.c.ρq_rai, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_rai += vtt

                @. ᶜa_scalar =
                    (Y.c.sgsʲs.:($$j).q_sno - specific(Y.c.ρq_sno, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_sno += vtt
            end
            @. ᶠu³_diff = ᶠu³⁰ - ᶠu³

            @. ᶜa_scalar = (ᶜq_liq⁰ - specific(Y.c.ρq_liq, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρq_liq += vtt

            @. ᶜa_scalar = (ᶜq_ice⁰ - specific(Y.c.ρq_ice, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρq_ice += vtt

            @. ᶜa_scalar = (ᶜq_rai⁰ - specific(Y.c.ρq_rai, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρq_rai += vtt

            @. ᶜa_scalar = (ᶜq_sno⁰ - specific(Y.c.ρq_sno, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρq_sno += vtt
        end
        # TODO - compute sedimentation and terminal velocities
        # TODO - add w q_tot, w h_tot terms
    end
    return nothing
end

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::DiagnosticEDMFX,
)

    turbconv_params = CAP.turbconv_params(p.params)
    a_max = CAP.max_area(turbconv_params)
    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding) = p.atmos.numerics
    (; ᶠu³, ᶜh_tot) = p.precomputed
    (; ᶜρaʲs, ᶜρʲs, ᶠu³ʲs, ᶜKʲs, ᶜmseʲs, ᶜq_totʲs) = p.precomputed
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    FT = eltype(Y)

    if p.atmos.edmfx_model.sgs_mass_flux isa Val{true}
        # energy
        ᶠu³_diff = p.scratch.ᶠtemp_CT3
        ᶜa_scalar = p.scratch.ᶜtemp_scalar
        for j in 1:n
            @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
            # @. ᶜa_scalar =
            #     (ᶜmseʲs.:($$j) + ᶜKʲs.:($$j) - ᶜh_tot) *
            #     draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j))
            # TODO: remove this filter when mass flux is treated implicitly
            @. ᶜa_scalar =
                (ᶜmseʲs.:($$j) + ᶜKʲs.:($$j) - ᶜh_tot) * min(
                    min(draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)), a_max),
                    FT(0.02) / max(
                        Geometry.WVector(ᶜinterp(ᶠu³_diff)).components.data.:1,
                        eps(FT),
                    ),
                )
            vtt = vertical_transport(
                ᶜρʲs.:($j),
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρe_tot += vtt
        end

        if !(p.atmos.moisture_model isa DryModel)
            # specific humidity
            for j in 1:n
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
                # @. ᶜa_scalar =
                #     (ᶜq_totʲs.:($$j) - specific(Y.c.ρq_tot, Y.c.ρ)) *
                #     draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j))
                # TODO: remove this filter when mass flux is treated implicitly
                @. ᶜa_scalar =
                    (ᶜq_totʲs.:($$j) - specific(Y.c.ρq_tot, Y.c.ρ)) * min(
                        min(draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)), a_max),
                        FT(0.02) / max(
                            Geometry.WVector(
                                ᶜinterp(ᶠu³_diff),
                            ).components.data.:1,
                            eps(FT),
                        ),
                    )
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_tot += vtt
            end
        end
    end
    return nothing
end

edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    FT = Spaces.undertype(axes(Y.c))
    (; dt, params) = p
    turbconv_params = CAP.turbconv_params(params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    (; ᶜρa⁰, ᶜu⁰, ᶜK⁰, ᶜmse⁰, ᶜq_tot⁰, ᶜtke⁰, ᶠu³ʲs, ᶠu³⁰, ᶜlinear_buoygrad, ᶜstrain_rate_norm, ᶜdetrʲs) = p.precomputed
    if (
        p.atmos.moisture_model isa NonEquilMoistModel &&
        p.atmos.precip_model isa Microphysics1Moment
    )
        (; ᶜq_liq⁰, ᶜq_ice⁰, ᶜq_rai⁰, ᶜq_sno⁰) = p.precomputed
    end
    (; ρatke_flux) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()

    if p.atmos.edmfx_model.sgs_diffusive_flux isa Val{true}
        (; ustar, obukhov_length) = p.precomputed.sfc_conditions
        (; params) = p

        n = n_mass_flux_subdomains(turbconv_model)

        ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar
        @. ᶜprandtl_nvec =
            turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm)

        sfc_tke = Fields.level(ᶜtke⁰, 1)
        z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
        ᶜz = Fields.coordinate_field(Y.c).z
        ᶜdz = Fields.Δz_field(axes(Y.c))

        ᶜtke_exch = p.scratch.ᶜtemp_scalar_2
        @. ᶜtke_exch = 0
        for j in 1:n
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
        ᶠρaK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_h = ᶠinterp(ᶜρa⁰) * ᶠinterp(ᶜK_h)
        ᶠρaK_u = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_u = ᶠinterp(ᶜρa⁰) * ᶠinterp(ᶜK_u)

        # energy
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(ᶠρaK_h * ᶠgradᵥ(ᶜmse⁰ + ᶜK⁰)))
        if use_prognostic_tke(turbconv_model)
            # turbulent transport (diffusive flux)
            # boundary condition for the diffusive flux
            ᶜdivᵥ_ρatke = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(ρatke_flux),
            )
            # relax tke to zero in one time step if tke < 0
            @. Yₜ.c.sgs⁰.ρatke -=
                ᶜdivᵥ_ρatke(-(ᶠρaK_u * ᶠgradᵥ(ᶜtke⁰))) + ifelse(
                    ᶜtke⁰ >= FT(0),
                    tke_dissipation(
                        turbconv_params,
                        Y.c.sgs⁰.ρatke,
                        ᶜtke⁰,
                        ᶜmixing_length,
                    ),
                    Y.c.sgs⁰.ρatke / float(dt),
                )
        end
        if !(p.atmos.moisture_model isa DryModel)
            # specific humidity
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq_tot(-(ᶠρaK_h * ᶠgradᵥ(ᶜq_tot⁰)))
            @. Yₜ.c.ρq_tot -= ᶜρχₜ_diffusion
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
        end
        if (
            p.atmos.moisture_model isa NonEquilMoistModel &&
            p.atmos.precip_model isa Microphysics1Moment
        )
            α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)

            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜdivᵥ_ρq = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq(-(ᶠρaK_h * ᶠgradᵥ(ᶜq_liq⁰)))
            @. Yₜ.c.ρq_liq -= ᶜρχₜ_diffusion

            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq(-(ᶠρaK_h * ᶠgradᵥ(ᶜq_ice⁰)))
            @. Yₜ.c.ρq_ice -= ᶜρχₜ_diffusion

            # TODO - do I need to change anything in the implicit solver
            # to include the α_vert_diff_tracer?
            @. ᶜρχₜ_diffusion =
                ᶜdivᵥ_ρq(-(ᶠρaK_h * α_vert_diff_tracer * ᶠgradᵥ(ᶜq_rai⁰)))
            @. Yₜ.c.ρq_rai -= ᶜρχₜ_diffusion

            @. ᶜρχₜ_diffusion =
                ᶜdivᵥ_ρq(-(ᶠρaK_h * α_vert_diff_tracer * ᶠgradᵥ(ᶜq_sno⁰)))
            @. Yₜ.c.ρq_sno -= ᶜρχₜ_diffusion
        end

        # momentum
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        ᶠstrain_rate .= compute_strain_rate_face(ᶜu⁰)
        @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Y.c.ρ)
    end
    return nothing
end

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::Union{EDOnlyEDMFX, DiagnosticEDMFX},
)

    FT = Spaces.undertype(axes(Y.c))
    (; dt, params) = p
    turbconv_params = CAP.turbconv_params(params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    (; ᶜu, ᶜh_tot, ᶜtke⁰, ᶜmixing_length) = p.precomputed
    (; ᶜK_u, ᶜK_h, ρatke_flux) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()

    if p.atmos.edmfx_model.sgs_diffusive_flux isa Val{true}
        ᶠρaK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_h = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h)
        ᶠρaK_u = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_u = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_u)

        # energy
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(ᶠρaK_h * ᶠgradᵥ(ᶜh_tot)))

        if use_prognostic_tke(turbconv_model)
            # turbulent transport (diffusive flux)
            # boundary condition for the diffusive flux
            ᶜdivᵥ_ρatke = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(ρatke_flux),
            )
            # relax tke to zero in one time step if tke < 0
            @. Yₜ.c.sgs⁰.ρatke -=
                ᶜdivᵥ_ρatke(-(ᶠρaK_u * ᶠgradᵥ(ᶜtke⁰))) + ifelse(
                    ᶜtke⁰ >= FT(0),
                    tke_dissipation(
                        turbconv_params,
                        Y.c.sgs⁰.ρatke,
                        ᶜtke⁰,
                        ᶜmixing_length,
                    ),
                    Y.c.sgs⁰.ρatke / float(dt),
                )
        end

        if !(p.atmos.moisture_model isa DryModel)
            # specific humidity
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )
            @. ᶜρχₜ_diffusion =
                ᶜdivᵥ_ρq_tot(-(ᶠρaK_h * ᶠgradᵥ(specific(Y.c.ρq_tot, Y.c.ρ))))
            @. Yₜ.c.ρq_tot -= ᶜρχₜ_diffusion
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
        end

        # momentum
        ᶠstrain_rate = p.scratch.ᶠtemp_UVWxUVW
        ᶠstrain_rate .= compute_strain_rate_face(ᶜu)
        @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Y.c.ρ)
    end

    # TODO: Add tracer flux

    return nothing
end
