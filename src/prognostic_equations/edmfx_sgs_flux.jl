#####
##### EDMF SGS flux
#####

edmfx_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing

function edmfx_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, turbconv_model::EDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; params, edmfx_upwinding) = p
    (; ᶠu³, ᶜh_tot, ᶜspecific) = p
    (; ᶠu³ʲs, ᶜh_totʲs, ᶜspecificʲs) = p
    (; ᶜρa⁰, ᶠu³⁰, ᶜspecific⁰, ᶜts⁰) = p
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(Y.c).J

    thermo_params = CAP.thermodynamics_params(params)
    ρe_tot_tendency = ρq_tot_tendency = p.ᶜtemp_scalar[colidx]
    if p.atmos.edmfx_sgs_flux
        ᶠu³_diff_colidx = p.ᶠtemp_CT3[colidx]
        ᶜh_tot_diff_colidx = ᶜq_tot_diff_colidx = p.ᶜtemp_scalar[colidx]
        for j in 1:n
            @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
            @. ᶜh_tot_diff_colidx = ᶜh_totʲs.:($$j)[colidx] - ᶜh_tot[colidx]
            vertical_transport!(
                ρe_tot_tendency,
                ᶜJ[colidx],
                Y.c.sgsʲs.:($j).ρa[colidx],
                ᶠu³_diff_colidx,
                ᶜh_tot_diff_colidx,
                dt,
                edmfx_upwinding,
            )
            @. Yₜ.c.ρe_tot[colidx] += ρe_tot_tendency
        end
        @. ᶠu³_diff_colidx = ᶠu³⁰[colidx] - ᶠu³[colidx]
        @. ᶜh_tot_diff_colidx =
            TD.total_specific_enthalpy.(
                thermo_params,
                ᶜts⁰[colidx],
                ᶜspecific⁰.e_tot[colidx],
            ) - ᶜh_tot[colidx]
        vertical_transport!(
            ρe_tot_tendency,
            ᶜJ[colidx],
            ᶜρa⁰[colidx],
            ᶠu³_diff_colidx,
            ᶜh_tot_diff_colidx,
            dt,
            edmfx_upwinding,
        )
        @. Yₜ.c.ρe_tot[colidx] += ρe_tot_tendency

        if !(p.atmos.moisture_model isa DryModel)
            for j in 1:n
                @. ᶠu³_diff_colidx = ᶠu³ʲs.:($$j)[colidx] - ᶠu³[colidx]
                @. ᶜq_tot_diff_colidx =
                    ᶜspecificʲs.:($$j).q_tot[colidx] - ᶜspecific.q_tot[colidx]
                vertical_transport!(
                    ρq_tot_tendency,
                    ᶜJ[colidx],
                    Y.c.sgsʲs.:($j).ρa[colidx],
                    ᶠu³_diff_colidx,
                    ᶜq_tot_diff_colidx,
                    dt,
                    edmfx_upwinding,
                )
                @. Yₜ.c.ρq_tot[colidx] += ρq_tot_tendency
            end
            @. ᶠu³_diff_colidx = ᶠu³⁰[colidx] - ᶠu³[colidx]
            @. ᶜq_tot_diff_colidx =
                ᶜspecific⁰.q_tot[colidx] - ᶜspecific.q_tot[colidx]
            vertical_transport!(
                ρq_tot_tendency,
                ᶜJ[colidx],
                ᶜρa⁰[colidx],
                ᶠu³_diff_colidx,
                ᶜq_tot_diff_colidx,
                dt,
                edmfx_upwinding,
            )
            @. Yₜ.c.ρq_tot[colidx] += ρq_tot_tendency
        end
    end

    # TODO: Add momentum flux

    # TODO: Add tracer flux

    return nothing
end
