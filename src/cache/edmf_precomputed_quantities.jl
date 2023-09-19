#####
##### Precomputed quantities
#####
import Thermodynamics as TD
import ClimaCore: Spaces, Fields

"""
    set_edmf_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities stored in `p` for edmfx.
"""
function set_edmf_precomputed_quantities!(Y, p, ᶠuₕ³, t)
    (; energy_form, moisture_model, turbconv_model) = p.atmos
    #EDMFX BCs only support total energy as state variable
    @assert energy_form isa TotalEnergy
    @assert !(moisture_model isa DryModel)

    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    (; dt) = p.simulation
    thermo_params = CAP.thermodynamics_params(params)
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_args = (thermo_params, energy_form, moisture_model)

    (; ᶜspecific, ᶜp, ᶜΦ, ᶜh_tot, ᶜρ_ref) = p
    (; ᶜspecific⁰, ᶜρa⁰, ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜts⁰, ᶜρ⁰, ᶜh_tot⁰) = p
    (; ᶜmixing_length, ᶜlinear_buoygrad, ᶜstrain_rate_norm, ᶜK_u, ᶜK_h) = p
    (;
        ᶜspecificʲs,
        ᶜuʲs,
        ᶠu³ʲs,
        ᶜKʲs,
        ᶜtsʲs,
        ᶜρʲs,
        ᶜh_totʲs,
        ᶜentrʲs,
        ᶜdetrʲs,
    ) = p
    (; ustar, obukhov_length, buoyancy_flux) = p.sfc_conditions

    @. ᶜspecific⁰ = specific_full_sgs⁰(Y.c, turbconv_model)
    @. ᶜρa⁰ = ρa⁰(Y.c)
    set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Y, turbconv_model)
    set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Y.c.uₕ, ᶠuₕ³)
    @. ᶜK⁰ += ᶜspecific⁰.tke
    @. ᶜts⁰ = ts_sgs(thermo_args..., ᶜspecific⁰, ᶜK⁰, ᶜΦ, ᶜp)
    @. ᶜρ⁰ = TD.air_density(thermo_params, ᶜts⁰)
    @. ᶜh_tot⁰ =
        TD.total_specific_enthalpy(thermo_params, ᶜts⁰, ᶜspecific⁰.e_tot)
    @. ᶜspecificʲs = specific_sgsʲs(Y.c, turbconv_model)
    for j in 1:n
        ᶜuʲ = ᶜuʲs.:($j)
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        ᶜKʲ = ᶜKʲs.:($j)
        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
        ᶜspecificʲ = ᶜspecificʲs.:($j)
        ᶜtsʲ = ᶜtsʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜh_totʲ = ᶜh_totʲs.:($j)

        set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³)
        @. ᶜtsʲ = ts_sgs(thermo_args..., ᶜspecificʲ, ᶜKʲ, ᶜΦ, ᶜp)
        @. ᶜρʲ = TD.air_density(thermo_params, ᶜtsʲ)

        # When ᶜe_intʲ = ᶜe_int and ᶜq_totʲ = ᶜq_tot, we still observe that
        # ᶜρʲ != ᶜρ. This is because the conversion from ᶜρ to ᶜp to ᶜρʲ
        # introduces a tiny round-off error of order epsilon to ᶜρʲ. If left
        # unchecked, this round-off error then changes the tendency of ᶠu₃ʲ,
        # which in turn introduces an error to ᶠu³ʲ, which then increases
        # the error in ᶜρʲ. For now, we will filter ᶜρʲ to fix this. Note
        # that this will no longer be necessary after we add diffusion.
        @. ᶜρʲ = ifelse(abs(ᶜρʲ - Y.c.ρ) <= 2 * eps(Y.c.ρ), Y.c.ρ, ᶜρʲ)

        # EDMFX boundary condition:

        # We need field_values everywhere because we are mixing
        # information from surface and first interior inside the
        # sgs_h/q_tot_first_interior_bc call.
        ᶜz_int_val =
            Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
        z_sfc_val = Fields.field_values(
            Fields.level(Fields.coordinate_field(Y.f).z, Fields.half),
        )
        ᶜρ_int_val = Fields.field_values(Fields.level(Y.c.ρ, 1))
        ᶜp_int_val = Fields.field_values(Fields.level(ᶜp, 1))
        (; ρ_flux_h_tot, ρ_flux_q_tot, ustar, obukhov_length) = p.sfc_conditions
        buoyancy_flux_val = Fields.field_values(buoyancy_flux)
        ρ_flux_h_tot_val = Fields.field_values(ρ_flux_h_tot)
        ρ_flux_q_tot_val = Fields.field_values(ρ_flux_q_tot)
        ustar_val = Fields.field_values(ustar)
        obukhov_length_val = Fields.field_values(obukhov_length)
        sfc_local_geometry_val = Fields.field_values(
            Fields.local_geometry_field(Fields.level(Y.f, Fields.half)),
        )

        # Based on boundary conditions for updrafts we overwrite
        # the first interior point for EDMFX ᶜh_totʲ...
        @. ᶜh_totʲ =
            TD.total_specific_enthalpy(thermo_params, ᶜtsʲ, ᶜspecificʲ.e_tot)
        ᶜh_tot_int_val = Fields.field_values(Fields.level(ᶜh_tot, 1))
        ᶜh_totʲ_int_val = Fields.field_values(Fields.level(ᶜh_totʲ, 1))
        @. ᶜh_totʲ_int_val = sgs_scalar_first_interior_bc(
            ᶜz_int_val - z_sfc_val,
            ᶜρ_int_val,
            ᶜh_tot_int_val,
            buoyancy_flux_val,
            ρ_flux_h_tot_val,
            ustar_val,
            obukhov_length_val,
            sfc_local_geometry_val,
        )

        # ... and the first interior point for EDMFX ᶜq_totʲ.
        ᶜq_tot_int_val = Fields.field_values(Fields.level(ᶜspecific.q_tot, 1))
        ᶜq_totʲ_int_val = Fields.field_values(Fields.level(ᶜspecificʲ.q_tot, 1))
        @. ᶜq_totʲ_int_val = sgs_scalar_first_interior_bc(
            ᶜz_int_val - z_sfc_val,
            ᶜρ_int_val,
            ᶜq_tot_int_val,
            buoyancy_flux_val,
            ρ_flux_q_tot_val,
            ustar_val,
            obukhov_length_val,
            sfc_local_geometry_val,
        )

        # Then overwrite the prognostic variables at first inetrior point.
        ᶜKʲ_int_val = Fields.field_values(Fields.level(ᶜKʲ, 1))
        ᶜΦ_int_val = Fields.field_values(Fields.level(ᶜΦ, 1))
        ᶜtsʲ_int_val = Fields.field_values(Fields.level(ᶜtsʲ, 1))
        @. ᶜtsʲ_int_val = TD.PhaseEquil_phq(
            thermo_params,
            ᶜp_int_val,
            ᶜh_totʲ_int_val - ᶜKʲ_int_val - ᶜΦ_int_val,
            ᶜq_totʲ_int_val,
        )
        sgsʲs_ρ_int_val = Fields.field_values(Fields.level(ᶜρʲs.:($j), 1))
        sgsʲs_ρa_int_val =
            Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρa, 1))
        sgsʲs_ρae_tot_int_val =
            Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρae_tot, 1))
        sgsʲs_ρaq_tot_int_val =
            Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρaq_tot, 1))

        turbconv_params = CAP.turbconv_params(params)
        @. sgsʲs_ρa_int_val =
            $(FT(turbconv_params.surface_area)) *
            TD.air_density(thermo_params, ᶜtsʲ_int_val)
        @. sgsʲs_ρae_tot_int_val =
            sgsʲs_ρa_int_val * TD.total_energy(
                thermo_params,
                ᶜtsʲ_int_val,
                ᶜKʲ_int_val,
                ᶜΦ_int_val,
            )
        @. sgsʲs_ρaq_tot_int_val = sgsʲs_ρa_int_val * ᶜq_totʲ_int_val
    end

    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶜlg = Fields.local_geometry_field(Y.c)

    for j in 1:n
        @. ᶜentrʲs.:($$j) = entrainment(
            params,
            ᶜz,
            z_sfc,
            ᶜp,
            Y.c.ρ,
            buoyancy_flux,
            Y.c.sgsʲs.:($$j).ρa / ᶜρʲs.:($$j),
            get_physical_w(ᶜuʲs.:($$j), ᶜlg),
            TD.relative_humidity(thermo_params, ᶜtsʲs.:($$j)),
            ᶜphysical_buoyancy(params, ᶜρ_ref, ᶜρʲs.:($$j)),
            get_physical_w(ᶜu⁰, ᶜlg),
            TD.relative_humidity(thermo_params, ᶜts⁰),
            ᶜphysical_buoyancy(params, ᶜρ_ref, ᶜρ⁰),
            dt,
            p.atmos.edmfx_entr_model,
        )
        @. ᶜdetrʲs.:($$j) = detrainment(
            params,
            ᶜz,
            z_sfc,
            ᶜp,
            Y.c.ρ,
            buoyancy_flux,
            Y.c.sgsʲs.:($$j).ρa / ᶜρʲs.:($$j),
            get_physical_w(ᶜuʲs.:($$j), ᶜlg),
            TD.relative_humidity(thermo_params, ᶜtsʲs.:($$j)),
            ᶜphysical_buoyancy(params, ᶜρ_ref, ᶜρʲs.:($$j)),
            get_physical_w(ᶜu⁰, ᶜlg),
            TD.relative_humidity(thermo_params, ᶜts⁰),
            ᶜphysical_buoyancy(params, ᶜρ_ref, ᶜρ⁰),
            dt,
            p.atmos.edmfx_detr_model,
        )
    end

    # First order approximation: Use environmental mean fields.
    @. ᶜlinear_buoygrad = buoyancy_gradients(
        params,
        moisture_model,
        EnvBuoyGrad(
            BuoyGradMean(),
            TD.air_temperature(thermo_params, ᶜts⁰),                           # t_sat
            TD.vapor_specific_humidity(thermo_params, ᶜts⁰),                   # qv_sat
            ᶜspecific⁰.q_tot,                                                  # qt_sat
            TD.dry_pottemp(thermo_params, ᶜts⁰),                               # θ_sat
            TD.liquid_ice_pottemp(thermo_params, ᶜts⁰),                        # θ_liq_ice_sat
            projected_vector_data(
                C3,
                ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ᶜts⁰))),
                ᶜlg,
            ),                                                                 # ∂θv∂z_unsat
            projected_vector_data(C3, ᶜgradᵥ(ᶠinterp(ᶜspecific⁰.q_tot)), ᶜlg), # ∂qt∂z_sat
            projected_vector_data(
                C3,
                ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts⁰))),
                ᶜlg,
            ),                                                                 # ∂θl∂z_sat
            ᶜp,                                                                # p
            ifelse(TD.has_condensate(thermo_params, ᶜts⁰), 1, 0),              # en_cld_frac
            ᶜρ⁰,                                                               # ρ
        ),
    )

    @. ᶜstrain_rate_norm = $(FT(1e-4))
    ᶜprandtl_nvec = p.ᶜtemp_scalar
    @. ᶜprandtl_nvec = FT(1) / 3
    ᶜtke_exch = p.ᶜtemp_scalar_2
    @. ᶜtke_exch = 0
    for j in 1:n
        @. ᶜtke_exch +=
            Y.c.sgsʲs.:($$j).ρa * ᶜdetrʲs.:($$j) / ᶜρa⁰ * (
                1 / 2 *
                (
                    get_physical_w(ᶜuʲs.:($$j), ᶜlg) - get_physical_w(ᶜu⁰, ᶜlg)
                )^2 - ᶜspecific⁰.tke
            )
    end

    sfc_tke = Fields.level(ᶜspecific⁰.tke, 1)
    @. ᶜmixing_length = mixing_length(
        p.params,
        ustar,
        ᶜz,
        z_sfc,
        ᶜdz,
        sfc_tke,
        ᶜlinear_buoygrad,
        ᶜspecific⁰.tke,
        obukhov_length,
        ᶜstrain_rate_norm,
        ᶜprandtl_nvec,
        ᶜtke_exch,
    )

    turbconv_params = CAP.turbconv_params(params)
    c_m = TCP.tke_ed_coeff(turbconv_params)
    @. ᶜK_u = c_m * ᶜmixing_length * sqrt(max(ᶜspecific⁰.tke, 0))
    # TODO: add Prantdl number
    @. ᶜK_h = ᶜK_u / ᶜprandtl_nvec

    return nothing
end
