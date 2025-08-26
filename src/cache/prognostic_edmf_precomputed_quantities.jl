#####
##### Precomputed quantities
#####
import NVTX
import Thermodynamics as TD
import ClimaCore: Spaces, Fields

"""
    set_prognostic_edmf_precomputed_quantities!(Y, p, ᶠuₕ³, t)

Updates the edmf environment precomputed quantities stored in `p` for prognostic edmfx.
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_environment!(
    Y,
    p,
    ᶠuₕ³,
    t,
)
    @assert !(p.atmos.moisture_model isa DryModel)

    thermo_params = CAP.thermodynamics_params(p.params)
    (; turbconv_model) = p.atmos
    (; ᶜΦ,) = p.core
    (; ᶜp, ᶜK) = p.precomputed
    (; ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜts⁰) = p.precomputed

    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))

    set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Y, turbconv_model)
    set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Y.c.uₕ, ᶠuₕ³)
    # @. ᶜK⁰ += ᶜtke⁰
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)

    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)

    if p.atmos.moisture_model isa NonEquilMoistModel && (
        p.atmos.microphysics_model isa Microphysics1Moment ||
        p.atmos.microphysics_model isa Microphysics2Moment
    )
        ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
        ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
        ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
        ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
        @. ᶜts⁰ = TD.PhaseNonEquil_phq(
            thermo_params,
            ᶜp,
            ᶜmse⁰ - ᶜΦ,
            TD.PhasePartition(ᶜq_tot⁰, ᶜq_liq⁰ + ᶜq_rai⁰, ᶜq_ice⁰ + ᶜq_sno⁰),
        )
    else

        @. ᶜts⁰ = TD.PhaseEquil_phq(thermo_params, ᶜp, ᶜmse⁰ - ᶜΦ, ᶜq_tot⁰)
    end
    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_draft!(Y, p, ᶠuₕ³, t)

Updates velocity and thermodynamics quantities in each SGS draft.
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_draft!(
    Y,
    p,
    ᶠuₕ³,
    t,
)
    (; moisture_model, turbconv_model) = p.atmos
    @assert !(moisture_model isa DryModel)

    n = n_mass_flux_subdomains(turbconv_model)
    thermo_params = CAP.thermodynamics_params(p.params)

    (; ᶜΦ,) = p.core
    (; ᶜp, ᶜuʲs, ᶠu³ʲs, ᶜKʲs, ᶠKᵥʲs, ᶜtsʲs, ᶜρʲs) = p.precomputed

    for j in 1:n
        ᶜuʲ = ᶜuʲs.:($j)
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        ᶜKʲ = ᶜKʲs.:($j)
        ᶠKᵥʲ = ᶠKᵥʲs.:($j)
        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
        ᶜtsʲ = ᶜtsʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot
        if p.atmos.moisture_model isa NonEquilMoistModel && (
            p.atmos.microphysics_model isa Microphysics1Moment ||
            p.atmos.microphysics_model isa Microphysics2Moment
        )
            ᶜq_liqʲ = Y.c.sgsʲs.:($j).q_liq
            ᶜq_iceʲ = Y.c.sgsʲs.:($j).q_ice
            ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
            ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
        end

        set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³)
        @. ᶠKᵥʲ = (adjoint(CT3(ᶠu₃ʲ)) * ᶠu₃ʲ) / 2
        if p.atmos.moisture_model isa NonEquilMoistModel && (
            p.atmos.microphysics_model isa Microphysics1Moment ||
            p.atmos.microphysics_model isa Microphysics2Moment
        )
            @. ᶜtsʲ = TD.PhaseNonEquil_phq(
                thermo_params,
                ᶜp,
                ᶜmseʲ - ᶜΦ,
                TD.PhasePartition(
                    ᶜq_totʲ,
                    ᶜq_liqʲ + ᶜq_raiʲ,
                    ᶜq_iceʲ + ᶜq_snoʲ,
                ),
            )
        else
            @. ᶜtsʲ = TD.PhaseEquil_phq(thermo_params, ᶜp, ᶜmseʲ - ᶜΦ, ᶜq_totʲ)
        end
        @. ᶜρʲ = TD.air_density(thermo_params, ᶜtsʲ)
    end
    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_bottom_bc!(Y, p, ᶠuₕ³, t)

Updates velocity and thermodynamics quantities at the surface in each SGS draft.
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_bottom_bc!(
    Y,
    p,
    t,
)
    (; moisture_model, turbconv_model) = p.atmos
    @assert !(moisture_model isa DryModel)

    FT = Spaces.undertype(axes(Y.c))
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_params = CAP.turbconv_params(p.params)

    (; ᶜΦ,) = p.core
    (; ᶜp, ᶜK, ᶜtsʲs, ᶜρʲs, ᶜts) = p.precomputed
    (; ustar, obukhov_length, buoyancy_flux) = p.precomputed.sfc_conditions

    for j in 1:n
        ᶜtsʲ = ᶜtsʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot
        if p.atmos.moisture_model isa NonEquilMoistModel && (
            p.atmos.microphysics_model isa Microphysics1Moment ||
            p.atmos.microphysics_model isa Microphysics2Moment
        )
            ᶜq_liqʲ = Y.c.sgsʲs.:($j).q_liq
            ᶜq_iceʲ = Y.c.sgsʲs.:($j).q_ice
            ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
            ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
        end

        # We need field_values everywhere because we are mixing
        # information from surface and first interior inside the
        # sgs_scalar_first_interior_bc call.
        ᶜz_int_val =
            Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
        z_sfc_val = Fields.field_values(
            Fields.level(Fields.coordinate_field(Y.f).z, Fields.half),
        )
        ᶜρ_int_val = Fields.field_values(Fields.level(Y.c.ρ, 1))
        ᶜp_int_val = Fields.field_values(Fields.level(ᶜp, 1))
        (; ρ_flux_h_tot, ρ_flux_q_tot, ustar, obukhov_length) =
            p.precomputed.sfc_conditions

        buoyancy_flux_val = Fields.field_values(buoyancy_flux)
        ρ_flux_h_tot_val = Fields.field_values(ρ_flux_h_tot)
        ρ_flux_q_tot_val = Fields.field_values(ρ_flux_q_tot)

        ustar_val = Fields.field_values(ustar)
        obukhov_length_val = Fields.field_values(obukhov_length)
        sfc_local_geometry_val = Fields.field_values(
            Fields.local_geometry_field(Fields.level(Y.f, Fields.half)),
        )

        # Based on boundary conditions for updrafts we overwrite
        # the first interior point for EDMFX ᶜmseʲ...
        ᶜaʲ_int_val = p.scratch.temp_data_level
        # TODO: replace this with the actual surface area fraction when
        # using prognostic surface area
        @. ᶜaʲ_int_val = FT(turbconv_params.surface_area)
        ᶜh_tot = @. lazy(
            TD.total_specific_enthalpy(
                thermo_params,
                ᶜts,
                specific(Y.c.ρe_tot, Y.c.ρ),
            ),
        )
        ᶜh_tot_int_val = Fields.field_values(Fields.level(ᶜh_tot, 1))
        ᶜK_int_val = Fields.field_values(Fields.level(ᶜK, 1))
        ᶜmseʲ_int_val = Fields.field_values(Fields.level(ᶜmseʲ, 1))
        @. ᶜmseʲ_int_val = sgs_scalar_first_interior_bc(
            ᶜz_int_val - z_sfc_val,
            ᶜρ_int_val,
            ᶜaʲ_int_val,
            ᶜh_tot_int_val - ᶜK_int_val,
            buoyancy_flux_val,
            ρ_flux_h_tot_val,
            ustar_val,
            obukhov_length_val,
            sfc_local_geometry_val,
        )

        # ... and the first interior point for EDMFX ᶜq_totʲ.

        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        ᶜq_tot_int_val = Fields.field_values(Fields.level(ᶜq_tot, 1))
        ᶜq_totʲ_int_val = Fields.field_values(Fields.level(ᶜq_totʲ, 1))
        @. ᶜq_totʲ_int_val = sgs_scalar_first_interior_bc(
            ᶜz_int_val - z_sfc_val,
            ᶜρ_int_val,
            ᶜaʲ_int_val,
            ᶜq_tot_int_val,
            buoyancy_flux_val,
            ρ_flux_q_tot_val,
            ustar_val,
            obukhov_length_val,
            sfc_local_geometry_val,
        )
        if p.atmos.moisture_model isa NonEquilMoistModel && (
            p.atmos.microphysics_model isa Microphysics1Moment ||
            p.atmos.microphysics_model isa Microphysics2Moment
        )
            # TODO - any better way to define the cloud and precip tracer flux?

            ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
            ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
            ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
            ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
            ᶜq_liq_int_val = Fields.field_values(Fields.level(ᶜq_liq, 1))
            ᶜq_liqʲ_int_val = Fields.field_values(Fields.level(ᶜq_liqʲ, 1))
            @. ᶜq_liqʲ_int_val = ᶜq_liq_int_val

            ᶜq_ice_int_val = Fields.field_values(Fields.level(ᶜq_ice, 1))
            ᶜq_iceʲ_int_val = Fields.field_values(Fields.level(ᶜq_iceʲ, 1))
            @. ᶜq_iceʲ_int_val = ᶜq_ice_int_val

            ᶜq_rai_int_val = Fields.field_values(Fields.level(ᶜq_rai, 1))
            ᶜq_raiʲ_int_val = Fields.field_values(Fields.level(ᶜq_raiʲ, 1))
            @. ᶜq_raiʲ_int_val = ᶜq_rai_int_val

            ᶜq_sno_int_val = Fields.field_values(Fields.level(ᶜq_sno, 1))
            ᶜq_snoʲ_int_val = Fields.field_values(Fields.level(ᶜq_snoʲ, 1))
            @. ᶜq_snoʲ_int_val = ᶜq_sno_int_val
        end
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.microphysics_model isa Microphysics2Moment

            ᶜn_liq = @. lazy(specific(Y.c.ρn_liq, Y.c.ρ))
            ᶜn_rai = @. lazy(specific(Y.c.ρn_rai, Y.c.ρ))
            ᶜn_liqʲ = Y.c.sgsʲs.:($j).n_liq
            ᶜn_raiʲ = Y.c.sgsʲs.:($j).n_rai

            ᶜn_liq_int_val = Fields.field_values(Fields.level(ᶜn_liq, 1))
            ᶜn_liqʲ_int_val = Fields.field_values(Fields.level(ᶜn_liqʲ, 1))
            @. ᶜn_liqʲ_int_val = ᶜn_liq_int_val

            ᶜn_rai_int_val = Fields.field_values(Fields.level(ᶜn_rai, 1))
            ᶜn_raiʲ_int_val = Fields.field_values(Fields.level(ᶜn_raiʲ, 1))
            @. ᶜn_raiʲ_int_val = ᶜn_rai_int_val

        end

        # Then overwrite the prognostic variables at first inetrior point.
        ᶜΦ_int_val = Fields.field_values(Fields.level(ᶜΦ, 1))
        ᶜtsʲ_int_val = Fields.field_values(Fields.level(ᶜtsʲ, 1))
        if p.atmos.moisture_model isa NonEquilMoistModel && (
            p.atmos.microphysics_model isa Microphysics1Moment ||
            p.atmos.microphysics_model isa Microphysics2Moment
        )
            @. ᶜtsʲ_int_val = TD.PhaseNonEquil_phq(
                thermo_params,
                ᶜp_int_val,
                ᶜmseʲ_int_val - ᶜΦ_int_val,
                TD.PhasePartition(
                    ᶜq_totʲ_int_val,
                    ᶜq_liqʲ_int_val + ᶜq_raiʲ_int_val,
                    ᶜq_iceʲ_int_val + ᶜq_snoʲ_int_val,
                ),
            )
        else
            @. ᶜtsʲ_int_val = TD.PhaseEquil_phq(
                thermo_params,
                ᶜp_int_val,
                ᶜmseʲ_int_val - ᶜΦ_int_val,
                ᶜq_totʲ_int_val,
            )
        end
        sgsʲs_ρ_int_val = Fields.field_values(Fields.level(ᶜρʲs.:($j), 1))
        sgsʲs_ρa_int_val =
            Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρa, 1))

        @. sgsʲs_ρ_int_val = TD.air_density(thermo_params, ᶜtsʲ_int_val)
        @. sgsʲs_ρa_int_val =
            $(FT(turbconv_params.surface_area)) *
            TD.air_density(thermo_params, ᶜtsʲ_int_val)
    end
    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_implicit_closures!(Y, p, t)

Updates the precomputed quantities stored in `p` for edmfx implicit closures.
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_implicit_closures!(
    Y,
    p,
    t,
)

    (; moisture_model, turbconv_model) = p.atmos
    @assert !(moisture_model isa DryModel)

    (; params) = p
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶠu₃⁰, ᶠnh_pressure₃_dragʲs) = p.precomputed
    ᶠlg = Fields.local_geometry_field(Y.f)

    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    # nonhydrostatic pressure closure drag term
    for j in 1:n
        if p.atmos.edmfx_model.nh_pressure isa Val{true}
            @. ᶠnh_pressure₃_dragʲs.:($$j) = ᶠupdraft_nh_pressure_drag(
                params,
                ᶠlg,
                Y.f.sgsʲs.:($$j).u₃,
                ᶠu₃⁰,
                scale_height,
            )
        else
            @. ᶠnh_pressure₃_dragʲs.:($$j) = C3(0)
        end
    end

    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_explicit_closures!(Y, p, t)

Updates the precomputed quantities stored in `p` for edmfx explicit closures.
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_explicit_closures!(
    Y,
    p,
    t,
)

    (; moisture_model, turbconv_model) = p.atmos
    @assert !(moisture_model isa DryModel)

    (; params) = p
    (; dt) = p
    (; ᶠgradᵥ_ᶜΦ, ᶜgradᵥ_ᶠΦ) = p.core
    thermo_params = CAP.thermodynamics_params(params)
    turbconv_params = CAP.turbconv_params(params)

    FT = eltype(params)
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜu, ᶜp, ᶠu³⁰, ᶜts⁰) = p.precomputed
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm, ρatke_flux) = p.precomputed
    (;
        ᶜuʲs,
        ᶜtsʲs,
        ᶠu³ʲs,
        ᶜρʲs,
        ᶜentrʲs,
        ᶜdetrʲs,
        ᶜturb_entrʲs,
        ᶠnh_pressure₃_buoyʲs,
    ) = p.precomputed
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions

    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))

    ᶜvert_div = p.scratch.ᶜtemp_scalar
    ᶜmassflux_vert_div = p.scratch.ᶜtemp_scalar_2
    ᶜw_vert_div = p.scratch.ᶜtemp_scalar_3
    for j in 1:n
        # entrainment/detrainment
        @. ᶜentrʲs.:($$j) = entrainment(
            thermo_params,
            turbconv_params,
            ᶜz,
            z_sfc,
            ᶜp,
            Y.c.ρ,
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
            get_physical_w(ᶜuʲs.:($$j), ᶜlg),
            TD.relative_humidity(thermo_params, ᶜtsʲs.:($$j)),
            vertical_buoyancy_acceleration(Y.c.ρ, ᶜρʲs.:($$j), ᶜgradᵥ_ᶠΦ, ᶜlg),
            get_physical_w(ᶜu, ᶜlg),
            TD.relative_humidity(thermo_params, ᶜts⁰),
            FT(0),
            max(ᶜtke⁰, 0),
            p.atmos.edmfx_model.entr_model,
        )

        @. ᶜentrʲs.:($$j) = limit_entrainment(
            ᶜentrʲs.:($$j),
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
            dt,
        )

        @. ᶜturb_entrʲs.:($$j) = turbulent_entrainment(
            turbconv_params,
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
        )

        @. ᶜturb_entrʲs.:($$j) =
            limit_turb_entrainment(ᶜentrʲs.:($$j), ᶜturb_entrʲs.:($$j), dt)

        @. ᶜvert_div = ᶜdivᵥ(ᶠinterp(ᶜρʲs.:($$j)) * ᶠu³ʲs.:($$j)) / ᶜρʲs.:($$j)
        @. ᶜmassflux_vert_div =
            ᶜdivᵥ(ᶠinterp(Y.c.sgsʲs.:($$j).ρa) * ᶠu³ʲs.:($$j))
        @. ᶜw_vert_div = ᶜdivᵥ(ᶠu³ʲs.:($$j))
        @. ᶜdetrʲs.:($$j) = detrainment(
            thermo_params,
            turbconv_params,
            ᶜz,
            z_sfc,
            ᶜp,
            Y.c.ρ,
            Y.c.sgsʲs.:($$j).ρa,
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
            get_physical_w(ᶜuʲs.:($$j), ᶜlg),
            TD.relative_humidity(thermo_params, ᶜtsʲs.:($$j)),
            vertical_buoyancy_acceleration(Y.c.ρ, ᶜρʲs.:($$j), ᶜgradᵥ_ᶠΦ, ᶜlg),
            get_physical_w(ᶜu, ᶜlg),
            TD.relative_humidity(thermo_params, ᶜts⁰),
            FT(0),
            ᶜentrʲs.:($$j),
            ᶜvert_div,
            ᶜmassflux_vert_div,
            ᶜw_vert_div,
            ᶜtke⁰,
            p.atmos.edmfx_model.detr_model,
        )

        @. ᶜdetrʲs.:($$j) = limit_detrainment(
            ᶜdetrʲs.:($$j),
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
            dt,
        )

        # The buoyancy term in the nonhydrostatic pressure closure is always applied
        # for prognostic edmf. The tendency is combined with the buoyancy term in the
        # updraft momentum equation in `edmfx_sgs_vertical_advection_tendency!`. This
        # term is still calculated here as it is used explicitly in the TKE equation.
        @. ᶠnh_pressure₃_buoyʲs.:($$j) = ᶠupdraft_nh_pressure_buoyancy(
            params,
            buoyancy(ᶠinterp(Y.c.ρ), ᶠinterp(ᶜρʲs.:($$j)), ᶠgradᵥ_ᶜΦ),
        )
    end

    (; ᶜgradᵥ_θ_virt⁰, ᶜgradᵥ_q_tot⁰, ᶜgradᵥ_θ_liq_ice⁰) = p.precomputed
    # First order approximation: Use environmental mean fields.
    @. ᶜgradᵥ_θ_virt⁰ = ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ᶜts⁰)))       # ∂θv∂z_unsat
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    @. ᶜgradᵥ_q_tot⁰ = ᶜgradᵥ(ᶠinterp(ᶜq_tot⁰))                                        # ∂qt∂z_sat
    @. ᶜgradᵥ_θ_liq_ice⁰ =
        ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts⁰)))                    # ∂θl∂z_sat
    @. ᶜlinear_buoygrad = buoyancy_gradients( # TODO - do we need to modify buoyancy gradients based on NonEq + 1M tracers?
        BuoyGradMean(),
        thermo_params,
        moisture_model,
        ᶜts⁰,
        C3,
        ᶜgradᵥ_θ_virt⁰,
        ᶜgradᵥ_q_tot⁰,
        ᶜgradᵥ_θ_liq_ice⁰,
        ᶜlg,
    )

    # TODO: Make strain_rate_norm calculation a function in eddy_diffusion_closures
    # TODO: Currently the shear production only includes vertical gradients
    ᶠu⁰ = p.scratch.ᶠtemp_C123
    @. ᶠu⁰ = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³⁰)
    ᶜstrain_rate = p.scratch.ᶜtemp_UVWxUVW
    ᶜstrain_rate .= compute_strain_rate_center(ᶠu⁰)
    @. ᶜstrain_rate_norm = norm_sqr(ᶜstrain_rate)

    ρatke_flux_values = Fields.field_values(ρatke_flux)
    ρa_sfc_values = Fields.field_values(Fields.level(ᶜρa⁰, 1)) # TODO: replace by surface value
    ustar_values = Fields.field_values(ustar)
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), half),
    )
    @. ρatke_flux_values = surface_flux_tke(
        turbconv_params,
        ρa_sfc_values,
        ustar_values,
        sfc_local_geometry_values,
    )

    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_precipitation!(Y, p, microphysics_model)

Updates the precomputed quantities stored in `p` for edmfx precipitation sources.
"""
function set_prognostic_edmf_precomputed_quantities_precipitation!(
    Y,
    p,
    ::NoPrecipitation,
)
    return nothing
end
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_precipitation!(
    Y,
    p,
    ::Microphysics0Moment,
)
    @assert !(p.atmos.moisture_model isa DryModel)

    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmp = CAP.microphysics_0m_params(params)
    (; ᶜts⁰, ᶜtsʲs, ᶜSqₜᵖʲs, ᶜSqₜᵖ⁰) = p.precomputed

    # Sources from the updrafts
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. ᶜSqₜᵖʲs.:($$j) = q_tot_0M_precipitation_sources(
            thp,
            cmp,
            dt,
            Y.c.sgsʲs.:($$j).q_tot,
            ᶜtsʲs.:($$j),
        )
    end
    # sources from the environment
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    @. ᶜSqₜᵖ⁰ = q_tot_0M_precipitation_sources(thp, cmp, dt, ᶜq_tot⁰, ᶜts⁰)
    return nothing
end
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_precipitation!(
    Y,
    p,
    ::Microphysics1Moment,
)
    @assert (p.atmos.moisture_model isa NonEquilMoistModel)

    (; params, dt) = p
    (; ᶜΦ,) = p.core
    thp = CAP.thermodynamics_params(params)
    cmp = CAP.microphysics_1m_params(params)
    cmc = CAP.microphysics_cloud_params(params)
    (; turbconv_model) = p.atmos

    (; ᶜSqₗᵖʲs, ᶜSqᵢᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs, ᶜρʲs, ᶜtsʲs) = p.precomputed
    (; ᶜSqₗᵖ⁰, ᶜSqᵢᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰, ᶜts⁰) = p.precomputed

    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜwₜʲs, ᶜwₕʲs) = p.precomputed

    # TODO - can I re-use them between js and env?
    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    FT = eltype(params)

    for j in 1:n

        # compute terminal velocity for precipitation
        @. ᶜwᵣʲs.:($$j) = CM1.terminal_velocity(
            cmp.pr,
            cmp.tv.rain,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
        )
        @. ᶜwₛʲs.:($$j) = CM1.terminal_velocity(
            cmp.ps,
            cmp.tv.snow,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_sno),
        )
        # compute sedimentation velocity for cloud condensate [m/s]
        @. ᶜwₗʲs.:($$j) = CMNe.terminal_velocity(
            cmc.liquid,
            cmc.Ch2022.rain,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_liq),
        )
        @. ᶜwᵢʲs.:($$j) = CMNe.terminal_velocity(
            cmc.ice,
            cmc.Ch2022.small_ice,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_ice),
        )
        # compute their contirbutions to energy and total water advection
        @. ᶜwₜʲs.:($$j) = ifelse(
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_tot > FT(0),
            (
                ᶜwₗʲs.:($$j) * Y.c.sgsʲs.:($$j).q_liq +
                ᶜwᵢʲs.:($$j) * Y.c.sgsʲs.:($$j).q_ice +
                ᶜwᵣʲs.:($$j) * Y.c.sgsʲs.:($$j).q_rai +
                ᶜwₛʲs.:($$j) * Y.c.sgsʲs.:($$j).q_sno
            ) / Y.c.sgsʲs.:($$j).q_tot,
            FT(0),
        )
        @. ᶜwₕʲs.:($$j) = ifelse(
            Y.c.sgsʲs.:($$j).ρa * abs(Y.c.sgsʲs.:($$j).mse) > FT(0),
            (
                ᶜwₗʲs.:($$j) *
                Y.c.sgsʲs.:($$j).q_liq *
                (Iₗ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwᵢʲs.:($$j) *
                Y.c.sgsʲs.:($$j).q_ice *
                (Iᵢ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwᵣʲs.:($$j) *
                Y.c.sgsʲs.:($$j).q_rai *
                (Iₗ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwₛʲs.:($$j) *
                Y.c.sgsʲs.:($$j).q_sno *
                (Iᵢ(thp, ᶜtsʲs.:($$j)) + ᶜΦ)
            ) / (Y.c.sgsʲs.:($$j).mse),
            FT(0),
        )

        # Precipitation sources and sinks from the updrafts
        compute_precipitation_sources!(
            ᶜSᵖ,
            ᶜSᵖ_snow,
            ᶜSqₗᵖʲs.:($j),
            ᶜSqᵢᵖʲs.:($j),
            ᶜSqᵣᵖʲs.:($j),
            ᶜSqₛᵖʲs.:($j),
            ᶜρʲs.:($j),
            Y.c.sgsʲs.:($j).q_tot,
            Y.c.sgsʲs.:($j).q_liq,
            Y.c.sgsʲs.:($j).q_ice,
            Y.c.sgsʲs.:($j).q_rai,
            Y.c.sgsʲs.:($j).q_sno,
            ᶜtsʲs.:($j),
            dt,
            cmp,
            thp,
        )
        compute_precipitation_sinks!(
            ᶜSᵖ,
            ᶜSqᵣᵖʲs.:($j),
            ᶜSqₛᵖʲs.:($j),
            ᶜρʲs.:($j),
            Y.c.sgsʲs.:($j).q_tot,
            Y.c.sgsʲs.:($j).q_liq,
            Y.c.sgsʲs.:($j).q_ice,
            Y.c.sgsʲs.:($j).q_rai,
            Y.c.sgsʲs.:($j).q_sno,
            ᶜtsʲs.:($j),
            dt,
            cmp,
            thp,
        )
        # Cloud formation from the updrafts
        @. ᶜSqₗᵖʲs.:($$j) += cloud_sources(
            cmc.liquid,
            thp,
            Y.c.sgsʲs.:($$j).q_tot,
            Y.c.sgsʲs.:($$j).q_liq,
            Y.c.sgsʲs.:($$j).q_ice,
            Y.c.sgsʲs.:($$j).q_rai,
            Y.c.sgsʲs.:($$j).q_sno,
            ᶜρʲs.:($$j),
            TD.air_temperature(thp, ᶜtsʲs.:($$j)),
            dt,
        )
        @. ᶜSqᵢᵖʲs.:($$j) += cloud_sources(
            cmc.ice,
            thp,
            Y.c.sgsʲs.:($$j).q_tot,
            Y.c.sgsʲs.:($$j).q_liq,
            Y.c.sgsʲs.:($$j).q_ice,
            Y.c.sgsʲs.:($$j).q_rai,
            Y.c.sgsʲs.:($$j).q_sno,
            ᶜρʲs.:($$j),
            TD.air_temperature(thp, ᶜtsʲs.:($$j)),
            dt,
        )
    end

    # Precipitation sources and sinks from the environment
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
    ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜts⁰))
    compute_precipitation_sources!(
        ᶜSᵖ,
        ᶜSᵖ_snow,
        ᶜSqₗᵖ⁰,
        ᶜSqᵢᵖ⁰,
        ᶜSqᵣᵖ⁰,
        ᶜSqₛᵖ⁰,
        ᶜρ⁰,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜq_rai⁰,
        ᶜq_sno⁰,
        ᶜts⁰,
        dt,
        cmp,
        thp,
    )
    compute_precipitation_sinks!(
        ᶜSᵖ,
        ᶜSqᵣᵖ⁰,
        ᶜSqₛᵖ⁰,
        ᶜρ⁰,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜq_rai⁰,
        ᶜq_sno⁰,
        ᶜts⁰,
        dt,
        cmp,
        thp,
    )
    # Cloud formation from the environment
    @. ᶜSqₗᵖ⁰ += cloud_sources(
        cmc.liquid,
        thp,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜq_rai⁰,
        ᶜq_sno⁰,
        ᶜρ⁰,
        TD.air_temperature(thp, ᶜts⁰),
        dt,
    )
    @. ᶜSqᵢᵖ⁰ += cloud_sources(
        cmc.ice,
        thp,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜq_rai⁰,
        ᶜq_sno⁰,
        ᶜρ⁰,
        TD.air_temperature(thp, ᶜts⁰),
        dt,
    )
    return nothing
end
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_precipitation!(
    Y,
    p,
    ::Microphysics2Moment,
)

    (; params, dt) = p
    (; ᶜΦ,) = p.core
    thp = CAP.thermodynamics_params(params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    cmc = CAP.microphysics_cloud_params(params)

    (;
        ᶜSqₗᵖʲs,
        ᶜSqᵢᵖʲs,
        ᶜSqᵣᵖʲs,
        ᶜSqₛᵖʲs,
        ᶜSnₗᵖʲs,
        ᶜSnᵣᵖʲs,
        ᶜρʲs,
        ᶜtsʲs,
        ᶜuʲs,
    ) = p.precomputed
    (; ᶜSqₗᵖ⁰, ᶜSqᵢᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰, ᶜSnₗᵖ⁰, ᶜSnᵣᵖ⁰, ᶜts⁰, ᶜu⁰) =
        p.precomputed
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜwₙₗʲs, ᶜwₙᵣʲs, ᶜwₜʲs, ᶜwₕʲs, ᶜuʲs) =
        p.precomputed

    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜS₂ᵖ = p.scratch.ᶜtemp_scalar_2

    # Get prescribed aerosol concentrations
    seasalt_num = p.scratch.ᶜtemp_scalar_3
    seasalt_mean_radius = p.scratch.ᶜtemp_scalar_4
    sulfate_num = p.scratch.ᶜtemp_scalar_5
    if (:tracers in propertynames(p)) &&
       (:prescribed_aerosols_field in propertynames(p.tracers))
        compute_prescribed_aerosol_properties!(
            seasalt_num,
            seasalt_mean_radius,
            sulfate_num,
            p.tracers.prescribed_aerosols_field,
            cm2p.aerosol,
        )
    else
        @. seasalt_num = 0
        @. seasalt_mean_radius = 0
        @. sulfate_num = 0
    end

    # Compute sources
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    FT = eltype(params)
    for j in 1:n

        # compute terminal velocity for precipitation
        # TODO sedimentation of snow is based on the 1M scheme
        @. ᶜwₙᵣʲs.:($$j) = getindex(
            CM2.rain_terminal_velocity(
                cm2p.sb,
                cm2p.rtv,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_rai),
            ),
            1,
        )
        @. ᶜwᵣʲs.:($$j) = getindex(
            CM2.rain_terminal_velocity(
                cm2p.sb,
                cm2p.rtv,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_rai),
            ),
            2,
        )
        @. ᶜwₛʲs.:($$j) = CM1.terminal_velocity(
            cm1p.ps,
            cm1p.tv.snow,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_sno),
        )
        # compute sedimentation velocity for cloud condensate [m/s]
        # TODO sedimentation of ice is based on the 1M scheme
        @. ᶜwₙₗʲs.:($$j) = getindex(
            CM2.cloud_terminal_velocity(
                cm2p.sb.pdf_c,
                cm2p.ctv,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_liq),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_liq),
            ),
            1,
        )
        @. ᶜwₗʲs.:($$j) = getindex(
            CM2.cloud_terminal_velocity(
                cm2p.sb.pdf_c,
                cm2p.ctv,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_liq),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_liq),
            ),
            2,
        )
        @. ᶜwᵢʲs.:($$j) = CMNe.terminal_velocity(
            cmc.ice,
            cmc.Ch2022.small_ice,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_ice),
        )
        # compute their contirbutions to energy and total water advection
        @. ᶜwₜʲs.:($$j) = ifelse(
            Y.c.sgsʲs.:($$j).ρa * Y.c.sgsʲs.:($$j).q_tot > FT(0),
            (
                ᶜwₗʲs.:($$j) * Y.c.sgsʲs.:($$j).q_liq +
                ᶜwᵢʲs.:($$j) * Y.c.sgsʲs.:($$j).q_ice +
                ᶜwᵣʲs.:($$j) * Y.c.sgsʲs.:($$j).q_rai +
                ᶜwₛʲs.:($$j) * Y.c.sgsʲs.:($$j).q_sno
            ) / Y.c.sgsʲs.:($$j).q_tot,
            FT(0),
        )
        @. ᶜwₕʲs.:($$j) = ifelse(
            Y.c.sgsʲs.:($$j).ρa * abs(Y.c.sgsʲs.:($$j).mse) > FT(0),
            (
                ᶜwₗʲs.:($$j) *
                Y.c.sgsʲs.:($$j).q_liq *
                (Iₗ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwᵢʲs.:($$j) *
                Y.c.sgsʲs.:($$j).q_ice *
                (Iᵢ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwᵣʲs.:($$j) *
                Y.c.sgsʲs.:($$j).q_rai *
                (Iₗ(thp, ᶜtsʲs.:($$j)) + ᶜΦ) +
                ᶜwₛʲs.:($$j) *
                Y.c.sgsʲs.:($$j).q_sno *
                (Iᵢ(thp, ᶜtsʲs.:($$j)) + ᶜΦ)
            ) / (Y.c.sgsʲs.:($$j).mse),
            FT(0),
        )

        # Precipitation sources and sinks from the updrafts
        compute_warm_precipitation_sources_2M!(
            ᶜSᵖ,
            ᶜS₂ᵖ,
            ᶜSnₗᵖʲs.:($j),
            ᶜSnᵣᵖʲs.:($j),
            ᶜSqₗᵖʲs.:($j),
            ᶜSqᵣᵖʲs.:($j),
            ᶜρʲs.:($j),
            Y.c.sgsʲs.:($j).n_liq,
            Y.c.sgsʲs.:($j).n_rai,
            Y.c.sgsʲs.:($j).q_tot,
            Y.c.sgsʲs.:($j).q_liq,
            Y.c.sgsʲs.:($j).q_ice,
            Y.c.sgsʲs.:($j).q_rai,
            Y.c.sgsʲs.:($j).q_sno,
            ᶜtsʲs.:($j),
            dt,
            cm2p,
            thp,
        )
        @. ᶜSqᵢᵖʲs.:($$j) = 0
        @. ᶜSqₛᵖʲs.:($$j) = 0
        # Cloud formation from the updrafts
        @. ᶜSqₗᵖʲs.:($$j) += cloud_sources(
            cmc.liquid,
            thp,
            Y.c.sgsʲs.:($$j).q_tot,
            Y.c.sgsʲs.:($$j).q_liq,
            Y.c.sgsʲs.:($$j).q_ice,
            Y.c.sgsʲs.:($$j).q_rai,
            Y.c.sgsʲs.:($$j).q_sno,
            ᶜρʲs.:($$j),
            TD.air_temperature(thp, ᶜtsʲs.:($$j)),
            dt,
        )
        @. ᶜSqᵢᵖʲs.:($$j) += cloud_sources(
            cmc.ice,
            thp,
            Y.c.sgsʲs.:($$j).q_tot,
            Y.c.sgsʲs.:($$j).q_liq,
            Y.c.sgsʲs.:($$j).q_ice,
            Y.c.sgsʲs.:($$j).q_rai,
            Y.c.sgsʲs.:($$j).q_sno,
            ᶜρʲs.:($$j),
            TD.air_temperature(thp, ᶜtsʲs.:($$j)),
            dt,
        )
        @. ᶜSnₗᵖʲs += aerosol_activation_sources(
            seasalt_num,
            seasalt_mean_radius,
            sulfate_num,
            Y.c.sgsʲs.:($$j).q_tot,
            Y.c.sgsʲs.:($$j).q_liq + Y.c.sgsʲs.:($$j).q_rai,
            Y.c.sgsʲs.:($$j).q_ice + Y.c.sgsʲs.:($$j).q_sno,
            Y.c.sgsʲs.:($$j).n_liq + Y.c.sgsʲs.:($$j).n_rai,
            ᶜρʲs.:($$j),
            max(0, w_component.(Geometry.WVector.(ᶜuʲs.:($$j)))),
            (cm2p,),
            thp,
            ᶜtsʲs.:($$j),
            dt,
        )
    end

    # Precipitation sources and sinks from the environment
    ᶜn_liq⁰ = ᶜspecific_env_value(@name(n_liq), Y, p)
    ᶜn_rai⁰ = ᶜspecific_env_value(@name(n_rai), Y, p)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
    ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜts⁰))
    compute_warm_precipitation_sources_2M!(
        ᶜSᵖ,
        ᶜS₂ᵖ,
        ᶜSnₗᵖ⁰,
        ᶜSnᵣᵖ⁰,
        ᶜSqₗᵖ⁰,
        ᶜSqᵣᵖ⁰,
        ᶜρ⁰,
        ᶜn_liq⁰,
        ᶜn_rai⁰,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜq_rai⁰,
        ᶜq_sno⁰,
        ᶜts⁰,
        dt,
        cm2p,
        thp,
    )
    @. ᶜSqᵢᵖ⁰ = 0
    @. ᶜSqₛᵖ⁰ = 0
    # Cloud formation from the environment
    @. ᶜSqₗᵖ⁰ += cloud_sources(
        cmc.liquid,
        thp,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜq_rai⁰,
        ᶜq_sno⁰,
        ᶜρ⁰,
        TD.air_temperature(thp, ᶜts⁰),
        dt,
    )
    @. ᶜSqᵢᵖ⁰ += cloud_sources(
        cmc.ice,
        thp,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜq_rai⁰,
        ᶜq_sno⁰,
        ᶜρ⁰,
        TD.air_temperature(thp, ᶜts⁰),
        dt,
    )
    @. ᶜSnₗᵖ⁰ += aerosol_activation_sources(
        seasalt_num,
        seasalt_mean_radius,
        sulfate_num,
        ᶜq_tot⁰,
        ᶜq_liq⁰ + ᶜq_rai⁰,
        ᶜq_ice⁰ + ᶜq_sno⁰,
        ᶜn_liq⁰ + ᶜn_rai⁰,
        ᶜρ⁰,
        w_component.(Geometry.WVector.(ᶜu⁰)),
        (cm2p,),
        thp,
        ᶜts⁰,
        dt,
    )
    return nothing
end
