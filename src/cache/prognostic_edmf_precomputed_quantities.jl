#####
##### Precomputed quantities
#####
import NVTX
import Thermodynamics as TD
import ClimaCore: Spaces, Fields

"""
    set_prognostic_edmf_precomputed_quantities!(Y, p, ᶠuₕ³, t)

Updates the edmf environment precomputed quantities stored in `p` for edmfx.
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
    (; ᶜp, ᶜh_tot, ᶜK) = p.precomputed
    (; ᶜtke⁰, ᶜρa⁰, ᶠu₃⁰, ᶜK⁰, ᶜts⁰, ᶜρ⁰, ᶜmse⁰, ᶜq_tot⁰) =
        p.precomputed
    if p.atmos.moisture_model isa NonEquilMoistModel &&
       p.atmos.precip_model isa Microphysics1Moment
        (; ᶜq_liq⁰, ᶜq_ice⁰, ᶜq_rai⁰, ᶜq_sno⁰) = p.precomputed
    end

    @. ᶜρa⁰ = ρa⁰(Y.c)
    @. ᶜtke⁰ = divide_by_ρa(Y.c.sgs⁰.ρatke, ᶜρa⁰, 0, Y.c.ρ, turbconv_model)
    @. ᶜmse⁰ = divide_by_ρa(
        Y.c.ρ * (ᶜh_tot - ᶜK) - ρamse⁺(Y.c.sgsʲs),
        ᶜρa⁰,
        Y.c.ρ * (ᶜh_tot - ᶜK),
        Y.c.ρ,
        turbconv_model,
    )
    @. ᶜq_tot⁰ = divide_by_ρa(
        Y.c.ρq_tot - ρaq_tot⁺(Y.c.sgsʲs),
        ᶜρa⁰,
        Y.c.ρq_tot,
        Y.c.ρ,
        turbconv_model,
    )
    if p.atmos.moisture_model isa NonEquilMoistModel &&
       p.atmos.precip_model isa Microphysics1Moment
        @. ᶜq_liq⁰ = divide_by_ρa(
            Y.c.ρq_liq - ρaq_liq⁺(Y.c.sgsʲs),
            ᶜρa⁰,
            Y.c.ρq_liq,
            Y.c.ρ,
            turbconv_model,
        )
        @. ᶜq_ice⁰ = divide_by_ρa(
            Y.c.ρq_ice - ρaq_ice⁺(Y.c.sgsʲs),
            ᶜρa⁰,
            Y.c.ρq_ice,
            Y.c.ρ,
            turbconv_model,
        )
        @. ᶜq_rai⁰ = divide_by_ρa(
            Y.c.ρq_rai - ρaq_rai⁺(Y.c.sgsʲs),
            ᶜρa⁰,
            Y.c.ρq_rai,
            Y.c.ρ,
            turbconv_model,
        )
        @. ᶜq_sno⁰ = divide_by_ρa(
            Y.c.ρq_sno - ρaq_sno⁺(Y.c.sgsʲs),
            ᶜρa⁰,
            Y.c.ρq_sno,
            Y.c.ρ,
            turbconv_model,
        )
    end
    
    ᶠu³⁰ = ᶠu³_lazy(Y.c.uₕ, Y.c.ρ, ᶠu₃⁰)
    set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Y, turbconv_model)

    set_velocity_quantities!(ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Y.c.uₕ, ᶠuₕ³, Y.c.ρ)

    # @. ᶜK⁰ += ᶜtke⁰
    if p.atmos.moisture_model isa NonEquilMoistModel &&
       p.atmos.precip_model isa Microphysics1Moment
        @. ᶜts⁰ = TD.PhaseNonEquil_phq(
            thermo_params,
            ᶜp,
            ᶜmse⁰ - ᶜΦ,
            TD.PhasePartition(ᶜq_tot⁰, ᶜq_liq⁰ + ᶜq_rai⁰, ᶜq_ice⁰ + ᶜq_sno⁰),
        )
    else
        @. ᶜts⁰ = TD.PhaseEquil_phq(thermo_params, ᶜp, ᶜmse⁰ - ᶜΦ, ᶜq_tot⁰)
    end
    @. ᶜρ⁰ = TD.air_density(thermo_params, ᶜts⁰)
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
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.precip_model isa Microphysics1Moment
            ᶜq_liqʲ = Y.c.sgsʲs.:($j).q_liq
            ᶜq_iceʲ = Y.c.sgsʲs.:($j).q_ice
            ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
            ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
        end

        set_velocity_quantities!(ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³, ᶜρʲ)

        @. ᶠKᵥʲ = (adjoint(CT3(ᶠu₃ʲ)) * ᶠu₃ʲ) / 2
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.precip_model isa Microphysics1Moment
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
    (; ᶜspecific, ᶜp, ᶜh_tot, ᶜK, ᶜtsʲs, ᶜρʲs) = p.precomputed
    (; ustar, obukhov_length, buoyancy_flux) = p.precomputed.sfc_conditions

    for j in 1:n
        ᶜtsʲ = ᶜtsʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.precip_model isa Microphysics1Moment
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
        ᶜq_tot_int_val = Fields.field_values(Fields.level(ᶜspecific.q_tot, 1))
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
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.precip_model isa Microphysics1Moment
            # TODO - any better way to define the cloud and precip tracer flux?
            ᶜq_liq_int_val =
                Fields.field_values(Fields.level(ᶜspecific.q_liq, 1))
            ᶜq_liqʲ_int_val = Fields.field_values(Fields.level(ᶜq_liqʲ, 1))
            @. ᶜq_liqʲ_int_val = ᶜq_liq_int_val

            ᶜq_ice_int_val =
                Fields.field_values(Fields.level(ᶜspecific.q_ice, 1))
            ᶜq_iceʲ_int_val = Fields.field_values(Fields.level(ᶜq_iceʲ, 1))
            @. ᶜq_iceʲ_int_val = ᶜq_ice_int_val

            ᶜq_rai_int_val =
                Fields.field_values(Fields.level(ᶜspecific.q_rai, 1))
            ᶜq_raiʲ_int_val = Fields.field_values(Fields.level(ᶜq_raiʲ, 1))
            @. ᶜq_raiʲ_int_val = ᶜq_rai_int_val

            ᶜq_sno_int_val =
                Fields.field_values(Fields.level(ᶜspecific.q_sno, 1))
            ᶜq_snoʲ_int_val = Fields.field_values(Fields.level(ᶜq_snoʲ, 1))
            @. ᶜq_snoʲ_int_val = ᶜq_sno_int_val
        end

        # Then overwrite the prognostic variables at first inetrior point.
        ᶜΦ_int_val = Fields.field_values(Fields.level(ᶜΦ, 1))
        ᶜtsʲ_int_val = Fields.field_values(Fields.level(ᶜtsʲ, 1))
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.precip_model isa Microphysics1Moment
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

    (; ᶜtke⁰, ᶜp, ᶜρa⁰, ᶜts⁰, ᶜq_tot⁰) = p.precomputed
    (;
        ᶜmixing_length_tuple,
        ᶜmixing_length,
        ᶜlinear_buoygrad,
        ᶜstrain_rate_norm,
        ᶜK_u,
        ᶜK_h,
        ρatke_flux,
    ) = p.precomputed
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

    ᶜvert_div = p.scratch.ᶜtemp_scalar
    ᶜmassflux_vert_div = p.scratch.ᶜtemp_scalar_2
    ᶜw_vert_div = p.scratch.ᶜtemp_scalar_3

    ᶜu = Base.materialize(ᶜu_lazy(Y.c.uₕ, Y.f.u₃))

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
    ᶠu³⁰ = ᶠu³_lazy(Y.c.uₕ, Y.c.ρ, Y.f.u₃)
    @. ᶠu⁰ = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³⁰)
    ᶜstrain_rate = p.scratch.ᶜtemp_UVWxUVW
    ᶜstrain_rate .= compute_strain_rate_center(ᶠu⁰)
    @. ᶜstrain_rate_norm = norm_sqr(ᶜstrain_rate)

    ᶜprandtl_nvec = p.scratch.ᶜtemp_scalar
    @. ᶜprandtl_nvec =
        turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm)

    ᶜtke_exch = p.scratch.ᶜtemp_scalar_2
    @. ᶜtke_exch = 0
    for j in 1:n
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        @. ᶜtke_exch +=
            Y.c.sgsʲs.:($$j).ρa * ᶜdetrʲs.:($$j) / ᶜρa⁰ *
            (1 / 2 * norm_sqr(ᶜinterp(ᶠu³⁰) - ᶜinterp(ᶠu³ʲs.:($$j))) - ᶜtke⁰)
    end

    sfc_tke = Fields.level(ᶜtke⁰, 1)
    @. ᶜmixing_length_tuple = mixing_length(
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
    )

    @. ᶜmixing_length = ᶜmixing_length_tuple.master

    @. ᶜK_u = eddy_viscosity(turbconv_params, ᶜtke⁰, ᶜmixing_length)
    @. ᶜK_h = eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec)

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
    set_prognostic_edmf_precomputed_quantities_precipitation!(Y, p, precip_model)

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
    (; ᶜts⁰, ᶜq_tot⁰, ᶜtsʲs, ᶜSqₜᵖʲs, ᶜSqₜᵖ⁰) = p.precomputed

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

    (; ᶜSqₗᵖʲs, ᶜSqᵢᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs, ᶜρʲs, ᶜtsʲs) = p.precomputed
    (; ᶜSqₗᵖ⁰, ᶜSqᵢᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰, ᶜρ⁰, ᶜts⁰) = p.precomputed
    (; ᶜq_tot⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜq_rai⁰, ᶜq_sno⁰) = p.precomputed

    # TODO - can I re-use them between js and env?
    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    for j in 1:n
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
