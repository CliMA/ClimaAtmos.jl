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

    thermo_params = CAP.thermodynamics_params(p.params)
    (; turbconv_model) = p.atmos
    (; ᶜΦ,) = p.core
    (; ᶜp, ᶜK) = p.precomputed
    (; ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜts⁰) = p.precomputed

    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Y, turbconv_model)
    set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Y.c.uₕ, ᶠuₕ³)
    # @. ᶜK⁰ += ᶜtke
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
    (; moisture_model, turbconv_model, microphysics_model) = p.atmos

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
        if moisture_model isa NonEquilMoistModel && (
            microphysics_model isa Microphysics1Moment ||
            microphysics_model isa Microphysics2Moment
        )
            ᶜq_liqʲ = Y.c.sgsʲs.:($j).q_liq
            ᶜq_iceʲ = Y.c.sgsʲs.:($j).q_ice
            ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
            ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
        end

        set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³)
        @. ᶠKᵥʲ = (adjoint(CT3(ᶠu₃ʲ)) * ᶠu₃ʲ) / 2
        if moisture_model isa NonEquilMoistModel && (
            microphysics_model isa Microphysics1Moment ||
            microphysics_model isa Microphysics2Moment
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
    set_prognostic_edmf_precomputed_quantities_implicit_closures!(Y, p, t)

Updates the precomputed quantities stored in `p` for edmfx implicit closures.
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_implicit_closures!(
    Y,
    p,
    t,
)

    (; turbconv_model) = p.atmos

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

    (; turbconv_model) = p.atmos

    (; params) = p
    (; dt) = p
    (; ᶠgradᵥ_ᶜΦ, ᶜgradᵥ_ᶠΦ) = p.core
    thermo_params = CAP.thermodynamics_params(params)
    turbconv_params = CAP.turbconv_params(params)

    FT = eltype(params)
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜu, ᶜp, ᶠu³, ᶜts, ᶠu³⁰, ᶜts⁰) = p.precomputed
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm, ρtke_flux) = p.precomputed
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
    ᶜaʲ_int_val = p.scratch.temp_data_level

    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜdz = Fields.Δz_field(axes(Y.c))
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))

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
            max(ᶜtke, 0),
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
            ᶜtke,
            p.atmos.edmfx_model.detr_model,
        )

        @. ᶜdetrʲs.:($$j) = limit_detrainment(
            ᶜdetrʲs.:($$j),
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
            dt,
        )

        # If the surface buoyancy flux is positive, increase entrainment in the first cell
        # so that the updraft area grows to at least `surface_area` within one timestep.
        # Otherwise (stable surface), leave entrainment unchanged.
        buoyancy_flux_val = Fields.field_values(p.precomputed.sfc_conditions.buoyancy_flux)
        sgsʲs_ρ_int_val = Fields.field_values(Fields.level(ᶜρʲs.:($j), 1))
        sgsʲs_ρa_int_val =
            Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρa, 1))
        @. ᶜaʲ_int_val = draft_area(sgsʲs_ρa_int_val, sgsʲs_ρ_int_val)
        entr_int_val = Fields.field_values(Fields.level(ᶜentrʲs.:($j), 1))
        detr_int_val = Fields.field_values(Fields.level(ᶜdetrʲs.:($j), 1))
        @. entr_int_val = limit_entrainment(
            ifelse(
                buoyancy_flux_val < 0 || ᶜaʲ_int_val >= $(FT(turbconv_params.surface_area)),
                entr_int_val,
                detr_int_val +
                ($(FT(turbconv_params.surface_area)) / ᶜaʲ_int_val - 1) / dt,
            ),
            ᶜaʲ_int_val,
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

    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice, cloud_diagnostics_tuple) =
        p.precomputed
    @. ᶜlinear_buoygrad = buoyancy_gradients( # TODO - do we need to modify buoyancy gradients based on NonEq + 1M tracers?
        BuoyGradMean(),
        thermo_params,
        ᶜts,
        cloud_diagnostics_tuple.cf,
        C3,
        ᶜgradᵥ_q_tot,
        ᶜgradᵥ_θ_liq_ice,
        ᶜlg,
    )

    # TODO: Make strain_rate_norm calculation a function in eddy_diffusion_closures
    # TODO: Currently the shear production only includes vertical gradients
    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ᶜstrain_rate = compute_strain_rate_center_vertical(ᶠu)
    @. ᶜstrain_rate_norm = norm_sqr(ᶜstrain_rate)

    ρtke_flux_values = Fields.field_values(ρtke_flux)
    ρa_sfc_values = Fields.field_values(Fields.level(ᶜρa⁰, 1)) # TODO: replace by surface value
    ustar_values = Fields.field_values(ustar)
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), half),
    )
    @. ρtke_flux_values = surface_flux_tke(
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

    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmp = CAP.microphysics_1m_params(params)
    cmc = CAP.microphysics_cloud_params(params)

    (; ᶜSqₗᵖʲs, ᶜSqᵢᵖʲs, ᶜSqᵣᵖʲs, ᶜSqₛᵖʲs, ᶜρʲs, ᶜtsʲs) = p.precomputed
    (; ᶜSqₗᵖ⁰, ᶜSqᵢᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰, ᶜts⁰) = p.precomputed

    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs) = p.precomputed

    # TODO - can I re-use them between js and env?
    ᶜSᵖ = p.scratch.ᶜtemp_scalar
    ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

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
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜwₙₗʲs, ᶜwₙᵣʲs, ᶜuʲs) =
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
