#####
##### Precomputed quantities
#####
import NVTX
import Thermodynamics as TD
import ClimaCore: Spaces, Fields

# Helper function for thermodynamic state from saturation adjustment.
# Returns a NamedTuple with T, q_liq, q_ice.
# Uses TD.ph() dispatch
function saturation_adjustment_tuple(thermo_params, ::TD.ph, p, h, q_tot)
    FT = eltype(thermo_params)
    sa_result = TD.saturation_adjustment(
        thermo_params,
        TD.ph(),
        p,
        h,
        q_tot;
        maxiter = 4,
        tol = FT(0),
    )
    return (; T = sa_result.T, q_liq = sa_result.q_liq, q_ice = sa_result.q_ice)
end

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
    (; ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed

    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Y, turbconv_model)
    set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Y.c.uₕ, ᶠuₕ³)
    # @. ᶜK⁰ += ᶜtke
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)

    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)

    if p.atmos.moisture_model isa NonEquilMoistModel && (
        p.atmos.microphysics_model isa
        Union{Microphysics1Moment, QuadratureMicrophysics{Microphysics1Moment}} ||
        p.atmos.microphysics_model isa
        Union{Microphysics2Moment, QuadratureMicrophysics{Microphysics2Moment}}
    )
        ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
        ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
        ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
        ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
        # Compute env thermodynamic state from primitives
        @. ᶜq_liq_rai⁰ = max(0, ᶜq_liq⁰ + ᶜq_rai⁰)
        @. ᶜq_ice_sno⁰ = max(0, ᶜq_ice⁰ + ᶜq_sno⁰)
        # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
        @. ᶜq_tot_safe⁰ = max(ᶜq_liq_rai⁰ + ᶜq_ice_sno⁰, ᶜq_tot⁰)
        ᶜh⁰ = @. lazy(ᶜmse⁰ - ᶜΦ)  # specific enthalpy
        @. ᶜT⁰ = TD.air_temperature(
            thermo_params,
            TD.ph(),
            ᶜh⁰,
            ᶜq_tot_safe⁰,
            ᶜq_liq_rai⁰,
            ᶜq_ice_sno⁰,
        )
    else
        # EquilMoistModel: use saturation adjustment to get T and phase partition
        @. ᶜq_tot_safe⁰ = max(0, ᶜq_tot⁰)
        (; ᶜsa_result) = p.precomputed
        h⁰ = @. lazy(ᶜmse⁰ - ᶜΦ)
        @. ᶜsa_result =
            saturation_adjustment_tuple(thermo_params, TD.ph(), ᶜp, h⁰, ᶜq_tot_safe⁰)
        @. ᶜT⁰ = ᶜsa_result.T
        @. ᶜq_liq_rai⁰ = ᶜsa_result.q_liq
        @. ᶜq_ice_sno⁰ = ᶜsa_result.q_ice
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
    (;
        ᶜp,
        ᶜuʲs,
        ᶠu³ʲs,
        ᶜKʲs,
        ᶠKᵥʲs,
        ᶜTʲs,
        ᶜq_tot_safeʲs,
        ᶜq_liq_raiʲs,
        ᶜq_ice_snoʲs,
        ᶜρʲs,
    ) = p.precomputed

    for j in 1:n
        ᶜuʲ = ᶜuʲs.:($j)
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        ᶜKʲ = ᶜKʲs.:($j)
        ᶠKᵥʲ = ᶠKᵥʲs.:($j)
        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜq_tot_safeʲ = ᶜq_tot_safeʲs.:($j)
        ᶜq_liq_raiʲ = ᶜq_liq_raiʲs.:($j)
        ᶜq_ice_snoʲ = ᶜq_ice_snoʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot

        set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³)
        @. ᶠKᵥʲ = (adjoint(CT3(ᶠu₃ʲ)) * ᶠu₃ʲ) / 2

        @. ᶜq_tot_safeʲ = max(0, ᶜq_totʲ)
        if moisture_model isa NonEquilMoistModel && (
            microphysics_model isa
            Union{Microphysics1Moment, QuadratureMicrophysics{Microphysics1Moment}} ||
            microphysics_model isa
            Union{Microphysics2Moment, QuadratureMicrophysics{Microphysics2Moment}}
        )
            ᶜq_liqʲ = Y.c.sgsʲs.:($j).q_liq
            ᶜq_iceʲ = Y.c.sgsʲs.:($j).q_ice
            ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
            ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
            @. ᶜq_liq_raiʲ = max(0, ᶜq_liqʲ + ᶜq_raiʲ)
            @. ᶜq_ice_snoʲ = max(0, ᶜq_iceʲ + ᶜq_snoʲ)
            # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
            @. ᶜq_tot_safeʲ = max(ᶜq_liq_raiʲ + ᶜq_ice_snoʲ, ᶜq_totʲ)
            ᶜhʲ = @. lazy(ᶜmseʲ - ᶜΦ)
            @. ᶜTʲ = TD.air_temperature(
                thermo_params,
                TD.ph(),
                ᶜhʲ,
                ᶜq_tot_safeʲ,
                ᶜq_liq_raiʲ,
                ᶜq_ice_snoʲ,
            )
        else
            # EquilMoistModel: use saturation adjustment
            (; ᶜsa_result) = p.precomputed
            @. ᶜsa_result = saturation_adjustment_tuple(
                thermo_params,
                TD.ph(),
                ᶜp,
                ᶜmseʲ - ᶜΦ,
                ᶜq_tot_safeʲ,
            )
            @. ᶜTʲ = ᶜsa_result.T
            @. ᶜq_liq_raiʲ = ᶜsa_result.q_liq
            @. ᶜq_ice_snoʲ = ᶜsa_result.q_ice
        end
        @. ᶜρʲ =
            TD.air_density(thermo_params, ᶜTʲ, ᶜp, ᶜq_tot_safeʲ, ᶜq_liq_raiʲ, ᶜq_ice_snoʲ)
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

    (; ᶜu, ᶜp, ᶠu³, ᶜT, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    (; ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm, ρtke_flux) = p.precomputed
    (;
        ᶜuʲs,
        ᶜTʲs,
        ᶜq_tot_safeʲs,
        ᶜq_liq_raiʲs,
        ᶜq_ice_snoʲs,
        ᶠu³ʲs,
        ᶜρʲs,
        ᶜentrʲs,
        ᶜdetrʲs,
        ᶜturb_entrʲs,
        ᶠρ_diffʲs,
    ) = p.precomputed
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    ᶜaʲ_int_val = p.scratch.temp_data_level

    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜlg = Fields.local_geometry_field(Y.c)
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
            TD.relative_humidity(
                thermo_params,
                ᶜTʲs.:($$j),
                ᶜp,
                ᶜq_tot_safeʲs.:($$j),
                ᶜq_liq_raiʲs.:($$j),
                ᶜq_ice_snoʲs.:($$j),
            ),
            vertical_buoyancy_acceleration(Y.c.ρ, ᶜρʲs.:($$j), ᶜgradᵥ_ᶠΦ, ᶜlg),
            get_physical_w(ᶜu, ᶜlg),
            TD.relative_humidity(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_safe⁰,
                ᶜq_liq_rai⁰,
                ᶜq_ice_sno⁰,
            ),
            FT(0),
            max(ᶜtke, 0),
            p.atmos.edmfx_model.entr_model,
        )

        @. ᶜturb_entrʲs.:($$j) = turbulent_entrainment(
            turbconv_params,
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
        )

        if p.atmos.sgs_entr_detr_mode == Explicit()
            @. ᶜentrʲs.:($$j) = limit_entrainment(
                ᶜentrʲs.:($$j),
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
                dt,
            )
            @. ᶜturb_entrʲs.:($$j) =
                limit_turb_entrainment(ᶜentrʲs.:($$j), ᶜturb_entrʲs.:($$j), dt)
        end

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
            TD.relative_humidity(
                thermo_params,
                ᶜTʲs.:($$j),
                ᶜp,
                ᶜq_tot_safeʲs.:($$j),
                ᶜq_liq_raiʲs.:($$j),
                ᶜq_ice_snoʲs.:($$j),
            ),
            vertical_buoyancy_acceleration(Y.c.ρ, ᶜρʲs.:($$j), ᶜgradᵥ_ᶠΦ, ᶜlg),
            get_physical_w(ᶜu, ᶜlg),
            TD.relative_humidity(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_safe⁰,
                ᶜq_liq_rai⁰,
                ᶜq_ice_sno⁰,
            ),
            FT(0),
            ᶜentrʲs.:($$j),
            ᶜvert_div,
            ᶜmassflux_vert_div,
            ᶜw_vert_div,
            ᶜtke,
            p.atmos.edmfx_model.detr_model,
        )

        if p.atmos.sgs_entr_detr_mode == Explicit()
            @. ᶜdetrʲs.:($$j) = limit_detrainment(
                ᶜdetrʲs.:($$j),
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
                dt,
            )
        else
            @. ᶜdetrʲs.:($$j) = limit_detrainment(
                ᶜdetrʲs.:($$j),
                ᶜentrʲs.:($$j),
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
                dt,
            )
        end

        # If the surface buoyancy flux is positive, adjust the first-cell updraft area toward `surface_area`:
        #   - if area < surface_area: increase entrainment to grow area toward `surface_area`
        #   - if area > surface_area: increase detrainment to decay area toward `surface_area`
        # If area is negative or surface buoyancy flux is non-positive (stable/neutral surface), leave
        # entrainment and detrainment unchanged.
        buoyancy_flux_val = Fields.field_values(p.precomputed.sfc_conditions.buoyancy_flux)
        sgsʲs_ρ_int_val = Fields.field_values(Fields.level(ᶜρʲs.:($j), 1))
        sgsʲs_ρa_int_val =
            Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρa, 1))
        # Seed a small positive updraft area fraction when surface buoyancy flux is positive.
        # This perturbation prevents the plume area from staying identically zero,
        # allowing entrainment to grow it to the prescribed surface area.
        @. sgsʲs_ρa_int_val += ifelse(buoyancy_flux_val < 0,
            0,
            max(0, sgsʲs_ρ_int_val * $(eps(FT)) - sgsʲs_ρa_int_val),
        )
        @. ᶜaʲ_int_val = draft_area(sgsʲs_ρa_int_val, sgsʲs_ρ_int_val)
        entr_int_val = Fields.field_values(Fields.level(ᶜentrʲs.:($j), 1))
        detr_int_val = Fields.field_values(Fields.level(ᶜdetrʲs.:($j), 1))
        @. entr_int_val = ifelse(
            buoyancy_flux_val < 0 || ᶜaʲ_int_val <= 0 ||
            ᶜaʲ_int_val >= $(FT(turbconv_params.surface_area)),
            entr_int_val,
            detr_int_val +
            ($(FT(turbconv_params.surface_area)) / ᶜaʲ_int_val - 1) / dt,
        )
        @. detr_int_val = ifelse(
            buoyancy_flux_val < 0 || ᶜaʲ_int_val <= 0 ||
            ᶜaʲ_int_val < $(FT(turbconv_params.surface_area)),
            detr_int_val,
            entr_int_val -
            ($(FT(turbconv_params.surface_area)) / ᶜaʲ_int_val - 1) / dt,
        )
        if p.atmos.sgs_entr_detr_mode == Explicit()
            @. entr_int_val = limit_entrainment(entr_int_val, ᶜaʲ_int_val, dt)
            @. detr_int_val = limit_detrainment(detr_int_val, ᶜaʲ_int_val, dt)
        else
            @. detr_int_val = limit_detrainment(detr_int_val, entr_int_val, ᶜaʲ_int_val, dt)
        end

        @. ᶠρ_diffʲs.:($$j) = min(0, ᶠinterp(ᶜρʲs.:($$j) - Y.c.ρ)) / ᶠinterp(ᶜρʲs.:($$j))
    end

    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice, ᶜcloud_fraction) =
        p.precomputed
    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    @. ᶜlinear_buoygrad = buoyancy_gradients( # TODO - do we need to modify buoyancy gradients based on NonEq + 1M tracers?
        BuoyGradMean(),
        thermo_params,
        ᶜT,
        Y.c.ρ,
        ᶜq_tot,
        ᶜq_liq_rai,
        ᶜq_ice_sno,
        ᶜcloud_fraction,
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
    ρ_sfc_values = Fields.field_values(Fields.level(Y.c.ρ, 1)) # TODO: replace by surface value
    ustar_values = Fields.field_values(ustar)
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), half),
    )
    @. ρtke_flux_values = surface_flux_tke(
        turbconv_params,
        ρ_sfc_values,
        ustar_values,
        sfc_local_geometry_values,
    )

    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_precipitation!(Y, p, microphysics_model)

Updates the precomputed microphysics tendency quantities stored in `p` for EDMFX.

# SGS Quadrature Design

For EDMF, microphysics tendencies are computed separately for updrafts and the environment:

**Updrafts** use direct BMT evaluation (no SGS quadrature) because:
1. Updrafts are coherent turbulent structures with more homogeneous thermodynamic properties
2. Updraft area fraction is usually small (~1-10%), so SGS variance within updrafts has limited 
impact on the grid-mean tendency

**Environment** uses SGS quadrature integration (when `QuadratureMicrophysics` is configured)
because the environment dominates the grid-mean variance. The quadrature captures subgrid-scale
fluctuations in temperature and moisture, which is important for threshold processes like
condensation/evaporation at cloud edges.
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
    ::Union{Microphysics0Moment, QuadratureMicrophysics{Microphysics0Moment}},
)

    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmp = CAP.microphysics_0m_params(params)
    (;
        ᶜT⁰,
        ᶜp,
        ᶜq_tot_safe⁰,
        ᶜq_liq_rai⁰,
        ᶜq_ice_sno⁰,
        ᶜTʲs,
        ᶜq_liq_raiʲs,
        ᶜq_ice_snoʲs,
        ᶜSqₜᵐʲs,
        ᶜSqₜᵐ⁰,
    ) = p.precomputed

    # Sources from the updrafts (direct BMT evaluation without quadrature)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜmp_tendency) = p.precomputed
    for j in 1:n
        # Materialize BMT result first to avoid NamedTuple property access in broadcast
        @. ᶜmp_tendency = BMT.bulk_microphysics_tendencies(
            BMT.Microphysics0Moment(),
            cmp, thp,
            ᶜTʲs.:($$j),
            ᶜq_liq_raiʲs.:($$j),
            ᶜq_ice_snoʲs.:($$j),
        )
        @. ᶜSqₜᵐʲs.:($$j) = limit_sink(
            ᶜmp_tendency.dq_tot_dt,
            Y.c.sgsʲs.:($$j).q_tot, dt,
        )
    end

    # Sources from the environment (with SGS quadrature integration)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰))

    # Get SGS quadrature from atmos config (GridMeanSGS if not using QuadratureMicrophysics)
    SG_quad = if p.atmos.microphysics_model isa QuadratureMicrophysics
        p.atmos.microphysics_model.quadrature
    else
        GridMeanSGS()
    end

    # Get T-based variances from cache (zero when SGS quadrature is off)
    FT = eltype(Y)
    if hasproperty(p.precomputed, :ᶜT′T′)
        (; ᶜT′T′, ᶜq′q′) = p.precomputed
    else
        ᶜT′T′ = FT(0)
        ᶜq′q′ = FT(0)
    end

    # Integrate 0M tendencies over SGS fluctuations
    @. ᶜmp_tendency = microphysics_tendencies_quadrature_0m(
        SG_quad,
        cmp,
        thp,
        ᶜρ⁰,
        ᶜT⁰,
        ᶜq_tot⁰,
        ᶜT′T′,
        ᶜq′q′,
        correlation_Tq(params),
    )
    @. ᶜSqₜᵐ⁰ = limit_sink(ᶜmp_tendency.dq_tot_dt, ᶜq_tot⁰, dt)
    return nothing
end
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_precipitation!(
    Y,
    p,
    ::Union{Microphysics1Moment, QuadratureMicrophysics{Microphysics1Moment}},
)

    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cmp = CAP.microphysics_1m_params(params)
    cmc = CAP.microphysics_cloud_params(params)

    (; ᶜSqₗᵐʲs, ᶜSqᵢᵐʲs, ᶜSqᵣᵐʲs, ᶜSqₛᵐʲs, ᶜρʲs, ᶜTʲs) = p.precomputed
    (; ᶜSqₗᵐ⁰, ᶜSqᵢᵐ⁰, ᶜSqᵣᵐ⁰, ᶜSqₛᵐ⁰, ᶜmp_tendency) = p.precomputed
    (; ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed

    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs) = p.precomputed

    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    for j in 1:n

        # compute terminal velocity for precipitation
        @. ᶜwᵣʲs.:($$j) = CM1.terminal_velocity(
            cmp.precip.rain,
            cmp.terminal_velocity.rain,
            max(zero(Y.c.ρ), ᶜρʲs.:($$j)),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
        )
        @. ᶜwₛʲs.:($$j) = CM1.terminal_velocity(
            cmp.precip.snow,
            cmp.terminal_velocity.snow,
            max(zero(Y.c.ρ), ᶜρʲs.:($$j)),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_sno),
        )
        # compute sedimentation velocity for cloud condensate [m/s]
        @. ᶜwₗʲs.:($$j) = CMNe.terminal_velocity(
            cmc.liquid,
            cmc.stokes,
            max(zero(Y.c.ρ), ᶜρʲs.:($$j)),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_liq),
        )
        @. ᶜwᵢʲs.:($$j) = CMNe.terminal_velocity(
            cmc.ice,
            cmc.Ch2022.small_ice,
            max(zero(Y.c.ρ), ᶜρʲs.:($$j)),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_ice),
        )

        # Microphysics tendencies from the updrafts (using fused BMT API)
        compute_1m_precipitation_tendencies!(
            ᶜSqₗᵐʲs.:($j),
            ᶜSqᵢᵐʲs.:($j),
            ᶜSqᵣᵐʲs.:($j),
            ᶜSqₛᵐʲs.:($j),
            ᶜmp_tendency,
            ᶜρʲs.:($j),
            Y.c.sgsʲs.:($j).q_tot,
            Y.c.sgsʲs.:($j).q_liq,
            Y.c.sgsʲs.:($j).q_ice,
            Y.c.sgsʲs.:($j).q_rai,
            Y.c.sgsʲs.:($j).q_sno,
            ᶜTʲs.:($j),
            dt,
            cmp,
            thp,
            p.atmos.microphysics_tendency_timestepping,
        )

        # Exact derivatives ∂(dqₓ/dt)/∂qₓ at updraft state for Jacobian
        (; ᶜ∂Sqₗʲs, ᶜ∂Sqᵢʲs, ᶜ∂Sqᵣʲs, ᶜ∂Sqₛʲs, ᶜmp_derivative) = p.precomputed
        @. ᶜmp_derivative = BMT.bulk_microphysics_derivatives(
            BMT.Microphysics1Moment(),
            cmp, thp,
            ᶜρʲs.:($$j),
            ᶜTʲs.:($$j),
            Y.c.sgsʲs.:($$j).q_tot,
            Y.c.sgsʲs.:($$j).q_liq,
            Y.c.sgsʲs.:($$j).q_ice,
            Y.c.sgsʲs.:($$j).q_rai,
            Y.c.sgsʲs.:($$j).q_sno,
        )
        @. ᶜ∂Sqₗʲs.:($$j) = ᶜmp_derivative.∂tendency_∂q_lcl
        @. ᶜ∂Sqᵢʲs.:($$j) = ᶜmp_derivative.∂tendency_∂q_icl
        @. ᶜ∂Sqᵣʲs.:($$j) = ᶜmp_derivative.∂tendency_∂q_rai
        @. ᶜ∂Sqₛʲs.:($$j) = ᶜmp_derivative.∂tendency_∂q_sno
    end

    # Microphysics tendencies from the environment (with SGS quadrature)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
    ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰))

    # Get SGS quadrature from atmos config (GridMeanSGS if not using QuadratureMicrophysics)
    SG_quad = if p.atmos.microphysics_model isa QuadratureMicrophysics
        p.atmos.microphysics_model.quadrature
    else
        GridMeanSGS()
    end

    # Get T-based variances from cache (zero when SGS quadrature is off)
    FT = eltype(Y)
    if hasproperty(p.precomputed, :ᶜT′T′)
        (; ᶜT′T′, ᶜq′q′) = p.precomputed
    else
        ᶜT′T′ = FT(0)
        ᶜq′q′ = FT(0)
    end

    # Integrate microphysics tendencies over SGS fluctuations
    # (writes into pre-allocated ᶜmp_tendency to avoid NamedTuple allocation)
    @. ᶜmp_tendency = microphysics_tendencies_quadrature(
        BMT.Microphysics1Moment(),
        SG_quad,
        cmp,
        thp,
        ᶜρ⁰,
        ᶜp,
        ᶜT⁰,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜq_ice⁰,
        ᶜq_rai⁰,
        ᶜq_sno⁰,
        ᶜT′T′,
        ᶜq′q′,
        correlation_Tq(p.params),
    )

    # Apply physically motivated tendency limits
    _apply_1m_limits!(
        ᶜmp_tendency, p.atmos.microphysics_tendency_timestepping,
        thp, ᶜq_tot⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜq_rai⁰, ᶜq_sno⁰, dt,
    )
    @. ᶜSqₗᵐ⁰ = ᶜmp_tendency.dq_lcl_dt
    @. ᶜSqᵢᵐ⁰ = ᶜmp_tendency.dq_icl_dt
    @. ᶜSqᵣᵐ⁰ = ᶜmp_tendency.dq_rai_dt
    @. ᶜSqₛᵐ⁰ = ᶜmp_tendency.dq_sno_dt
    return nothing
end
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_precipitation!(
    Y,
    p,
    ::Union{Microphysics2Moment, QuadratureMicrophysics{Microphysics2Moment}},
)

    (; params, dt) = p
    thp = CAP.thermodynamics_params(params)
    cm1p = CAP.microphysics_1m_params(p.params)
    cm2p = CAP.microphysics_2m_params(p.params)
    cmc = CAP.microphysics_cloud_params(params)

    (;
        ᶜSqₗᵐʲs,
        ᶜSqᵢᵐʲs,
        ᶜSqᵣᵐʲs,
        ᶜSqₛᵐʲs,
        ᶜSnₗᵐʲs,
        ᶜSnᵣᵐʲs,
        ᶜρʲs,
        ᶜTʲs,
        ᶜuʲs,
    ) = p.precomputed
    (; ᶜSqₗᵐ⁰, ᶜSqᵢᵐ⁰, ᶜSqᵣᵐ⁰, ᶜSqₛᵐ⁰, ᶜSnₗᵐ⁰, ᶜSnᵣᵐ⁰, ᶜu⁰, ᶜmp_tendency) =
        p.precomputed
    (; ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
    (; ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs, ᶜwₙₗʲs, ᶜwₙᵣʲs, ᶜuʲs) =
        p.precomputed

    # Get prescribed aerosol concentrations
    seasalt_num = p.scratch.ᶜtemp_scalar_3
    seasalt_mean_radius = p.scratch.ᶜtemp_scalar_4
    sulfate_num = p.scratch.ᶜtemp_scalar_5
    if hasproperty(p, :tracers) &&
       hasproperty(p.tracers, :prescribed_aerosols_field)
        compute_prescribed_aerosol_properties!(
            seasalt_num,
            seasalt_mean_radius,
            sulfate_num,
            p.tracers.prescribed_aerosols_field,
            params.prescribed_aerosol_params,
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
                cm2p.warm_rain.seifert_beheng,
                cmc.Ch2022.rain,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_rai),
            ),
            1,
        )
        @. ᶜwᵣʲs.:($$j) = getindex(
            CM2.rain_terminal_velocity(
                cm2p.warm_rain.seifert_beheng,
                cmc.Ch2022.rain,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_rai),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_rai),
            ),
            2,
        )
        @. ᶜwₛʲs.:($$j) = CM1.terminal_velocity(
            cm1p.precip.snow,
            cm1p.terminal_velocity.snow,
            ᶜρʲs.:($$j),
            max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_sno),
        )
        # compute sedimentation velocity for cloud condensate [m/s]
        # TODO sedimentation of ice is based on the 1M scheme
        @. ᶜwₙₗʲs.:($$j) = getindex(
            CM2.cloud_terminal_velocity(
                cm2p.warm_rain.seifert_beheng.pdf_c,
                cmc.stokes,
                max(zero(Y.c.ρ), Y.c.sgsʲs.:($$j).q_liq),
                ᶜρʲs.:($$j),
                max(zero(Y.c.ρ), ᶜρʲs.:($$j) * Y.c.sgsʲs.:($$j).n_liq),
            ),
            1,
        )
        @. ᶜwₗʲs.:($$j) = getindex(
            CM2.cloud_terminal_velocity(
                cm2p.warm_rain.seifert_beheng.pdf_c,
                cmc.stokes,
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

        # Microphysics tendencies from the updrafts (using fused BMT API)
        # Note: ice and snow tendencies are zero in warm rain 2M scheme
        compute_2m_precipitation_tendencies!(
            ᶜSqₗᵐʲs.:($j),
            ᶜSnₗᵐʲs.:($j),
            ᶜSqᵣᵐʲs.:($j),
            ᶜSnᵣᵐʲs.:($j),
            ᶜmp_tendency,
            ᶜρʲs.:($j),
            Y.c.sgsʲs.:($j).q_tot,
            Y.c.sgsʲs.:($j).q_liq,
            Y.c.sgsʲs.:($j).n_liq,
            Y.c.sgsʲs.:($j).q_rai,
            Y.c.sgsʲs.:($j).n_rai,
            ᶜTʲs.:($j),
            dt,
            cm2p,
            thp,
        )
        @. ᶜSqᵢᵐʲs.:($$j) = 0
        @. ᶜSqₛᵐʲs.:($$j) = 0
        ᶜwʲ = @. lazy(max(0, w_component(Geometry.WVector(ᶜuʲs.:($$j)))))
        @. ᶜSnₗᵐʲs += aerosol_activation_sources(
            (cmc.activation,),  # TODO: remove parenthesis once CMP parameter types are Base.broadcastable
            seasalt_num,
            seasalt_mean_radius,
            sulfate_num,
            Y.c.sgsʲs.:($$j).q_tot,
            Y.c.sgsʲs.:($$j).q_liq + Y.c.sgsʲs.:($$j).q_rai,
            Y.c.sgsʲs.:($$j).q_ice + Y.c.sgsʲs.:($$j).q_sno,
            Y.c.sgsʲs.:($$j).n_liq + Y.c.sgsʲs.:($$j).n_rai,
            ᶜρʲs.:($$j),
            ᶜwʲ,
            (cm2p,),
            thp,
            ᶜTʲs.:($$j),
            ᶜp,
            dt,
            (params.prescribed_aerosol_params,),
        )
    end

    # Microphysics tendencies from the environment (with SGS quadrature)
    ᶜn_liq⁰ = ᶜspecific_env_value(@name(n_liq), Y, p)
    ᶜn_rai⁰ = ᶜspecific_env_value(@name(n_rai), Y, p)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜq_liq⁰ = ᶜspecific_env_value(@name(q_liq), Y, p)
    ᶜq_ice⁰ = ᶜspecific_env_value(@name(q_ice), Y, p)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
    ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰))

    # Get SGS quadrature from atmos config (GridMeanSGS if not using QuadratureMicrophysics)
    SG_quad = if p.atmos.microphysics_model isa QuadratureMicrophysics
        p.atmos.microphysics_model.quadrature
    else
        GridMeanSGS()
    end

    # Integrate microphysics tendencies over SGS fluctuations
    # (writes into pre-allocated ᶜmp_tendency to avoid NamedTuple allocation)
    @. ᶜmp_tendency = microphysics_tendencies_quadrature_2m(
        SG_quad,
        cm2p,
        thp,
        ᶜρ⁰,
        ᶜT⁰,
        ᶜq_tot⁰,
        ᶜq_liq⁰,
        ᶜn_liq⁰,
        ᶜq_rai⁰,
        ᶜn_rai⁰,
    )

    # Apply coupled limiting directly 
    ᶜf_liq = @. lazy(
        coupled_sink_limit_factor(
            ᶜmp_tendency.dq_lcl_dt, ᶜmp_tendency.dn_lcl_dt, ᶜq_liq⁰, ᶜn_liq⁰, dt,
        ),
    )
    ᶜf_rai = @. lazy(
        coupled_sink_limit_factor(
            ᶜmp_tendency.dq_rai_dt, ᶜmp_tendency.dn_rai_dt, ᶜq_rai⁰, ᶜn_rai⁰, dt,
        ),
    )
    @. ᶜSqₗᵐ⁰ = ᶜmp_tendency.dq_lcl_dt * ᶜf_liq
    @. ᶜSnₗᵐ⁰ = ᶜmp_tendency.dn_lcl_dt * ᶜf_liq
    @. ᶜSqᵣᵐ⁰ = ᶜmp_tendency.dq_rai_dt * ᶜf_rai
    @. ᶜSnᵣᵐ⁰ = ᶜmp_tendency.dn_rai_dt * ᶜf_rai
    @. ᶜSqᵢᵐ⁰ = 0
    @. ᶜSqₛᵐ⁰ = 0
    ᶜw⁰ = @. lazy(w_component(Geometry.WVector(ᶜu⁰)))
    @. ᶜSnₗᵐ⁰ += aerosol_activation_sources(
        (cmc.activation,),
        seasalt_num,
        seasalt_mean_radius,
        sulfate_num,
        ᶜq_tot⁰,
        ᶜq_liq⁰ + ᶜq_rai⁰,
        ᶜq_ice⁰ + ᶜq_sno⁰,
        ᶜn_liq⁰ + ᶜn_rai⁰,
        ᶜρ⁰,
        ᶜw⁰,
        (cm2p,),
        thp,
        ᶜT⁰,
        ᶜp,
        dt,
        (params.prescribed_aerosol_params,),
    )
    return nothing
end
