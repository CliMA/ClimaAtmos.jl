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
    (; ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) =
        p.precomputed

    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Y, turbconv_model)
    set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Y.c.uₕ, ᶠuₕ³)
    # @. ᶜK⁰ += ᶜtke
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)

    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)

    if p.atmos.microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), Y, p)
        ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), Y, p)
        ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
        ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
        # Compute env thermodynamic state from primitives
        @. ᶜq_liq⁰ = max(0, ᶜq_lcl⁰ + ᶜq_rai⁰)
        @. ᶜq_ice⁰ = max(0, ᶜq_icl⁰ + ᶜq_sno⁰)
        # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
        @. ᶜq_tot_nonneg⁰ = max(ᶜq_liq⁰ + ᶜq_ice⁰, ᶜq_tot⁰)
        ᶜh⁰ = @. lazy(ᶜmse⁰ - ᶜΦ)  # specific enthalpy
        T_min_sgs = CAP.T_min_sgs(p.params)
        @. ᶜT⁰ = max(
            T_min_sgs,
            TD.air_temperature(
                thermo_params,
                TD.ph(),
                ᶜh⁰,
                ᶜq_tot_nonneg⁰,
                ᶜq_liq⁰,
                ᶜq_ice⁰,
            ),
        )
    else
        # EquilibriumMicrophysics0M: use saturation adjustment to get T and phase partition
        @. ᶜq_tot_nonneg⁰ = max(0, ᶜq_tot⁰)
        (; ᶜsa_result) = p.precomputed
        h⁰ = @. lazy(ᶜmse⁰ - ᶜΦ)
        @. ᶜsa_result =
            saturation_adjustment_tuple(thermo_params, TD.ph(), ᶜp, h⁰, ᶜq_tot_nonneg⁰)
        @. ᶜT⁰ = ᶜsa_result.T
        @. ᶜq_liq⁰ = ᶜsa_result.q_liq
        @. ᶜq_ice⁰ = ᶜsa_result.q_ice
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
    (; microphysics_model, turbconv_model) = p.atmos

    n = n_mass_flux_subdomains(turbconv_model)
    thermo_params = CAP.thermodynamics_params(p.params)

    (; ᶜΦ) = p.core
    (;
        ᶜp,
        ᶜuʲs,
        ᶠu³ʲs,
        ᶜKʲs,
        ᶠKᵥʲs,
        ᶜTʲs,
        ᶜq_tot_nonnegʲs,
        ᶜq_liqʲs,
        ᶜq_iceʲs,
        ᶜρʲs,
    ) = p.precomputed

    for j in 1:n
        ᶜuʲ = ᶜuʲs.:($j)
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        ᶜKʲ = ᶜKʲs.:($j)
        ᶠKᵥʲ = ᶠKᵥʲs.:($j)
        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜq_tot_nonnegʲ = ᶜq_tot_nonnegʲs.:($j)
        ᶜq_liqʲ = ᶜq_liqʲs.:($j)
        ᶜq_iceʲ = ᶜq_iceʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot

        set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³)
        @. ᶠKᵥʲ = (adjoint(CT3(ᶠu₃ʲ)) * ᶠu₃ʲ) / 2

        @. ᶜq_tot_nonnegʲ = max(0, ᶜq_totʲ)
        if microphysics_model isa Union{
            NonEquilibriumMicrophysics1M,
            NonEquilibriumMicrophysics2M,
        }
            ᶜq_lclʲ = Y.c.sgsʲs.:($j).q_lcl
            ᶜq_iclʲ = Y.c.sgsʲs.:($j).q_icl
            ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
            ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
            @. ᶜq_liqʲ = max(0, ᶜq_lclʲ + ᶜq_raiʲ)
            @. ᶜq_iceʲ = max(0, ᶜq_iclʲ + ᶜq_snoʲ)
            # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
            @. ᶜq_tot_nonnegʲ = max(ᶜq_liqʲ + ᶜq_iceʲ, ᶜq_totʲ)
            ᶜhʲ = @. lazy(ᶜmseʲ - ᶜΦ)
            T_min_sgs = CAP.T_min_sgs(p.params)
            @. ᶜTʲ = max(
                T_min_sgs,
                TD.air_temperature(
                    thermo_params,
                    TD.ph(),
                    ᶜhʲ,
                    ᶜq_tot_nonnegʲ,
                    ᶜq_liqʲ,
                    ᶜq_iceʲ,
                ),
            )
        else
            # EquilibriumMicrophysics0M: use saturation adjustment
            (; ᶜsa_result) = p.precomputed
            @. ᶜsa_result = saturation_adjustment_tuple(
                thermo_params,
                TD.ph(),
                ᶜp,
                ᶜmseʲ - ᶜΦ,
                ᶜq_tot_nonnegʲ,
            )
            @. ᶜTʲ = ᶜsa_result.T
            @. ᶜq_liqʲ = ᶜsa_result.q_liq
            @. ᶜq_iceʲ = ᶜsa_result.q_ice
        end
        @. ᶜρʲ =
            TD.air_density(
                thermo_params,
                ᶜTʲ,
                ᶜp,
                ᶜq_tot_nonnegʲ,
                ᶜq_liqʲ,
                ᶜq_iceʲ,
            )
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
    (; ᶜgradᵥ_ᶠΦ) = p.core
    thermo_params = CAP.thermodynamics_params(params)
    turbconv_params = CAP.turbconv_params(params)

    FT = eltype(params)
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜp, ᶠu³) = p.precomputed
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; ᶜstrain_rate_norm, ρtke_flux) = p.precomputed
    (;
        ᶜuʲs,
        ᶜTʲs,
        ᶜq_tot_nonnegʲs,
        ᶜq_liqʲs,
        ᶜq_iceʲs,
        ᶜρʲs,
        ᶜentr_vel_scaleʲs,
        ᶜarea_bounding_entr_detrʲs,
        ᶜturb_entrʲs,
        ᶜρ_diffʲs,
    ) = p.precomputed
    (; ustar) = p.precomputed.sfc_conditions

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜdz = Fields.Δz_field(axes(Y.c))
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))

    for j in 1:n
        # Compute the entrainment velocity scale and the signed area-bounding rate.
        # The environment velocity is passed as w⁰ = 0 to the coefficient model;
        # using the true (ᶜwʲ - ᶜw⁰) difference would introduce residual forcing
        # when ᶜwʲ ≈ 0, which can spuriously grow the area fraction and destabilize
        # otherwise trivial updrafts. The total entrainment rate is then assembled
        # at every tendency call site (`edmfx_entr_detr_tendency!` and the
        # implicit ρa solve) via `compute_entrainment` using the
        # (then-updated) updraft velocity |wʲ|.
        @. ᶜentr_vel_scaleʲs.:($$j) = entrainment_velocity_scale(
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
                ᶜq_tot_nonnegʲs.:($$j),
                ᶜq_liqʲs.:($$j),
                ᶜq_iceʲs.:($$j),
            ),
            vertical_buoyancy_acceleration(Y.c.ρ, ᶜρʲs.:($$j), ᶜgradᵥ_ᶠΦ, ᶜlg),
            FT(0),
            TD.relative_humidity(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_nonneg⁰,
                ᶜq_liq⁰,
                ᶜq_ice⁰,
            ),
            FT(0),
            max(ᶜtke, 0),
            p.atmos.edmfx_model.entr_model,
        )
        @. ᶜturb_entrʲs.:($$j) = turbulent_entrainment(
            turbconv_params,
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
        )

        @. ᶜarea_bounding_entr_detrʲs.:($$j) = area_bounding_entr_detr(
            turbconv_params,
            draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
        )

        @. ᶜρ_diffʲs.:($$j) = (ᶜρʲs.:($$j) - Y.c.ρ) / ᶜρʲs.:($$j)
    end

    # Surface BC payload per updraft at level 1 (capped mass flux +
    # buoyant-air mse/q_tot values).
    set_edmfx_surface_conditions!(Y, p)

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
