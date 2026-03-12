#####
##### Precomputed quantities
#####
import NVTX
import Thermodynamics as TD
import ClimaCore: Spaces, Fields, RecursiveApply

###
### Helper functions for the diagnosic edmf integral
###
# ᶠJ     - Jacobian at half level below (-1/2)
# ᶜJₚ    - Jacobian at previous level below (-1)
# ᶠJₚ    - Jacobian at previous half level below (-3/2)
# ᶜρaʲₚ  - updraft density * area at previous level below (-1)
# ᶜρʲₚ   - updraft density at previous level below (-1)
# ᶜρₚ    - environment density at previous level below (-1)
# ᶜ∇ϕ₃ₚ  - covariant geopotential gradient at previous level below (-1)
# ᶠu³ʲₚ  - contravariant updraft velocity at previous half level below [1/s] (-3/2)
# ᶜϵʲₚ   - entrainment at previous level (-1)
# ᶜδʲₚ   - detrainment at previous level (-1)
# ᶜϵₜʲₚ  - turbulent entrainment at previous level (-1)
# ᶜSʲₚ   - microphysics sources and sinks at previous level (-1)
# ᶜtracerʲₚ - updraft property at previous level (-1)
# ᶜtracerₚ  - environment property at previous level (-1)

# Advection of area, mse and tracers
function diag_edmf_advection(ᶠJ, ᶠJₚ, ᶜρaʲₚ, ᶠu³ʲₚ, ᶜtracerʲₚ)
    return (1 / ᶠJ) * (ᶠJₚ * ᶜρaʲₚ * ᶠu³ʲₚ * ᶜtracerʲₚ)
end
# Entrainment/detrainment of area, mse and tracers
# Note that updraft area entrainment does not include turbulent entrainment.
# In order to re-use the same function for all tracers, we pass in ones
# as updraft and environment tracers for area fraction.
function entr_detr(ᶠJ, ᶜJₚ, ᶜρaʲₚ, ᶜϵʲₚ, ᶜδʲₚ, ᶜϵₜʲₚ, ᶜtracerₚ, ᶜtracerʲₚ)
    return (1 / ᶠJ) * (
        ᶜJₚ * ᶜρaʲₚ * ((ᶜϵʲₚ + ᶜϵₜʲₚ) * ᶜtracerₚ - (ᶜδʲₚ + ᶜϵₜʲₚ) * ᶜtracerʲₚ)
    )
end
# Buoyancy term for mse
function mse_buoyancy(ᶠJ, ᶜJₚ, ᶜρaʲₚ, ᶠu³ʲₚ, ᶜρʲₚ, ᶜρₚ, ᶜ∇Φ₃ₚ)
    return (1 / ᶠJ) * (ᶜJₚ * ᶜρaʲₚ * ᶠu³ʲₚ * (ᶜρʲₚ - ᶜρₚ) / ᶜρʲₚ * ᶜ∇Φ₃ₚ)
end
# Microphysics sources
function microphysics_sources(ᶠJ, ᶜJₚ, ᶜρaʲₚ, ᶜSʲₚ)
    return (1 / ᶠJ) * (ᶜJₚ * ᶜρaʲₚ * ᶜSʲₚ)
end

@inline function kinetic_energy(
    uₕ_level,
    u³_halflevel,
    local_geometry_level,
    local_geometry_halflevel,
)
    return (
        dot(
            C123(uₕ_level, local_geometry_level),
            CT123(uₕ_level, local_geometry_level),
        ) +
        dot(
            C123(u³_halflevel, local_geometry_halflevel),
            CT123(u³_halflevel, local_geometry_halflevel),
        ) +
        2 * dot(
            CT123(uₕ_level, local_geometry_level),
            C123(u³_halflevel, local_geometry_halflevel),
        )
    ) / 2
end

NVTX.@annotate function set_diagnostic_edmfx_draft_quantities_level!(
    thermo_params,
    sa_result_level,
    ρ_level,
    T_level,
    q_tot_safe_level,
    q_liq_rai_level,
    q_ice_sno_level,
    mse_level,
    q_tot_level,
    p_level,
    Φ_level,
)
    FT = eltype(thermo_params)

    @. q_tot_safe_level = max(0, q_tot_level)
    # Perform saturation adjustment to get T, q_liq, q_ice
    @. sa_result_level = saturation_adjustment_tuple(
        thermo_params,
        TD.ph(),
        p_level,
        mse_level - Φ_level,
        q_tot_safe_level,
    )
    # Extract primitive values from thermo state
    @. T_level = sa_result_level.T
    @. q_liq_rai_level = sa_result_level.q_liq
    @. q_ice_sno_level = sa_result_level.q_ice
    # Compute density from primitive values
    @. ρ_level = TD.air_density(
        thermo_params,
        T_level,
        p_level,
        q_tot_safe_level,
        q_liq_rai_level,
        q_ice_sno_level,
    )
    return nothing
end
NVTX.@annotate function set_diagnostic_edmfx_draft_quantities_level!(
    thermo_params,
    ρ_level,
    T_level,
    q_tot_safe_level,
    q_liq_rai_level,
    q_ice_sno_level,
    mse_level,
    q_tot_level,
    q_liq_level,
    q_ice_level,
    q_rai_level,
    q_sno_level,
    p_level,
    Φ_level,
)
    FT = eltype(thermo_params)
    @. q_liq_rai_level = max(0, q_liq_level + q_rai_level)
    @. q_ice_sno_level = max(0, q_ice_level + q_sno_level)
    # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
    @. q_tot_safe_level = max(q_liq_rai_level + q_ice_sno_level, q_tot_level)
    @. T_level = TD.air_temperature(
        thermo_params,
        TD.ph(),
        mse_level - Φ_level,
        q_tot_safe_level,
        q_liq_rai_level,
        q_ice_sno_level,
    )
    @. ρ_level = TD.air_density(
        thermo_params,
        T_level,
        p_level,
        q_tot_safe_level,
        q_liq_rai_level,
        q_ice_sno_level,
    )
    return nothing
end

NVTX.@annotate function set_diagnostic_edmfx_env_quantities_level!(
    ρ_level,
    ρaʲs_level,
    u³_halflevel,
    u³ʲs_halflevel,
    u³⁰_halflevel,
    uₕ_level,
    K⁰_level,
    local_geometry_level,
    local_geometry_halflevel,
    turbconv_model,
)
    @. u³⁰_halflevel = specific(
        ρ_level * u³_halflevel -
        unrolled_dotproduct(ρaʲs_level, u³ʲs_halflevel),
        ρ_level,
        ρ_level * u³_halflevel,
        ρ_level,
        turbconv_model,
    )
    @. K⁰_level = kinetic_energy(
        uₕ_level,
        u³⁰_halflevel,
        local_geometry_level,
        local_geometry_halflevel,
    )
    return nothing
end

"""
    set_diagnostic_edmf_precomputed_quantities_bottom_bc!(Y, p, t)

Updates the bottom boundary conditions in precomputed quantities
stored in `p` for diagnostic edmfx.
"""
NVTX.@annotate function set_diagnostic_edmf_precomputed_quantities_bottom_bc!(
    Y,
    p,
    t,
)
    (; turbconv_model, microphysics_model) = p.atmos
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜΦ) = p.core
    (; ᶜp, ᶠu³, ᶜK, ᶜh_tot) = p.precomputed
    (; ustar, obukhov_length, buoyancy_flux, ρ_flux_h_tot, ρ_flux_q_tot) =
        p.precomputed.sfc_conditions
    (; ᶜρaʲs, ᶠu³ʲs, ᶜKʲs, ᶜmseʲs, ᶜq_totʲs, ᶜρʲs) = p.precomputed
    (; ᶜTʲs, ᶜq_tot_safeʲs, ᶜq_liq_raiʲs, ᶜq_ice_snoʲs) = p.precomputed
    (; ᶠu³⁰, ᶜK⁰) = p.precomputed

    (; params) = p

    thermo_params = CAP.thermodynamics_params(params)
    turbconv_params = CAP.turbconv_params(params)

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))

    ρ_int_level = Fields.field_values(Fields.level(Y.c.ρ, 1))
    uₕ_int_level = Fields.field_values(Fields.level(Y.c.uₕ, 1))
    u³_int_halflevel = Fields.field_values(Fields.level(ᶠu³, half))
    h_tot_int_level = Fields.field_values(Fields.level(ᶜh_tot, 1))
    K_int_level = Fields.field_values(Fields.level(ᶜK, 1))
    q_tot_int_level = Fields.field_values(Fields.level(ᶜq_tot, 1))

    p_int_level = Fields.field_values(Fields.level(ᶜp, 1))
    Φ_int_level = Fields.field_values(Fields.level(ᶜΦ, 1))

    local_geometry_int_level =
        Fields.field_values(Fields.level(Fields.local_geometry_field(Y.c), 1))
    local_geometry_int_halflevel = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), half),
    )
    z_int_level =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
    z_sfc_halflevel =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.f).z, half))
    buoyancy_flux_sfc_halflevel = Fields.field_values(buoyancy_flux)
    ρ_flux_h_tot_sfc_halflevel = Fields.field_values(ρ_flux_h_tot)
    ρ_flux_q_tot_sfc_halflevel = Fields.field_values(ρ_flux_q_tot)
    ustar_sfc_halflevel = Fields.field_values(ustar)
    obukhov_length_sfc_halflevel = Fields.field_values(obukhov_length)

    if microphysics_model isa NonEquilibriumMicrophysics1M
        (; ᶜq_liqʲs, ᶜq_iceʲs, ᶜq_raiʲs, ᶜq_snoʲs) = p.precomputed

        ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
        ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
        ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
        ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))

        q_liq_int_level = Fields.field_values(Fields.level(ᶜq_liq, 1))
        q_ice_int_level = Fields.field_values(Fields.level(ᶜq_ice, 1))
        q_rai_int_level = Fields.field_values(Fields.level(ᶜq_rai, 1))
        q_sno_int_level = Fields.field_values(Fields.level(ᶜq_sno, 1))

        # TODO consider adding them to p.precomputed.sfc_conditions
        # Though they will always be zero.
        ρ_flux_q_liq_sfc_halflevel = FT(0)
        ρ_flux_q_ice_sfc_halflevel = FT(0)
        ρ_flux_q_rai_sfc_halflevel = FT(0)
        ρ_flux_q_sno_sfc_halflevel = FT(0)
    end

    # boundary condition
    for j in 1:n
        ᶜρaʲ = ᶜρaʲs.:($j)
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        ᶜKʲ = ᶜKʲs.:($j)
        ᶜmseʲ = ᶜmseʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜq_totʲ = ᶜq_totʲs.:($j)
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜq_tot_safeʲ = ᶜq_tot_safeʲs.:($j)
        ᶜq_liq_raiʲ = ᶜq_liq_raiʲs.:($j)
        ᶜq_ice_snoʲ = ᶜq_ice_snoʲs.:($j)

        ρaʲ_int_level = Fields.field_values(Fields.level(ᶜρaʲ, 1))
        u³ʲ_int_halflevel = Fields.field_values(Fields.level(ᶠu³ʲ, half))
        Kʲ_int_level = Fields.field_values(Fields.level(ᶜKʲ, 1))
        mseʲ_int_level = Fields.field_values(Fields.level(ᶜmseʲ, 1))
        q_totʲ_int_level = Fields.field_values(Fields.level(ᶜq_totʲ, 1))
        ρʲ_int_level = Fields.field_values(Fields.level(ᶜρʲ, 1))
        Tʲ_int_level = Fields.field_values(Fields.level(ᶜTʲ, 1))
        q_tot_safeʲ_int_level = Fields.field_values(Fields.level(ᶜq_tot_safeʲ, 1))
        q_liq_raiʲ_int_level = Fields.field_values(Fields.level(ᶜq_liq_raiʲ, 1))
        q_ice_snoʲ_int_level = Fields.field_values(Fields.level(ᶜq_ice_snoʲ, 1))

        @. u³ʲ_int_halflevel = CT3(
            Geometry.WVector($(FT(0)), local_geometry_int_halflevel),
            local_geometry_int_halflevel,
        )
        @. mseʲ_int_level = sgs_scalar_first_interior_bc(
            z_int_level - z_sfc_halflevel,
            ρ_int_level,
            FT(turbconv_params.surface_area),
            h_tot_int_level - K_int_level,
            buoyancy_flux_sfc_halflevel,
            ρ_flux_h_tot_sfc_halflevel,
            ustar_sfc_halflevel,
            obukhov_length_sfc_halflevel,
            local_geometry_int_halflevel,
        )
        @. q_totʲ_int_level = sgs_scalar_first_interior_bc(
            z_int_level - z_sfc_halflevel,
            ρ_int_level,
            FT(turbconv_params.surface_area),
            q_tot_int_level,
            buoyancy_flux_sfc_halflevel,
            ρ_flux_q_tot_sfc_halflevel,
            ustar_sfc_halflevel,
            obukhov_length_sfc_halflevel,
            local_geometry_int_halflevel,
        )

        @. Kʲ_int_level = kinetic_energy(
            uₕ_int_level,
            u³ʲ_int_halflevel,
            local_geometry_int_level,
            local_geometry_int_halflevel,
        )

        if microphysics_model isa NonEquilibriumMicrophysics1M
            ᶜq_liqʲ = ᶜq_liqʲs.:($j)
            ᶜq_iceʲ = ᶜq_iceʲs.:($j)
            ᶜq_raiʲ = ᶜq_raiʲs.:($j)
            ᶜq_snoʲ = ᶜq_snoʲs.:($j)

            q_liqʲ_int_level = Fields.field_values(Fields.level(ᶜq_liqʲ, 1))
            q_iceʲ_int_level = Fields.field_values(Fields.level(ᶜq_iceʲ, 1))
            q_raiʲ_int_level = Fields.field_values(Fields.level(ᶜq_raiʲ, 1))
            q_snoʲ_int_level = Fields.field_values(Fields.level(ᶜq_snoʲ, 1))

            @. q_liqʲ_int_level = sgs_scalar_first_interior_bc(
                z_int_level - z_sfc_halflevel,
                ρ_int_level,
                FT(turbconv_params.surface_area),
                q_liq_int_level,
                buoyancy_flux_sfc_halflevel,
                ρ_flux_q_liq_sfc_halflevel,
                ustar_sfc_halflevel,
                obukhov_length_sfc_halflevel,
                local_geometry_int_halflevel,
            )
            @. q_iceʲ_int_level = sgs_scalar_first_interior_bc(
                z_int_level - z_sfc_halflevel,
                ρ_int_level,
                FT(turbconv_params.surface_area),
                q_ice_int_level,
                buoyancy_flux_sfc_halflevel,
                ρ_flux_q_ice_sfc_halflevel,
                ustar_sfc_halflevel,
                obukhov_length_sfc_halflevel,
                local_geometry_int_halflevel,
            )
            @. q_raiʲ_int_level = sgs_scalar_first_interior_bc(
                z_int_level - z_sfc_halflevel,
                ρ_int_level,
                FT(turbconv_params.surface_area),
                q_rai_int_level,
                buoyancy_flux_sfc_halflevel,
                ρ_flux_q_rai_sfc_halflevel,
                ustar_sfc_halflevel,
                obukhov_length_sfc_halflevel,
                local_geometry_int_halflevel,
            )
            @. q_snoʲ_int_level = sgs_scalar_first_interior_bc(
                z_int_level - z_sfc_halflevel,
                ρ_int_level,
                FT(turbconv_params.surface_area),
                q_sno_int_level,
                buoyancy_flux_sfc_halflevel,
                ρ_flux_q_sno_sfc_halflevel,
                ustar_sfc_halflevel,
                obukhov_length_sfc_halflevel,
                local_geometry_int_halflevel,
            )
        end

        if microphysics_model isa NonEquilibriumMicrophysics1M
            set_diagnostic_edmfx_draft_quantities_level!(
                thermo_params,
                ρʲ_int_level,
                Tʲ_int_level,
                q_tot_safeʲ_int_level,
                q_liq_raiʲ_int_level,
                q_ice_snoʲ_int_level,
                mseʲ_int_level,
                q_totʲ_int_level,
                q_liqʲ_int_level,
                q_iceʲ_int_level,
                q_raiʲ_int_level,
                q_snoʲ_int_level,
                p_int_level,
                Φ_int_level,
            )
        else
            (; ᶜsa_result) = p.precomputed
            sa_result_int_level = Fields.field_values(Fields.level(ᶜsa_result, 1))
            set_diagnostic_edmfx_draft_quantities_level!(
                thermo_params,
                sa_result_int_level,
                ρʲ_int_level,
                Tʲ_int_level,
                q_tot_safeʲ_int_level,
                q_liq_raiʲ_int_level,
                q_ice_snoʲ_int_level,
                mseʲ_int_level,
                q_totʲ_int_level,
                p_int_level,
                Φ_int_level,
            )
        end

        @. ρaʲ_int_level = ρʲ_int_level * FT(turbconv_params.surface_area)
    end

    ρaʲs_int_level = Fields.field_values(Fields.level(ᶜρaʲs, 1))
    u³ʲs_int_halflevel = Fields.field_values(Fields.level(ᶠu³ʲs, half))
    u³⁰_int_halflevel = Fields.field_values(Fields.level(ᶠu³⁰, half))
    K⁰_int_level = Fields.field_values(Fields.level(ᶜK⁰, 1))
    set_diagnostic_edmfx_env_quantities_level!(
        ρ_int_level,
        ρaʲs_int_level,
        u³_int_halflevel,
        u³ʲs_int_halflevel,
        u³⁰_int_halflevel,
        uₕ_int_level,
        K⁰_int_level,
        local_geometry_int_level,
        local_geometry_int_halflevel,
        turbconv_model,
    )

    return nothing
end

function compute_u³ʲ_u³ʲ(
    u³ʲ_prev_halflevel,
    J_prev_halflevel,
    J_halflevel,
    J_prev_level,
    ∇Φ³_data_prev_level,
    ρʲ_prev_level,
    ρ_prev_level,
    entrʲ_prev_level,
    turb_entrʲ_prev_level,
    u³⁰_data_prev_halflevel,
    nh_pressure³_buoyʲ_data_prev_halflevel,
    nh_pressure³_dragʲ_data_prev_halflevel,
)
    u³ʲ_u³ʲ =
        (1 / (J_halflevel^2)) *
        (J_prev_halflevel^2 * u³ʲ_prev_halflevel * u³ʲ_prev_halflevel)

    u³ʲ_u³ʲ -=
        (1 / (J_halflevel^2)) * (
            J_prev_level^2 *
            2 *
            (
                ∇Φ³_data_prev_level * (ρʲ_prev_level - ρ_prev_level) /
                ρʲ_prev_level
            )
        )

    u³ʲ_u³ʲ +=
        (1 / (J_halflevel^2)) * (
            J_prev_level^2 *
            2 *
            (
                (entrʲ_prev_level + turb_entrʲ_prev_level) *
                u³⁰_data_prev_halflevel -
                (entrʲ_prev_level + turb_entrʲ_prev_level) * u³ʲ_prev_halflevel
            )
        )

    u³ʲ_u³ʲ -=
        (1 / (J_halflevel^2)) * (
            J_prev_level^2 *
            2 *
            (
                nh_pressure³_buoyʲ_data_prev_halflevel +
                nh_pressure³_dragʲ_data_prev_halflevel
            )
        )
    return u³ʲ_u³ʲ
end

NVTX.@annotate function set_diagnostic_edmf_precomputed_quantities_do_integral!(
    Y,
    p,
    t,
)
    (; turbconv_model, microphysics_model) = p.atmos
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜdz = Fields.Δz_field(axes(Y.c))
    (; params) = p
    (; dt) = p
    (; ᶜΦ, ᶜgradᵥ_ᶠΦ) = p.core
    (; ᶜp, ᶠu³, ᶜT, ᶜh_tot, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno, ᶜK) = p.precomputed
    (;
        ᶜρaʲs,
        ᶠu³ʲs,
        ᶜKʲs,
        ᶜmseʲs,
        ᶜq_totʲs,
        ᶜρʲs,
        ᶜentrʲs,
        ᶜdetrʲs,
        ᶜturb_entrʲs,
        ᶠnh_pressure³_buoyʲs,
        ᶠnh_pressure³_dragʲs,
    ) = p.precomputed
    (; ᶜTʲs, ᶜq_tot_safeʲs, ᶜq_liq_raiʲs, ᶜq_ice_snoʲs) = p.precomputed
    (; ᶠu³⁰, ᶜK⁰) = p.precomputed

    if microphysics_model isa NonEquilibriumMicrophysics1M
        ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
        ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
        ᶜq_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
        ᶜq_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))

        (; ᶜq_liqʲs, ᶜq_iceʲs, ᶜq_raiʲs, ᶜq_snoʲs) = p.precomputed
    end

    thermo_params = CAP.thermodynamics_params(params)
    microphys_0m_params = CAP.microphysics_0m_params(params)
    microphys_1m_params = CAP.microphysics_1m_params(params)
    cloud_params = CAP.microphysics_cloud_params(params)
    turbconv_params = CAP.turbconv_params(params)

    ᶠΦ = p.scratch.ᶠtemp_scalar
    @. ᶠΦ = CAP.grav(params) * ᶠz
    ᶜ∇Φ³ = p.scratch.ᶜtemp_CT3
    @. ᶜ∇Φ³ = CT3(ᶜgradᵥ(ᶠΦ))
    @. ᶜ∇Φ³ += CT3(gradₕ(ᶜΦ))
    ᶜ∇Φ₃ = p.scratch.ᶜtemp_C3
    @. ᶜ∇Φ₃ = ᶜgradᵥ(ᶠΦ)

    z_sfc_halflevel =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.f).z, half))

    # integral
    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))

    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))

    for i in 2:Spaces.nlevels(axes(Y.c))
        ρ_level = Fields.field_values(Fields.level(Y.c.ρ, i))
        uₕ_level = Fields.field_values(Fields.level(Y.c.uₕ, i))
        u³_halflevel = Fields.field_values(Fields.level(ᶠu³, i - half))
        K_level = Fields.field_values(Fields.level(ᶜK, i))
        h_tot_level = Fields.field_values(Fields.level(ᶜh_tot, i))
        q_tot_level = Fields.field_values(Fields.level(ᶜq_tot, i))
        p_level = Fields.field_values(Fields.level(ᶜp, i))
        Φ_level = Fields.field_values(Fields.level(ᶜΦ, i))
        local_geometry_level = Fields.field_values(
            Fields.level(Fields.local_geometry_field(Y.c), i),
        )
        local_geometry_halflevel = Fields.field_values(
            Fields.level(Fields.local_geometry_field(Y.f), i - half),
        )
        ∂x∂ξ_level = local_geometry_level.∂x∂ξ.components.data
        end_index = fieldcount(eltype(∂x∂ξ_level)) # This will be 4 in 2D and 9 in 3D.
        ∂x³∂ξ³_level = ∂x∂ξ_level.:($end_index)

        Φ_prev_level = Fields.field_values(Fields.level(ᶜΦ, i - 1))
        ∇Φ³_prev_level = Fields.field_values(Fields.level(ᶜ∇Φ³, i - 1))
        ∇Φ³_data_prev_level = ∇Φ³_prev_level.components.data.:1
        ∇Φ₃_prev_level = Fields.field_values(Fields.level(ᶜ∇Φ₃, i - 1))
        ∇Φ₃_data_prev_level = ∇Φ₃_prev_level.components.data.:1
        ρ_prev_level = Fields.field_values(Fields.level(Y.c.ρ, i - 1))
        u³_prev_halflevel = Fields.field_values(Fields.level(ᶠu³, i - 1 - half))
        u³⁰_prev_halflevel =
            Fields.field_values(Fields.level(ᶠu³⁰, i - 1 - half))
        u³⁰_data_prev_halflevel = u³⁰_prev_halflevel.components.data.:1
        K_prev_level = Fields.field_values(Fields.level(ᶜK, i - 1))
        h_tot_prev_level = Fields.field_values(Fields.level(ᶜh_tot, i - 1))
        q_tot_prev_level = Fields.field_values(Fields.level(ᶜq_tot, i - 1))
        T_prev_level = Fields.field_values(Fields.level(ᶜT, i - 1))
        q_tot_safe_prev_level = Fields.field_values(Fields.level(ᶜq_tot_safe, i - 1))
        q_liq_rai_prev_level = Fields.field_values(Fields.level(ᶜq_liq_rai, i - 1))
        q_ice_sno_prev_level = Fields.field_values(Fields.level(ᶜq_ice_sno, i - 1))
        p_prev_level = Fields.field_values(Fields.level(ᶜp, i - 1))
        z_prev_level = Fields.field_values(Fields.level(ᶜz, i - 1))
        dz_prev_level = Fields.field_values(Fields.level(ᶜdz, i - 1))

        if microphysics_model isa NonEquilibriumMicrophysics1M
            q_liq_level = Fields.field_values(Fields.level(ᶜq_liq, i))
            q_liq_prev_level = Fields.field_values(Fields.level(ᶜq_liq, i - 1))

            q_ice_level = Fields.field_values(Fields.level(ᶜq_ice, i))
            q_ice_prev_level = Fields.field_values(Fields.level(ᶜq_ice, i - 1))

            q_rai_level = Fields.field_values(Fields.level(ᶜq_rai, i))
            q_rai_prev_level = Fields.field_values(Fields.level(ᶜq_rai, i - 1))

            q_sno_level = Fields.field_values(Fields.level(ᶜq_sno, i))
            q_sno_prev_level = Fields.field_values(Fields.level(ᶜq_sno, i - 1))
        end

        local_geometry_prev_level = Fields.field_values(
            Fields.level(Fields.local_geometry_field(Y.c), i - 1),
        )
        local_geometry_prev_halflevel = Fields.field_values(
            Fields.level(Fields.local_geometry_field(Y.f), i - 1 - half),
        )
        for j in 1:n
            ᶜρaʲ = ᶜρaʲs.:($j)
            ᶠu³ʲ = ᶠu³ʲs.:($j)
            ᶜKʲ = ᶜKʲs.:($j)
            ᶜmseʲ = ᶜmseʲs.:($j)
            ᶜρʲ = ᶜρʲs.:($j)
            ᶜq_totʲ = ᶜq_totʲs.:($j)
            ᶜTʲ = ᶜTʲs.:($j)
            ᶜq_tot_safeʲ = ᶜq_tot_safeʲs.:($j)
            ᶜq_liq_raiʲ = ᶜq_liq_raiʲs.:($j)
            ᶜq_ice_snoʲ = ᶜq_ice_snoʲs.:($j)
            ᶜentrʲ = ᶜentrʲs.:($j)
            ᶜdetrʲ = ᶜdetrʲs.:($j)
            ᶜturb_entrʲ = ᶜturb_entrʲs.:($j)
            ᶠnh_pressure³_buoyʲ = ᶠnh_pressure³_buoyʲs.:($j)
            ᶠnh_pressure³_dragʲ = ᶠnh_pressure³_dragʲs.:($j)

            if microphysics_model isa EquilibriumMicrophysics0M
                ᶜmp_tendencyʲ = p.precomputed.ᶜmp_tendencyʲs.:($j)
            end
            if microphysics_model isa NonEquilibriumMicrophysics1M
                ᶜq_liqʲ = ᶜq_liqʲs.:($j)
                ᶜq_iceʲ = ᶜq_iceʲs.:($j)
                ᶜq_raiʲ = ᶜq_raiʲs.:($j)
                ᶜq_snoʲ = ᶜq_snoʲs.:($j)
                ᶜmp_tendencyʲ = p.precomputed.ᶜmp_tendencyʲs.:($j)
            end

            ρaʲ_level = Fields.field_values(Fields.level(ᶜρaʲ, i))
            u³ʲ_halflevel = Fields.field_values(Fields.level(ᶠu³ʲ, i - half))
            Kʲ_level = Fields.field_values(Fields.level(ᶜKʲ, i))
            mseʲ_level = Fields.field_values(Fields.level(ᶜmseʲ, i))
            q_totʲ_level = Fields.field_values(Fields.level(ᶜq_totʲ, i))
            ρʲ_level = Fields.field_values(Fields.level(ᶜρʲ, i))
            Tʲ_level = Fields.field_values(Fields.level(ᶜTʲ, i))
            q_tot_safeʲ_level = Fields.field_values(Fields.level(ᶜq_tot_safeʲ, i))
            q_liq_raiʲ_level = Fields.field_values(Fields.level(ᶜq_liq_raiʲ, i))
            q_ice_snoʲ_level = Fields.field_values(Fields.level(ᶜq_ice_snoʲ, i))

            ρaʲ_prev_level = Fields.field_values(Fields.level(ᶜρaʲ, i - 1))
            u³ʲ_prev_halflevel =
                Fields.field_values(Fields.level(ᶠu³ʲ, i - 1 - half))
            u³ʲ_data_prev_halflevel = u³ʲ_prev_halflevel.components.data.:1
            mseʲ_prev_level = Fields.field_values(Fields.level(ᶜmseʲ, i - 1))
            q_totʲ_prev_level =
                Fields.field_values(Fields.level(ᶜq_totʲ, i - 1))
            ρʲ_prev_level = Fields.field_values(Fields.level(ᶜρʲ, i - 1))
            Tʲ_prev_level = Fields.field_values(Fields.level(ᶜTʲ, i - 1))
            q_tot_safeʲ_prev_level = Fields.field_values(Fields.level(ᶜq_tot_safeʲ, i - 1))
            q_liq_raiʲ_prev_level = Fields.field_values(Fields.level(ᶜq_liq_raiʲ, i - 1))
            q_ice_snoʲ_prev_level = Fields.field_values(Fields.level(ᶜq_ice_snoʲ, i - 1))
            ᶜgradᵥ_ᶠΦ_prev_level =
                Fields.field_values(Fields.level(ᶜgradᵥ_ᶠΦ, i - 1))
            entrʲ_prev_level = Fields.field_values(Fields.level(ᶜentrʲ, i - 1))
            detrʲ_prev_level = Fields.field_values(Fields.level(ᶜdetrʲ, i - 1))
            turb_entrʲ_prev_level =
                Fields.field_values(Fields.level(ᶜturb_entrʲ, i - 1))
            nh_pressure³_buoyʲ_prev_halflevel = Fields.field_values(
                Fields.level(ᶠnh_pressure³_buoyʲ, i - 1 - half),
            )
            nh_pressure³_dragʲ_prev_halflevel = Fields.field_values(
                Fields.level(ᶠnh_pressure³_dragʲ, i - 1 - half),
            )
            scale_height =
                CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)

            ᶜmp_tendencyʲ_prev_level =
                if microphysics_model isa MoistMicrophysics
                    Fields.field_values(Fields.level(ᶜmp_tendencyʲ, i - 1))
                else
                    Ref(nothing)
                end
            if microphysics_model isa NonEquilibriumMicrophysics1M
                q_liqʲ_level = Fields.field_values(Fields.level(ᶜq_liqʲ, i))
                q_iceʲ_level = Fields.field_values(Fields.level(ᶜq_iceʲ, i))
                q_raiʲ_level = Fields.field_values(Fields.level(ᶜq_raiʲ, i))
                q_snoʲ_level = Fields.field_values(Fields.level(ᶜq_snoʲ, i))

                q_liqʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜq_liqʲ, i - 1))
                q_iceʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜq_iceʲ, i - 1))
                q_raiʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜq_raiʲ, i - 1))
                q_snoʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜq_snoʲ, i - 1))
            end

            tke_prev_level = Fields.field_values(Fields.level(ᶜtke, i - 1))

            # Use a temperature floor to prevent DomainError in
            # saturation_vapor_pressure when updraft thermodynamics
            # produce unphysical temperatures.
            rhʲ_prev_level = p.scratch.temp_data_level_4
            @. rhʲ_prev_level = TD.relative_humidity(
                thermo_params,
                max(Tʲ_prev_level, CAP.T_min_sgs(params)),
                p_prev_level,
                q_tot_safeʲ_prev_level,
                q_liq_raiʲ_prev_level,
                q_ice_snoʲ_prev_level,
            )
            rh_prev_level = p.scratch.temp_data_level_5
            @. rh_prev_level = TD.relative_humidity(
                thermo_params,
                max(T_prev_level, CAP.T_min_sgs(params)),
                p_prev_level,
                q_tot_safe_prev_level,
                q_liq_rai_prev_level,
                q_ice_sno_prev_level,
            )

            @. entrʲ_prev_level = entrainment(
                thermo_params,
                turbconv_params,
                z_prev_level,
                z_sfc_halflevel,
                p_prev_level,
                ρ_prev_level,
                draft_area(ρaʲ_prev_level, ρʲ_prev_level),
                get_physical_w(
                    u³ʲ_prev_halflevel,
                    local_geometry_prev_halflevel,
                ),
                rhʲ_prev_level,
                vertical_buoyancy_acceleration(
                    ρ_prev_level,
                    ρʲ_prev_level,
                    ᶜgradᵥ_ᶠΦ_prev_level,
                    local_geometry_prev_halflevel,
                ),
                get_physical_w(
                    u³_prev_halflevel,
                    local_geometry_prev_halflevel,
                ),
                rh_prev_level,
                FT(0),
                tke_prev_level,
                p.atmos.edmfx_model.entr_model,
            )

            @. turb_entrʲ_prev_level = turbulent_entrainment(
                turbconv_params,
                draft_area(ρaʲ_prev_level, ρʲ_prev_level),
            )

            # We don't have an upper limit to entrainment for the first level
            # (calculated at i=2), as the vertical velocity at the first level is zero
            if i > 2
                @. entrʲ_prev_level = limit_entrainment(
                    entrʲ_prev_level,
                    draft_area(ρaʲ_prev_level, ρʲ_prev_level),
                    get_physical_w(
                        u³ʲ_prev_halflevel,
                        local_geometry_prev_halflevel,
                    ),
                    dz_prev_level,
                )

                @. turb_entrʲ_prev_level = limit_turb_entrainment(
                    entrʲ_prev_level,
                    turb_entrʲ_prev_level,
                    get_physical_w(
                        u³ʲ_prev_halflevel,
                        local_geometry_prev_halflevel,
                    ),
                    dz_prev_level,
                )
            end
            @. entrʲ_prev_level = limit_entrainment(
                entrʲ_prev_level,
                draft_area(ρaʲ_prev_level, ρʲ_prev_level),
                dt,
            )

            # TODO: use updraft top instead of scale height
            if p.atmos.edmfx_model.nh_pressure isa Val{true}
                @. nh_pressure³_buoyʲ_prev_halflevel =
                    ᶠupdraft_nh_pressure_buoyancy(
                        params,
                        -∇Φ³_prev_level * (ρʲ_prev_level - ρ_prev_level) /
                        ρʲ_prev_level,
                    )
                @. nh_pressure³_dragʲ_prev_halflevel =
                    ᶠupdraft_nh_pressure_drag(
                        params,
                        local_geometry_prev_halflevel,
                        u³ʲ_prev_halflevel,
                        u³⁰_prev_halflevel,
                        scale_height,
                    )
            else
                @. nh_pressure³_buoyʲ_prev_halflevel = CT3(0)
                @. nh_pressure³_dragʲ_prev_halflevel = CT3(0)
            end

            nh_pressure³_buoyʲ_data_prev_halflevel =
                nh_pressure³_buoyʲ_prev_halflevel.components.data.:1
            nh_pressure³_dragʲ_data_prev_halflevel =
                nh_pressure³_dragʲ_prev_halflevel.components.data.:1

            # Microphysics sources and sinks. To be applied in updraft continuity,
            # moisture and energy equations for updrafts and grid mean.

            # 0-moment microphysics: sink of q_tot from precipitation removal
            if microphysics_model isa EquilibriumMicrophysics0M
                @. ᶜmp_tendencyʲ_prev_level.dq_tot_dt =
                    BMT.bulk_microphysics_tendencies(
                        BMT.Microphysics0Moment(),
                        microphys_0m_params,
                        thermo_params,
                        Tʲ_prev_level,
                        q_liq_raiʲ_prev_level,
                        q_ice_snoʲ_prev_level,
                    )
                @. ᶜmp_tendencyʲ_prev_level.dq_tot_dt = limit_sink(
                    ᶜmp_tendencyʲ_prev_level.dq_tot_dt,
                    q_totʲ_prev_level, dt,
                )
                @. ᶜmp_tendencyʲ_prev_level.e_tot_hlpr =
                    e_tot_0M_precipitation_sources_helper(
                        thermo_params,
                        Tʲ_prev_level,
                        q_liq_rai_prev_level,
                        q_ice_sno_prev_level,
                        Φ_prev_level,
                    )
                # 1-moment microphysics: cloud water (liquid and ice) and
                # precipitation (rain and snow) tendencies. q_tot is constant, because
                # all the species are considered a part of the working fluid.
            elseif microphysics_model isa NonEquilibriumMicrophysics1M
                # Microphysics tendencies from the updrafts (using fused BMT API)
                compute_1m_precipitation_tendencies!(
                    ᶜmp_tendencyʲ_prev_level,
                    ρʲ_prev_level,
                    q_totʲ_prev_level,
                    q_liqʲ_prev_level,
                    q_iceʲ_prev_level,
                    q_raiʲ_prev_level,
                    q_snoʲ_prev_level,
                    Tʲ_prev_level,
                    dt,
                    microphys_1m_params,
                    thermo_params,
                    p.atmos.microphysics_tendency_timestepping,
                )
            end

            u³ʲ_datau³ʲ_data = p.scratch.temp_data_level
            @. u³ʲ_datau³ʲ_data = compute_u³ʲ_u³ʲ(
                u³ʲ_data_prev_halflevel,
                local_geometry_prev_halflevel.J,
                local_geometry_halflevel.J,
                local_geometry_prev_level.J,
                ∇Φ³_data_prev_level,
                ρʲ_prev_level,
                ρ_prev_level,
                entrʲ_prev_level,
                turb_entrʲ_prev_level,
                u³⁰_data_prev_halflevel,
                nh_pressure³_buoyʲ_data_prev_halflevel,
                nh_pressure³_dragʲ_data_prev_halflevel,
            )

            # get u³ʲ to calculate divergence term for detrainment,
            # u³ʲ will be clipped later after we get area fraction
            minimum_value = FT(1e-6)
            @. u³ʲ_halflevel = ifelse(
                ((u³ʲ_datau³ʲ_data < 10 * ∇Φ³_data_prev_level * eps(FT))),
                u³_halflevel,
                CT3(sqrt(max(FT(0), u³ʲ_datau³ʲ_data))),
            )

            u³ʲ_data_halflevel = u³ʲ_halflevel.components.data.:1
            vert_div_level = p.scratch.temp_data_level_2
            @. vert_div_level =
                (
                    local_geometry_halflevel.J * u³ʲ_data_halflevel * ρ_level -
                    local_geometry_prev_level.J *
                    u³ʲ_data_prev_halflevel *
                    ρ_prev_level
                ) / local_geometry_level.J / ρ_level
            w_vert_div_level = p.scratch.temp_data_level_3
            @. w_vert_div_level =
                (
                    local_geometry_halflevel.J * u³ʲ_data_halflevel -
                    local_geometry_prev_level.J * u³ʲ_data_prev_halflevel
                ) / local_geometry_level.J

            @. detrʲ_prev_level = detrainment(
                thermo_params,
                turbconv_params,
                z_prev_level,
                z_sfc_halflevel,
                p_prev_level,
                ρ_prev_level,
                ρaʲ_prev_level,
                draft_area(ρaʲ_prev_level, ρʲ_prev_level),
                get_physical_w(u³ʲ_prev_halflevel, local_geometry_prev_halflevel),
                rhʲ_prev_level,
                vertical_buoyancy_acceleration(
                    ρ_prev_level,
                    ρʲ_prev_level,
                    ᶜgradᵥ_ᶠΦ_prev_level,
                    local_geometry_prev_halflevel,
                ),
                get_physical_w(u³_prev_halflevel, local_geometry_prev_halflevel),
                rh_prev_level,
                FT(0),
                entrʲ_prev_level,
                vert_div_level,
                FT(0), # mass flux divergence is not implemented for diagnostic edmf
                w_vert_div_level,
                tke_prev_level,
                p.atmos.edmfx_model.detr_model,
            )

            @. detrʲ_prev_level = limit_detrainment(
                detrʲ_prev_level,
                draft_area(ρaʲ_prev_level, ρʲ_prev_level),
                get_physical_w(
                    u³ʲ_prev_halflevel,
                    local_geometry_prev_halflevel,
                ),
                dz_prev_level,
                dt,
            )

            ρaʲu³ʲ_data = p.scratch.temp_data_level_2
            ρaʲu³ʲ_datamse = ρaʲu³ʲ_dataq_tot = p.scratch.temp_data_level_3

            ###
            ### Area fraction
            ###
            @. ρaʲu³ʲ_data =
                diag_edmf_advection(
                    local_geometry_halflevel.J,
                    local_geometry_prev_halflevel.J,
                    ρaʲ_prev_level,
                    u³ʲ_data_prev_halflevel,
                    FT(1),
                ) + entr_detr(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    entrʲ_prev_level,
                    detrʲ_prev_level,
                    turb_entrʲ_prev_level,
                    FT(1),
                    FT(1),
                )
            if microphysics_model isa EquilibriumMicrophysics0M
                @. ρaʲu³ʲ_data += microphysics_sources(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    ᶜmp_tendencyʲ_prev_level.dq_tot_dt,
                )
            end

            # Change current level velocity and density * area fraction
            kill_updraft = @. lazy(
                (u³ʲ_datau³ʲ_data < 10 * ∇Φ³_data_prev_level * eps(FT)) |
                (ρaʲu³ʲ_data < (minimum_value / ∂x³∂ξ³_level)),
            )
            @. u³ʲ_halflevel = ifelse(
                kill_updraft,
                u³_halflevel,
                CT3(sqrt(max(FT(0), u³ʲ_datau³ʲ_data))),
            )
            @. ρaʲ_level = ifelse(
                kill_updraft,
                FT(0),
                ρaʲu³ʲ_data / sqrt(max(FT(0), u³ʲ_datau³ʲ_data)),
            )

            ###
            ### Moist static energy
            ###
            @. ρaʲu³ʲ_datamse =
                diag_edmf_advection(
                    local_geometry_halflevel.J,
                    local_geometry_prev_halflevel.J,
                    ρaʲ_prev_level,
                    u³ʲ_data_prev_halflevel,
                    mseʲ_prev_level,
                ) +
                mse_buoyancy(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    u³ʲ_data_prev_halflevel,
                    ρʲ_prev_level,
                    ρ_prev_level,
                    ∇Φ₃_data_prev_level,
                ) +
                entr_detr(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    entrʲ_prev_level,
                    detrʲ_prev_level,
                    turb_entrʲ_prev_level,
                    h_tot_prev_level - K_prev_level,
                    mseʲ_prev_level,
                )
            if microphysics_model isa EquilibriumMicrophysics0M
                @. ρaʲu³ʲ_datamse += microphysics_sources(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    ᶜmp_tendencyʲ_prev_level.dq_tot_dt *
                    ᶜmp_tendencyʲ_prev_level.e_tot_hlpr,
                )
            end
            @. mseʲ_level = ifelse(
                kill_updraft,
                h_tot_level - K_level,
                ρaʲu³ʲ_datamse / ρaʲu³ʲ_data,
            )

            ###
            ### Total water
            ###
            @. ρaʲu³ʲ_dataq_tot =
                diag_edmf_advection(
                    local_geometry_halflevel.J,
                    local_geometry_prev_halflevel.J,
                    ρaʲ_prev_level,
                    u³ʲ_data_prev_halflevel,
                    q_totʲ_prev_level,
                ) + entr_detr(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    entrʲ_prev_level,
                    detrʲ_prev_level,
                    turb_entrʲ_prev_level,
                    q_tot_prev_level,
                    q_totʲ_prev_level,
                )
            if microphysics_model isa EquilibriumMicrophysics0M
                @. ρaʲu³ʲ_dataq_tot += microphysics_sources(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    ᶜmp_tendencyʲ_prev_level.dq_tot_dt,
                )
            end
            @. q_totʲ_level = ifelse(
                kill_updraft,
                q_tot_level,
                ρaʲu³ʲ_dataq_tot / ρaʲu³ʲ_data,
            )

            ###
            ### 1-moment microphysics tracers
            ###
            if microphysics_model isa NonEquilibriumMicrophysics1M
                # TODO - loop over tracres
                ρaʲu³ʲ_dataq_ = p.scratch.temp_data_level_3
                @. ρaʲu³ʲ_dataq_ =
                    diag_edmf_advection(
                        local_geometry_halflevel.J,
                        local_geometry_prev_halflevel.J,
                        ρaʲ_prev_level,
                        u³ʲ_data_prev_halflevel,
                        q_liqʲ_prev_level,
                    ) +
                    entr_detr(
                        local_geometry_halflevel.J,
                        local_geometry_prev_level.J,
                        ρaʲ_prev_level,
                        entrʲ_prev_level,
                        detrʲ_prev_level,
                        turb_entrʲ_prev_level,
                        q_liq_prev_level,
                        q_liqʲ_prev_level,
                    ) +
                    microphysics_sources(
                        local_geometry_halflevel.J,
                        local_geometry_prev_level.J,
                        ρaʲ_prev_level,
                        ᶜmp_tendencyʲ_prev_level.dq_lcl_dt,
                    )
                @. q_liqʲ_level = ifelse(
                    kill_updraft,
                    q_liq_level,
                    ρaʲu³ʲ_dataq_ / ρaʲu³ʲ_data,
                )

                @. ρaʲu³ʲ_dataq_ =
                    diag_edmf_advection(
                        local_geometry_halflevel.J,
                        local_geometry_prev_halflevel.J,
                        ρaʲ_prev_level,
                        u³ʲ_data_prev_halflevel,
                        q_iceʲ_prev_level,
                    ) +
                    entr_detr(
                        local_geometry_halflevel.J,
                        local_geometry_prev_level.J,
                        ρaʲ_prev_level,
                        entrʲ_prev_level,
                        detrʲ_prev_level,
                        turb_entrʲ_prev_level,
                        q_ice_prev_level,
                        q_iceʲ_prev_level,
                    ) +
                    microphysics_sources(
                        local_geometry_halflevel.J,
                        local_geometry_prev_level.J,
                        ρaʲ_prev_level,
                        ᶜmp_tendencyʲ_prev_level.dq_icl_dt,
                    )
                @. q_iceʲ_level = ifelse(
                    kill_updraft,
                    q_ice_level,
                    ρaʲu³ʲ_dataq_ / ρaʲu³ʲ_data,
                )

                @. ρaʲu³ʲ_dataq_ =
                    diag_edmf_advection(
                        local_geometry_halflevel.J,
                        local_geometry_prev_halflevel.J,
                        ρaʲ_prev_level,
                        u³ʲ_data_prev_halflevel,
                        q_raiʲ_prev_level,
                    ) +
                    entr_detr(
                        local_geometry_halflevel.J,
                        local_geometry_prev_level.J,
                        ρaʲ_prev_level,
                        entrʲ_prev_level,
                        detrʲ_prev_level,
                        turb_entrʲ_prev_level,
                        q_rai_prev_level,
                        q_raiʲ_prev_level,
                    ) +
                    microphysics_sources(
                        local_geometry_halflevel.J,
                        local_geometry_prev_level.J,
                        ρaʲ_prev_level,
                        ᶜmp_tendencyʲ_prev_level.dq_rai_dt,
                    )
                @. q_raiʲ_level = ifelse(
                    kill_updraft,
                    q_rai_level,
                    ρaʲu³ʲ_dataq_ / ρaʲu³ʲ_data,
                )

                @. ρaʲu³ʲ_dataq_ =
                    diag_edmf_advection(
                        local_geometry_halflevel.J,
                        local_geometry_prev_halflevel.J,
                        ρaʲ_prev_level,
                        u³ʲ_data_prev_halflevel,
                        q_snoʲ_prev_level,
                    ) +
                    entr_detr(
                        local_geometry_halflevel.J,
                        local_geometry_prev_level.J,
                        ρaʲ_prev_level,
                        entrʲ_prev_level,
                        detrʲ_prev_level,
                        turb_entrʲ_prev_level,
                        q_sno_prev_level,
                        q_snoʲ_prev_level,
                    ) +
                    microphysics_sources(
                        local_geometry_halflevel.J,
                        local_geometry_prev_level.J,
                        ρaʲ_prev_level,
                        ᶜmp_tendencyʲ_prev_level.dq_sno_dt,
                    )
                @. q_snoʲ_level = ifelse(
                    kill_updraft,
                    q_sno_level,
                    ρaʲu³ʲ_dataq_ / ρaʲu³ʲ_data,
                )
            end

            # set updraft to grid-mean if vertical velocity is too small
            if i > 2
                kill_updraft_2 = @. lazy(
                    u³ʲ_data_prev_halflevel * u³ʲ_data_prev_halflevel <
                    ∇Φ³_data_prev_level * (ρʲ_prev_level - ρ_prev_level) /
                    ρʲ_prev_level,
                )
                @. ρaʲ_level = ifelse(kill_updraft_2, FT(0), ρaʲ_level)
                @. u³ʲ_halflevel =
                    ifelse(kill_updraft_2, u³_halflevel, u³ʲ_halflevel)
                @. mseʲ_level =
                    ifelse(kill_updraft_2, h_tot_level - K_level, mseʲ_level)
                @. q_totʲ_level =
                    ifelse(kill_updraft_2, q_tot_level, q_totʲ_level)
                if microphysics_model isa NonEquilibriumMicrophysics1M
                    @. q_liqʲ_level =
                        ifelse(kill_updraft_2, q_liq_level, q_liqʲ_level)
                    @. q_iceʲ_level =
                        ifelse(kill_updraft_2, q_ice_level, q_iceʲ_level)
                    @. q_raiʲ_level =
                        ifelse(kill_updraft_2, q_rai_level, q_raiʲ_level)
                    @. q_snoʲ_level =
                        ifelse(kill_updraft_2, q_sno_level, q_snoʲ_level)
                end
            end

            @. Kʲ_level = kinetic_energy(
                uₕ_level,
                u³ʲ_halflevel,
                local_geometry_level,
                local_geometry_halflevel,
            )
            if microphysics_model isa NonEquilibriumMicrophysics1M
                set_diagnostic_edmfx_draft_quantities_level!(
                    thermo_params,
                    ρʲ_level,
                    Tʲ_level,
                    q_tot_safeʲ_level,
                    q_liq_raiʲ_level,
                    q_ice_snoʲ_level,
                    mseʲ_level,
                    q_totʲ_level,
                    q_liqʲ_level,
                    q_iceʲ_level,
                    q_raiʲ_level,
                    q_snoʲ_level,
                    p_level,
                    Φ_level,
                )
            else
                (; ᶜsa_result) = p.precomputed
                sa_result_level = Fields.field_values(Fields.level(ᶜsa_result, i))
                set_diagnostic_edmfx_draft_quantities_level!(
                    thermo_params,
                    sa_result_level,
                    ρʲ_level,
                    Tʲ_level,
                    q_tot_safeʲ_level,
                    q_liq_raiʲ_level,
                    q_ice_snoʲ_level,
                    mseʲ_level,
                    q_totʲ_level,
                    p_level,
                    Φ_level,
                )
            end
        end
        ρaʲs_level = Fields.field_values(Fields.level(ᶜρaʲs, i))
        u³ʲs_halflevel = Fields.field_values(Fields.level(ᶠu³ʲs, i - half))
        u³⁰_halflevel = Fields.field_values(Fields.level(ᶠu³⁰, i - half))
        K⁰_level = Fields.field_values(Fields.level(ᶜK⁰, i))
        set_diagnostic_edmfx_env_quantities_level!(
            ρ_level,
            ρaʲs_level,
            u³_halflevel,
            u³ʲs_halflevel,
            u³⁰_halflevel,
            uₕ_level,
            K⁰_level,
            local_geometry_level,
            local_geometry_halflevel,
            turbconv_model,
        )
    end
    return nothing
end

"""
    set_diagnostic_edmf_precomputed_quantities_top_bc!(Y, p, t)

Updates the top boundary condition of precomputed quantities stored in `p` for diagnostic edmfx.
"""
NVTX.@annotate function set_diagnostic_edmf_precomputed_quantities_top_bc!(
    Y,
    p,
    t,
)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜentrʲs, ᶜdetrʲs, ᶜturb_entrʲs) = p.precomputed
    (; ᶠu³⁰, ᶠu³ʲs, ᶜuʲs, ᶠnh_pressure³_buoyʲs, ᶠnh_pressure³_dragʲs) =
        p.precomputed
    (; microphysics_model) = p.atmos

    # set values for the top level
    i_top = Spaces.nlevels(axes(Y.c))
    u³⁰_halflevel = Fields.field_values(Fields.level(ᶠu³⁰, i_top + half))
    @. u³⁰_halflevel = CT3(0)

    for j in 1:n
        ᶜuʲ = ᶜuʲs.:($j)
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        ᶠnh_pressure³_buoyʲ = ᶠnh_pressure³_buoyʲs.:($j)
        ᶠnh_pressure³_dragʲ = ᶠnh_pressure³_dragʲs.:($j)
        ᶜentrʲ = ᶜentrʲs.:($j)
        ᶜdetrʲ = ᶜdetrʲs.:($j)
        ᶜturb_entrʲ = ᶜturb_entrʲs.:($j)

        u³ʲ_halflevel = Fields.field_values(Fields.level(ᶠu³ʲ, i_top + half))
        @. u³ʲ_halflevel = CT3(0)
        nh_pressure³_buoyʲ_halflevel =
            Fields.field_values(Fields.level(ᶠnh_pressure³_buoyʲ, i_top - half))
        @. nh_pressure³_buoyʲ_halflevel = CT3(0)
        nh_pressure³_buoyʲ_halflevel =
            Fields.field_values(Fields.level(ᶠnh_pressure³_buoyʲ, i_top + half))
        @. nh_pressure³_buoyʲ_halflevel = CT3(0)
        nh_pressure³_dragʲ_halflevel =
            Fields.field_values(Fields.level(ᶠnh_pressure³_dragʲ, i_top - half))
        @. nh_pressure³_dragʲ_halflevel = CT3(0)
        nh_pressure³_dragʲ_halflevel =
            Fields.field_values(Fields.level(ᶠnh_pressure³_dragʲ, i_top + half))
        @. nh_pressure³_dragʲ_halflevel = CT3(0)

        entrʲ_level = Fields.field_values(Fields.level(ᶜentrʲ, i_top))
        detrʲ_level = Fields.field_values(Fields.level(ᶜdetrʲ, i_top))
        turb_entrʲ_level = Fields.field_values(Fields.level(ᶜturb_entrʲ, i_top))
        fill!(entrʲ_level, RecursiveApply.rzero(eltype(entrʲ_level)))
        fill!(detrʲ_level, RecursiveApply.rzero(eltype(detrʲ_level)))
        fill!(turb_entrʲ_level, RecursiveApply.rzero(eltype(turb_entrʲ_level)))
        @. ᶜuʲ = C123(Y.c.uₕ) + ᶜinterp(C123(ᶠu³ʲ))

        if microphysics_model isa EquilibriumMicrophysics0M
            ᶜmp_tendencyʲ = p.precomputed.ᶜmp_tendencyʲs.:($j)
            ᶜmp_tendencyʲ_level = Fields.field_values(Fields.level(ᶜmp_tendencyʲ, i_top))
            @. ᶜmp_tendencyʲ_level.dq_tot_dt = 0
            @. ᶜmp_tendencyʲ_level.e_tot_hlpr = 0
        end
        if microphysics_model isa NonEquilibriumMicrophysics1M
            ᶜmp_tendencyʲ = p.precomputed.ᶜmp_tendencyʲs.:($j)
            ᶜmp_tendencyʲ_level = Fields.field_values(Fields.level(ᶜmp_tendencyʲ, i_top))
            @. ᶜmp_tendencyʲ_level.dq_lcl_dt = 0
            @. ᶜmp_tendencyʲ_level.dq_icl_dt = 0
            @. ᶜmp_tendencyʲ_level.dq_rai_dt = 0
            @. ᶜmp_tendencyʲ_level.dq_sno_dt = 0
        end
    end
    return nothing
end

"""
    set_diagnostic_edmf_precomputed_quantities_env_closures!(Y, p, t)

Updates the environment closures in precomputed quantities stored in `p` for diagnostic edmfx.
"""
NVTX.@annotate function set_diagnostic_edmf_precomputed_quantities_env_closures!(
    Y,
    p,
    t,
)
    (; params) = p
    (; ᶜT, ᶜq_liq_rai, ᶜq_ice_sno, ᶠu³) = p.precomputed
    (; ustar) = p.precomputed.sfc_conditions
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
    (; ρtke_flux) = p.precomputed
    turbconv_params = CAP.turbconv_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    ᶜlg = Fields.local_geometry_field(Y.c)

    if p.atmos.turbconv_model isa DiagnosticEDMFX
        (; ᶠu³⁰, ᶜu⁰) = p.precomputed
        @. ᶜu⁰ = C123(Y.c.uₕ) + ᶜinterp(C123(ᶠu³⁰))
    end  # Set here, but used in a different function

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    @. ᶜlinear_buoygrad = buoyancy_gradients(
        BuoyGradMean(),
        thermo_params,
        ᶜT,
        Y.c.ρ,
        ᶜq_tot,
        ᶜq_liq_rai,
        ᶜq_ice_sno,
        p.precomputed.ᶜcloud_fraction,
        C3,
        p.precomputed.ᶜgradᵥ_q_tot,
        p.precomputed.ᶜgradᵥ_θ_liq_ice,
        ᶜlg,
    )

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
