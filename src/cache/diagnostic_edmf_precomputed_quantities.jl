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
    ts_level,
    ρ_level,
    mse_level,
    q_tot_level,
    p_level,
    Φ_level,
)
    FT = eltype(thermo_params)
    @. ts_level = TD.PhaseEquil_phq(
        thermo_params,
        p_level,
        mse_level - Φ_level,
        q_tot_level,
        8,
        FT(0.0003),
    )
    @. ρ_level = TD.air_density(thermo_params, ts_level)
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
    (; turbconv_model) = p.atmos
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜΦ) = p.core
    (; ᶜp, ᶠu³, ᶜK) = p.precomputed
    (; ustar, obukhov_length, buoyancy_flux, ρ_flux_h_tot, ρ_flux_q_tot) =
        p.precomputed.sfc_conditions
    (; ᶜρaʲs, ᶠu³ʲs, ᶜKʲs, ᶜmseʲs, ᶜq_totʲs, ᶜtsʲs, ᶜρʲs) = p.precomputed
    (; ᶠu³⁰, ᶜK⁰) = p.precomputed

    (; params) = p

    thermo_params = CAP.thermodynamics_params(params)
    turbconv_params = CAP.turbconv_params(params)
    ᶜts = p.precomputed.ᶜts   #TODO replace

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    ᶜe_tot = @. lazy(specific(Y.c.ρe_tot, Y.c.ρ))
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜe_tot))

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

    # boundary condition
    for j in 1:n
        ᶜρaʲ = ᶜρaʲs.:($j)
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        ᶜKʲ = ᶜKʲs.:($j)
        ᶜmseʲ = ᶜmseʲs.:($j)
        ᶜtsʲ = ᶜtsʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜq_totʲ = ᶜq_totʲs.:($j)

        ρaʲ_int_level = Fields.field_values(Fields.level(ᶜρaʲ, 1))
        u³ʲ_int_halflevel = Fields.field_values(Fields.level(ᶠu³ʲ, half))
        Kʲ_int_level = Fields.field_values(Fields.level(ᶜKʲ, 1))
        mseʲ_int_level = Fields.field_values(Fields.level(ᶜmseʲ, 1))
        q_totʲ_int_level = Fields.field_values(Fields.level(ᶜq_totʲ, 1))
        tsʲ_int_level = Fields.field_values(Fields.level(ᶜtsʲ, 1))
        ρʲ_int_level = Fields.field_values(Fields.level(ᶜρʲ, 1))

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
        set_diagnostic_edmfx_draft_quantities_level!(
            thermo_params,
            tsʲ_int_level,
            ρʲ_int_level,
            mseʲ_int_level,
            q_totʲ_int_level,
            p_int_level,
            Φ_int_level,
        )
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
    dt = float(dt)
    (; ᶜΦ, ᶜgradᵥ_ᶠΦ) = p.core
    (; ᶜp, ᶠu³, ᶜts, ᶜK) = p.precomputed
    (;
        ᶜρaʲs,
        ᶠu³ʲs,
        ᶜKʲs,
        ᶜmseʲs,
        ᶜq_totʲs,
        ᶜtsʲs,
        ᶜρʲs,
        ᶜentrʲs,
        ᶜdetrʲs,
        ᶜturb_entrʲs,
        ᶠnh_pressure³_buoyʲs,
        ᶠnh_pressure³_dragʲs,
    ) = p.precomputed
    (; ᶠu³⁰, ᶜK⁰) = p.precomputed



    if microphysics_model isa Microphysics1Moment
        q_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
        q_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
    end

    thermo_params = CAP.thermodynamics_params(params)
    microphys_0m_params = CAP.microphysics_0m_params(params)
    microphys_1m_params = CAP.microphysics_1m_params(params)
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
    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(
            thermo_params,
            ᶜts,
            specific(Y.c.ρe_tot, Y.c.ρ),
        ),
    )

    ᶜtke⁰ = @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, Y.c.ρ, turbconv_model))

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
        ts_prev_level = Fields.field_values(Fields.level(ᶜts, i - 1))
        p_prev_level = Fields.field_values(Fields.level(ᶜp, i - 1))
        z_prev_level = Fields.field_values(Fields.level(ᶜz, i - 1))
        dz_prev_level = Fields.field_values(Fields.level(ᶜdz, i - 1))

        if microphysics_model isa Microphysics1Moment
            q_rai_prev_level = Fields.field_values(Fields.level(q_rai, i - 1))
            q_sno_prev_level = Fields.field_values(Fields.level(q_sno, i - 1))
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
            ᶜtsʲ = ᶜtsʲs.:($j)
            ᶜρʲ = ᶜρʲs.:($j)
            ᶜq_totʲ = ᶜq_totʲs.:($j)
            ᶜentrʲ = ᶜentrʲs.:($j)
            ᶜdetrʲ = ᶜdetrʲs.:($j)
            ᶜturb_entrʲ = ᶜturb_entrʲs.:($j)
            ᶠnh_pressure³_buoyʲ = ᶠnh_pressure³_buoyʲs.:($j)
            ᶠnh_pressure³_dragʲ = ᶠnh_pressure³_dragʲs.:($j)

            if microphysics_model isa
               Union{Microphysics0Moment, Microphysics1Moment}
                ᶜS_q_totʲ = p.precomputed.ᶜSqₜᵖʲs.:($j)
            end
            if microphysics_model isa Microphysics1Moment
                ᶜS_q_raiʲ = p.precomputed.ᶜSqᵣᵖʲs.:($j)
                ᶜS_q_snoʲ = p.precomputed.ᶜSqₛᵖʲs.:($j)
                ᶜS_e_totʲ = p.precomputed.ᶜSeₜᵖʲs.:($j)
                ᶜSᵖ = p.scratch.ᶜtemp_scalar
                ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2
                ᶜq_liqʲ = ᶜq_liqʲs.:($j)
                ᶜq_iceʲ = ᶜq_iceʲs.:($j)
            end

            ρaʲ_level = Fields.field_values(Fields.level(ᶜρaʲ, i))
            u³ʲ_halflevel = Fields.field_values(Fields.level(ᶠu³ʲ, i - half))
            Kʲ_level = Fields.field_values(Fields.level(ᶜKʲ, i))
            mseʲ_level = Fields.field_values(Fields.level(ᶜmseʲ, i))
            q_totʲ_level = Fields.field_values(Fields.level(ᶜq_totʲ, i))
            tsʲ_level = Fields.field_values(Fields.level(ᶜtsʲ, i))
            ρʲ_level = Fields.field_values(Fields.level(ᶜρʲ, i))

            ρaʲ_prev_level = Fields.field_values(Fields.level(ᶜρaʲ, i - 1))
            u³ʲ_prev_halflevel =
                Fields.field_values(Fields.level(ᶠu³ʲ, i - 1 - half))
            u³ʲ_data_prev_halflevel = u³ʲ_prev_halflevel.components.data.:1
            mseʲ_prev_level = Fields.field_values(Fields.level(ᶜmseʲ, i - 1))
            q_totʲ_prev_level =
                Fields.field_values(Fields.level(ᶜq_totʲ, i - 1))
            ρʲ_prev_level = Fields.field_values(Fields.level(ᶜρʲ, i - 1))
            ᶜgradᵥ_ᶠΦ_prev_level =
                Fields.field_values(Fields.level(ᶜgradᵥ_ᶠΦ, i - 1))
            tsʲ_prev_level = Fields.field_values(Fields.level(ᶜtsʲ, i - 1))
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

            S_q_totʲ_prev_level =
                if microphysics_model isa
                   Union{Microphysics0Moment, Microphysics1Moment}
                    Fields.field_values(Fields.level(ᶜS_q_totʲ, i - 1))
                else
                    Ref(nothing)
                end
            if microphysics_model isa Microphysics1Moment
                S_q_raiʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜS_q_raiʲ, i - 1))
                S_q_snoʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜS_q_snoʲ, i - 1))
                S_e_totʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜS_e_totʲ, i - 1))
                Sᵖ_prev_level = Fields.field_values(Fields.level(ᶜSᵖ, i - 1))
                Sᵖ_snow_prev_level =
                    Fields.field_values(Fields.level(ᶜSᵖ_snow, i - 1))
                q_liqʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜq_liqʲ, i - 1))
                q_iceʲ_prev_level =
                    Fields.field_values(Fields.level(ᶜq_iceʲ, i - 1))
            end

            tke_prev_level = Fields.field_values(Fields.level(ᶜtke⁰, i - 1))

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
                TD.relative_humidity(thermo_params, tsʲ_prev_level),
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
                TD.relative_humidity(thermo_params, ts_prev_level),
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

            # Updraft q_tot sources from precipitation formation
            # To be applied in updraft continuity, moisture and energy
            # for updrafts and grid mean
            if microphysics_model isa Microphysics0Moment
                @. S_q_totʲ_prev_level = q_tot_0M_precipitation_sources(
                    thermo_params,
                    microphys_0m_params,
                    dt,
                    q_totʲ_prev_level,
                    tsʲ_prev_level,
                )
            elseif microphysics_model isa Microphysics1Moment
                compute_precipitation_sources!(
                    Sᵖ_prev_level,
                    Sᵖ_snow_prev_level,
                    S_q_totʲ_prev_level,
                    S_q_raiʲ_prev_level,
                    S_q_snoʲ_prev_level,
                    S_e_totʲ_prev_level,
                    ρʲ_prev_level,
                    q_totʲ_prev_level,
                    q_liqʲ_prev_level,
                    q_iceʲ_prev_level,
                    q_rai_prev_level,
                    q_sno_prev_level,
                    tsʲ_prev_level,
                    Φ_prev_level,
                    dt,
                    microphys_1m_params,
                    thermo_params,
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
                CT3(sqrt(max(0, u³ʲ_datau³ʲ_data))),
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

            @. detrʲ_prev_level = detrainment_from_thermo_state(
                thermo_params,
                turbconv_params,
                z_prev_level,
                z_sfc_halflevel,
                p_prev_level,
                ρ_prev_level,
                ρaʲ_prev_level,
                tsʲ_prev_level,
                ρʲ_prev_level,
                u³ʲ_prev_halflevel,
                local_geometry_prev_halflevel,
                u³_prev_halflevel,
                ts_prev_level,
                FT(0),
                entrʲ_prev_level,
                vert_div_level,
                FT(0), # mass flux divergence is not implemented for diagnostic edmf
                w_vert_div_level,
                tke_prev_level,
                ᶜgradᵥ_ᶠΦ_prev_level,
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
            if microphysics_model isa Microphysics0Moment
                @. ρaʲu³ʲ_data += microphysics_sources(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    S_q_totʲ_prev_level,
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
            if microphysics_model isa Microphysics0Moment
                @. ρaʲu³ʲ_datamse += microphysics_sources(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    S_q_totʲ_prev_level *
                    e_tot_0M_precipitation_sources_helper(
                        thermo_params,
                        tsʲ_prev_level,
                        Φ_prev_level,
                    ),
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
            if microphysics_model isa Microphysics0Moment
                @. ρaʲu³ʲ_dataq_tot += microphysics_sources(
                    local_geometry_halflevel.J,
                    local_geometry_prev_level.J,
                    ρaʲ_prev_level,
                    S_q_totʲ_prev_level,
                )
            end
            @. q_totʲ_level = ifelse(
                kill_updraft,
                q_tot_level,
                ρaʲu³ʲ_dataq_tot / ρaʲu³ʲ_data,
            )

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
            end

            @. Kʲ_level = kinetic_energy(
                uₕ_level,
                u³ʲ_halflevel,
                local_geometry_level,
                local_geometry_halflevel,
            )
            set_diagnostic_edmfx_draft_quantities_level!(
                thermo_params,
                tsʲ_level,
                ρʲ_level,
                mseʲ_level,
                q_totʲ_level,
                p_level,
                Φ_level,
            )
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

        if microphysics_model isa
           Union{Microphysics0Moment, Microphysics1Moment}
            ᶜS_q_totʲ = p.precomputed.ᶜSqₜᵖʲs.:($j)
            S_q_totʲ_level = Fields.field_values(Fields.level(ᶜS_q_totʲ, i_top))
            @. S_q_totʲ_level = 0
        end
        if microphysics_model isa Microphysics1Moment
            ᶜS_q_raiʲ = p.precomputed.ᶜSqᵣᵖʲs.:($j)
            ᶜS_q_snoʲ = p.precomputed.ᶜSqₛᵖʲs.:($j)
            ᶜS_e_totʲ = p.precomputed.ᶜSeₜᵖʲs.:($j)
            S_q_raiʲ_level = Fields.field_values(Fields.level(ᶜS_q_raiʲ, i_top))
            S_q_snoʲ_level = Fields.field_values(Fields.level(ᶜS_q_snoʲ, i_top))
            S_e_totʲ_level = Fields.field_values(Fields.level(ᶜS_e_totʲ, i_top))
            @. S_q_raiʲ_level = 0
            @. S_q_snoʲ_level = 0
            @. S_e_totʲ_level = 0
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
    (; moisture_model, turbconv_model, microphysics_model) = p.atmos
    n = n_mass_flux_subdomains(turbconv_model)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜdz = Fields.Δz_field(axes(Y.c))
    (; params) = p
    (; dt) = p
    (; ᶜp, ᶜu, ᶜts) = p.precomputed
    (; ustar, obukhov_length) = p.precomputed.sfc_conditions
    (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
    (; ρatke_flux) = p.precomputed
    turbconv_params = CAP.turbconv_params(params)
    thermo_params = CAP.thermodynamics_params(params)
    ᶜlg = Fields.local_geometry_field(Y.c)

    if p.atmos.turbconv_model isa DiagnosticEDMFX
        (; ᶠu³⁰, ᶜu⁰) = p.precomputed
    elseif p.atmos.turbconv_model isa EDOnlyEDMFX
        ᶠu³⁰ = p.precomputed.ᶠu³
        ᶜu⁰ = ᶜu
    end
    @. ᶜu⁰ = C123(Y.c.uₕ) + ᶜinterp(C123(ᶠu³⁰))  # Set here, but used in a different function

    @. ᶜlinear_buoygrad = buoyancy_gradients(
        BuoyGradMean(),
        thermo_params,
        moisture_model,
        ᶜts,
        C3,
        p.precomputed.ᶜgradᵥ_θ_virt,    # ∂θv∂z_unsat
        p.precomputed.ᶜgradᵥ_q_tot,     # ∂qt∂z_sat
        p.precomputed.ᶜgradᵥ_θ_liq_ice, # ∂θl∂z_sat
        ᶜlg,
    )

    # TODO: Currently the shear production only includes vertical gradients
    ᶠu⁰ = p.scratch.ᶠtemp_C123
    @. ᶠu⁰ = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³⁰)
    ᶜstrain_rate = p.scratch.ᶜtemp_UVWxUVW
    ᶜstrain_rate .= compute_strain_rate_center(ᶠu⁰)
    @. ᶜstrain_rate_norm = norm_sqr(ᶜstrain_rate)

    ρatke_flux_values = Fields.field_values(ρatke_flux)
    ρ_int_values = Fields.field_values(Fields.level(Y.c.ρ, 1))
    u_int_values = Fields.field_values(Fields.level(ᶜu, 1))
    ustar_values = Fields.field_values(ustar)
    int_local_geometry_values =
        Fields.field_values(Fields.level(Fields.local_geometry_field(Y.c), 1))
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), half),
    )
    @. ρatke_flux_values = surface_flux_tke(
        turbconv_params,
        ρ_int_values,
        ustar_values,
        sfc_local_geometry_values,
    )
    return nothing
end

"""
    set_diagnostic_edmf_precomputed_quantities_env_precip!(Y, p, t)

Updates the precipitation sources in precomputed quantities stored in `p` for diagnostic edmfx.
"""
NVTX.@annotate function set_diagnostic_edmf_precomputed_quantities_env_precipitation!(
    Y,
    p,
    t,
    ::NoPrecipitation,
)
    return nothing
end
NVTX.@annotate function set_diagnostic_edmf_precomputed_quantities_env_precipitation!(
    Y,
    p,
    t,
    microphysics_model::Microphysics0Moment,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    microphys_0m_params = CAP.microphysics_0m_params(p.params)
    (; dt) = p
    (; ᶜts, ᶜSqₜᵖ⁰) = p.precomputed

    # Environment precipitation sources (to be applied to grid mean)
    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    @. ᶜSqₜᵖ⁰ = q_tot_0M_precipitation_sources(
        thermo_params,
        microphys_0m_params,
        dt,
        ᶜq_tot,
        ᶜts,
    )
    return nothing
end
NVTX.@annotate function set_diagnostic_edmf_precomputed_quantities_env_precipitation!(
    Y,
    p,
    t,
    microphysics_model::Microphysics1Moment,
)
    error("Not implemented yet")
    #thermo_params = CAP.thermodynamics_params(p.params)
    #microphys_1m_params = CAP.microphysics_1m_params(p.params)

    #(; ᶜts, ᶜSqₜᵖ⁰, ᶜSeₜᵖ⁰, ᶜSqᵣᵖ⁰, ᶜSqₛᵖ⁰) = p.precomputed
    #(; ᶜqᵣ, ᶜqₛ) = p.precomputed

    #ᶜSᵖ = p.scratch.ᶜtemp_scalar
    #ᶜSᵖ_snow = p.scratch.ᶜtemp_scalar_2

    ## Environment precipitation sources (to be applied to grid mean)
    #compute_precipitation_sources!(
    #    ᶜSᵖ,
    #    ᶜSᵖ_snow,
    #    ᶜSqₜᵖ⁰,
    #    ᶜSqᵣᵖ⁰,
    #    ᶜSqₛᵖ⁰,
    #    ᶜSeₜᵖ⁰,
    #    Y.c.ρ,
    #    ᶜqᵣ,
    #    ᶜqₛ,
    #    ᶜts,
    #    p.core.ᶜΦ,
    #    p.dt,
    #    microphys_1m_params,
    #    thermo_params,
    #)
    return nothing
end
