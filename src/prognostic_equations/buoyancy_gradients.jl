import Thermodynamics.Parameters as TDP
get_t_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.air_temperature(thermo_params, x.ts)
get_p(thermo_params, x::EnvBuoyGradVars) = TD.air_pressure(thermo_params, x.ts)
get_ρ(thermo_params, x::EnvBuoyGradVars) = TD.air_density(thermo_params, x.ts)
get_en_cld_frac(thermo_params, x::EnvBuoyGradVars) =
    ifelse(TD.has_condensate(thermo_params, x.ts), 1, 0)
get_θ_liq_ice_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.liquid_ice_pottemp(thermo_params, x.ts)
get_qt_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.total_specific_humidity(thermo_params, x.ts)
get_ql_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.liquid_specific_humidity(thermo_params, x.ts)
get_qi_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.ice_specific_humidity(thermo_params, x.ts)
get_qv_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.vapor_specific_humidity(thermo_params, x.ts)
get_θ_sat(thermo_params, x::EnvBuoyGradVars) =
    TD.dry_pottemp(thermo_params, x.ts)
get_∂θl∂z_sat(_, x::EnvBuoyGradVars) = x.∂θl∂z_sat
get_∂qt∂z_sat(_, x::EnvBuoyGradVars) = x.∂qt∂z_sat
get_∂θv∂z_unsat(_, x::EnvBuoyGradVars) = x.∂θv∂z_unsat

"""
    buoyancy_gradients(
        ::AbstractEnvBuoyGradClosure,
        thermo_params,
        moisture_model,
        bg_model::EnvBuoyGradVars,
    )

Returns the vertical buoyancy gradients in the environment, as well as in its dry and cloudy volume fractions.
The dispatch on EnvBuoyGradVars type is performed at the EnvBuoyGradVars construction time, and the analytical solutions
used here are consistent for both mean fields and conditional fields obtained from assumed distributions
over the conserved thermodynamic variables.
"""
function buoyancy_gradients end

function buoyancy_gradients(
    ebgc::AbstractEnvBuoyGradClosure,
    thermo_params,
    moisture_model,
    ts,
    ::Type{C3},
    ∂θv∂z_unsat,
    ∂qt∂z_sat,
    ∂θl∂z_sat,
    ᶜlg,
) where {C3}
    return buoyancy_gradients(
        ebgc,
        thermo_params,
        moisture_model,
        EnvBuoyGradVars(
            ts,
            projected_vector_buoy_grad_vars(
                C3,
                ∂θv∂z_unsat,
                ∂qt∂z_sat,
                ∂θl∂z_sat,
                ᶜlg,
            ),
        ),
    )
end

function buoyancy_gradients(
    ebgc::AbstractEnvBuoyGradClosure,
    thermo_params,
    moisture_model,
    bg_model::EnvBuoyGradVars,
)
    FT = eltype(bg_model)

    g = TDP.grav(thermo_params)
    molmass_ratio = TDP.molmass_ratio(thermo_params)
    R_d = TDP.R_d(thermo_params)
    R_v = TDP.R_v(thermo_params)

    phase_part = TD.PhasePartition(FT(0), FT(0), FT(0)) # assuming R_d = R_m
    p = get_p(thermo_params, bg_model)
    Π = TD.exner_given_pressure(thermo_params, p, phase_part)

    ∂b∂θv = g * (R_d * get_ρ(thermo_params, bg_model) / p) * Π
    θ_liq_ice_sat = get_θ_liq_ice_sat(thermo_params, bg_model)
    qt_sat = get_qt_sat(thermo_params, bg_model)

    if get_en_cld_frac(thermo_params, bg_model) > 0.0
        ts_sat = if moisture_model isa DryModel
            TD.PhaseDry_pθ(thermo_params, p, θ_liq_ice_sat)
        elseif moisture_model isa EquilMoistModel
            TD.PhaseEquil_pθq(thermo_params, p, θ_liq_ice_sat, qt_sat)
        elseif moisture_model isa NonEquilMoistModel
            TD.PhaseNonEquil_pθq(
                thermo_params,
                p,
                θ_liq_ice_sat,
                TD.PhasePartition(
                    qt_sat,
                    get_ql_sat(thermo_params, bg_model),
                    get_qi_sat(thermo_params, bg_model),
                ),
            )
        else
            error("Unsupported moisture model")
        end

        phase_part = TD.PhasePartition(thermo_params, ts_sat)
        lh = TD.latent_heat_liq_ice(thermo_params, phase_part)
        cp_m = TD.cp_m(thermo_params, ts_sat)
        t_sat = get_t_sat(thermo_params, bg_model)
        qv_sat = get_qv_sat(thermo_params, bg_model)
        # TODO - double check if this is assuming liquid only?
        ∂b∂θl_sat = (
            ∂b∂θv *
            (1 + molmass_ratio * (1 + lh / R_v / t_sat) * qv_sat - qt_sat) /
            (1 + lh * lh / cp_m / R_v / t_sat / t_sat * qv_sat)
        )
        ∂b∂qt_sat =
            (lh / cp_m / t_sat * ∂b∂θl_sat - ∂b∂θv) *
            get_θ_sat(thermo_params, bg_model)
    else
        ∂b∂θl_sat = FT(0)
        ∂b∂qt_sat = FT(0)
    end

    ∂b∂z = buoyancy_gradient_chain_rule(
        ebgc,
        bg_model,
        thermo_params,
        ∂b∂θv,
        ∂b∂θl_sat,
        ∂b∂qt_sat,
    )
    return ∂b∂z
end

"""
    buoyancy_gradient_chain_rule(
        ::AbstractEnvBuoyGradClosure,
        bg_model::EnvBuoyGradVars,
        thermo_params,
        ∂b∂θv,
        ∂b∂θl_sat,
        ∂b∂qt_sat,
    ) where

Returns the vertical buoyancy gradients in the environment, as well as in its dry and cloudy volume fractions,
from the partial derivatives with respect to thermodynamic variables in dry and cloudy volumes.
"""
function buoyancy_gradient_chain_rule(
    ::AbstractEnvBuoyGradClosure,
    bg_model::EnvBuoyGradVars,
    thermo_params,
    ∂b∂θv,
    ∂b∂θl_sat,
    ∂b∂qt_sat,
)
    FT = eltype(thermo_params)
    en_cld_frac = get_en_cld_frac(thermo_params, bg_model)
    if en_cld_frac > FT(0)
        ∂b∂z_θl_sat = ∂b∂θl_sat * get_∂θl∂z_sat(thermo_params, bg_model)
        ∂b∂z_qt_sat = ∂b∂qt_sat * get_∂qt∂z_sat(thermo_params, bg_model)
    else
        ∂b∂z_θl_sat = FT(0)
        ∂b∂z_qt_sat = FT(0)
    end

    ∂b∂z_unsat =
        en_cld_frac < FT(1) ? ∂b∂θv * get_∂θv∂z_unsat(thermo_params, bg_model) :
        FT(0)

    ∂b∂z_sat = ∂b∂z_θl_sat + ∂b∂z_qt_sat
    ∂b∂z = (1 - en_cld_frac) * ∂b∂z_unsat + en_cld_frac * ∂b∂z_sat

    return ∂b∂z
end
