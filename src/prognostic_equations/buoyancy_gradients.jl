"""
    buoyancy_gradients(
        params,
        moisture_model,
        bg_model::EnvBuoyGrad{FT, EBG}
    ) where {FT <: Real, EBG <: AbstractEnvBuoyGradClosure}

Returns the vertical buoyancy gradients in the environment, as well as in its dry and cloudy volume fractions.
The dispatch on EnvBuoyGrad type is performed at the EnvBuoyGrad construction time, and the analytical solutions
used here are consistent for both mean fields and conditional fields obtained from assumed distributions
over the conserved thermodynamic variables.
"""
function buoyancy_gradients(
    params,
    moisture_model,
    bg_model::EnvBuoyGrad{FT, EBG},
) where {FT <: Real, EBG <: AbstractEnvBuoyGradClosure}

    thermo_params = CAP.thermodynamics_params(params)
    g = CAP.grav(params)
    molmass_ratio = CAP.molmass_ratio(params)
    R_d = CAP.R_d(params)
    R_v = CAP.R_v(params)

    phase_part = TD.PhasePartition(FT(0), FT(0), FT(0)) # assuming R_d = R_m
    Π = TD.exner_given_pressure(thermo_params, bg_model.p, phase_part)

    ∂b∂θv = g * (R_d * bg_model.ρ / bg_model.p) * Π

    if bg_model.en_cld_frac > 0.0
        ts_sat = if moisture_model isa DryModel
            TD.PhaseDry_pθ(thermo_params, bg_model.p, bg_model.θ_liq_ice_sat)
        elseif moisture_model isa EquilMoistModel
            TD.PhaseEquil_pθq(
                thermo_params,
                bg_model.p,
                bg_model.θ_liq_ice_sat,
                bg_model.qt_sat,
            )
        elseif moisture_model isa NonEquilMoistModel
            TD.PhaseNonEquil_pθq(
                thermo_params,
                bg_model.p,
                bg_model.θ_liq_ice_sat,
                TD.PhasePartition(
                    bg_model.qt_sat,
                    bg_model.ql_sat,
                    bg_model.qi_sat,
                ),
            )
        else
            error("Unsupported moisture model")
        end

        phase_part =
            if moisture_model isa DryModel || moisture_model isa EquilMoistModel
                TD.PhasePartition(thermo_params, ts_sat)
            elseif moisture_model isa NonEquilMoistModel
                TD.PhasePartition(
                    bg_model.qt_sat,
                    bg_model.ql_sat,
                    bg_model.qi_sat,
                )
            end

        lh = TD.latent_heat_liq_ice(thermo_params, phase_part)
        cp_m = TD.cp_m(thermo_params, ts_sat)
        # TODO - double check if this is assuming liquid only?
        ∂b∂θl_sat = (
            ∂b∂θv * (
                1 +
                molmass_ratio *
                (1 + lh / R_v / bg_model.t_sat) *
                bg_model.qv_sat - bg_model.qt_sat
            ) / (
                1 +
                lh * lh / cp_m / R_v / bg_model.t_sat / bg_model.t_sat *
                bg_model.qv_sat
            )
        )
        ∂b∂qt_sat =
            (lh / cp_m / bg_model.t_sat * ∂b∂θl_sat - ∂b∂θv) * bg_model.θ_sat
    else
        ∂b∂θl_sat = FT(0)
        ∂b∂qt_sat = FT(0)
    end

    ∂b∂z = buoyancy_gradient_chain_rule(bg_model, ∂b∂θv, ∂b∂θl_sat, ∂b∂qt_sat)
    return ∂b∂z
end

"""
    buoyancy_gradient_chain_rule(
        bg_model::EnvBuoyGrad{FT, EBG},
        ∂b∂θv::FT,
        ∂b∂θl_sat::FT,
        ∂b∂qt_sat::FT,
    ) where {FT <: Real, EBG <: AbstractEnvBuoyGradClosure}

Returns the vertical buoyancy gradients in the environment, as well as in its dry and cloudy volume fractions,
from the partial derivatives with respect to thermodynamic variables in dry and cloudy volumes.
"""
function buoyancy_gradient_chain_rule(
    bg_model::EnvBuoyGrad{FT, EBG},
    ∂b∂θv::FT,
    ∂b∂θl_sat::FT,
    ∂b∂qt_sat::FT,
) where {FT <: Real, EBG <: AbstractEnvBuoyGradClosure}
    if bg_model.en_cld_frac > FT(0)
        ∂b∂z_θl_sat = ∂b∂θl_sat * bg_model.∂θl∂z_sat
        ∂b∂z_qt_sat = ∂b∂qt_sat * bg_model.∂qt∂z_sat
    else
        ∂b∂z_θl_sat = FT(0)
        ∂b∂z_qt_sat = FT(0)
    end

    ∂b∂z_unsat =
        bg_model.en_cld_frac < FT(1) ? ∂b∂θv * bg_model.∂θv∂z_unsat : FT(0)

    ∂b∂z_sat = ∂b∂z_θl_sat + ∂b∂z_qt_sat
    ∂b∂z =
        (1 - bg_model.en_cld_frac) * ∂b∂z_unsat +
        bg_model.en_cld_frac * ∂b∂z_sat

    return ∂b∂z
end
