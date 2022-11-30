function thermo_state_pθq(
    param_set::APS,
    p::FT,
    θ_liq_ice::FT,
    q_tot::FT,
) where {FT}
    # config = (50, 1e-3, RootSolvers.RegulaFalsiMethod)
    # config = (50, 1e-3, RootSolvers.NewtonMethodAD)
    config = ()
    thermo_params = TCP.thermodynamics_params(param_set)
    return TD.PhaseEquil_pθq(thermo_params, p, θ_liq_ice, q_tot, config...)
end
function thermo_state_pθq(
    param_set::APS,
    p::FT,
    θ_liq_ice::FT,
    q_tot::FT,
    q_liq::FT,
    q_ice::FT,
) where {FT}
    config = ()
    q = TD.PhasePartition(q_tot, q_liq, q_ice)
    thermo_params = TCP.thermodynamics_params(param_set)
    return TD.PhaseNonEquil_pθq(thermo_params, p, θ_liq_ice, q, config...)
end

geopotential(thermo_params::TD.Parameters.ThermodynamicsParameters, z::Real) =
    z * TD.Parameters.grav(thermo_params)

function enthalpy(h_tot::FT, e_kin::FT, e_pot::FT) where {FT}
    return h_tot - e_kin - e_pot
end

function enthalpy(mse::FT, e_pot::FT) where {FT}
    return mse - e_pot
end
