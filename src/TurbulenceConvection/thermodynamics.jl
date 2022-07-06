function thermo_state_pθq(param_set::APS, p::FT, θ_liq_ice::FT, q_tot::FT) where {FT}
    # config = (50, 1e-3, RootSolvers.RegulaFalsiMethod)
    # config = (50, 1e-3, RootSolvers.NewtonMethodAD)
    config = ()
    thermo_params = TCP.thermodynamics_params(param_set)
    return TD.PhaseEquil_pθq(thermo_params, p, θ_liq_ice, q_tot, config...)
end
function thermo_state_pθq(param_set::APS, p::FT, θ_liq_ice::FT, q_tot::FT, q_liq::FT, q_ice::FT) where {FT}
    config = ()
    q = TD.PhasePartition(q_tot, q_liq, q_ice)
    thermo_params = TCP.thermodynamics_params(param_set)
    return TD.PhaseNonEquil_pθq(thermo_params, p, θ_liq_ice, q, config...)
end


function thermo_state_peq(param_set::APS, p::FT, e_int::FT, q_tot::FT) where {FT}
    # config = (50, 1e-3, RootSolvers.RegulaFalsiMethod)
    # config = (50, 1e-3, RootSolvers.NewtonMethodAD)
    config = ()
    thermo_params = TCP.thermodynamics_params(param_set)
    return TD.PhaseEquil_peq(thermo_params, p, e_int, q_tot, config...)
end

function thermo_state_peq(param_set::APS, p::FT, e_int::FT, q_tot::FT, q_liq::FT, q_ice::FT) where {FT}
    config = ()
    q = TD.PhasePartition(q_tot, q_liq, q_ice)
    thermo_params = TCP.thermodynamics_params(param_set)
    return TD.PhaseNonEquil_peq(thermo_params, p, e_int, q, config...)
end

function thermo_state_phq(param_set::APS, p::FT, h::FT, q_tot::FT) where {FT}
    config = ()
    thermo_params = TCP.thermodynamics_params(param_set)
    return TD.PhaseEquil_phq(thermo_params, p, h, q_tot, config...)
end

function thermo_state_phq(param_set::APS, p::FT, h::FT, q_tot::FT, q_liq::FT, q_ice::FT) where {FT}
    config = ()
    q = TD.PhasePartition(q_tot, q_liq, q_ice)
    thermo_params = TCP.thermodynamics_params(param_set)
    return TD.PhaseNonEquil_phq(thermo_params, p, h, q, config...)
end

function geopotential(param_set, z::Real)
    FT = eltype(param_set)
    grav = FT(TCP.grav(param_set))
    return grav * z
end

# TODO: move to Thermodynamics.jl
function total_enthalpy(param_set::APS, e_tot::FT, ts) where {FT}
    thermo_params = TCP.thermodynamics_params(param_set)
    Rm = TD.gas_constant_air(thermo_params, ts)
    T = TD.air_temperature(thermo_params, ts)
    return e_tot + Rm * T
end

function total_enthalpy(param_set::APS, e_tot::FT, p::FT, ρ::FT) where {FT}
    return e_tot + p / ρ
end

function enthalpy(h_tot::FT, e_kin::FT, e_pot::FT) where {FT}
    return h_tot - e_kin - e_pot
end

function enthalpy(mse::FT, e_pot::FT) where {FT}
    return mse - e_pot
end
