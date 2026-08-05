abstract type AbstractTerminalVelocityMode end
struct DiagnosticTerminalVelocity <: AbstractTerminalVelocityMode end
struct FixedTerminalVelocity <: AbstractTerminalVelocityMode end
#TODO fixed-velocity sedimentation of 2M/P3 tracers not implemented
Base.broadcastable(x::AbstractTerminalVelocityMode) = tuple(x)

"""
    velocity_mode(atmos, name)

Select the per-species terminal velocity mode from `atmos` based on the tracer
`name` (a `MatrixFields.FieldName`).

Each hydrometeor species has its own toggle in `AtmosWater`, so e.g. rain can
use `DiagnosticTerminalVelocity` while the others remain
`FixedTerminalVelocity`.
"""
velocity_mode(atmos, ::MatrixFields.FieldName{(:q_lcl,)}) =
    atmos.terminal_velocity_liquid
velocity_mode(atmos, ::MatrixFields.FieldName{(:q_icl,)}) =
    atmos.terminal_velocity_ice
velocity_mode(atmos, ::MatrixFields.FieldName{(:q_rai,)}) =
    atmos.terminal_velocity_rain
velocity_mode(atmos, ::MatrixFields.FieldName{(:q_sno,)}) =
    atmos.terminal_velocity_snow

# Liquid, 1M
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::FixedTerminalVelocity,
    ::MatrixFields.FieldName{(:q_lcl,)},
    params,
    args...,
) = CAP.fixed_cloud_liquid_terminal_velocity(params)

terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_lcl,)},
    params,
    cmc,
    cmp,
    ρ,
    q,
) = CMNe.terminal_velocity(cmc.liquid, cmc.stokes, ρ, q)

# Ice, 1M
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::FixedTerminalVelocity,
    ::MatrixFields.FieldName{(:q_icl,)},
    params,
    args...,
) = CAP.fixed_cloud_ice_terminal_velocity(params)

terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_icl,)},
    params,
    cmc,
    cmp,
    ρ,
    q,
) = CMNe.terminal_velocity(cmc.ice, cmc.Ch2022.small_ice, ρ, q)

# Rain, 1M
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::FixedTerminalVelocity,
    ::MatrixFields.FieldName{(:q_rai,)},
    params,
    args...,
) = CAP.fixed_rain_terminal_velocity(params)

terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_rai,)},
    params,
    cmc,
    cmp,
    ρ,
    q,
) = CM1.terminal_velocity(cmp.precip.rain, cmp.terminal_velocity.rain, ρ, q)

# Snow, 1M
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::FixedTerminalVelocity,
    ::MatrixFields.FieldName{(:q_sno,)},
    params,
    args...,
) = CAP.fixed_snow_terminal_velocity(params)

terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_sno,)},
    params,
    cmc,
    cmp,
    ρ,
    q,
) = CM1.terminal_velocity(cmp.precip.snow, cmp.terminal_velocity.snow, ρ, q)

"""
    gs_terminal_velocity(
        ::NonEquilibriumMicrophysics1M,
        ::AbstractTerminalVelocityMode,
        var_name,
        params,
        ρwχ,
        ρχ,
    )

Return the grid-scale terminal velocity.

  - For `FixedTerminalVelocity`, returns the prescribed constant value
    from `params`.
  - For `DiagnosticTerminalVelocity`, returns the mass-weighted velocity
    `ρwχ / ρχ`.

In the diagnostic case, the result is clamped to avoid spurious negative
velocities arising from numerical errors when `ρχ` is small.
"""
gs_terminal_velocity(
    cm_1m::NonEquilibriumMicrophysics1M,
    tv_mode::FixedTerminalVelocity,
    var_name,
    params,
    args...,
) = terminal_velocity(cm_1m, tv_mode, var_name, params)

gs_terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    var_name,
    params,
    ρwχ,
    ρχ::FT,
) where {FT} = ifelse(ρχ > ϵ_numerics(FT), max(ρwχ / ρχ, zero(ρχ)), zero(ρχ))
