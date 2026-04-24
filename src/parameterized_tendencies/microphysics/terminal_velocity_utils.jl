abstract type AbstractTerminalVelocityMode end
struct DiagnosticTerminalVelocity <: AbstractTerminalVelocityMode end
struct FixedTerminalVelocity{FT} <: AbstractTerminalVelocityMode
    liquid::FT
    ice::FT
    rain::FT
    snow::FT
    #TODO fixed-velocity sedimentation of 2M/P3 tracers not implemented 
end
Base.broadcastable(x::AbstractTerminalVelocityMode) = tuple(x)

# Liquid, 1M
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    mode::FixedTerminalVelocity,
    ::MatrixFields.FieldName{(:q_lcl,)},
    args...,
) = mode.liquid

terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_lcl,)},
    cmc,
    cmp,
    ρ,
    q,
) = CMNe.terminal_velocity(cmc.liquid, cmc.stokes, ρ, q)

# Ice, 1M
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    mode::FixedTerminalVelocity,
    ::MatrixFields.FieldName{(:q_icl,)},
    args...,
) = mode.ice

terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_icl,)},
    cmc,
    cmp,
    ρ,
    q,
) = CMNe.terminal_velocity(cmc.ice, cmc.Ch2022.small_ice, ρ, q)

# Rain, 1M
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    mode::FixedTerminalVelocity,
    ::MatrixFields.FieldName{(:q_rai,)},
    args...,
) = mode.rain

terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_rai,)},
    cmc,
    cmp,
    ρ,
    q,
) = CM1.terminal_velocity(cmp.precip.rain, cmp.terminal_velocity.rain, ρ, q)

# Snow, 1M
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    mode::FixedTerminalVelocity,
    ::MatrixFields.FieldName{(:q_sno,)},
    args...,
) = mode.snow

terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_sno,)},
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
        ρwχ,
        ρχ::FT,
    )

Return the grid-scale terminal velocity.

- For `FixedTerminalVelocity`, returns the prescribed constant value.
- For `DiagnosticTerminalVelocity`, returns the mass-weighted velocity
  `ρwχ / ρχ`.

In the diagnostic case, the result is clamped to avoid spurious negative
velocities arising from numerical errors when `ρχ` is small.
"""
gs_terminal_velocity(
    cm_1m::NonEquilibriumMicrophysics1M,
    tv_mode::FixedTerminalVelocity,
    var_name,
    args...,
) = terminal_velocity(cm_1m, tv_mode, var_name, args...)

gs_terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    var_name,
    ρwχ,
    ρχ::FT,
) where {FT} = ifelse(ρχ > ϵ_numerics(FT), max(ρwχ / ρχ, zero(ρχ)), zero(ρχ))
