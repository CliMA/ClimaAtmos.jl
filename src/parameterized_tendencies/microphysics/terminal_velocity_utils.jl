"""
    AbstractTerminalVelocityMode

Strategy for setting the sedimentation velocity of the water species.

Subtypes:

  - [`DiagnosticTerminalVelocity`](@ref): velocity diagnosed by CloudMicrophysics
    from the local density and specific humidity.
  - [`FixedTerminalVelocity`](@ref): prescribed constant velocity per species.
"""
abstract type AbstractTerminalVelocityMode end

"""
    DiagnosticTerminalVelocity <: AbstractTerminalVelocityMode

Diagnose the mass-weighted terminal velocity of each species from the local state
using the CloudMicrophysics size distributions.
"""
struct DiagnosticTerminalVelocity <: AbstractTerminalVelocityMode end

"""
    FixedTerminalVelocity{FT} <: AbstractTerminalVelocityMode

Prescribed, state-independent terminal velocity for each 1-moment species, used in
idealized tests where the sedimentation rate is to be controlled directly.

Sign convention: the stored values are downward fall speeds and must be
non-negative.

# Fields

  - `liquid`: Cloud liquid fall speed [m/s].
  - `ice`: Cloud ice fall speed [m/s].
  - `rain`: Rain fall speed [m/s].
  - `snow`: Snow fall speed [m/s].
"""
struct FixedTerminalVelocity{FT} <: AbstractTerminalVelocityMode
    liquid::FT
    ice::FT
    rain::FT
    snow::FT
    #TODO fixed-velocity sedimentation of 2M/P3 tracers not implemented
end
Base.broadcastable(x::AbstractTerminalVelocityMode) = tuple(x)

# Liquid, 1M
"""
    terminal_velocity(microphysics_model, tv_mode, var_name, cmc, cmp, ρ, q)

Return the terminal velocity of the 1-moment species selected by `var_name`.

Dispatches on the `MatrixFields.FieldName` `var_name`, one of `@name(q_lcl)`,
`@name(q_icl)`, `@name(q_rai)`, `@name(q_sno)`, and on `tv_mode`: a
[`FixedTerminalVelocity`](@ref) returns the corresponding prescribed constant and
ignores the state, while [`DiagnosticTerminalVelocity`](@ref) calls the matching
CloudMicrophysics closure (Stokes for cloud liquid, Chen 2022 small-ice for cloud
ice, the 1M rain and snow distributions for precipitation).

# Arguments

  - `microphysics_model`: Microphysics model; only `NonEquilibriumMicrophysics1M`
    is implemented.
  - `tv_mode`: Terminal velocity mode, dispatched on.
  - `var_name`: `MatrixFields.FieldName` naming the species.
  - `cmc`, `cmp`: Cloud condensate and precipitation parameter sets.
  - `ρ`: Air density [kg/m³].
  - `q`: Specific humidity of the species [kg/kg].

# Returns

Terminal velocity [m/s], positive downward.
"""
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
        ::NonEquilibriumMicrophysics1M, tv_mode, var_name, ρwχ, ρχ,
    )

Return the grid-scale terminal velocity of a 1-moment species from subdomain fluxes.

With [`FixedTerminalVelocity`](@ref) the prescribed constant is returned and the
flux arguments are ignored. With [`DiagnosticTerminalVelocity`](@ref) the
mass-weighted average `ρwχ / ρχ` is formed, so that the grid-scale flux equals the
sum of the subdomain fluxes. Two guards protect that division: the result is zero
where `ρχ ≤ ϵ_numerics(FT)`, and negatives from numerical error are clipped to
zero, keeping the grid-scale value bracketed by the subdomain velocities.

# Arguments

  - `tv_mode`: Terminal velocity mode, dispatched on.
  - `var_name`: `MatrixFields.FieldName` naming the species.
  - `ρwχ`: Area-weighted sum of subdomain sedimentation fluxes [kg/m²/s].
  - `ρχ`: Area-weighted sum of subdomain species masses [kg/m³].

# Returns

Terminal velocity [m/s], non-negative and positive downward.

Called from `set_precipitation_velocities!`.
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
