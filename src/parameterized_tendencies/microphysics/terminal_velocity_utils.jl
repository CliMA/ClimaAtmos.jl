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
"""
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

# The 1M terminal velocity varies as q^(1/8), so ∂w/∂q ∝ q^(-7/8).
# As q goes to 0, it creates a large error.
# Evaluating w at max(q, RAIN_Q_FLOOR) makes ∂w/∂q zero, and removes the issue.
# 1e-8 corresponds to a minimum fall speed of ~0.8 m/s.
terminal_velocity(
    ::NonEquilibriumMicrophysics1M,
    ::DiagnosticTerminalVelocity,
    ::MatrixFields.FieldName{(:q_rai,)},
    params,
    cmc,
    cmp,
    ρ,
    q::FT,
) where {FT} = CM1.terminal_velocity(
    cmp.precip.rain,
    cmp.terminal_velocity.rain,
    ρ,
    max(q, FT(1e-8)),
)

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
        ::NonEquilibriumMicrophysics1M, tv_mode, var_name, params, ρwχ, ρχ,
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
  - `params`: Cloud microphysics parameters
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
