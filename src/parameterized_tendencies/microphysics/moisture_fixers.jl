# ============================================================================
# Moisture Fixers for Microphysics
# ============================================================================
# Functions for correcting negative moisture values and ensuring physical
# consistency of water species.

import ClimaCore.MatrixFields as MF

"""
    tracer_nonnegativity_vapor_tendency(q, q_vap, dt)

Compute the tendency that restores a negative tracer to zero by borrowing from vapor.

A tracer driven negative by numerical error is restored over one timestep, at rate
`-q/dt`, but never faster than the vapor can supply. The vapor budget is shared by
passing `n = 5` to `limit`, so each of the four mass tracers plus a margin
can draw at most a fifth of the available vapor per timestep even when all are
corrected simultaneously.

# Arguments

  - `q`: Tracer specific humidity, possibly negative [kg/kg].
  - `q_vap`: Vapor specific humidity available as the source [kg/kg].
  - `dt`: Model timestep [s].

# Returns

A non-negative tendency [kg/kg/s]: zero when `q ≥ 0`, otherwise `min(-q/dt, limit(q_vap, dt, 5))`.
"""
@inline function tracer_nonnegativity_vapor_tendency(q, q_vap, dt)
    # -min(0, q/dt) gives positive tendency when q < 0
    return min(-min(zero(q), q / dt), limit(q_vap, dt, 5))
end

# Default: no correction (dry model, equilibrium moisture, etc.)
tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, _) = nothing

"""
    tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, microphysics_model)

Add tendencies that restore negative water mass tracers, borrowing from vapor.

Each of the four mass tracers `ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno` that has gone
negative receives a positive tendency restoring it to zero over `p.dt`, capped by
the vapor available for sharing (see
`tracer_nonnegativity_vapor_tendency`). Grid-mean vapor is diagnosed as
`q_tot - q_lcl - q_icl - q_rai - q_sno`. Number concentrations and the P3 tracers
are not corrected here.

Only the `NonEquilibriumMicrophysics1M` and `NonEquilibriumMicrophysics2M` methods
do work; all other microphysics models fall back to a no-op. Even for 1M/2M the
function returns immediately unless `p.atmos.water.tracer_nonnegativity_method` is
a `TracerNonnegativityVaporTendency`.

`ρq_tot` is deliberately left untouched: vapor is diagnostic, so raising a tracer
at fixed total water removes exactly the same mass from vapor and total water is
conserved. The `limit` cap is what keeps the implied vapor sink from driving vapor
negative in turn.

# Arguments

  - `Yₜ`: Tendency state vector, mutated in place.
  - `Y`: Current state vector.
  - `p`: Cache; reads `p.atmos.water.tracer_nonnegativity_method` and `p.dt`.
  - `t`: Current simulation time [s].
  - `microphysics_model`: Microphysics model, dispatched on.

Mutates `Yₜ.c.ρq_lcl`, `Yₜ.c.ρq_icl`, `Yₜ.c.ρq_rai`, and `Yₜ.c.ρq_sno`; the return
value is unused.
"""
function tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t,
    ::Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    },
)
    p.atmos.water.tracer_nonnegativity_method isa TracerNonnegativityVaporTendency || return

    moisture_species = (
        MF.@name(ρq_lcl), MF.@name(ρq_icl),
        MF.@name(ρq_rai), MF.@name(ρq_sno),
    )

    # Compute vapor specific humidity: q_vap = q_tot - q_lcl - q_icl - q_rai - q_sno
    q_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    q_lcl = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
    q_icl = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
    q_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    q_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
    q_vap = @. lazy(q_tot - q_lcl - q_icl - q_rai - q_sno)

    MF.unrolled_foreach(moisture_species) do ρq_name
        ᶜρq = MF.get_field(Y.c, ρq_name)
        ᶜρqₜ = MF.get_field(Yₜ.c, ρq_name)
        ᶜq = @. lazy(specific(ᶜρq, Y.c.ρ))
        # Add positive tendency to restore negative tracers using mass from vapor
        @. ᶜρqₜ += Y.c.ρ * tracer_nonnegativity_vapor_tendency(ᶜq, q_vap, p.dt)
    end
end
