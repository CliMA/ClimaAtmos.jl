# ============================================================================
# Moisture Fixers for Microphysics
# ============================================================================
# Functions for correcting negative moisture values and ensuring physical
# consistency of water species.

import ClimaCore.MatrixFields as MF

"""
    tracer_nonnegativity_vapor_tendency(q, q_vap, dt)

Compute a tendency to restore negative tracer values by borrowing from vapor.

When a tracer `q` becomes negative (due to numerical errors), this function
returns a positive tendency to restore it toward zero, limited by available
vapor `q_vap`.

# Arguments

  - `q`: Tracer specific humidity (may be negative) [kg/kg]
  - `q_vap`: Vapor specific humidity (source for correction) [kg/kg]
  - `dt`: Model timestep [s]

# Returns

Tendency [kg/kg/s] to add to tracer:

  - If `q >= 0`: Returns `0` (no correction needed)
  - If `q < 0`: Returns positive tendency limited by available vapor

# Notes

    # -min(0, q/dt) gives positive tendency when q < 0

Uses `n=5` in `limit()` to share vapor among multiple tracers that may need correction.
"""
@inline function tracer_nonnegativity_vapor_tendency(q, q_vap, dt)
    # -min(0, q/dt) gives positive tendency when q < 0
    return min(-min(zero(q), q / dt), limit(q_vap, dt, 5))
end

# Default: no correction (dry model, equilibrium moisture, etc.)
tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, _) = nothing

"""
    tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, microphysics_model)

Apply tracer nonnegativity corrections by borrowing mass from vapor.

For `NonEquilibriumMicrophysics` (1M/2M): if any cloud/precipitation
tracer (q_liq, q_ice, q_rai, q_sno) is negative, adds a positive tendency
sourced from grid-mean vapor.

# Arguments

  - `Yₜ`: Tendency state vector (modified in place)
  - `Y`: State vector
  - `p`: Cache containing `atmos`, `dt`, etc.
  - `t`: Current time
  - `microphysics_model`: Microphysics model (dispatched on `NonEquilibriumMicrophysics1M`
    or `NonEquilibriumMicrophysics2M`)

# Modifies

  - `Yₜ.c.ρq_lcl`, `Yₜ.c.ρq_icl`, `Yₜ.c.ρq_rai`, `Yₜ.c.ρq_sno`

# Notes

Only active when `p.atmos.water.tracer_nonnegativity_method` is `TracerNonnegativityVaporTendency`.
"""
function tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t,
    microphysics_model::Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    },
)
    p.atmos.water.tracer_nonnegativity_method isa TracerNonnegativityVaporTendency || return

    # Restore negative water-mass tracers from vapor. 1M carries cloud ice
    # `ρq_icl` and snow `ρq_sno`; 2M+P3 carries a single ice mass `ρq_ice`
    # (no snow). The branch is constant-folded per concrete model type.
    q_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    q_lcl = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
    q_rai = @. lazy(specific(Y.c.ρq_rai, Y.c.ρ))
    if microphysics_model isa NonEquilibriumMicrophysics2M
        moisture_species =
            (MF.@name(ρq_lcl), MF.@name(ρq_ice), MF.@name(ρq_rai))
        q_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
        q_vap = @. lazy(q_tot - q_lcl - q_ice - q_rai)
    else  # NonEquilibriumMicrophysics1M
        moisture_species = (
            MF.@name(ρq_lcl), MF.@name(ρq_icl),
            MF.@name(ρq_rai), MF.@name(ρq_sno),
        )
        q_icl = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
        q_sno = @. lazy(specific(Y.c.ρq_sno, Y.c.ρ))
        q_vap = @. lazy(q_tot - q_lcl - q_icl - q_rai - q_sno)
    end

    MF.unrolled_foreach(moisture_species) do ρq_name
        ᶜρq = MF.get_field(Y.c, ρq_name)
        ᶜρqₜ = MF.get_field(Yₜ.c, ρq_name)
        ᶜq = @. lazy(specific(ᶜρq, Y.c.ρ))
        # Add positive tendency to restore negative tracers using mass from vapor
        @. ᶜρqₜ += Y.c.ρ * tracer_nonnegativity_vapor_tendency(ᶜq, q_vap, p.dt)
    end
end
