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
Uses `n=5` in `limit()` to share vapor among multiple tracers that may need correction.
"""
@inline function tracer_nonnegativity_vapor_tendency(q, q_vap, dt)
    # -min(0, q/dt) gives positive tendency when q < 0
    return min(-min(zero(q), q / dt), limit(q_vap, dt, 5))
end

# Default: no correction (dry model, equilibrium moisture, etc.)
tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, _) = nothing

"""
    tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, mm)

Apply tracer nonnegativity corrections by borrowing mass from vapor.

For `NonEquilibriumMicrophysics` (1M/2M): if any cloud/precipitation
tracer (q_liq, q_ice, q_rai, q_sno) is negative, adds a positive tendency
sourced from grid-mean vapor.

# Arguments
- `Yₜ`: Tendency state vector (modified in place)
- `Y`: State vector
- `p`: Cache containing `atmos`, `dt`, etc.
- `t`: Current time
- `mm`: Microphysics model (dispatched on `NonEquilibriumMicrophysics1M`
  or `NonEquilibriumMicrophysics2M`)

# Modifies
- `Yₜ.c.ρq_lcl`, `Yₜ.c.ρq_icl`, `Yₜ.c.ρq_rai`, `Yₜ.c.ρq_sno` (if `NonEquilibriumMicrophysics1M`)
- `Yₜ.c.ρq_lcl`, `Yₜ.c.ρq_ice`, `Yₜ.c.ρq_rai` (if `NonEquilibriumMicrophysics2M`)

# Notes
Only active when `p.atmos.water.tracer_nonnegativity_method` is `TracerNonnegativityVaporTendency`.
"""
function tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, 
    mm::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
)
    p.atmos.water.tracer_nonnegativity_method isa TracerNonnegativityVaporTendency || return
    moisture_species = condensate_names(mm)
    generic_tracer_nonnegativity_vapor_tendency!(Yₜ.c, Y.c, moisture_species, p.dt)
end


function generic_tracer_nonnegativity_vapor_tendency!(ᶜYₜ, ᶜY, moisture_species, dt)
    ᶜρqs = UU.unrolled_map(Base.Fix1(MF.get_field, ᶜY), moisture_species)  # TODO: Test that this is valid code
    ᶜρq_cond = @. lazy(UU.unrolled_sum(ᶜρqs))

    # Compute vapor specific humidity: q_vap = (ρq_tot - ρq_cond) / ρ
    ᶜq_vap = @. lazy(specific(ᶜY.ρq_tot - ᶜρq_cond, ᶜY.ρ))

    MF.unrolled_foreach(moisture_species) do ρq_name
        ᶜρq = MF.get_field(ᶜY, ρq_name)
        ᶜρqₜ = MF.get_field(ᶜYₜ, ρq_name)
        ᶜq = @. lazy(specific(ᶜρq, ᶜY.ρ))
        # Add positive tendency to restore negative tracers using mass from vapor
        @. ᶜρqₜ += ᶜY.ρ * tracer_nonnegativity_vapor_tendency(ᶜq, ᶜq_vap, dt)
    end
end
