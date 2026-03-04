# Scalar helpers for the Jacobian diagonal computation. Extracting the
# computation into a scalar function avoids calling `abs` twice in a single
# broadcast expression, which can prevent broadcast-fusion type inference and
# cause heap allocations in the surrounding function.
@inline function _jac_coeff(Sq, q)
    FT = typeof(Sq)
    ε = q_min(FT)
    aq = abs(q)
    # Use S/|q| as an approximation for ∂S/∂q.  The approximation is exact
    # for accretion-type processes where S ∝ q (∂S/∂q = S/q), and provides
    # a useful damping estimate for other processes.
    #
    # For sinks (Sq < 0): S/max(|q|,ε) creates a strong barrier that prevents
    # q from going negative in the Newton solver, and is bounded by |S|/ε as
    # |q| → 0.
    #
    # For sources (Sq > 0) with |q| > ε: S/q captures accretion-type
    # ∂S/∂q = S/q exactly, converting the otherwise unstable Picard iteration
    # (q^{k+1} = q^n + dtγ·S(q^k), which diverges when dtγ·k_acc·q_liq > 1)
    # into a one-step Newton solve, even when dtγ·S/q > 1 (fast conversion).
    #
    # For sources (Sq > 0) with |q| ≈ 0: return zero to avoid the 1/ε
    # singularity.  In this regime (e.g. autoconversion creating the first
    # rain drops from q_rain = 0), S is independent of q, so J = I is exact
    # and the Newton step q^{k+1} = q^n + dtγ·S converges in one iteration.
    return ifelse(Sq >= zero(FT) && aq <= ε, zero(FT), Sq / max(aq, ε))
end

@inline function _jac_coeff_from_ratio(Sq, ρq, ρ)
    FT = typeof(Sq)
    ε = q_min(FT)
    q = ρq / ρ
    aq = abs(q)
    return ifelse(Sq >= zero(FT) && aq <= ε, zero(FT), Sq / max(aq, ε))
end

"""
    add_microphysics_jacobian_entry!(∂, dtγ, Sq_field, q_field)

Broadcast-level helper that adds `dtγ * DiagonalMatrixRow(Sq/|q|)` to the
matrix block `∂`.

The coefficient `Sq/|q|` approximates `∂S/∂q`:
- **Sinks** (`Sq < 0`): returns `Sq / max(|q|, ε)`, providing a barrier that
  prevents negative `q` and is bounded by `|Sq|/ε` as `|q| → 0`.
- **Sources** (`Sq > 0`) with `|q| > ε`: returns `Sq / |q|`, which equals
  `∂S/∂q` exactly for accretion-type processes (`S ∝ q`), converting the
  otherwise unstable Picard iteration into a one-step Newton solve.
- **Sources** (`Sq > 0`) with `|q| ≈ 0`: returns zero to avoid the `S/ε`
  singularity; the `J = I` approximation is exact here (autoconversion is
  independent of the target species).

`Sq_field` and `q_field` must be in the *same* units — either both specific
(per-mass) or both density-weighted — so that the ratio `Sq / |q|` gives the
correct Jacobian diagonal entry.

!!! note
    This function performs the entire `@.` broadcast internally so that all
    field operations fuse into a single kernel, avoiding allocations from
    broadcast fusion barriers (e.g., through lazy fields).
"""
@inline function add_microphysics_jacobian_entry!(∂, dtγ, Sq_field, q_field)
    @. ∂ += $(dtγ) * DiagonalMatrixRow(_jac_coeff(Sq_field, q_field))
    return nothing
end

"""
    add_microphysics_jacobian_entry!(∂, dtγ, Sq, ρq, ρ)

Broadcast-level helper for *grid-mean* variables where the tendency `Sq` is
already in specific (per-mass) form but the state `ρq` is density-weighted.
The state is divided by `ρ` inside the fused broadcast to form the specific
value `q = ρq / ρ`, giving the Jacobian diagonal `Sq / |q|`.

!!! note
    The division by `ρ` is performed inside the broadcast to avoid creating
    intermediate lazy fields, which would break broadcast fusion and cause
    heap allocations. Consider moving the division outside the broadcast to 
    have only one method for this function. 
"""
@inline function add_microphysics_jacobian_entry!(∂, dtγ, Sq, ρq, ρ)
    @. ∂ += $(dtγ) * DiagonalMatrixRow(_jac_coeff_from_ratio(Sq, ρq, ρ))
    return nothing
end
