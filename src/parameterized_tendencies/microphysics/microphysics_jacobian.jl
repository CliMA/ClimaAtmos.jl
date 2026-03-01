# Scalar helpers for the Jacobian diagonal computation. Extracting the
# computation into a scalar function avoids calling `abs` twice in a single
# broadcast expression, which can prevent broadcast-fusion type inference and
# cause heap allocations in the surrounding function.
#
# The 2-argument overloads compute ε internally to avoid passing a scalar
# through the broadcast, which would otherwise allocate a Ref wrapper.
@inline function _jac_coeff(Sq, q, ε)
    aq = abs(q)
    return ifelse(aq > ε, Sq / aq, zero(Sq))
end
@inline function _jac_coeff(Sq, q)
    ε = ϵ_numerics(typeof(Sq))
    return _jac_coeff(Sq, q, ε)
end

@inline function _jac_coeff_from_ratio(Sq, ρq, ρ, ε)
    q = ρq / ρ
    aq = abs(q)
    return ifelse(aq > ε, Sq / aq, zero(Sq))
end
@inline function _jac_coeff_from_ratio(Sq, ρq, ρ)
    ε = ϵ_numerics(typeof(Sq))
    return _jac_coeff_from_ratio(Sq, ρq, ρ, ε)
end

"""
    add_microphysics_jacobian_entry!(∂, dtγ, Sq_field, q_field)

Broadcast-level helper that adds `dtγ * DiagonalMatrixRow(Sq/|q|)` to the
matrix block `∂`.  The contribution is suppressed when `|q|` is below a
numerical threshold.

`Sq_field` and `q_field` must be in the *same* units — either both specific
(per-mass) or both density-weighted — so that the ratio `Sq / |q|` gives the
correct Jacobian diagonal entry.

!!! note
    This function performs the entire `@.` broadcast internally so that all
    field operations fuse into a single kernel, avoiding allocations from
    broadcast fusion barriers (e.g., through lazy fields).
"""
@inline function add_microphysics_jacobian_entry!(∂, dtγ, Sq_field, q_field)
    @. ∂ += dtγ * DiagonalMatrixRow(_jac_coeff(Sq_field, q_field))
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
    heap allocations.
"""
@inline function add_microphysics_jacobian_entry!(∂, dtγ, Sq, ρq, ρ)
    @. ∂ += dtγ * DiagonalMatrixRow(_jac_coeff_from_ratio(Sq, ρq, ρ))
    return nothing
end
