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
    FT = eltype(Sq_field)
    ε = ϵ_numerics(FT)
    @. ∂ +=
        dtγ * DiagonalMatrixRow(
            ifelse(abs(q_field) > ε, Sq_field / abs(q_field), zero(FT)),
        )
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
    FT = eltype(Sq)
    ε = ϵ_numerics(FT)
    @. ∂ += dtγ * DiagonalMatrixRow(
        ifelse(abs(ρq / ρ) > ε, Sq / abs(ρq / ρ), zero(FT)),
    )
    return nothing
end
