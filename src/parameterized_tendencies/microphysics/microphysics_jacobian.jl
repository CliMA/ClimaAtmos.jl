"""
    microphysics_jacobian_diagonal(Sq, q)

Diagonal Jacobian entry `∂(ρ·Sq)/∂(ρq) ≈ Sq/q` for microphysics tendencies.

Linearizes the tendency assuming `S ∝ q`, giving `Sq/q` for both sources
and sinks.  Guards against division by zero when `q ≈ 0` using
[`ϵ_numerics`](@ref).
"""
@inline function microphysics_jacobian_diagonal(Sq, q)
    FT = typeof(Sq)
    q_safe = max(abs(q), ϵ_numerics(FT))
    return Sq / q_safe
end
