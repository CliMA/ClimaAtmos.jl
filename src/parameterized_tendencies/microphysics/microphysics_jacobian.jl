"""
    microphysics_jacobian_diagonal(Sq, q)

Diagonal Jacobian entry `∂(ρ·Sq)/∂(ρq) ≈ Sq/q` for microphysics tendencies.

Linearizes the tendency assuming `S ∝ q`, giving `Sq/q` for both sources
and sinks.  When `|q|` is below a threshold the linearization is unreliable
(the tendency does not really depend on `q` for near-zero condensate),
so the contribution is suppressed to prevent Newton-solver divergence.
"""
@inline function microphysics_jacobian_diagonal(Sq, q)
    FT = typeof(Sq)
    q_threshold = ϵ_numerics(FT))

    return ifelse(abs(q) > q_threshold, Sq / abs(q), zero(Sq))
end
