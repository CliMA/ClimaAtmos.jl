"""
    JacobianAlgorithm

A description of how to compute the matrix ``∂R/∂Y``, where ``R(Y)`` denotes the
residual of an implicit step with the state ``Y``. Concrete implementations of
this abstract type should define 3 methods:
 - `jacobian_cache(alg::JacobianAlgorithm, Y, atmos)`
 - `update_jacobian!(alg::JacobianAlgorithm, cache, Y, p, dtγ, t)`
 - `invert_jacobian!(alg::JacobianAlgorithm, cache, ΔY, R)`

See [Implicit Solver](@ref) for additional background information.
"""
abstract type JacobianAlgorithm end

"""
    Jacobian(alg, Y, atmos)

Wrapper for a [`JacobianAlgorithm`](@ref) and its cache, which it uses to update
and invert the Jacobian.
"""
struct Jacobian{A <: JacobianAlgorithm, C}
    alg::A
    cache::C
end
function Jacobian(alg, Y, atmos)
    krylov_cache = (; ΔY_krylov = similar(Y), R_krylov = similar(Y))
    cache = (; jacobian_cache(alg, Y, atmos)..., krylov_cache...)
    return Jacobian(alg, cache)
end

# ClimaTimeSteppers.jl calls zero(jac_prototype) to initialize the Jacobian, but
# we don't need to allocate a second Jacobian for this (in particular, the exact
# Jacobian can be very expensive to allocate).
Base.zero(jacobian::Jacobian) = jacobian

safe_float(dtγ, Y) = eltype(Y)(float(dtγ)) # Convert dtγ to the eltype of Y.

# ClimaTimeSteppers.jl calls this to set the Jacobian before each linear solve.
NVTX.@annotate update_jacobian!(jacobian, Y, p, dtγ, t) =
    update_jacobian!(jacobian.alg, jacobian.cache, Y, p, safe_float(dtγ, Y), t)

# ClimaTimeSteppers.jl calls this to perform each linear solve.
NVTX.@annotate LinearAlgebra.ldiv!(
    ΔY::Fields.FieldVector,
    jacobian::Jacobian,
    R::Fields.FieldVector,
) = invert_jacobian!(jacobian.alg, jacobian.cache, ΔY, R)

# This is called by Krylov.jl from inside ClimaTimeSteppers.jl. See
# https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a related
# issue that requires the same workaround.
function LinearAlgebra.ldiv!(
    ΔY::AbstractVector,
    jacobian::Jacobian,
    R::AbstractVector,
)
    (; ΔY_krylov, R_krylov) = jacobian.cache
    R_krylov .= R
    LinearAlgebra.ldiv!(ΔY_krylov, jacobian, R_krylov)
    ΔY .= ΔY_krylov
end
