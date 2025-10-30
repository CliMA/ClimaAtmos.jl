"""
    JacobianAlgorithm

A description of how to compute the matrix ``∂R/∂Y``, where ``R(Y)`` denotes the
residual of an implicit step with the state ``Y``. Concrete implementations of
this abstract type should define 3 methods:
 - `jacobian_cache(alg::JacobianAlgorithm, Y, atmos; [verbose])`
 - `update_jacobian!(alg::JacobianAlgorithm, cache, Y, p, dtγ, t)`
 - `invert_jacobian!(alg::JacobianAlgorithm, cache, ΔY, R)`
To facilitate debugging, concrete implementations should also define
 - `first_column_block_arrays(alg::JacobianAlgorithm, Y, p, dtγ, t)`

See [Implicit Solver](@ref) for additional background information.
"""
abstract type JacobianAlgorithm end

abstract type SparseJacobian <: JacobianAlgorithm end

"""
    Jacobian(alg, Y, atmos; [verbose])

Wrapper for a [`JacobianAlgorithm`](@ref) and its cache, which it uses to update
and invert the Jacobian. The optional `verbose` flag specifies whether debugging
information should be printed during initialization.
"""
struct Jacobian{A <: JacobianAlgorithm, C}
    alg::A
    cache::C
end
function Jacobian(alg, Y, atmos; verbose = false)
    krylov_cache = (; ΔY_krylov = similar(Y), R_krylov = similar(Y))
    cache = (; jacobian_cache(alg, Y, atmos; verbose)..., krylov_cache...)
    return Jacobian(alg, cache)
end

# Ignore the verbose flag in jacobian_cache when it is not needed.
jacobian_cache(alg, Y, atmos; verbose) = jacobian_cache(alg, Y, atmos)

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

# This defines a standardized format for comparing different types of Jacobians.
function first_column_block_arrays(alg::SparseJacobian, Y, p, dtγ, t)
    scalar_names = scalar_field_names(Y)
    column_Y = first_column_view(Y)
    column_p = first_column_view(p)
    column_cache = jacobian_cache(alg, column_Y, p.atmos; verbose = false)

    update_jacobian!(alg, column_cache, column_Y, column_p, dtγ, t)
    column_∂R_∂Y = column_cache.matrix

    block_arrays = Dict()
    for block_key in Iterators.product(scalar_names, scalar_names)
        block_key in keys(column_∂R_∂Y) || continue
        block_value = Base.materialize(column_∂R_∂Y[block_key])
        block_arrays[block_key] =
            block_value isa Fields.Field ?
            MatrixFields.column_field2array(block_value) : block_value
    end
    return block_arrays
end
function first_column_rescaling_arrays(Y, p, t)
    scalar_names = scalar_field_names(Y)
    column_Y = first_column_view(Y)
    column_p = first_column_view(p)
    column_Yₜ = similar(column_Y)

    implicit_tendency!(column_Yₜ, column_Y, column_p, t)

    rescaling_arrays = Dict()
    for block_key in Iterators.product(scalar_names, scalar_names)
        (block_row_name, block_column_name) = block_key
        block_row_Yₜ_values =
            parent(MatrixFields.get_field(column_Yₜ, block_row_name))
        block_column_Yₜ_values =
            parent(MatrixFields.get_field(column_Yₜ, block_column_name))
        safe_inverse = x -> iszero(x) || issubnormal(x) ? zero(x) : inv(x)
        rescaling_arrays[block_key] =
            Array(safe_inverse.(block_row_Yₜ_values) .* block_column_Yₜ_values')
    end
    return rescaling_arrays
end
