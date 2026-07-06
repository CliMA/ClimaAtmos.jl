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
    # Host-side counters of Jacobian updates and linear solves, incremented in
    # the update_jacobian! and ldiv! entry points that ClimaTimeSteppers
    # calls. Without a Krylov method, each linear solve is one Newton
    # iteration, so these measure how many iterations a convergence checker
    # actually takes (with a Krylov method, ldiv! is instead called once per
    # preconditioner application).
    counter_cache =
        (; newton_counters = (; updates = Ref(0), linear_solves = Ref(0)))
    cache = (;
        jacobian_cache(alg, Y, atmos; verbose)...,
        krylov_cache...,
        counter_cache...,
    )
    return Jacobian(alg, cache)
end

reset_newton_counters!(jacobian::Jacobian) = (
    jacobian.cache.newton_counters.updates[] = 0;
    jacobian.cache.newton_counters.linear_solves[] = 0;
    nothing
)

# Ignore the verbose flag in jacobian_cache when it is not needed.
jacobian_cache(alg, Y, atmos; verbose) =
    jacobian_cache(alg, Y, atmos)

# ClimaTimeSteppers.jl calls zero(jac_prototype) to initialize the Jacobian, but
# we don't need to allocate a second Jacobian for this (in particular, the exact
# Jacobian can be very expensive to allocate).
Base.zero(jacobian::Jacobian) = jacobian

# ClimaTimeSteppers.jl calls this to set the Jacobian before each linear solve.
NVTX.@annotate function update_jacobian!(jacobian, Y, p, dtγ, t)
    jacobian.cache.newton_counters.updates[] += 1
    update_jacobian!(jacobian.alg, jacobian.cache, Y, p, eltype(Y)(dtγ), t)
end

# ClimaTimeSteppers.jl calls this to perform each linear solve.
NVTX.@annotate function LinearAlgebra.ldiv!(
    ΔY::Fields.FieldVector,
    jacobian::Jacobian,
    R::Fields.FieldVector,
)
    jacobian.cache.newton_counters.linear_solves[] += 1
    invert_jacobian!(jacobian.alg, jacobian.cache, ΔY, R)
end

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

"""
    column_jacobian_cache(alg, Y, atmos)

A Jacobian cache built on the first-column view of `Y`, used by the debug
comparison. Algorithms whose caches depend on the grid geometry (which the
column view loses) should override this to resolve geometry-derived quantities
from the full grid of `Y` before slicing, so that the column Jacobian stays
consistent with the simulation's Jacobian.
"""
column_jacobian_cache(alg::JacobianAlgorithm, Y, atmos) =
    jacobian_cache(alg, first_column_view(Y), atmos; verbose = false)

# This defines a standardized format for comparing different types of Jacobians.
function first_column_block_arrays(alg::SparseJacobian, Y, p, dtγ, t)
    scalar_names = scalar_field_names(Y)
    column_Y = first_column_view(Y)
    column_p = first_column_view(p)
    column_cache = column_jacobian_cache(alg, Y, p.atmos)

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
