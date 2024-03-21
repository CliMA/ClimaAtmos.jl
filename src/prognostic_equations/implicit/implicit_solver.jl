import BlockArrays, ForwardDiff
import LinearAlgebra: I, Adjoint, LU, lu, lu!, ldiv!
import ClimaCore.MatrixFields: @name

"""
    JacobianAlgorithm

A description of how to compute the matrix ``∂R/∂Y``, where ``R(Y)`` denotes the
residual of an implicit step with the state ``Y``. Concrete implementations of
this abstract type should define 4 methods:
 - `jacobian_cache(alg::JacobianAlgorithm, Y, p)`
 - `always_update_exact_jacobian(alg::JacobianAlgorithm)`
 - `factorize_exact_jacobian!(alg::JacobianAlgorithm, cache, Y, p, dtγ, t)`
 - `approximate_jacobian!(alg::JacobianAlgorithm, cache, Y, p, dtγ, t)`
 - `invert_jacobian!(alg::JacobianAlgorithm, cache, x, b)`

# Background

When we use an implicit or split implicit-explicit (IMEX) timestepping scheme,
we end up with a nonlinear equation of the form ``R(Y) = 0``, where
```math
    R(Y) = Y_{imp}(Y) - Y = \\hat{Y} + Δt * T_{imp}(Y) - Y.
```
In this expression, ``Y_{imp}(Y)`` denotes the state at some time ``t + Δt``.
This can be expressed as the sum of ``\\hat{Y}``, the contribution from the
state at time ``t`` (and possibly also at earlier times, depending on the order
of the timestepping scheme), and ``Δt * T_{imp}(Y)``, the contribution from the
implicit tendency ``T_{imp}`` between times ``t`` and ``t + Δt``. The new state
at the end of each implicit step in the timestepping scheme is the value of
``Y`` that solves this equation, i.e., the value of ``Y`` that is consistent
with the state ``Y_{imp}(Y)`` predicted by the implicit step.

Note: When we use a higher-order timestepping scheme, the full step ``Δt`` is
divided into several sub-steps or "stages", where the duration of stage ``i`` is
``Δt * γ_i`` for some constant ``γ_i`` between 0 and 1.

In order to solve this equation using Newton's method, we must specify the
derivative ``∂R/∂Y``. Since ``\\hat{Y}`` does not depend on ``Y`` (it is only a
function of the state at or before time ``t``), this derivative is
```math
    R'(Y) = Δt * T_{imp}'(Y) - I.
```
In addition, we must specify how to divide ``R(Y)`` by this derivative, i.e.,
how to solve the linear equation
```math
    R'(Y) * ΔY = R(Y).
```

Note: This equation comes from assuming that there is some ``ΔY`` such that
``R(Y - ΔY) = 0`` and making the first-order approximation
```math
    R(Y - ΔY) \\approx R(Y) - R'(Y) * ΔY.
```

After initializing ``Y`` to ``Y[0] = \\hat{Y}``, Newton's method executes the
following steps:
- Compute the derivative ``R'(Y[0])``.
- Compute the implicit tendency ``T_{imp}(Y[0])`` and use it to get ``R(Y[0])``.
- Solve the linear equation ``R'(Y[0]) * ΔY[0] = R(Y[0])`` for ``ΔY[0]``.
- Update ``Y`` to ``Y[1] = Y[0] - ΔY[0]``.
If the number of Newton iterations is limited to 1, this new value of ``Y`` is
taken to be the solution of the implicit equation. Otherwise, this sequence of
steps is repeated, i.e., ``ΔY[1]`` is computed and used to update ``Y`` to
``Y[2] = Y[1] - ΔY[1]``, then ``ΔY[2]`` is computed and used to update ``Y`` to
``Y[3] = Y[2] - ΔY[2]``, and so on. The iterative process is terminated either
when the residual ``R(Y)`` is sufficiently close to 0 (according to the
convergence condition passed to Newton's method), or when the maximum number of
iterations is reached.
"""
abstract type JacobianAlgorithm end

struct ImplicitEquationJacobian{A <: JacobianAlgorithm, C}
    alg::A
    cache::C
end
function ImplicitEquationJacobian(alg, Y, p)
    cache = (;
        jacobian_cache(alg, Y, p)...,
        x_krylov = similar(Y),
        b_krylov = similar(Y),
    )
    return ImplicitEquationJacobian(alg, cache)
end

# We only use A, but ClimaTimeSteppers.jl currently requires us to pass
# jac_prototype and then calls similar(jac_prototype) to obtain A. Since
# this was only done for compatibility with OrdinaryDiffEq.jl, which we are no
# longer using, we should fix this in a new version of ClimaTimeSteppers.jl.
Base.similar(A::ImplicitEquationJacobian) = A

# This is called from a callback when always_update_exact_jacobian is false.
NVTX.@annotate function update_exact_jacobian!(A, Y, p, dtγ, t)
    FT = eltype(Y)
    factorize_exact_jacobian!(A.alg, A.cache, Y, p, FT(dtγ), t)
end

# This is passed to ClimaTimeSteppers.jl and called on each Newton iteration.
NVTX.@annotate function update_jacobian!(A, Y, p, dtγ, t)
    FT = eltype(Y)
    if always_update_exact_jacobian(A.alg)
        factorize_exact_jacobian!(A.alg, A.cache, Y, p, FT(dtγ), t)
    end
    approximate_jacobian!(A.alg, A.cache, Y, p, FT(dtγ), t)
end

# This is called by ClimaTimeSteppers.jl on each Newton iteration.
NVTX.@annotate ldiv!(
    x::Fields.FieldVector,
    A::ImplicitEquationJacobian,
    b::Fields.FieldVector,
) = invert_jacobian!(A.alg, A.cache, x, b)

# This is called by Krylov.jl from inside ClimaTimeSteppers.jl. See
# https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a related
# issue that requires the same workaround.
function ldiv!(
    x::AbstractVector,
    A::ImplicitEquationJacobian,
    b::AbstractVector,
)
    A.cache.b_krylov .= b
    ldiv!(A.cache.x_krylov, A, A.cache.b_krylov)
    x .= A.cache.x_krylov
end
