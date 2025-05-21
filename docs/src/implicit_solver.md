# Implicit Solver

When we use an implicit or split implicit-explicit (IMEX) timestepping scheme,
we end up with a nonlinear equation of the form ``R(Y) = 0``, where
```math
    R(Y) = Y_{imp}(Y) - Y = Y_{prev} + Δt * T_{imp}(Y) - Y.
```
In this expression, ``Y_{imp}(Y)`` denotes the state at some time ``t + Δt``.
This can be expressed as the sum of ``Y_{prev}``, the contribution from the
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
derivative ``∂R/∂Y``. Since ``Y_{prev}`` does not depend on ``Y`` (it is only a
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
    R(Y - ΔY) \approx R(Y) - R'(Y) * ΔY.
```

After initializing ``Y`` to ``Y[0] = Y_{prev}``, Newton's method executes the
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

In ClimaAtmos, the derivative ``∂R/∂Y`` is represented as a
[`ClimaAtmos.Jacobian`](@ref), and the method for computing it is given by a
[`ClimaAtmos.JacobianAlgorithm`](@ref).
