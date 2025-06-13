# Implicit Solver

The state ``Y`` is evolved using a split implicit-explicit (IMEX) timestepping
scheme, which separates the tendency ``T(Y) = \partial Y/\partial t`` into
implicit (fast) and explicit (slow) components,
```math
T(Y) = T_{imp}(Y) + T_{exp}(Y).
```
For an implicit step from time ``t`` to ``t + \Delta t``, we begin with the
state ``Y_{prev}`` from the explicit step at time ``t`` (which also includes
information from all previous times before ``t``), and we find a state ``Y``
that solves the implicit equation
```math
Y = Y_{prev} + \Delta t * T_{imp}(Y),
```
where ``\Delta t * T_{imp}(Y)`` is a linear approximation of the state change
due to the implicit tendency between times ``t`` and ``t + \Delta t``. Solving
this equation amounts to finding a root of the residual function
```math
R(Y) = Y_{prev} + \Delta t * T_{imp}(Y) - Y,
```
since any state ``Y`` that satisfies ``R(Y) = 0`` is consistent with the linear
approximation of the implicit state change.

*Note:* When we use a higher-order timestepping scheme, the full step
``\Delta t`` is divided into several sub-steps or "stages", where the duration
of stage ``i`` is ``\Delta t * γ_i`` for some constant ``γ_i`` between 0 and 1.

To find the root of ``R(Y)`` using Newton's method, we must specify the
derivative ``\partial R/\partial Y``. Since ``Y_{prev}`` does not depend on
``Y`` (it is only a function of the state at or before time ``t``), this
derivative is given by
```math
R'(Y) = \Delta t * \frac{\partial T_{imp}}{\partial Y} - I.
```
For each state ``Y``, Newton's method computes an update ``\Delta Y`` that
brings ``R(Y)`` closer to 0 by solving the linear equation
```math
R'(Y) * \Delta Y = R(Y).
```

*Note:* This equation comes from assuming that there is some ``\Delta Y``
for which ``R(Y - \Delta Y) = 0`` and approximating
```math
R(Y - \Delta Y) \approx R(Y) - R'(Y) * \Delta Y.
```

After initializing ``Y`` to ``Y[0] = Y_{prev}``, Newton's method executes the
following steps:
1. Compute the residual ``R(Y[0])`` and its derivative ``R'(Y[0])``.
2. Solve ``R'(Y[0]) * \Delta Y[0] = R(Y[0])`` for ``\Delta Y[0]``.
3. Update ``Y`` to ``Y[1] = Y[0] - \Delta Y[0]``.

If the number of Newton iterations is limited to 1, this new value of ``Y`` is
taken to be the solution of the implicit equation. Otherwise, this sequence of
steps is repeated, i.e., ``\Delta Y[1]`` is computed and ``Y`` is updated to
``Y[2] = Y[1] - \Delta Y[1]``, then ``\Delta Y[2]`` is computed and ``Y`` is
updated to ``Y[3] = Y[2] - \Delta Y[2]``, and so on until the maximum number of
iterations is reached.

## Computing the Jacobian

The derivative ``\partial R/\partial Y`` is represented as a
[`ClimaAtmos.Jacobian`](@ref), and the method for computing it and solving its
linear equation is given by a [`ClimaAtmos.JacobianAlgorithm`](@ref).

### Manual Differentiation

By making certain assumptions about the physical significance of each block in
the Jacobian (see Yatunin et al., Appendix F), we can obtain a sparse matrix
structure that allows for an efficient linear solver. Specifically, the memory
required for the sparse matrix and the time required for the linear solver both
scale linearly with the number of values in each column.

To populate the nonzero entries of this sparse matrix, the
[`ClimaAtmos.ManualSparseJacobian`](@ref) specifies approximate derivatives for
all possible configurations of the atmosphere model, which are analytically
derived from expressions used to compute the implicit tendency. This algorithm
also provides flags for zeroing out blocks of the sparse matrix, where each flag
corresponds to the implicit treatment of some particular physical process.

### Dense Automatic Differentiation

Another way to compute the Jacobian is through automatic differentiation. This
involves replacing all real numbers with dual numbers of the form
```math
x^D = x + x'_1 * \varepsilon_1 + x'_2 * \varepsilon_2 + \ldots,
```
where 
- ``x`` and ``x'_i`` are real numbers, and
- ``\varepsilon_i`` is an infinitesimal number with the property that
  ``\varepsilon_i * \varepsilon_j = 0``.

The [`ClimaAtmos.AutoDenseJacobian`](@ref) defines the dual counterpart of each
column's state vector ``Y`` as
```math
Y^D = Y + \varepsilon =
\begin{pmatrix}
    Y_1 + \varepsilon_1 \\
    Y_2 + \varepsilon_2 \\
    \vdots \\
    Y_N + \varepsilon_N
\end{pmatrix},
```
where ``N`` is the number of values in the column. If ``x`` is a function of
``Y``, evaluating ``x(Y^D)`` gives us
```math
x^D = x + \frac{\partial x}{\partial Y} * \varepsilon =
x + \frac{\partial x}{\partial Y_1} * \varepsilon_1 +
    \frac{\partial x}{\partial Y_2} * \varepsilon_2 + \ldots +
    \frac{\partial x}{\partial Y_N} * \varepsilon_N,
```
so that each coefficient ``x'_i`` is a component of ``\partial x/\partial Y``.

*Note:* When there are many values in each column, the dual number components
``\varepsilon_i`` are split into batches to reduce compilation latency and
runtime. Each batch requires a separate evaluation of ``x(Y^D)``, but that
evaluation involves fewer derivative coefficients and less compiled code.

After we initialize ``Y^D`` as shown above, we evaluate ``T_{imp}(Y^D)`` to
obtain a dense representation of the matrix ``\partial T_{imp}/\partial Y``,
```math
T_{imp}^D = T_{imp} + \frac{\partial T_{imp}}{\partial Y} * \varepsilon =
\begin{pmatrix}
    T_{imp, 1} + \frac{\partial T_{imp, 1}}{\partial Y_1} * \varepsilon_1 +
    \frac{\partial T_{imp, 1}}{\partial Y_2} * \varepsilon_2 + \ldots +
    \frac{\partial T_{imp, 1}}{\partial Y_N} * \varepsilon_N \\
    T_{imp, 2} + \frac{\partial T_{imp, 2}}{\partial Y_1} * \varepsilon_1 +
    \frac{\partial T_{imp, 2}}{\partial Y_2} * \varepsilon_2 + \ldots +
    \frac{\partial T_{imp, 2}}{\partial Y_N} * \varepsilon_N \\
    \vdots \\
    T_{imp, N} + \frac{\partial T_{imp, N}}{\partial Y_1} * \varepsilon_1 +
    \frac{\partial T_{imp, N}}{\partial Y_2} * \varepsilon_2 + \ldots +
    \frac{\partial T_{imp, N}}{\partial Y_N} * \varepsilon_N
\end{pmatrix},
```
where the entry in the ``i``-th row and ``j``-th column of
``\partial T_{imp}/\partial Y`` is the coefficient of ``\varepsilon_j`` in
``T_{imp, i}^D``.

*Note:* The implicit tendency is evaluated in two function calls. The first
function ``p_{imp}(Y)`` computes cached values that are treated implicitly, and
the second function ``T_{imp}(Y, p_{imp})`` computes the tendency itself. So, we
first evaluate ``p_{imp}(Y^D)`` to get ``p_{imp}^D``, and then we evaluate
``T_{imp}(Y^D, p_{imp}^D)``.

The full dense Jacobian ``\partial R/\partial Y`` is computed as
``\Delta t * \partial T_{imp}/\partial Y - I``, and its linear equation is
solved using [LU factorization](https://en.wikipedia.org/wiki/LU_decomposition).
The memory required for the dense matrix scales in proportion to ``N^2``, and
the time required for LU factorization scales as ``N^3``.

## See also
- [Yatunin, D, et al., "The CliMA atmosphere dynamical core: Concepts, numerics, and scaling"](https://doi.org/10.22541/essoar.173940262.23304403/v1), Section 5 and Appendix F
- [Documentation for ClimaTimeSteppers.jl](https://clima.github.io/ClimaTimeSteppers.jl/dev/algorithm_formulations/ode_solvers/)
