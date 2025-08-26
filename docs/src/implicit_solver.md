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

# Jacobian Algorithms

The derivative ``\partial R/\partial Y`` is represented as a
[`ClimaAtmos.Jacobian`](@ref), and the method for computing it and solving its
linear equation is given by a [`ClimaAtmos.JacobianAlgorithm`](@ref).

## Manual Differentiation

By making certain assumptions about the physical significance of each block in
the Jacobian (see Yatunin et al., Appendix F), we can obtain a sparse matrix
structure that allows for an efficient linear solver. Specifically, the time and
memory required to compute the sparse matrix and the time required to run the
linear solver all scale linearly with respect to the number of values in each
column's state vector.

To populate the nonzero entries of this sparse matrix, the
[`ClimaAtmos.ManualSparseJacobian`](@ref) specifies approximate derivatives for
all possible configurations of the atmosphere model, which are analytically
derived from expressions used to compute the implicit tendency. This algorithm
also provides flags for zeroing out blocks of the sparse matrix, where each flag
corresponds to the implicit treatment of some particular physical process.

## Automatic Differentiation

Another way to compute the Jacobian is through automatic differentiation. This
involves replacing all real numbers in the prognostic state with dual numbers of
the form
```math
x^D =
x + \hat{x}^1 * \varepsilon_1 + \hat{x}^2 * \varepsilon_2 + \ldots +
    \hat{x}^n * \varepsilon_n,
```
where 
- ``x`` and ``\hat{x}^i`` are real numbers, and
- ``\varepsilon_i`` is an infinitesimal number with the property that
  ``\varepsilon_i * \varepsilon_j = 0``.

Passing the dual number ``x + \hat{x} * \varepsilon`` to any function ``f(x)``
yields
```math
f(x + \hat{x} * \varepsilon) =
f(x) + \frac{\partial f(x)}{\partial x} * \hat{x} * \varepsilon.
```
By extension, passing the dual vector ``X + \hat{X} * \mathcal{E}``, where
```math
X =
\begin{pmatrix}
    X_1 \\
    X_2 \\
    \vdots \\
    X_N
\end{pmatrix},
\mathcal{E} =
\begin{pmatrix}
    \varepsilon_1 \\
    \varepsilon_2 \\
    \vdots \\
    \varepsilon_n
\end{pmatrix},
\textrm{ and }
\hat{X} =
\begin{pmatrix}
    \hat{X}^1_1 & \hat{X}^2_1 & \ldots & \hat{X}^n_1 \\
    \hat{X}^1_2 & \hat{X}^2_2 & \ldots & \hat{X}^n_2 \\
    \vdots & \vdots & \ddots & \vdots \\
    \hat{X}^1_N & \hat{X}^2_N & \ldots & \hat{X}^n_N
\end{pmatrix},
```
to any function ``f(X)`` yields
```math
f(X + \hat{X} * \mathcal{E}) =
f(X) + \frac{\partial f(X)}{\partial X} * \hat{X} * \mathcal{E}.
```
If the dual vector is
```math
Y^D = Y + P * \mathcal{E},
```
passing it to the implicit tendency ``T_{imp}(Y)`` yields the dual tendency
```math
T_{imp}^D = T_{imp}(Y^D) =
T_{imp}(Y) + \frac{\partial T_{imp}(Y)}{\partial Y} * P * \mathcal{E},
```
so ``P`` acts as a right preconditioner for ``\partial T_{imp}(Y)/\partial Y``.
The tendency derivative can be extracted from the ``\varepsilon`` components of
the dual tendency by inverting the preconditioner, after which a multiplication
by ``\Delta t`` and subtraction of ``I`` gives the full Jacobian matrix
``\partial R(Y)/\partial Y``.

To be more precise, the implicit tendency is evaluated in two function calls.
The first function, ``p_{imp}(Y)``, computes cached values that are treated
implicitly, and the second function, ``T_{imp}(Y, p_{imp})``, computes the
tendency itself. Further generalizing the property of
``\varepsilon_i * \varepsilon_j = 0`` to functions of two vectors, passing
``A + \hat{A} * \mathcal{E}`` and ``B + \hat{B} * \mathcal{E}`` to any function
``f(A, B)`` yields
```math
f(A + \hat{A} * \mathcal{E}, B + \hat{B} * \mathcal{E}) =
f(A, B) +
\left(
    \frac{\partial f(A, B)}{\partial A} * \hat{A} +
    \frac{\partial f(A, B)}{\partial B} * \hat{B}
\right) * \mathcal{E}.
```
So, the dual tendency ``T_{imp}^D`` is computed in two steps, first evaluating
``p_{imp}(Y^D)`` to get
```math
p_{imp}^D =
p_{imp}(Y) + \frac{\partial p_{imp}(Y)}{\partial Y} * P * \mathcal{E},
```
and then evaluating ``T_{imp}(Y^D, p_{imp}^D)`` to get
```math
T_{imp}^D =
T_{imp}(Y, p_{imp}(Y)) +
\left(
    \frac{\partial T_{imp}(Y, p_{imp}(Y))}{\partial Y} +
    \frac{\partial T_{imp}(Y, p_{imp}(Y))}{\partial p_{imp}(Y)} *
    \frac{\partial p_{imp}(Y)}{\partial Y}
\right) * P * \mathcal{E}.
```
In other words, the single-argument tendency derivative
``\partial T_{imp}(Y)/\partial Y`` is really a shorthand for
```math
\frac{\partial T_{imp}(Y)}{\partial Y} =
\frac{\partial T_{imp}(Y, p_{imp}(Y))}{\partial Y} +
\frac{\partial T_{imp}(Y, p_{imp}(Y))}{\partial p_{imp}(Y)} *
\frac{\partial p_{imp}(Y)}{\partial Y}.
```

### Dense Automatic Differentiation

The simplest form of automatic differentiation uses a dense representation of
the tendency matrix. When the number of ``\varepsilon`` components, ``n``, is
equal to the number of values in each column's state vector, ``N``, this
involves setting ``P`` to the ``N \times N`` identity matrix, so that the dual
counterpart of each column's state vector is
```math
Y^D = Y + \mathcal{E} =
\begin{pmatrix}
    Y_1 + \varepsilon_1 \\
    Y_2 + \varepsilon_2 \\
    \vdots \\
    Y_N + \varepsilon_N
\end{pmatrix}.
```
Evaluating ``T_{imp}(Y)`` on this input yields the dual tendency
```math
T_{imp}^D = T_{imp}(Y) + \frac{\partial T_{imp}(Y)}{\partial Y} * \mathcal{E} =
\begin{pmatrix}
    T_{imp, 1}(Y) + \frac{\partial T_{imp, 1}(Y)}{\partial Y_1} * \varepsilon_1 +
    \frac{\partial T_{imp, 1}(Y)}{\partial Y_2} * \varepsilon_2 + \ldots +
    \frac{\partial T_{imp, 1}(Y)}{\partial Y_N} * \varepsilon_N \\
    T_{imp, 2}(Y) + \frac{\partial T_{imp, 2}(Y)}{\partial Y_1} * \varepsilon_1 +
    \frac{\partial T_{imp, 2}(Y)}{\partial Y_2} * \varepsilon_2 + \ldots +
    \frac{\partial T_{imp, 2}(Y)}{\partial Y_N} * \varepsilon_N \\
    \vdots \\
    T_{imp, N}(Y) + \frac{\partial T_{imp, N}(Y)}{\partial Y_1} * \varepsilon_1 +
    \frac{\partial T_{imp, N}(Y)}{\partial Y_2} * \varepsilon_2 + \ldots +
    \frac{\partial T_{imp, N}(Y)}{\partial Y_N} * \varepsilon_N
\end{pmatrix},
```
where the entry in the ``i``-th row and ``j``-th column of
``\partial T_{imp}(Y)/\partial Y`` is the coefficient of ``\varepsilon_j`` in
the ``i``-th value of ``T_{imp}^D``.

When there are many values in each column, setting ``n = N`` can lead to
excessive compilation latency, and often also poor performance. To compensate
for this, the [`ClimaAtmos.AutoDenseJacobian`](@ref) splits the ``N`` values in
each column's state vector into partitions of length ``n < N``, where the
default value of ``n`` is 32. The partitioning is implemented by setting ``P``
to ``N \times n`` slices of the identity matrix (with the last slice possibly
containing fewer than ``n`` columns), so that ``T_{imp}^D`` only contains an
``N \times n`` slice of the matrix ``\partial T_{imp}(Y)/\partial Y``. Computing
``T_{imp}^D`` for all partitions yields the full derivative matrix.

With ``\partial R(Y)/\partial Y`` specified as a dense matrix, the time and
memory required to compute it scale in proportion to ``N^2``. The linear
equation ``R'(Y) * \Delta Y = R(Y)`` is solved by
[LU factorization](https://en.wikipedia.org/wiki/LU_decomposition), where the
time required to compute the L and U factors scales as ``N^3``. After the
factors are computed, the time required to invert them scales as ``N^2``.

### Sparse Automatic Differentiation

The `AutoDenseJacobian` can be sped up by copying its entries into the
`ManualSparseJacobian`. This allows the matrix to be inverted using an efficient
linear solver whose runtime scales in proportion to ``N``, rather than an LU
factorization that scales as ``N^3``. Although this improves performance on
CPUs, it still results in poor runtimes compared to the sparse representation.
This is especially the case on GPUs, where performance is primarily determined
by memory requirements. Introducing sparsity only to avoid the factorization
does not reduce the memory requirements that scale as ``N^2``.

To make the memory requirements of automatic differentiation scale linearly with
respect to ``N``, ``P`` can be set to an ``N \times c`` column coloring matrix
for the tendency derivative, so that ``c`` is the smallest value for which
``\partial T_{imp}(Y)/\partial Y * P`` is a lossless representation of the
nonzero entries in ``\partial T_{imp}(Y)/\partial Y``. Specifically, ``P`` is a
binary matrix, where a 1 in row ``i`` and column ``j`` means that the tendency
derivative column corresponding to ``Y_i`` is assigned color ``j``. Ideally,
``P`` should be chosen so that no two values ``Y_a`` and``Y_b`` can be assigned
the same color if ``\partial T_{imp, i}(Y)/\partial Y_a`` and
``\partial T_{imp, i}(Y)/\partial Y_b`` are both nonzero in any row ``i``. For
any such matrix ``P``, ``\partial T_{imp}(Y)/\partial Y * P`` uniquely
represents every nonzero value in the derivative matrix, so the entries of the
derivative matrix can be extracted from it without any errors.

This requirement on ``P`` can be loosened so that ``Y_a`` and ``Y_b`` can still
be assigned the same color as long as ``\partial T_{imp, i}(Y)/\partial Y_a``
and ``\partial T_{imp, i}(Y)/\partial Y_b`` do not have similar magnitudes.
If the derivative with respect to ``Y_b`` is negligibly small compared to the
derivative with respect to ``Y_a``, then assigning ``Y_b`` the same color as
``Y_a`` might not be an issue, since this will amount to replacing the
derivative ``\partial T_{imp, i}(Y)/\partial Y_a`` in ``T_{imp}^D`` with the sum  
``\partial T_{imp, i}(Y)/\partial Y_a + \partial T_{imp, i}(Y)/\partial Y_b``,
which should be approximately equal to ``\partial T_{imp, i}(Y)/\partial Y_a``.
When the derivatives with respect to ``Y_a`` and ``Y_b`` have comparable
magnitudes, though, the sum can no longer be used to approximate either of them,
and the only workaround is to assign them distinct colors.

Most of the derivatives that are ignored by the`ManualSparseJacobian` are
negligibly small, so they can safely be excluded from the sparsity pattern used
to assign column colors. However, some of the derivatives can be ignored based
on the inputs to the linear equation in which they are used, but they do not
have small magnitudes. For example, derivatives with respect to `ρ` tend to be
much larger than derivatives with respect to `ρe_tot`, since a adding one
kilogram of air to a cubic meter will typically have a more significant effect
than adding one Joule of energy. In physical simulations, though, changes of
`δρ = 1 kg/m^3` tend to be much less common than changes of `δρe_tot = 1 J/m^3`.
Generally, the amount by which `δρ` is smaller than `δρe_tot` exceeds the amount
by which derivatives with respect to `ρ` are larger than derivatives with
respect to `ρe_tot`, which means that those derivatives can be ignored when
solving the linear equation.

To avoid introducing errors to ``\partial T_{imp}(Y)/\partial Y``, the locations
of all non-negligible derivatives must be included in the sparsity pattern for
assigning column colors, even if those derivatives can be ignored when solving
the linear equation. In many cases, this requires the introduction of additional
colors, but sometimes the coloring can be extended to include these derivatives
without using more colors.

The memory requirements of this algorithm scale as ``N * c``, which can limit
the range of resolutions for which the sparse representation fits in GPU memory
when ``c`` is large. To avoid this, the [`ClimaAtmos.AutoSparseJacobian`](@ref)
splits the ``c`` colors into partitions of length ``n < c``. Each partition sets
``P`` to an ``N \times n`` slice of the column coloring matrix, so that
``T_{imp}^D`` contains an ``N \times n`` slice of the tendency derivative's
sparse representation. Computing ``T_{imp}^D`` for all partitions yields the
full tendency derivative. On GPUs, the number of partitions is the smallest
value for which the sparse representation fits in GPU memory (using a limit of
twice the memory that is currently free); on CPUs, it is assumed that the sparse
representation will always fit in memory, so only a single partition is used.

When running the `AutoSparseJacobian`, care should be taken to ensure that its
entries are not polluted by non-negligible derivatives from ignored blocks (or
from ignored bands within nonzero blocks). Whenever debugging reveals a
difference between the nonzero values generated by sparse and dense automatic
differentiation, ignored derivatives are almost always at fault. The default
"padding bands" added to the coloring sparsity pattern should handle most
non-negligible derivatives (such as the aforementioned derivatives with respect
to `ρ`), but new variables and tendencies may require additional padding bands.
For cases where the new padding bands that need to be added are not known in
advance, the `AutoSparseJacobian` also has an option to add a fixed number of
padding bands to every Jacobian block.

#### Debugging Sparse Automatic Differentiation Errors

If setting `use_auto_jacobian = true` makes a simulation unstable or leads to
inaccurate results, set `debug_jacobian = true` and compare the different
approximations of each Jacobian block:
- When a block differs between two algorithms, check whether the difference
  is significant (i.e., whether its normalized magnitude exceeds `1/dt`).
- If the `AutoSparseJacobian` and `ManualSparseJacobian` agree on a block but
  significantly differ from the `AutoDenseJacobian`:
  - Add bands that are missing from the sparsity pattern in this block to the
    `ManualSparseJacobian`, which also adds them to the `AutoSparseJacobian`.
- If the `AutoSparseJacobian` and `AutoDenseJacobian` agree on a block but
  significantly differ from the `ManualSparseJacobian`, and if the manual
  approximation is more accurate than the automatic value:
  - Determine which tendency term's derivative is responsible for the erroneous
    automatic value.
  - If possible, rewrite the tendency term so that it generates a more accurate
    derivative using dual numbers.
  - If this is not possible, add a new method for the tendency term that
    specializes on dual numbers with the tag [`ClimaAtmos.Jacobian`](@ref),
    overwriting the derivative automatically generated by `ForwardDiff.jl`.
- Otherwise, if the `AutoSparseJacobian` and `AutoDenseJacobian` significantly
  differ for a block:
  - Set `auto_jacobian_padding_bands` to a large number, and check whether this
    discrepancy between the sparse and dense values disappears.
  - If padding bands do not resolve the discrepancy:
    - Add non-padding bands that are missing from this block to the
      `ManualSparseJacobian`, which also adds them to the `AutoSparseJacobian`.
  - If padding bands resolve the discrepancy:
    - Find all differences between the sparsity patterns of the sparse and dense
      Jacobians in the same row as this block.
    - If any blocks (or bands within a block) are missing from this block's row
      of the sparse Jacobian, check whether they have unnormalized magnitudes
      that are significant in comparison to this block.
    - Extend the default padding bands of the `AutoSparseJacobian` so they cover
      every significant unnormalized value that could be affecting this block,
      and reset `auto_jacobian_padding_bands` to use the default padding bands.

## See also
- [Yatunin, D, et al., "The CliMA atmosphere dynamical core: Concepts, numerics, and scaling"](https://doi.org/10.22541/essoar.173940262.23304403/v1), Section 5 and Appendix F
- [Documentation for ClimaTimeSteppers.jl](https://clima.github.io/ClimaTimeSteppers.jl/dev/algorithm_formulations/ode_solvers/)
