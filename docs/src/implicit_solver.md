# Implicit Solver

This page documents the solve at the heart of the timestepping split introduced
in [Discretization and Operators](discretization.md): the vertical terms of the
[governing equations](equations.md) are treated implicitly, and this is how that
implicit system is solved.

## The implicit equation

The tendency ``T(Y) = \partial Y / \partial t`` is split into an implicit
(fast) and an explicit (slow) part,

```math
T(Y) = T_{imp}(Y) + T_{exp}(Y).
```

An implicit step starts from a state ``Y_{prev}``, assembled by the timestepper
from the explicit stages, and looks for a state ``Y`` that satisfies

```math
Y = Y_{prev} + \Delta t \, T_{imp}(Y).
```

Equivalently, ``Y`` is a root of the residual

```math
R(Y) = Y_{prev} + \Delta t \, T_{imp}(Y) - Y.
```

A higher-order Runge–Kutta scheme takes several stages per step, and the
coefficient multiplying the tendency in stage ``i`` is ``\Delta t \gamma_i``
rather than ``\Delta t``, where ``\gamma_i`` is the diagonal implicit
coefficient of the tableau. Everything below carries over with ``\Delta t``
replaced by ``\Delta t \gamma_i``; in the code this product is the argument
`dtγ`.

The root is found with Newton's method. Because ``Y_{prev}`` does not depend on
``Y``, the derivative of the residual is

```math
R'(Y) = \Delta t \, \frac{\partial T_{imp}}{\partial Y} - I .
```

Starting from ``Y[0] = Y_{prev}``, each Newton iteration evaluates the residual
and its derivative, solves the linear system

```math
R'(Y[k]) \, \Delta Y[k] = R(Y[k]) ,
```

and updates ``Y[k+1] = Y[k] - \Delta Y[k]``. The update follows from the
linearization ``R(Y - \Delta Y) \approx R(Y) - R'(Y) \Delta Y``, set to zero.

By default a single iteration is taken (`max_newton_iters_ode`), so ``Y[1]`` is
the solution of the implicit step. Iterating to a tolerance instead is available
through `use_newton_rtol` and `newton_rtol`. The linear system is solved
directly by default; setting `use_krylov_method` solves it instead with a
Jacobian-free Krylov method, which forms Jacobian-vector products by finite
differences and uses the Jacobian below as a left preconditioner.

## Jacobian algorithms

The matrix ``\partial R / \partial Y`` is held in a
[`ClimaAtmos.Jacobian`](@ref), and the method used to fill it and to solve its
linear system is a [`ClimaAtmos.JacobianAlgorithm`](@ref). Three are available:

| Algorithm                                 | Entries from                     | Selected by                                 |
|:----------------------------------------- |:-------------------------------- |:------------------------------------------- |
| [`ClimaAtmos.ManualSparseJacobian`](@ref) | Analytically derived derivatives | default                                     |
| [`ClimaAtmos.AutoSparseJacobian`](@ref)   | Automatic differentiation        | `use_auto_jacobian: true`                   |
| [`ClimaAtmos.AutoDenseJacobian`](@ref)    | Automatic differentiation        | `use_dense_jacobian: true` (takes priority) |

`update_jacobian_every` controls how often the matrix is refreshed — once per
linear solve (`"solve"`, the default), once per stage (`"stage"`), or once per
timestep (`"dt"`).

### Manual differentiation

Assumptions about the physical significance of each block in the Jacobian (see
[Yatunin2026](@cite), Appendix F) give a sparse matrix structure that admits an
efficient linear solver. The time and memory needed to build the matrix, and the
time needed to solve with it, then all scale linearly with the number ``N`` of
values in a column's state vector.

The [`ClimaAtmos.ManualSparseJacobian`](@ref) fills the nonzero entries with
approximate derivatives, derived analytically from the expressions that compute
the implicit tendency, and zeroes out blocks belonging to processes that are not
treated implicitly. Which blocks are nonzero follows from the `AtmosModel` —
the presence of topography, the diffusion mode, and the prognostic variables in
``Y`` — and is fixed when the cache is built, not configured by the user.

When grid-scale diffusion is implicit, the linear solve is itself approximate;
`approximate_linear_solve_iters` sets how many iterations it takes.

### Automatic differentiation

The alternative is to differentiate the tendency automatically, by replacing
every real number in the prognostic state with a dual number

```math
x^D = x + \hat{x}^1 \varepsilon_1 + \hat{x}^2 \varepsilon_2 + \ldots +
      \hat{x}^n \varepsilon_n ,
```

where ``x`` and ``\hat{x}^i`` are real and the ``\varepsilon_i`` are
infinitesimals satisfying ``\varepsilon_i \varepsilon_j = 0``. Passing
``x + \hat{x} \varepsilon`` to a function ``f`` yields

```math
f(x + \hat{x} \varepsilon) =
f(x) + \frac{\partial f(x)}{\partial x} \hat{x} \varepsilon ,
```

and the same holds componentwise for a vector ``X`` of length ``N`` seeded with
an ``N \times n`` matrix ``\hat{X}``:

```math
f(X + \hat{X} \mathcal{E}) =
f(X) + \frac{\partial f(X)}{\partial X} \hat{X} \mathcal{E} ,
\quad \mathcal{E} = (\varepsilon_1, \ldots, \varepsilon_n)^T .
```

Seeding the state as ``Y^D = Y + P \mathcal{E}`` and passing it to the implicit
tendency gives

```math
T_{imp}^D = T_{imp}(Y^D) =
T_{imp}(Y) + \frac{\partial T_{imp}(Y)}{\partial Y} P \mathcal{E} .
```

The ``\varepsilon`` components of ``T_{imp}^D`` therefore hold the compressed
derivative ``(\partial T_{imp} / \partial Y) P``, not the derivative itself. The
seed matrix ``P`` is chosen so that the individual entries can be recovered from
that product; multiplying by ``\Delta t`` and subtracting ``I`` then gives
``\partial R / \partial Y``. The two algorithms below differ only in the choice
of ``P``.

The tendency is evaluated in two calls rather than one: ``p_{imp}(Y)`` computes
the cached quantities that are treated implicitly, and ``T_{imp}(Y, p_{imp})``
computes the tendency from them. Dual numbers propagate through both, so
evaluating ``p_{imp}(Y^D)`` and then ``T_{imp}(Y^D, p_{imp}^D)`` accumulates the
chain rule automatically:

```math
\frac{\partial T_{imp}(Y)}{\partial Y} =
\frac{\partial T_{imp}(Y, p_{imp}(Y))}{\partial Y} +
\frac{\partial T_{imp}(Y, p_{imp}(Y))}{\partial p_{imp}(Y)}
\frac{\partial p_{imp}(Y)}{\partial Y} .
```

The single-argument notation ``\partial T_{imp}(Y)/\partial Y`` used above is
shorthand for this sum.

### Dense automatic differentiation

The simplest choice is ``P = I``, the ``N \times N`` identity, so that
``Y^D_i = Y_i + \varepsilon_i``. The coefficient of ``\varepsilon_j`` in the
``i``-th component of ``T_{imp}^D`` is then the entry in row ``i`` and column
``j`` of ``\partial T_{imp} / \partial Y``, read off directly.

Carrying ``n = N`` dual components at once is expensive to compile and often
slow to run, so [`ClimaAtmos.AutoDenseJacobian`](@ref) splits them into
partitions of ``n < N`` components, with ``n = 32`` by default. Each partition
sets ``P`` to an ``N \times n`` slice of the identity and recovers the
corresponding slice of the derivative matrix; the partitions together give the
full matrix.

Storing ``\partial R / \partial Y`` densely costs time and memory proportional
to ``N^2``. The linear system is solved by
[LU factorization](https://en.wikipedia.org/wiki/LU_decomposition), factorized
and back-substituted in parallel across columns; forming the factors costs
``N^3`` and applying them ``N^2``.

This algorithm is a reference rather than a production choice: it makes no
assumptions about sparsity, so it is the standard against which the two sparse
algorithms are checked.

### Sparse automatic differentiation

Introducing sparsity only to avoid the LU factorization would not help much. The
dense matrix could be copied into the sparse structure and inverted with the
linear solver that scales as ``N``, which is faster on CPUs, but the ``N^2``
memory of the dense evaluation would remain — and on GPUs, where memory traffic
dominates, that is what matters.

Making the memory scale linearly with ``N`` requires a smaller seed matrix. With
``P`` an ``N \times c`` column coloring matrix, a binary matrix in which a 1 in
row ``i`` and column ``j`` assigns the derivative column of ``Y_i`` to color
``j``, the compressed product ``(\partial T_{imp} / \partial Y) P`` sums the
columns sharing each color. Two entries ``Y_a`` and ``Y_b`` of the state can
share a color only if ``\partial T_{imp,i} / \partial Y_a`` and
``\partial T_{imp,i} / \partial Y_b`` are never both nonzero in the same row
``i``; the smallest ``c`` for which this holds makes the compression lossless,
and every entry can be recovered exactly.

The requirement can be relaxed: ``Y_a`` and ``Y_b`` may share a color if their
derivatives never have comparable magnitudes. Summing a derivative with one that
is negligible beside it leaves the first essentially unchanged. When the two are
comparable, the sum approximates neither, and they need distinct colors.

Most of the derivatives the `ManualSparseJacobian` ignores are negligibly small
and can safely be left out of the coloring sparsity pattern. Some, though, are
ignored for a different reason and are not small. Derivatives with respect to
`ρ` tend to be much larger than derivatives with respect to `ρe_tot` — adding a
kilogram of air to a cubic meter does more than adding a joule of energy — but
in a simulation, changes of `δρ = 1 kg/m^3` are correspondingly rarer than
changes of `δρe_tot = 1 J/m^3`. The disparity in the perturbations outweighs the
disparity in the derivatives, so those entries can be dropped from the linear
solve even though they are large.

They cannot be dropped from the coloring. Every non-negligible derivative has to
appear in the sparsity pattern used to assign colors, whether or not it is used
in the solve; otherwise it pollutes the entries that share its color. Including
them sometimes costs additional colors and sometimes fits within the existing
ones.

[`ClimaAtmos.AutoSparseJacobian`](@ref) takes its sparsity structure and its
linear solver from a `ManualSparseJacobian`, and colors that structure with
[SparseMatrixColorings.jl](https://github.com/JuliaDiff/SparseMatrixColorings.jl).
Its memory scales as ``N c``, which can still exceed GPU memory when ``c`` is
large, so the ``c`` colors are split into partitions of ``n < c``. On GPUs the
number of partitions is the smallest for which the result fits, allowing twice
the memory currently free to leave room for garbage collection; on CPUs a single
partition is always used.

To keep the entries clean, the algorithm adds "padding bands" to the coloring
sparsity pattern, covering the large-but-ignored derivatives described above.
The per-block defaults handle most cases, but new variables and tendencies may
need more. When the bands that are needed are not known in advance,
`auto_jacobian_padding_bands` adds a fixed number to every block.

## Debugging sparse automatic differentiation

Whenever the sparse and dense algorithms disagree on a nonzero entry, an ignored
derivative is almost always the cause.

Setting `debug_jacobian: true` prints block-by-block summary tables comparing
the available algorithms in the first column of the final state. The tables come
from `print_jacobian_summary` in
[post_processing/jacobian_summary.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/post_processing/jacobian_summary.jl),
which [.buildkite/ci_driver.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/.buildkite/ci_driver.jl)
calls after the run; they are skipped when `use_dense_jacobian` is `true`, since
the dense matrix is the reference. Magnitudes are reported per block as an RMS,
both unnormalized and normalized to units of s⁻¹.

If `use_auto_jacobian: true` makes a simulation unstable or inaccurate, work
through the tables as follows.

  - When a block differs between two algorithms, check whether the difference is
    significant, that is, whether its normalized magnitude exceeds ``1/\Delta t``.
  - If the `AutoSparseJacobian` and `ManualSparseJacobian` agree on a block but
    differ significantly from the `AutoDenseJacobian`, add the bands missing
    from that block's sparsity pattern to the `ManualSparseJacobian`; the
    `AutoSparseJacobian` inherits them.
  - If the two automatic algorithms agree but differ significantly from the
    `ManualSparseJacobian`, and the manual value is the more accurate one, find
    the tendency term whose derivative is at fault. Rewrite the term so that it
    differentiates accurately if possible; otherwise add a method for it that
    specializes on dual numbers tagged with [`ClimaAtmos.Jacobian`](@ref),
    overriding what `ForwardDiff.jl` generates.
  - If only the sparse and dense automatic algorithms differ, set
    `auto_jacobian_padding_bands` to a large number and check whether the
    discrepancy disappears.
      + If it does not, the block is missing non-padding bands; add them to the
        `ManualSparseJacobian`.
      + If it does, compare the sparse and dense sparsity patterns across the
        whole row containing this block. Any block or band missing from the
        sparse pattern whose unnormalized magnitude is significant beside this
        block is a candidate. Extend the default padding bands to cover it, then
        reset `auto_jacobian_padding_bands` to the defaults.

## Where this is implemented

| Concept                    | Source                                                                                                                                                                                                                                                                     |
|:-------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implicit tendency          | [src/prognostic_equations/implicit/implicit_tendency.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/implicit_tendency.jl)                                                                                                          |
| Jacobian wrapper and types | [src/prognostic_equations/implicit/jacobian.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/jacobian.jl)                                                                                                                            |
| Manual sparse algorithm    | [src/prognostic_equations/implicit/manual_sparse_jacobian.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/manual_sparse_jacobian.jl)                                                                                                |
| Automatic algorithms       | [auto_dense_jacobian.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/auto_dense_jacobian.jl), [auto_sparse_jacobian.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/auto_sparse_jacobian.jl) |
| Dual-number support        | [src/prognostic_equations/implicit/autodiff_utils.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/prognostic_equations/implicit/autodiff_utils.jl)                                                                                                                |
| Timestepper interface      | [ClimaTimeSteppers.jl](https://clima.github.io/ClimaTimeSteppers.jl/stable/)                                                                                                                                                                                               |

The tendency split itself is described in
[Discretization and Operators](discretization.md), and the representation of
time in [Integer Time (ITime)](itime.md). The configuration keys named above are
listed in [Configuration Options](configuration_options.md).
