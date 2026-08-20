# Integer Time (ITime)

Simulation time in ClimaAtmos is counted in integers, not in floating-point
seconds. The type that does this is `ITime`, provided by the
[TimeManager module](https://clima.github.io/ClimaUtilities.jl/dev/timemanager/)
of ClimaUtilities and shared across CliMA components.

For the timestepping scheme that advances the state over that time, see
[Discretization and Operators](discretization.md) and
[Implicit Solver](implicit_solver.md).

## Why not floating-point time

Accumulating a timestep in floating point drifts, and eventually stops
advancing altogether.

Drift appears immediately, because most decimal timesteps are not exactly
representable in binary:

```@repl
0.1 + 0.1 + 0.1 == 0.3
```

With `t = 0` and `dt = 0.1`, the simulation time is already wrong after three
steps, and the error accumulates over the run.

Worse, once the elapsed time is large enough that the timestep falls below the
spacing between adjacent representable numbers, the addition has no effect at
all:

```@repl
Float32(16777216) + Float32(1) == Float32(16777216)
```

Counting seconds in `Float32`, time stops incrementing after about 194 days.

An inaccurate simulation time is not a cosmetic problem. Dates derived from it
are wrong, and diagnostics keyed to those dates are written at the wrong times.
Counting integer periods avoids both failure modes exactly.

## How ITime represents time

An `ITime` has three fields: a `counter`, a `period`, and an optional `epoch`.
The counter holds the number of periods elapsed since the epoch.

```@repl example
using ClimaUtilities.TimeManager, Dates # ITime is from ClimaUtilities
x = ITime(3, period = Minute(1), epoch = DateTime(2010))
counter(x)
period(x)
epoch(x)
```

An `ITime` behaves like a number carrying units. Addition and subtraction work
as expected; multiplying two `ITime`s is undefined, and dividing them gives a
plain number rather than an `ITime`.

```@repl example
y = ITime(60, period = Second(1), epoch = DateTime(2010))
x + y
x - y
x / y
```

The [TimeManager API](https://clima.github.io/ClimaUtilities.jl/dev/timemanager/#TimeManager-API)
lists the full set of supported operations.

## Using ITime in a simulation

ClimaAtmos always uses `ITime` for simulation time. The timestep, start time,
and end time are converted when the simulation is built, whether they are given
as strings such as `"30secs"` or as numbers, and are then promoted to a common
period.

Two consequences are worth knowing about.

!!! note "Rounding to the period"

    Multiplying an `ITime` `t` by a floating-point number `a` rounds `a * t` to
    the nearest integer counter, keeping the same period and epoch. The
    simulation therefore advances at the resolution of that period, which can
    slightly change surface conditions and any forcing or tendency that depends
    explicitly on time.

!!! note "Different results from `float` on `ITime`"

    `float` on an `ITime` returns a `Float64`. A simulation using `Float32`
    throughout will therefore not reproduce one using `ITime` for time and
    `Float32` for everything else.

## Developing with ITime

Three functions cover most needs. `float(t)` converts an `ITime` to a number of
seconds, `date(t)` gives the corresponding date, and `promote` reconciles two
`ITime`s whose periods differ.

```@repl example
float(x)
date(x)
promote(x, y)
```

## Where this is implemented

| Concept                     | Source                                                                                                                    |
|:--------------------------- |:------------------------------------------------------------------------------------------------------------------------- |
| `ITime` type and operations | [ClimaUtilities.TimeManager](https://clima.github.io/ClimaUtilities.jl/dev/timemanager/)                                  |
| Conversion at setup         | [src/simulation/AtmosSimulations.jl](https://github.com/CliMA/ClimaAtmos.jl/blob/main/src/simulation/AtmosSimulations.jl) |

The `dt`, `t_start`, and `t_end` configuration keys accept either strings or
numbers; see [Configuration Options](configuration_options.md).
