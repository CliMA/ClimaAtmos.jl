# ITime

`ITime`, or _integer time_, is a time type used by CliMA simulations to keep track
of simulation time. For more information, refer to the
[TimeManager section](https://clima.github.io/ClimaUtilities.jl/dev/timemanager/)
in ClimaUtilities.

## Why not use floating point for simulation time?

Due to floating point errors, time can easily be inaccurate or stop incrementing
(especially with Float32). For instance, consider the example below.

```@repl
0.1 + 0.1 + 0.1 == 0.3
```

If `t = 0` and `dt = 0.1`, then the simulation time is already wrong after
three time steps. This error can accumulate over the course of a simulation.

Additionally, time can stop incrementing as seen below.

```@repl
Float32(16777216) + Float32(1) == Float32(16777216)
```

In the expression above, if the number represents seconds, then time stops
incrementing after about 194 days.

These issues propagate and lead to problems, as we cannot rely on the
simulation time being accurate. For instance, dates will always be wrong when
converting from simulation time to date. Since dates are wrong, the diagnostics
are saved one timestep later than they should be.

`ITime` addresses these issues.

## Introduction to ITime

`ITime` consists of three fields: `counter`, `period`, and `epoch`. The counter
keeps track of the number of `period`s since the `epoch` if it exists. See the
examples below of constructing an `ITime`.

```@repl example
using ClimaUtilities.TimeManager, Dates # ITime is from ClimaUtilities
x = ITime(3, period = Minute(1), epoch = DateTime(2010))
counter(x)
period(x)
epoch(x)
```

An `ITime` behaves like a number with units. Addition and subtraction work as
expected, but multiplication between `ITime`s is not defined, and division
results in a number rather than an `ITime`. For more
information about what functions are available for `ITime`, see the
[API](https://clima.github.io/ClimaUtilities.jl/dev/timemanager/#TimeManager-API)
at ClimaUtilities.

```@repl example
y = ITime(60, period = Second(1), epoch = DateTime(2010))
x + y
x - y
x / y
```

## How do I use ITime in my simulation?

In this section, we address how to use `ITime` instead of floating point for
time in a ClimaAtmos simulation and how to write code with `ITime` in mind.

ClimaAtmos always uses `ITime` for simulation time: the time step, start time,
and end time are converted to `ITime` when the simulation is built, regardless of
whether they are provided as strings (e.g. `"30secs"`) or numbers.

!!! note "Different results from rounding using `ITime`"

    If `a` is a floating point number and `t` is an `ITime`, then we round
    `a * t` to the nearest integer for the `counter`, while keeping the same
    `period` and `epoch` if it exists. As a result, the simulation will run at a
    resolution of the period used for `ITime`. This can lead to slight
    differences in the surface conditions and the time dependent forcing and
    tendencies that explicitly depend on time.

!!! note "Different results from `float` on `ITime`"

    Using `float` on an `ITime` returns a `Float64`. As such, a simulation
    running Float32 and a simulation running `ITime` for time and `Float32` for
    everything else will return different results.

## Developing with ITime

Some helpful functions when working with `ITime`s are `float`, `date`, and
`promote`.

To convert an `ITime` `t` to a number of seconds, use the function `float` on
`t`. To get the current date, use the function `date` on `t`. Finally, the
types of two `ITime`s might not match (e.g. the periods are different). To
handle this, use `promote` on the two `ITime`s. For more information about
developing with `ITime`, see the
`ITime` [documentation](https://clima.github.io/ClimaUtilities.jl/dev/timemanager/)
in ClimaUtilities.

```@repl example
float(x)
date(x)
promote(x, y)
```
