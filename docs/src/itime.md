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

If `t = 0` and `dt = 0.1`, then the time of the simulation is already wrong
when the simulation done just three time steps. This can easily accumulate into
a larger error.

Additionally, time can stop incrementing as seen below.
```@repl
Float32(16777216) + Float32(1) == Float32(16777216)
```

In the expression above, if the number represents seconds, then time stops
incrementing after about 194 days.

These issues propagate and lead to problems as we cannot reliably depend on the
simulation time to be accurate. For instance, dates will always be wrong when
converting from simulation time to date. Since dates are wrong, the diagnostics
are saved after one timestep than they should be.

These are the issues that `ITime` aims to solve.

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

`ITime`s can be thought of as a number with units. Hence, addition and
subtraction is what we expected, but multiplication between `ITime`s is not
defined and division results in a number rather than an `ITime`. For more
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

If you are running a simulation from a YAML file, you can simply set `use_itime`
to `true` to enable `ITime`. If you do not want to use `ITime`, then set
`use_itime` to `false` to not use `ITime` which will use floating point instead.

!!! note "Different results from rounding using `ITime`"
    If `a` is a floating point number and `t` is an `ITime`, then we round
    `a * t` to the nearest integer for the `counter`, while keeping the same
    `period` and `epoch` if it exists. As a result, the simulation will run at a
    resolution of the period used for `ITime`. This could leads to slight
    differences in the surface conditions and the time dependent forcing and
    tendencies that explicitly depend on time.

!!! note "Different results from `float` on `ITime`"
    Using `float` on an `ITime` returns a `Float64`. As such, a simulation
    running Float32 and a simulation running `ITime` for time and `Float32` for
    everything else will return different results.

## Developing with ITime

Some helpful functions when working with `ITime`s are `float`, `date`, and
`promote`.

When working with `ITime`, you might need `t`, an `ITime`, to be the number of
seconds. This can be done by using the function `float` on `t`. For other cases,
you might need the current date which you can get by using the function `date`
on `t`. Finally, when working with `ITime`s, the types of the `ITime`s might not
match (e.g. the periods are different). To handle this, you can use `promote` on
the two `ITime`s. For more information about developing with `ITime`, see the
`ITime` [documentation](https://clima.github.io/ClimaUtilities.jl/dev/timemanager/) 
in ClimaUtilities.

``` @repl example
float(x)
date(x)
promote(x, y)
```
