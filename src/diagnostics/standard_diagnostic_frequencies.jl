# This file is included by default_diagnostics.jl.
#
# Standard frequency helpers. Each family comes in a plural form, which takes any number of
# short names and returns a vector, and a singular form, which takes one short name and
# returns that single `ScheduledDiagnostic`. All of them are thin wrappers around
# `common_diagnostics`, differing only in the accumulation period and the reduction.

"""
    monthly_maxs(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, taking the maximum over each month.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = monthly_maxs(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
monthly_maxs(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(Month(1), max, output_writer, start_date, t_start, short_names...)
"""
    monthly_max(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` taking the maximum of a variable over each month.

Single-variable form of `monthly_maxs`; despite the plural argument name, `short_names`
is one short name.

# Examples

```julia
diagnostic = monthly_max(Float64, "ts"; output_writer, start_date, t_start)
```
"""
monthly_max(FT, short_names; output_writer, start_date, t_start) =
    monthly_maxs(FT, short_names; output_writer, start_date, t_start)[1]

"""
    monthly_mins(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, taking the minimum over each month.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = monthly_mins(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
monthly_mins(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(Month(1), min, output_writer, start_date, t_start, short_names...)
"""
    monthly_min(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` taking the minimum of a variable over each month.

Single-variable form of `monthly_mins`; despite the plural argument name, `short_names`
is one short name.

# Examples

```julia
diagnostic = monthly_min(Float64, "ts"; output_writer, start_date, t_start)
```
"""
monthly_min(FT, short_names; output_writer, start_date, t_start) =
    monthly_mins(FT, short_names; output_writer, start_date, t_start)[1]

# An average is just a sum with a normalization before output
"""
    monthly_averages(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, averaging over each month.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = monthly_averages(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
monthly_averages(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(
        Month(1),
        (+),
        output_writer,
        start_date,
        t_start,
        short_names...,
        ;
        pre_output_hook! = average_pre_output_hook!,
    )

# An average is just a sum with a normalization before output
"""
    monthly_average(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` averaging a variable over each month.

Single-variable form of `monthly_averages`; despite the plural argument name,
`short_names` is one short name.

# Examples

```julia
diagnostic = monthly_average(Float64, "ts"; output_writer, start_date, t_start)
```
"""
monthly_average(FT, short_names; output_writer, start_date, t_start) =
    monthly_averages(FT, short_names; output_writer, start_date, t_start)[1]

"""
    tendaily_maxs(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, taking the maximum over ten days.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = tendaily_maxs(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
tendaily_maxs(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(Day(10), max, output_writer, start_date, t_start, short_names...)
"""
    tendaily_max(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` taking the maximum of a variable over ten days.

Single-variable form of `tendaily_maxs`; despite the plural argument name, `short_names`
is one short name.

# Examples

```julia
diagnostic = tendaily_max(Float64, "ts"; output_writer, start_date, t_start)
```
"""
tendaily_max(FT, short_names; output_writer, start_date, t_start) =
    tendaily_maxs(FT, short_names; output_writer, start_date, t_start)[1]

"""
    tendaily_mins(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, taking the minimum over ten days.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = tendaily_mins(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
tendaily_mins(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(Day(10), min, output_writer, start_date, t_start, short_names...)
"""
    tendaily_min(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` taking the minimum of a variable over ten days.

Single-variable form of `tendaily_mins`; despite the plural argument name, `short_names`
is one short name.

# Examples

```julia
diagnostic = tendaily_min(Float64, "ts"; output_writer, start_date, t_start)
```
"""
tendaily_min(FT, short_names; output_writer, start_date, t_start) =
    tendaily_mins(FT, short_names; output_writer, start_date, t_start)[1]

# An average is just a sum with a normalization before output
"""
    tendaily_averages(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, averaging over ten days.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = tendaily_averages(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
tendaily_averages(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(
        Day(10),
        (+),
        output_writer,
        start_date,
        t_start,
        short_names...;
        pre_output_hook! = average_pre_output_hook!,
    )
# An average is just a sum with a normalization before output
"""
    tendaily_average(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` averaging a variable over ten days.

Single-variable form of `tendaily_averages`; despite the plural argument name,
`short_names` is one short name.

# Examples

```julia
diagnostic = tendaily_average(Float64, "ts"; output_writer, start_date, t_start)
```
"""
tendaily_average(FT, short_names; output_writer, start_date, t_start) =
    tendaily_averages(FT, short_names; output_writer, start_date, t_start)[1]

"""
    daily_maxs(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, taking the maximum over each day.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = daily_maxs(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
daily_maxs(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(Day(1), max, output_writer, start_date, t_start, short_names...)
"""
    daily_max(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` taking the maximum of a variable over each day.

Single-variable form of `daily_maxs`; despite the plural argument name, `short_names` is
one short name.

# Examples

```julia
diagnostic = daily_max(Float64, "ts"; output_writer, start_date, t_start)
```
"""
daily_max(FT, short_names; output_writer, start_date, t_start) =
    daily_maxs(FT, short_names; output_writer, start_date, t_start)[1]

"""
    daily_mins(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, taking the minimum over each day.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = daily_mins(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
daily_mins(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(Day(1), min, output_writer, start_date, t_start, short_names...)
"""
    daily_min(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` taking the minimum of a variable over each day.

Single-variable form of `daily_mins`; despite the plural argument name, `short_names` is
one short name.

# Examples

```julia
diagnostic = daily_min(Float64, "ts"; output_writer, start_date, t_start)
```
"""
daily_min(FT, short_names; output_writer, start_date, t_start) =
    daily_mins(FT, short_names; output_writer, start_date, t_start)[1]

# An average is just a sum with a normalization before output
"""
    daily_averages(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, averaging over each day.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = daily_averages(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
daily_averages(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(
        Day(1),
        (+),
        output_writer,
        start_date,
        t_start,
        short_names...;
        pre_output_hook! = average_pre_output_hook!,
    )
# An average is just a sum with a normalization before output
"""
    daily_average(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` averaging a variable over each day.

Single-variable form of `daily_averages`; despite the plural argument name, `short_names`
is one short name.

# Examples

```julia
diagnostic = daily_average(Float64, "ts"; output_writer, start_date, t_start)
```
"""
daily_average(FT, short_names; output_writer, start_date, t_start) =
    daily_averages(FT, short_names; output_writer, start_date, t_start)[1]

"""
    hourly_maxs(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, taking the maximum over each hour.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = hourly_maxs(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
hourly_maxs(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(Hour(1), max, output_writer, start_date, t_start, short_names...)

"""
    hourly_max(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` taking the maximum of a variable over each hour.

Single-variable form of `hourly_maxs`; despite the plural argument name, `short_names` is
one short name.

# Examples

```julia
diagnostic = hourly_max(Float64, "ts"; output_writer, start_date, t_start)
```
"""
hourly_max(FT, short_names; output_writer, start_date, t_start) =
    hourly_maxs(FT, short_names; output_writer, start_date, t_start)[1]

"""
    hourly_mins(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, taking the minimum over each hour.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = hourly_mins(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
hourly_mins(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(Hour(1), min, output_writer, start_date, t_start, short_names...)
"""
    hourly_min(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` taking the minimum of a variable over each hour.

Single-variable form of `hourly_mins`; despite the plural argument name, `short_names` is
one short name.

# Examples

```julia
diagnostic = hourly_min(Float64, "ts"; output_writer, start_date, t_start)
```
"""
hourly_min(FT, short_names; output_writer, start_date, t_start) =
    hourly_mins(FT, short_names; output_writer, start_date, t_start)[1]

# An average is just a sum with a normalization before output
"""
    hourly_averages(FT, short_names...; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` per short name, averaging over each hour.

# Arguments

  - `FT`: Floating-point type of the simulation. Accepted for signature uniformity across
    the frequency helpers and currently unused.
  - `short_names...`: Short names of registered diagnostics, e.g. `"rhoa"`, `"ta"`.

# Keyword Arguments

  - `output_writer`: `ClimaDiagnostics` writer bound to every returned diagnostic.
  - `start_date`: `Dates.DateTime` assigned to the start of the simulation.
  - `t_start`: Start time of the simulation [s], or an `ITime`.

# Examples

```julia
diagnostics = hourly_averages(Float64, "rhoa", "ta"; output_writer, start_date, t_start)
```
"""
hourly_averages(FT, short_names...; output_writer, start_date, t_start) =
    common_diagnostics(
        Hour(1),
        (+),
        output_writer,
        start_date,
        t_start,
        short_names...;
        pre_output_hook! = average_pre_output_hook!,
    )

"""
    hourly_average(FT, short_names; output_writer, start_date, t_start)

Return one `ScheduledDiagnostic` averaging a variable over each hour.

Single-variable form of `hourly_averages`; despite the plural argument name,
`short_names` is one short name.

# Examples

```julia
diagnostic = hourly_average(Float64, "ts"; output_writer, start_date, t_start)
```
"""
hourly_average(FT, short_names; output_writer, start_date, t_start) =
    hourly_averages(FT, short_names; output_writer, start_date, t_start)[1]
