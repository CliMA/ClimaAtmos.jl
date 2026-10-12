import Dates: Year
import ClimaUtilities
import ClimaUtilities.TimeVaryingInputs
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput, LinearInterpolation
import Interpolations as Intp

"""
    ozone_cache(Y, start_date) -> (; o3, prescribed_o3_timevaryinginput)

Build the cache for prescribed ozone: a field `o3` holding the ozone volume
mixing ratio [mol/mol] on the center space, and the `TimeVaryingInput` that
regrids the `"vmro3"` variable of the ozone concentration file onto that space.
Regridding extrapolates periodically in longitude and flatly in latitude and
height, and interpolates linearly in time from `start_date`.
Called from `tracer_cache`; `o3` is updated in the radiation callback.
"""
function ozone_cache(Y, start_date)
    o3 = similar(Y.c.ρ)
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
    prescribed_o3_timevaryinginput = TimeVaryingInput(
        AA.ozone_concentration_file_path(; context = ClimaComms.context(Y.c)),
        "vmro3",
        axes(o3);
        reference_date = start_date,
        regridder_type = :InterpolationsRegridder,
        regridder_kwargs = (; extrapolation_bc),
        method = LinearInterpolation(),
    )
    return (; o3, prescribed_o3_timevaryinginput)
end

"""
    co2_cache(Y, start_date) -> (; co2, prescribed_co2_timevaryinginput)

Build the cache for prescribed CO₂. Because CO₂ is treated as well mixed, `co2`
is a one-element mutable array (not a field) holding the volume mixing ratio
[mol/mol], updated in place by `evaluate!`.

The monthly time series is read from the CO₂ concentration text file, converted
from ppm to a mole fraction, and dated to the 15th of each month, since the file
records only year and month. Called from `tracer_cache`.
"""
function co2_cache(Y, start_date)
    FT = Spaces.undertype(axes(Y.c))
    # ClimaUtilities < v0.1.21 can only write to Arrays that are on the same
    # device as the space
    ArrayType =
        pkgversion(ClimaUtilities) < v"0.1.21" ? ClimaComms.array_type(Y.c) :
        Array
    # co2 is well mixed, so it is just a number, but we create a mutable object
    # to update it with `evaluate!`
    co2 = ArrayType([zero(FT)])

    years = Int[]
    months = Int[]
    CO2_vals = FT[]
    open(
        AA.co2_concentration_file_path(; context = ClimaComms.context(Y.c)),
        "r",
    ) do file
        for line in eachline(file)
            # Skip comments
            startswith(line, '#') && continue
            parts = split(line)
            push!(years, parse(Int, parts[1]))
            push!(months, parse(Int, parts[2]))
            # convert from ppm to fraction, data is in fourth column of the text file
            push!(CO2_vals, parse(Float64, parts[4]) / 1_000_000)
        end
    end
    # The text file only has month and year, so we set the day to 15th of the month
    CO2_dates = Dates.DateTime.(years, months, 15)
    CO2_times =
        ClimaUtilities.Utils.period_to_seconds_float.(CO2_dates .- start_date)
    prescribed_co2_timevaryinginput = TimeVaryingInput(CO2_times, CO2_vals)
    return (; co2, prescribed_co2_timevaryinginput)
end

"""
    prescribed_aerosol_cache(Y, prescribed_aerosol_names, start_date)

MERRA-2 inputs for the prescribed aerosol bins listed in the
`prescribed_aerosols` config: a center Field of per-bin concentrations,
zero-initialized and updated in
`update_prescribed_aerosol_concentrations!`, and per-bin
`TimeVaryingInput`s.
"""
function prescribed_aerosol_cache(Y, prescribed_aerosol_names, start_date)
    prescribed_names = Symbol.(Tuple(prescribed_aerosol_names))
    isempty(prescribed_names) && return (;)
    # The keys in the merra2_aerosols.nc file have to match the species' bin
    # names. The file also has to be defined on the globe and provide time
    # series of lon-lat-z data.
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
    prescribed_aerosol_timevaryinginputs = NamedTuple{prescribed_names}(
        map(prescribed_names) do name
            TimeVaryingInput(
                AA.aerosol_concentration_file_path(;
                    context = ClimaComms.context(Y.c),
                ),
                string(name),
                axes(Y.c);
                reference_date = start_date,
                regridder_type = :InterpolationsRegridder,
                regridder_kwargs = (; extrapolation_bc),
                method = LinearInterpolation(),
            )
        end,
    )
    # The field is evaluated only in the RRTMGP radiation callback
    # (`update_prescribed_aerosol_concentrations!`). It is zero-initialized, so
    # it reads zero before the first callback and for the whole run with gray
    # or no radiation, including in the 2M aerosol activation.
    # TODO: evaluate the aerosol inputs outside the radiation callback.
    prescribed_aerosols_field = similar(
        Y.c,
        NamedTuple{prescribed_names, NTuple{length(prescribed_names), eltype(Y.c.ρ)}},
    )
    fill!(parent(prescribed_aerosols_field), zero(eltype(Y.c.ρ)))
    return (; prescribed_aerosols_field, prescribed_aerosol_timevaryinginputs)
end

"""
    tracer_cache(Y, prescribed_aerosol_names, time_varying_trace_gases, start_date)

Build `p.tracers`, the cache of prescribed aerosol and trace-gas inputs.

Prescribed aerosol inputs are built by `prescribed_aerosol_cache`.
Ozone and CO₂ are added by `ozone_cache` and `co2_cache` when `"O3"` and `"CO2"` appear in
`time_varying_trace_gases`.

# Returns

A `NamedTuple` that is empty when no tracer inputs are requested, and otherwise
merges the prescribed-aerosol, ozone, and CO₂ entries.
"""
function tracer_cache(Y, prescribed_aerosol_names, time_varying_trace_gases, start_date)
    prescribed_cache =
        prescribed_aerosol_cache(Y, prescribed_aerosol_names, start_date)

    if :O3 in Symbol.(time_varying_trace_gases)
        o3_cache = ozone_cache(Y, start_date)
    else
        o3_cache = (;)
    end

    if :CO2 in Symbol.(time_varying_trace_gases)
        co2_cache_nt = co2_cache(Y, start_date)
    else
        co2_cache_nt = (;)
    end

    return (; prescribed_cache..., o3_cache..., co2_cache_nt...)
end
