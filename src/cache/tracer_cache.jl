import Dates: Year
import ClimaUtilities
import ClimaUtilities.TimeVaryingInputs
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput, LinearInterpolation
import Interpolations as Intp

ozone_cache(_, _, _) = (;)
function ozone_cache(::PrescribedOzone, Y, start_date)
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

co2_cache(_, _, _) = (;)
function co2_cache(::MaunaLoaCO2, Y, start_date)
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

function tracer_cache(Y, atmos, prescribed_aerosol_names, start_date)
    if !isempty(prescribed_aerosol_names)
        target_space = axes(Y.c)

        # Take the aerosol concentration file, read the keys with names matching
        # the ones passed in the prescribed_aerosol_names option, and create a
        # NamedTuple that uses the same keys and has as values the TimeVaryingInputs
        # for those variables.
        #
        # The keys in the merra2_aerosols.nc file have to match the ones passed with the
        # configuration. The file also has to be defined on the globe and provide
        # time series of lon-lat-z data.
        prescribed_aerosol_names_as_symbols = Symbol.(prescribed_aerosol_names)
        target_space = axes(Y.c)
        extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
        timevaryinginputs = [
            TimeVaryingInput(
                AA.aerosol_concentration_file_path(;
                    context = ClimaComms.context(Y.c),
                ),
                name,
                target_space;
                reference_date = start_date,
                regridder_type = :InterpolationsRegridder,
                regridder_kwargs = (; extrapolation_bc),
                method = LinearInterpolation(),
            ) for name in prescribed_aerosol_names
        ]

        # Field is updated in the radiation callback
        prescribed_aerosols_field = similar(
            Y.c,
            NamedTuple{
                prescribed_aerosol_names_as_symbols,
                NTuple{
                    length(prescribed_aerosol_names_as_symbols),
                    eltype(Y.c.ρ),
                },
            },
        )
        prescribed_aerosol_timevaryinginputs =
            (; zip(prescribed_aerosol_names_as_symbols, timevaryinginputs)...)
        aerosol_cache =
            (; prescribed_aerosols_field, prescribed_aerosol_timevaryinginputs)
    else
        aerosol_cache = (;)
    end
    o3_cache = ozone_cache(atmos.ozone, Y, start_date)
    co2_cache_nt = co2_cache(atmos.co2, Y, start_date)
    return (; aerosol_cache..., o3_cache..., co2_cache_nt...)
end
