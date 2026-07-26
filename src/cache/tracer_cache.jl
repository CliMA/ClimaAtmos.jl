import Dates: Year
import ClimaUtilities
import ClimaUtilities.TimeVaryingInputs
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput, LinearInterpolation
import Interpolations as Intp

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

prescribed_bin_names(::Union{Nothing, AbstractPrognosticAerosol}) = ()
prescribed_bin_names(m::AbstractPrescribedAerosol) = bin_names(m)
prescribed_bin_names(a::AtmosAerosols) = foldl(
    (names, m) -> (names..., prescribed_bin_names(m)...),
    values(species_models(a));
    init = (),
)

"""
    prescribed_aerosol_cache(Y, aerosol_model::AtmosAerosols, start_date)

MERRA-2 inputs for prescribed aerosol species: a center Field of
per-bin concentrations (updated in `update_aerosol_concentrations!`) and
per bin `TimeVaryingInput`s. Returns (;) when no species is prescribed.
"""
function prescribed_aerosol_cache(Y, aerosol_model::AtmosAerosols, start_date)
    prescribed_names = prescribed_bin_names(aerosol_model)
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
    prescribed_aerosols_field =
        similar(
            Y.c,
            NamedTuple{prescribed_names, NTuple{length(prescribed_names), eltype(Y.c.ρ)}},
        )
    return (;
        prescribed_aerosols_field,
        prescribed_aerosol_timevaryinginputs,
    )
end

function tracer_cache(Y, aerosol_model::AtmosAerosols, time_varying_trace_gases, start_date)
    aerosol_cache = prescribed_aerosol_cache(Y, aerosol_model, start_date)

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

    return (; aerosol_cache..., o3_cache..., co2_cache_nt...)
end
