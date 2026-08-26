"""
Optional: write the sampled SSCF forcing to a canonical ClimaColumn NetCDF file.

Nothing in the run path uses this. It exists so the forcing can be inspected with ordinary NetCDF
tools, handed to another model, or driven back through ClimaAtmos's stock file-backed
`Setups.ForcingFromFile` as an independent check that the in-memory path agrees with the file path.
"""

using ClimaAtmos: ClimaAtmos as CA

"""
    write_climacolumn(FT, case, path; z, dt_sec, start_date, overwrite)

Write the forcing arrays for `case` to `path` in the ClimaColumn schema and return `path`.

`z` defaults to the model's own levels at the default resolution, so the written file matches what
a default run is driven by.
"""
function write_climacolumn(
    ::Type{FT},
    case::SocratesCase,
    path::AbstractString;
    z::AbstractVector = socrates_z(FT, case),
    dt_sec::Real = DEFAULT_FORCING_DT,
    start_date::Dates.DateTime = simulation_start_date(case),
    overwrite::Bool = false,
) where {FT <: AbstractFloat}
    if isfile(path) && !overwrite
        CA.ColumnDatasets.ClimaColumnFiles.is_conforming(path) && return path
        error("$path exists and is not a conforming ClimaColumn file; pass `overwrite = true`")
    end
    arrays = socrates_forcing_arrays(FT, case; z, dt_sec, start_date)
    mkpath(dirname(abspath(path)))
    times_dt = arrays.start_date .+ Dates.Millisecond.(round.(Int, 1000 .* arrays.times))
    CA.ColumnDatasets.ClimaColumnFiles.write_column_forcing_file(
        path,
        FT;
        z = arrays.z,
        time = times_dt,
        time_attrib = [
            "units" => "seconds since 1970-01-01T00:00:00",
            "calendar" => "proleptic_gregorian",
        ],
        column_vars = arrays.column,
        surface_vars = arrays.surface,
        site_latitude = arrays.lat,
        site_longitude = arrays.lon,
    )
    return path
end

write_climacolumn(case::SocratesCase, path::AbstractString; kwargs...) =
    write_climacolumn(Float64, case, path; kwargs...)