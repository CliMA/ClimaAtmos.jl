"""
    ClimaColumnFiles

The native CliMA column forcing format: pure 1D `(z, time)` column variables
plus `(time,)` surface variables, canonical CMIP short names with SI `units`
attributes, a strictly ascending height coordinate `z` [m], CF time, and the
global attributes `site_latitude` and `site_longitude`. Files are recognized
by this structure, so any file in the right shape works without special
markers.
"""
module ClimaColumnFiles

import NCDatasets as NC
import Dates

import ..ColumnDatasets:
    AbstractColumnFormat,
    format_name,
    format_variable_name,
    height_profile,
    site_location,
    validate,
    CANONICAL_COLUMN_VARS,
    CANONICAL_SURFACE_VARS

"""
SI units of the canonical variables, written as each variable's `units`
attribute by [`write_column_forcing_file`](@ref).
"""
const CANONICAL_UNITS = Dict(
    "ta" => "K",
    "hus" => "kg kg-1",
    "ua" => "m s-1",
    "va" => "m s-1",
    "wa" => "m s-1",
    "rho" => "kg m-3",
    "tntha" => "K s-1",
    "tnhusha" => "kg kg-1 s-1",
    "tntva" => "K s-1",
    "tnhusva" => "kg kg-1 s-1",
    "clw" => "kg kg-1",
    "cli" => "kg kg-1",
    "ts" => "K",
    "hfls" => "W m-2",
    "hfss" => "W m-2",
    "coszen" => "1",
    "rsdt" => "W m-2",
)

"""
    ClimaColumnFile

Dataset-format singleton for files following the ClimaColumn schema.
"""
struct ClimaColumnFile <: AbstractColumnFormat end

format_name(::ClimaColumnFile) = "ClimaColumn"

format_variable_name(::ClimaColumnFile, name::Symbol) = string(name)

height_profile(::ClimaColumnFile, ds, options) = Float64.(vec(ds["z"][:]))

site_location(::ClimaColumnFile, ds) = (;
    latitude = Float64(ds.attrib["site_latitude"]),
    longitude = Float64(ds.attrib["site_longitude"]),
)

"""
    is_conforming(path)

Whether `path` is a file that fully conforms to the ClimaColumn schema, i.e.
[`validate`](@ref) passes (native `(z, time)` layout, strictly-ascending `z`,
canonical SI units, site attributes). Returns `false` rather than throwing, so
this can gate cache reuse: a file left by an older writer in a stale layout — or
a structurally-native file that fails validation (e.g. unsorted `z`) — is
treated as non-conforming and regenerated instead of read (which would fail
loudly in `ColumnDataset`).
"""
function is_conforming(path)
    isfile(path) || return false
    return try
        validate(ClimaColumnFile(), path)
        true
    catch err
        err isa ErrorException || rethrow()
        false
    end
end

"""
    validate(::ClimaColumnFile, path)

Check that `path` follows the ClimaColumn schema and throw a descriptive
error otherwise. Checks the `(z, time)` layout, a strictly ascending `z`
coordinate with at least two levels, a CF-decodable `time` coordinate,
exact canonical SI `units` on all recognized data variables, and the site
location attributes.
"""
function validate(::ClimaColumnFile, path)
    problems = String[]
    NC.NCDataset(path, "r") do ds
        for (attrib, lo, hi) in
            (("site_latitude", -90, 90), ("site_longitude", -180, 180))
            if !haskey(ds.attrib, attrib)
                push!(problems, "missing global attribute `$attrib`")
            elseif !(lo <= ds.attrib[attrib] <= hi)
                push!(
                    problems,
                    "`$attrib` must be in [$lo, $hi] degrees; got $(ds.attrib[attrib])",
                )
            end
        end
        if haskey(ds, "z")
            z = vec(ds["z"][:])
            length(z) >= 2 || push!(problems, "`z` must have at least 2 levels")
            issorted(z; lt = <=) ||
                push!(problems, "`z` must be strictly ascending")
            get(ds["z"].attrib, "units", nothing) == "m" ||
                push!(problems, "`z` must have units `m`")
        else
            push!(problems, "missing coordinate variable `z`")
        end
        if haskey(ds, "time")
            eltype(vec(ds["time"][:])) <: Dates.AbstractDateTime ||
                push!(problems, "`time` must be CF-decodable to dates")
        else
            push!(problems, "missing coordinate variable `time`")
        end
        check_variables!(
            problems,
            ds,
            (CANONICAL_COLUMN_VARS..., :rho, :clw, :cli),
            ("z", "time"),
            "column",
        )
        check_variables!(problems, ds, CANONICAL_SURFACE_VARS, ("time",), "surface")
    end
    isempty(problems) || error(
        "Column forcing file $path does not follow the ClimaColumn \
         schema:\n - " * join(problems, "\n - "),
    )
    return nothing
end

# Check the dimensions and canonical SI units of each recognized variable in
# `names` that the file carries, appending a message per violation.
function check_variables!(problems, ds, names, expected_dims, kind)
    for name in names
        haskey(ds, string(name)) || continue
        var = ds[string(name)]
        NC.dimnames(var) == expected_dims || push!(
            problems,
            "$kind variable `$name` must have dimensions $expected_dims",
        )
        expected_units = variable_units(name)
        actual_units = get(var.attrib, "units", nothing)
        actual_units == expected_units || push!(
            problems,
            "$kind variable `$name` must have units `$expected_units`; " *
            "found `$(something(actual_units, "missing"))`",
        )
    end
    return nothing
end

variable_units(name) = get(CANONICAL_UNITS, String(name)) do
    error(
        "No canonical units registered for variable `$name`; add it to \
         ClimaColumnFiles.CANONICAL_UNITS",
    )
end

"""
    write_column_forcing_file(path, FT;
        z, time, time_attrib, column_vars, surface_vars,
        site_latitude, site_longitude)

Write a ClimaColumn schema file: the one producer implementation shared by
the ERA5 generator and any future converter.

Arguments:

  - `z`: Strictly ascending heights [m] of the column levels.
  - `time`: Vector of `DateTime`s; written with `time_attrib` (CF units and
    calendar).
  - `column_vars`: `name => (z × time) matrix` for the column variables.
  - `surface_vars`: `name => (time,) vector` for the surface variables.
  - `site_latitude`, `site_longitude`: Site coordinates [degrees].

Variable names must have units registered in `CANONICAL_UNITS`.
"""
function write_column_forcing_file(
    path,
    FT;
    z,
    time,
    time_attrib,
    column_vars,
    surface_vars,
    site_latitude,
    site_longitude,
)
    ds = NC.NCDataset(path, "c")
    try
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["site_latitude"] = site_latitude
        ds.attrib["site_longitude"] = site_longitude

        NC.defDim(ds, "z", length(z))
        NC.defDim(ds, "time", length(time))

        NC.defVar(
            ds,
            "z",
            FT.(z),
            ("z",),
            attrib = [
                "units" => "m",
                "long_name" => "height above mean sea level",
            ],
        )
        NC.defVar(ds, "time", time, ("time",), attrib = time_attrib)
        for (name, data) in column_vars
            NC.defVar(
                ds,
                String(name),
                FT.(data),
                ("z", "time"),
                attrib = ["units" => variable_units(name)],
            )
        end
        for (name, data) in surface_vars
            NC.defVar(
                ds,
                String(name),
                FT.(data),
                ("time",),
                attrib = ["units" => variable_units(name)],
            )
        end
    finally
        close(ds)
    end
    return path
end

end # module
