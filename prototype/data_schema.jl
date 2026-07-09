#=
Step 1 of the SCM data-schema plan (docs/src/data_contract_proposal.md): define the
schema concretely — a validator and a tiny writer — before building the reader on top.

  - `SCM_SCHEMA` / `COLUMN_VARS` / `SURFACE_VARS`: the canonical vocabulary + units.
  - `validate_contract(path)`: the machine-readable spec. Returns (; errors, warnings);
    a file is conforming iff `errors` is empty.
  - `write_conforming_column(path; ...)`: produces a minimal conforming 1-D `(z, time)`
    column file — the reference for what "conforming" means.
  - `main()`: writes an example conforming file, asserts it validates, and asserts a few
    malformed variants are rejected.

Intended to move to ClimaUtilities once the reader is built on it.

Run:
    julia --project=.buildkite prototype/data_schema.jl
=#

import NCDatasets as NC

const SCHEMA_VERSION = "1.0"
const SURFACE_MODES = ("prescribed_sst", "prescribed_fluxes", "interactive")

# Canonical column variables => (units, long_name). All optional per file: the reader
# builds only the terms that are present. These are the *allowed* names/units.
const COLUMN_VARS = Dict(
    "ta" => ("K", "air_temperature"),
    "hus" => ("kg kg-1", "specific_humidity"),
    "ua" => ("m s-1", "eastward_wind"),
    "va" => ("m s-1", "northward_wind"),
    "wa" => ("m s-1", "upward_air_velocity"),
    "rho" => ("kg m-3", "air_density"),
    "tntha" => ("K s-1", "tendency_of_air_temperature_due_to_horizontal_advection"),
    "tnhusha" =>
        ("kg kg-1 s-1", "tendency_of_specific_humidity_due_to_horizontal_advection"),
    "tntva" => ("K s-1", "tendency_of_air_temperature_due_to_vertical_advection"),
    "tnhusva" =>
        ("kg kg-1 s-1", "tendency_of_specific_humidity_due_to_vertical_advection"),
)

# Canonical surface variables => (units, long_name). Stored on `(time,)`.
const SURFACE_VARS = Dict(
    "ts" => ("K", "surface_temperature"),
    "hfls" => ("W m-2", "surface_upward_latent_heat_flux"),
    "hfss" => ("W m-2", "surface_upward_sensible_heat_flux"),
    "coszen" => ("1", "cosine_of_solar_zenith_angle"),
    "rsdt" => ("W m-2", "toa_incoming_shortwave_flux"),
)

const KNOWN_VARS = merge(COLUMN_VARS, SURFACE_VARS)

# ---------------------------------------------------------------------------
# Validator — the machine-readable form of docs/src/data_contract.md (SCM subset).
# ---------------------------------------------------------------------------
"""
    validate_contract(path) -> (; errors, warnings)

Check a NetCDF file against the SCM forcing schema. `errors` block conformance;
`warnings` are advisory (e.g. extra variables, missing `long_name`). A file is
conforming iff `isempty(errors)`.
"""
function validate_contract(path)
    errors = String[]
    warnings = String[]
    NC.NCDataset(path) do ds
        # 1. conformance marker
        if !haskey(ds.attrib, "clima_data_schema")
            push!(errors, "missing global attribute `clima_data_schema`")
        elseif ds.attrib["clima_data_schema"] != SCHEMA_VERSION
            push!(errors, "`clima_data_schema` must be \"$SCHEMA_VERSION\"")
        end

        # 2. required column dimensions
        for d in ("z", "time")
            haskey(ds.dim, d) || push!(errors, "missing dimension `$d`")
        end

        # 3. coordinate variables exist and are strictly ascending
        for d in ("z", "time")
            haskey(ds.dim, d) || continue
            if !haskey(ds, d)
                push!(errors, "dimension `$d` has no coordinate variable")
            else
                issorted(ds[d][:]) || push!(errors, "coordinate `$d` is not ascending")
            end
        end

        # 4. coordinate units: z in metres, time CF ("... since ...")
        if haskey(ds, "z") && get(ds["z"].attrib, "units", "") != "m"
            push!(errors, "`z` must have `units = \"m\"`")
        end
        if haskey(ds, "time") &&
           !occursin("since", get(ds["time"].attrib, "units", ""))
            push!(errors, "`time` must be CF (`units = \"seconds since <epoch>\"`)")
        end

        # 5. data variables: canonical name + correct SI units
        for vname in keys(ds)
            vname in ("z", "time") && continue
            if !haskey(KNOWN_VARS, vname)
                push!(warnings, "variable `$vname` is not in the canonical vocabulary")
                continue
            end
            want_units = KNOWN_VARS[vname][1]
            got_units = get(ds[vname].attrib, "units", nothing)
            if isnothing(got_units)
                push!(errors, "`$vname` is missing a `units` attribute")
            elseif got_units != want_units
                push!(errors, "`$vname` units \"$got_units\" != expected \"$want_units\"")
            end
            haskey(ds[vname].attrib, "long_name") ||
                push!(warnings, "`$vname` is missing `long_name`")
        end

        # 6. surface_mode, if present, must be recognized
        if haskey(ds.attrib, "surface_mode") &&
           !(ds.attrib["surface_mode"] in SURFACE_MODES)
            push!(errors, "`surface_mode` must be one of $(SURFACE_MODES)")
        end

        # 7. column files carry their site location
        for a in ("site_latitude", "site_longitude")
            haskey(ds.attrib, a) || push!(errors, "missing global attribute `$a`")
        end
    end
    return (; errors, warnings)
end

is_conforming(path) = isempty(validate_contract(path).errors)

# ---------------------------------------------------------------------------
# Writer — the reference for what a conforming column file looks like.
# ---------------------------------------------------------------------------
"""
    write_conforming_column(path; z, time, site_latitude, site_longitude, kwargs...)

Write a minimal conforming 1-D `(z, time)` SCM forcing file. `column` maps canonical
column-variable names to a `(z, time)` matrix (or a length-`z` vector broadcast in
time); `surface` maps canonical surface-variable names to a length-`time` vector.
"""
function write_conforming_column(
    path;
    z,
    time,
    site_latitude,
    site_longitude,
    epoch = "2007-07-01",
    calendar = "proleptic_gregorian",
    surface_mode = "interactive",
    column = Dict{String, Any}(),
    surface = Dict{String, Any}(),
    FT = Float64,
)
    nz, nt = length(z), length(time)
    NC.NCDataset(path, "c") do ds
        NC.defDim(ds, "z", nz)
        NC.defDim(ds, "time", nt)
        NC.defVar(ds, "z", FT.(collect(z)), ("z",);
            attrib = Dict("units" => "m", "positive" => "up"))
        NC.defVar(ds, "time", FT.(collect(time)), ("time",);
            attrib = Dict("units" => "seconds since $epoch", "calendar" => calendar))

        for (name, data) in column
            haskey(COLUMN_VARS, name) || error("unknown column variable `$name`")
            units, long_name = COLUMN_VARS[name]
            arr =
                data isa AbstractMatrix ? data :
                repeat(reshape(collect(data), nz, 1), 1, nt)
            v = NC.defVar(ds, name, FT, ("z", "time");
                attrib = Dict("units" => units, "long_name" => long_name))
            v[:, :] = FT.(arr)
        end
        for (name, data) in surface
            haskey(SURFACE_VARS, name) || error("unknown surface variable `$name`")
            units, long_name = SURFACE_VARS[name]
            v = NC.defVar(ds, name, FT, ("time",);
                attrib = Dict("units" => units, "long_name" => long_name))
            v[:] = FT.(collect(data))
        end

        ds.attrib["clima_data_schema"] = SCHEMA_VERSION
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["surface_mode"] = surface_mode
        ds.attrib["site_latitude"] = FT(site_latitude)
        ds.attrib["site_longitude"] = FT(site_longitude)
    end
    return path
end

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
function _selftest()
    nz, nt = 40, 7
    z = collect(range(0.0, 3000.0, length = nz))
    time = collect(0.0:3600.0:((nt - 1) * 3600.0))

    # A conforming file: ta/hus/ua/va + ts, no advection/subsidence (those terms absent).
    good = tempname() * ".nc"
    write_conforming_column(
        good;
        z,
        time,
        site_latitude = 17.0,
        site_longitude = -149.0,
        surface_mode = "prescribed_sst",
        column = Dict(
            "ta" => fill(290.0, nz, nt),
            "hus" => fill(0.01, nz, nt),
            "ua" => fill(5.0, nz, nt),
            "va" => fill(0.0, nz, nt),
        ),
        surface = Dict("ts" => fill(300.0, nt)),
    )
    res = validate_contract(good)
    @info "conforming example" errors = res.errors warnings = res.warnings
    @assert isempty(res.errors) "example file should be conforming: $(res.errors)"

    # Malformed: descending z.
    bad_z = tempname() * ".nc"
    write_conforming_column(bad_z; z = reverse(z), time,
        site_latitude = 0.0, site_longitude = 0.0,
        column = Dict("ta" => fill(290.0, nz, nt)))
    @assert any(occursin("ascending", e) for e in validate_contract(bad_z).errors)

    # Malformed: wrong units on `ta` (write by hand to bypass the writer's canonical units).
    bad_units = tempname() * ".nc"
    NC.NCDataset(bad_units, "c") do ds
        NC.defDim(ds, "z", nz)
        NC.defDim(ds, "time", nt)
        NC.defVar(ds, "z", z, ("z",); attrib = Dict("units" => "m"))
        NC.defVar(ds, "time", time, ("time",);
            attrib = Dict("units" => "seconds since 2007-07-01"))
        v = NC.defVar(ds, "ta", Float64, ("z", "time"); attrib = Dict("units" => "degC"))
        v[:, :] = fill(290.0, nz, nt)
        ds.attrib["clima_data_schema"] = "1.0"
        ds.attrib["site_latitude"] = 0.0
        ds.attrib["site_longitude"] = 0.0
    end
    @assert any(occursin("units", e) for e in validate_contract(bad_units).errors)

    # Malformed: missing conformance marker.
    bad_attr = tempname() * ".nc"
    NC.NCDataset(bad_attr, "c") do ds
        NC.defDim(ds, "z", nz)
        NC.defDim(ds, "time", nt)
        NC.defVar(ds, "z", z, ("z",); attrib = Dict("units" => "m"))
        NC.defVar(ds, "time", time, ("time",);
            attrib = Dict("units" => "seconds since 2007-07-01"))
    end
    @assert any(
        occursin("clima_data_schema", e) for e in validate_contract(bad_attr).errors
    )

    @info "PROTOTYPE PASSED: validator accepts a conforming file and rejects malformed ones"
    foreach(p -> rm(p; force = true), (good, bad_z, bad_units, bad_attr))
end

# Run the self-test only when executed directly, so this file can be `include`d.
if abspath(PROGRAM_FILE) == @__FILE__
    _selftest()
end
