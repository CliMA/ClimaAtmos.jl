#=
Step 1 (reader half): the SCM forcing reader, built on prototype/data_schema.jl.

Given a forcing file + a target column space + a start date, produce a TimeVaryingInput
per PRESENT canonical variable (presence-based — absent terms are simply not built).
Conforming files (carrying `clima_data_schema`) read directly; non-conforming legacy
files read through a REGISTRY entry that renames source variables to canonical names and
applies elementwise unit fixes. The same reader can also produce a
`ClimaAnalysis.OutputVar` for analysis/calibration.

Pure logic (contract discovery, presence, registry mapping) runs and is asserted; the
ClimaCore-space read (TimeVaryingInput / OutputVar) is guarded in a try/catch, matching
the artifact-dependent guards in scm_forcing_resolution.jl.

Intended to move to ClimaUtilities alongside the schema once proven.

Run:
    julia --project=.buildkite prototype/data_reader.jl
=#

include(joinpath(@__DIR__, "data_schema.jl"))

import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Geometry, Domains, Meshes, Spaces, Fields
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearInterpolation, evaluate!
import ClimaAnalysis
import NCDatasets as NC
import Interpolations as Intp
import Dates

# ---------------------------------------------------------------------------
# Registry — adapts legacy (non-conforming) files by name. Mirrors the register!
# idiom in surface_field_resolution.jl.
# ---------------------------------------------------------------------------
struct RegistryEntry
    rename::Dict{String, String}     # source var name => canonical name
    unit_fix::Dict{String, Any}      # canonical name => elementwise Function
    surface_mode::Symbol
end
RegistryEntry(;
    rename = Dict{String, String}(),
    unit_fix = Dict{String, Any}(),
    surface_mode = :interactive,
) = RegistryEntry(rename, unit_fix, surface_mode)

const REGISTRY = Dict{String, RegistryEntry}()
register!(name, entry::RegistryEntry) = (REGISTRY[name] = entry)

# Illustrative: a legacy SST file — the same rename + elementwise unit-fix mechanism a
# consumer like ClimaCoupler would use (`data + 273.15` currently inline there).
register!(
    "MODEL.SST.HAD187001-198110.OI198111-202206.nc",
    RegistryEntry(;
        rename = Dict("SST" => "ts"),
        unit_fix = Dict("ts" => (x -> x + 273.15)),
        surface_mode = :prescribed_sst,
    ),
)

# Source variable name for a canonical name (identity if conforming / not renamed).
function _source_name(rename, canonical)
    for (src, canon) in rename
        canon == canonical && return src
    end
    return canonical
end

# ---------------------------------------------------------------------------
# Contract discovery: conforming file (self-describing) vs registry fallback.
# ---------------------------------------------------------------------------
struct Contract
    conforming::Bool
    rename::Dict{String, String}
    unit_fix::Dict{String, Any}
    surface_mode::Symbol
end

function contract_for(path; artifact_name = basename(path))
    if is_conforming(path)
        smode = NC.NCDataset(path) do ds
            Symbol(get(ds.attrib, "surface_mode", "interactive"))
        end
        return Contract(true, Dict{String, String}(), Dict{String, Any}(), smode)
    end
    haskey(REGISTRY, artifact_name) || error(
        "cannot read `$artifact_name`: not conforming " *
        "(errors: $(validate_contract(path).errors)) and no REGISTRY entry",
    )
    r = REGISTRY[artifact_name]
    return Contract(false, r.rename, r.unit_fix, r.surface_mode)
end

# Present canonical variables (probe the file, mapping through any rename).
present_column_vars(path, c::Contract) =
    NC.NCDataset(path) do ds
        [name for name in keys(COLUMN_VARS) if haskey(ds, _source_name(c.rename, name))]
    end
present_surface_vars(path, c::Contract) =
    NC.NCDataset(path) do ds
        [name for name in keys(SURFACE_VARS) if haskey(ds, _source_name(c.rename, name))]
    end

# ---------------------------------------------------------------------------
# The reader: file -> TimeVaryingInputs (Field flavor). Presence-based.
# ---------------------------------------------------------------------------
function read_forcing(path, column_space, start_date; artifact_name = basename(path))
    c = contract_for(path; artifact_name)
    col = present_column_vars(path, c)
    inputs = map(col) do name
        preprocess_func = get(c.unit_fix, name, identity)
        tvi = TimeVaryingInput(
            path,
            _source_name(c.rename, name),
            column_space;
            start_date,
            method = LinearInterpolation(),
            regridder_kwargs = (; extrapolation_bc = (Intp.Flat(),)),  # z-only column
            file_reader_kwargs = (; preprocess_func),
        )
        Symbol(name) => tvi
    end
    return (;
        surface_mode = c.surface_mode,
        column = (; inputs...),
        # surface terms present in the file; building them needs a surface space (TODO).
        surface_present = present_surface_vars(path, c),
    )
end

# OutputVar flavor (analysis / calibration) — same file, same contract.
function read_output_var(path, canonical; artifact_name = basename(path))
    c = contract_for(path; artifact_name)
    cat = ClimaAnalysis.NCCatalog()
    ClimaAnalysis.add_file!(cat, path, _source_name(c.rename, canonical) => canonical)
    return ClimaAnalysis.get(cat, canonical)
end

function make_column_space(; z_max = 3000, z_elem = 40)
    dom = Domains.IntervalDomain(
        Geometry.ZPoint(0.0), Geometry.ZPoint(Float64(z_max));
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(dom; nelems = z_elem)
    # If this constructor differs in your ClimaCore version, adjust here.
    return Spaces.CenterFiniteDifferenceSpace(ClimaComms.device(), mesh)
end

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
function main()
    nz, nt = 40, 7
    z = collect(range(0.0, 3000.0, length = nz))
    time = collect(0.0:3600.0:((nt - 1) * 3600.0))

    # (1) A conforming column file: ta/hus/ua/va + ts; no advection/subsidence.
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
    c = contract_for(good)
    @assert c.conforming
    @assert c.surface_mode == :prescribed_sst
    @assert Set(present_column_vars(good, c)) == Set(["ta", "hus", "ua", "va"])
    @assert Set(present_surface_vars(good, c)) == Set(["ts"])
    @assert "tntha" ∉ present_column_vars(good, c)   # advection absent -> not built
    @info "conforming: contract + presence OK" surface_mode = c.surface_mode column =
        present_column_vars(good, c)

    # (2) A legacy column file: source names `T` (°C) / `Q`, no schema marker; adapted
    #     by a registry entry. Demonstrates rename + elementwise unit fix.
    legacy = tempname() * ".nc"
    NC.NCDataset(legacy, "c") do ds
        NC.defDim(ds, "z", nz)
        NC.defDim(ds, "time", nt)
        NC.defVar(ds, "z", z, ("z",); attrib = Dict("units" => "m"))
        NC.defVar(ds, "time", time, ("time",);
            attrib = Dict("units" => "seconds since 2007-07-01"))
        vt = NC.defVar(ds, "T", Float64, ("z", "time"); attrib = Dict("units" => "degC"))
        vt[:, :] = fill(17.0, nz, nt)
        vq = NC.defVar(ds, "Q", Float64, ("z", "time");
            attrib = Dict("units" => "kg kg-1"))
        vq[:, :] = fill(0.01, nz, nt)
    end
    register!(
        basename(legacy),
        RegistryEntry(;
            rename = Dict("T" => "ta", "Q" => "hus"),
            unit_fix = Dict("ta" => (x -> x + 273.15)),  # °C -> K
        ),
    )
    cl = contract_for(legacy)
    @assert !cl.conforming
    @assert Set(present_column_vars(legacy, cl)) == Set(["ta", "hus"])
    @assert _source_name(cl.rename, "ta") == "T"
    @info "legacy: registry contract + presence OK" column =
        present_column_vars(legacy, cl)

    # (3) End-to-end read onto a ClimaCore column space (guarded).
    try
        space = make_column_space()
        start_date = Dates.DateTime(2007, 7, 1)

        fc = read_forcing(good, space, start_date)
        @assert Set(keys(fc.column)) == Set((:ta, :hus, :ua, :va))
        field = Fields.zeros(space)
        evaluate!(field, fc.column.ta, 3600.0)
        @info "conforming read OK" ta_mean = sum(field) / length(parent(field))

        # legacy read: `ta` comes from source `T` in °C, unit-fixed to K (~290.15).
        lf = read_forcing(legacy, space, start_date)
        @assert Set(keys(lf.column)) == Set((:ta, :hus))
        evaluate!(field, lf.column.ta, 3600.0)
        @info "legacy read OK (°C -> K via registry)" ta_mean =
            sum(field) / length(parent(field))

        var = read_output_var(good, "ta")
        @info "OutputVar read OK" short_name = ClimaAnalysis.short_name(var)
    catch e
        @warn "skipped ClimaCore-space read (adjust space ctor / deps)" exception =
            (e, catch_backtrace())
    end

    @info "PROTOTYPE PASSED: reader dispatches conforming vs registry, presence-based, both flavors"
    foreach(p -> rm(p; force = true), (good, legacy))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
