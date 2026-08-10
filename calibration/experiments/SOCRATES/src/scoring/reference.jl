"""
Atlas LES reference data as `ClimaAnalysis.OutputVar`s.

Two sources are supported and must agree:

  - `:processed` — the reduced files under `calibrate/reference/Atlas_LES/`, already SI with time
    in seconds and the vertical dimension named `zc`.
  - `:sscf` — the raw Atlas SAM output from the SSCF artifact. Mixing ratios are `g/kg`, water
    paths `g/m²`, time is in **days**, the vertical dimension is `z`, and absent data is flagged by
    `missing_value = -9999` and (for some variables) a CF `_FillValue`. All of that is handled here.

Variables are named by their **ClimaAtmos short name**, with units strings matching what ClimaAtmos
writes, so the reference and the model output are directly comparable and `GEnsembleBuilder`'s
short-name and units checks pass.
"""

using ClimaAnalysis: ClimaAnalysis
using NCDatasets: NCDatasets as NC

# `OutputVar` binds the i-th key of `dims` to `size(data)[i]`, so the dimension container must be
# ordered — a plain `Dict` would silently mislabel axes of equal length. ClimaAnalysis imports the
# type into its submodules, so it is reachable without adding a direct dependency (the same path
# ClimaCalibrate uses for `ClimaAnalysis.Var.Metadata`).

"""
    LES_VARIABLE

Atmos short name → (LES variable, kind). `:profile` variables are `(z, time)` mass fractions;
`:path` variables are `(time,)` column integrals.

The liquid path maps to `CWP` ("Cloud Water Path"), not `LWP` ("GCSS Liquid Water Path"). ClimaAtmos's
`lwp` is `∫ρq_lcl dz`, cloud liquid only, and `CWP` is that same integral: it reproduces `∫ρ·QCL dz`
from the LES profiles to a constant 0.16 % (scatter 4e-5), whereas `LWP` runs ~1 % higher with an excess
correlating (+0.84) with precipitating area.

`rwp`/`swp` are weak constraints: the reported `RWP`/`SWP` sit ~10 % and ~15 % above the column integral
of `QPL`/`QPI` with flow-dependent scatter, so the precipitation signal is carried by the `husra`/`hussn`
profiles instead.
"""
const LES_VARIABLE = Base.ImmutableDict(
    "clw" => ("QCL", :profile),
    "cli" => ("QCI", :profile),
    "husra" => ("QPL", :profile),
    "hussn" => ("QPI", :profile),
    "lwp" => ("CWP", :path),
    "iwp" => ("IWP", :path),
    "rwp" => ("RWP", :path),
    "swp" => ("SWP", :path),
)

"""The scored variables, in a stable order."""
const REFERENCE_VARS =
    ("clw", "cli", "husra", "hussn", "lwp", "iwp", "rwp", "swp")

"""Units ClimaAtmos writes for each kind, so the reference matches the model side exactly."""
const KIND_UNITS = Base.ImmutableDict(:profile => "kg kg^-1", :path => "kg m^-2")

"""Sentinel the raw Atlas files use for absent data, in addition to any CF `_FillValue`."""
const ATLAS_MISSING_VALUE = -9999.0

"""Directory holding the reduced `<case>_processed.nc` reference files."""
processed_reference_dir() =
    joinpath(dirname(dirname(@__DIR__)), "reference", "Atlas_LES")

"""Path of the reduced reference file for `case`."""
processed_reference_path(case::SocratesCase) =
    joinpath(processed_reference_dir(), string(case_name(case), "_processed.nc"))

"""
    les_outputvars(case; source, vars)

The Atlas LES reference for `case`, as a `Dict` of Atmos short name → `ClimaAnalysis.OutputVar`.

Profiles carry dimensions `("z", "time")`; paths carry `("time",)`. `time` is elapsed seconds from
the start of the LES run, which is also the model's `t = 0`, so the two clocks align without any
date arithmetic.
"""
function les_outputvars(
    case::SocratesCase;
    source::Symbol = :processed,
    vars = REFERENCE_VARS,
)
    source in (:processed, :sscf) ||
        error("Unknown LES reference source `:$source`; expected `:processed` or `:sscf`.")
    raw = source === :processed ? _read_processed(case, vars) : _read_sscf(case, vars)
    return Dict{String, ClimaAnalysis.OutputVar}(
        name => _outputvar(name, last(LES_VARIABLE[name]), raw) for name in vars
    )
end

function _outputvar(name, kind, raw)
    attribs = Dict{String, Any}(
        "short_name" => name,
        "units" => KIND_UNITS[kind],
        "long_name" => "Atlas LES $name",
    )
    if kind === :profile
        dims = ClimaAnalysis.Var.OrderedDict("z" => raw.z, "time" => raw.time)
        dim_attribs =
            ClimaAnalysis.Var.OrderedDict("z" => Dict("units" => "m"), "time" => Dict("units" => "s"))
    else
        dims = ClimaAnalysis.Var.OrderedDict("time" => raw.time)
        dim_attribs = ClimaAnalysis.Var.OrderedDict("time" => Dict("units" => "s"))
    end
    return ClimaAnalysis.OutputVar(attribs, dims, dim_attribs, raw.data[name])
end

"""
    reference_on_levels(var, levels; rtol = DEFAULT_EDGE_RTOL)

`var` resampled in altitude onto `levels` [m]; returned unchanged when it has no altitude dimension.

Two things widen the range a level may be requested at, for two different reasons:

  - A cell-centred profile carries its value across the whole cell, so the reference is extended to
    the outer faces of its end cells. A level in an end half-cell then reads that cell's value
    instead of being rejected, which is the correct reading of cell-centred data — `resampled_as`
    otherwise throws for anything past the outermost *centres*.
  - The faces themselves are only known to the precision the altitudes were stored at. The Atlas
    files record z as Float32, so a face derived from them sits ~1e-3 m from the same face derived
    from `SSCF.default_new_z`. A level within `rtol` of an end half-cell's thickness is therefore
    taken as being *at* the boundary and clamped onto it; the returned altitudes are the levels as
    requested, since the clamp moves them by far less than any physical scale.

Beyond that band the level is genuinely outside the LES column and is an error.
"""
function reference_on_levels(
    var::ClimaAnalysis.OutputVar,
    levels;
    rtol::Real = DEFAULT_EDGE_RTOL,
)
    ClimaAnalysis.has_altitude(var) || return var
    padded = _pad_to_cell_extent(var)
    zp = padded.dims[ClimaAnalysis.altitude_name(padded)]
    lo, hi = first(zp), last(zp)
    tol_lo, tol_hi = rtol * (zp[2] - lo), rtol * (hi - zp[end - 1])
    requested = collect(Float64, levels)
    outside = filter(v -> v < lo - tol_lo || v > hi + tol_hi, requested)
    isempty(outside) || error(
        "Levels $outside m lie outside the Atlas LES column, which spans $(lo) to $(hi) m \
         including its end cells (tolerance $(tol_lo) / $(tol_hi) m), so the reference has no value \
         there.",
    )
    resampled = ClimaAnalysis.resampled_as(padded; z = clamp.(requested, lo, hi))
    return _with_altitude(resampled, requested)
end

"""
Fraction of an end half-cell's thickness within which a requested level counts as being on the
boundary. Covers the disagreement between faces derived from Float32 and Float64 altitudes while
staying orders of magnitude below any physical scale.
"""
const DEFAULT_EDGE_RTOL = 1.0e-4

# Relabel the altitude dimension, so a level clamped onto the boundary is still reported at the value
# it was asked for and matches the model's own grid exactly.
function _with_altitude(var::ClimaAnalysis.OutputVar, z)
    z_name = ClimaAnalysis.altitude_name(var)
    dims = ClimaAnalysis.Var.OrderedDict(
        name => name == z_name ? collect(Float64, z) : collect(d) for (name, d) in var.dims
    )
    return ClimaAnalysis.OutputVar(var.attributes, dims, var.dim_attributes, var.data)
end

# Extend a cell-centred profile to the outer faces of its end cells, each edge carrying the adjacent
# centre's value, so the interpolant is flat across the half-cell the data still describes.
function _pad_to_cell_extent(var::ClimaAnalysis.OutputVar)
    z_name = ClimaAnalysis.altitude_name(var)
    z = collect(Float64, var.dims[z_name])
    faces = SM.faces_from_centers(z)
    axis = var.dim2index[z_name]
    n = size(var.data, axis)
    data = cat(
        selectdim(var.data, axis, 1:1),
        var.data,
        selectdim(var.data, axis, n:n);
        dims = axis,
    )
    dims = ClimaAnalysis.Var.OrderedDict(
        name => name == z_name ? vcat(first(faces), z, last(faces)) : collect(d) for
        (name, d) in var.dims
    )
    return ClimaAnalysis.OutputVar(var.attributes, dims, var.dim_attributes, data)
end

"""
    les_raw_profiles(case, names) -> (; z, time, data)

Named variables read straight out of the processed reference file, without going through
[`LES_VARIABLE`](@ref).

For diagnostics that are not scored — the SAM microphysics process rates, for instance — where the
variable has no Atmos short name. Absent data is masked exactly as [`les_outputvars`](@ref) masks it,
and `time` is elapsed seconds. Names missing from the file are skipped rather than raising, since
which rates a file carries varies.
"""
function les_raw_profiles(case::SocratesCase, names)
    path = processed_reference_path(case)
    isfile(path) || error("Processed Atlas LES reference not found: $path")
    return NC.NCDataset(path, "r") do ds
        z = Float64.(vec(Array(ds["zc"])))
        time = _elapsed(Float64.(vec(Array(ds["time"]))))
        data = Dict{String, Array{Float64}}()
        for name in names
            haskey(ds, name) || continue
            a = _clean(Array(ds[name]), 1.0)
            # A `(time, zc)` CDL declaration is read transposed; orient on the time axis.
            data[name] = size(a, 1) == length(time) ? permutedims(a) : a
        end
        (; z, time, data)
    end
end

# --- readers ------------------------------------------------------------------------------- #

function _read_processed(case::SocratesCase, vars)
    path = processed_reference_path(case)
    isfile(path) || error(
        "Processed Atlas LES reference not found: $path. Generate it, or read the raw artifact \
         with `source = :sscf`.",
    )
    return NC.NCDataset(path, "r") do ds
        _read_dataset(ds, vars, "zc", 1.0, identity, path)
    end
end

function _read_sscf(case::SocratesCase, vars)
    les = SSCF.open_atlas_les_output(case.flight_number, case.forcing_type)
    # g/kg -> kg/kg and g/m^2 -> kg/m^2 are both a factor 1/1000; time is stored in days.
    return _read_dataset(les.data, vars, "z", 1.0e-3, t -> t .* 86400, "raw SSCF artifact")
end

function _read_dataset(ds, vars, z_name, scale, time_transform, where)
    z = Float64.(vec(Array(ds[z_name])))
    time = _elapsed(time_transform(Float64.(vec(Array(ds["time"])))))
    data = Dict{String, Array{Float64}}()
    for name in vars
        haskey(LES_VARIABLE, name) ||
            error("No LES mapping for Atmos short name `$name`")
        les_name, kind = LES_VARIABLE[name]
        haskey(ds, les_name) ||
            error("$where has no variable `$les_name` (needed for `$name`)")
        data[name] = _clean(Array(ds[les_name]), scale)
        expected = kind === :profile ? (length(z), length(time)) : (length(time),)
        size(data[name]) == expected || error(
            "$where variable `$les_name` has size $(size(data[name])); expected $expected. A \
             `(time, z)` CDL declaration is read transposed, so a mismatch here means the file \
             layout differs from what this reader assumes.",
        )
    end
    return (; z, time, data)
end

"""Elapsed seconds from the first sample, so `t = 0` is the start of the LES run."""
_elapsed(t) = t .- first(t)

"""
    _clean(a, scale)

`a` as `Float64` with absent data replaced by `NaN`, then scaled by `scale`.

NCDatasets maps a CF `_FillValue` to `missing`, which would make a bare `Float64.(…)` throw; the
Atlas `missing_value = -9999` sentinel is not masked automatically at all and would otherwise enter
a mean silently.
"""
function _clean(a::AbstractArray, scale::Real)
    out = Array{Float64}(undef, size(a))
    @inbounds for i in eachindex(a, out)
        v = a[i]
        out[i] = if v === missing
            NaN
        else
            x = Float64(v)
            x == ATLAS_MISSING_VALUE ? NaN : scale * x
        end
    end
    return out
end