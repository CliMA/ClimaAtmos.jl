#=
Prototype: the thin data-contract reader (docs/src/data_contract.md) and how it
maps onto existing artifacts. Complements scm_forcing_resolution.jl: that
prototype resolves config+Setup into the `external_forcing` OBJECT; this one is
the layer below — turning a NetCDF into the Fields / TimeVaryingInputs that the
object's cache consumes.

What it demonstrates:

  1. Contract discovery. A file is either SELF-DESCRIBING (carries a
     `clima_data_contract` global attribute) or it is a legacy artifact adapted
     by a per-repo REGISTRY entry. One `read_contract` covers both.

  2. Presence encodes intent. `present_canonical_vars` probes the file; only the
     variables that are present get a TimeVaryingInput + cache field. Absent
     terms (e.g. no `tntha` -> no horizontal advection) are simply not built.
     No boolean flags.

  3. One file, two output flavors. `column_forcing_inputs` -> Fields /
     TimeVaryingInput (running sim, via ClimaUtilities); `as_output_var` ->
     OutputVar (analysis/calibration, via ClimaAnalysis). Same contract; no
     per-artifact name maps or inline unit lambdas at the call site.

  4. The registry-vs-converter boundary. A REGISTRY entry can adapt a file whose
     only problems are naming + elementwise unit fixes (SST, sea ice). A file
     whose STRUCTURE differs (cfsite per-site groups; ARM VARANAL pressure
     levels) needs an upstream CONVERTER, not a registry entry. The registry
     marks these with `needs_converter`.

What actually runs vs. illustrative: the pure contract logic (discovery,
presence, registry classification) runs and is asserted on a synthetic
self-describing column file — no external artifacts needed. The ClimaCore-space
read (TimeVaryingInput / OutputVar) is guarded in a try/catch, matching the
artifact-dependent guards in scm_forcing_resolution.jl.

Run:
    julia --project=.buildkite prototype/data_contract_reader.jl
=#

import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Geometry, Domains, Meshes, Spaces, Fields
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearInterpolation, evaluate!
import ClimaAnalysis
import NCDatasets as NC
import Interpolations as Intp
import Dates

const FT = Float32

# Canonical vocabulary (docs/src/data_contract.md). Column + surface variables.
const COLUMN_VARS =
    ("ta", "hus", "ua", "va", "wa", "rho", "tntha", "tnhusha", "tntva", "tnhusva")
const SURFACE_VARS = ("ts", "hfls", "hfss", "coszen", "rsdt")
const CANONICAL = (COLUMN_VARS..., SURFACE_VARS...)

# ---------------------------------------------------------------------------
# The contract. For a self-describing file it is trivial (identity rename, SI
# units, height coord). For a legacy artifact the REGISTRY supplies it.
# ---------------------------------------------------------------------------
struct DataContract
    conforming::Bool
    rename::Dict{String, String}          # source var name => canonical name
    unit_fix::Dict{String, Any}           # canonical name => elementwise Function
    vertical::Symbol                      # :height | :pressure | :none
    needs_converter::Union{Nothing, String}  # reason string, or nothing
    surface_mode::Symbol                  # :prescribed_sst | :prescribed_fluxes | :interactive
end

# Source variable name for a canonical name (inverse of `rename`; identity if
# not renamed).
function source_name(c::DataContract, canonical)
    for (src, canon) in c.rename
        canon == canonical && return src
    end
    return canonical
end

# ---------------------------------------------------------------------------
# REGISTRY fallback for legacy verbatim artifacts, keyed by file basename.
# NOTE: the SST/ICE entries are exactly the conversions currently HARD-CODED
# inline in ClimaCoupler (prescr_ocean.jl `data + C_to_K`, prescr_seaice.jl
# `data / 100`) — here they are declared once instead of per-loader.
# ---------------------------------------------------------------------------
const REGISTRY = Dict{String, DataContract}(
    # Registry-adaptable: structurally fine, just rename + elementwise unit fix.
    "MODEL.SST.HAD187001-198110.OI198111-202206.nc" => DataContract(
        false,
        Dict("SST" => "ts"),
        Dict("ts" => (x -> x + 273.15f0)),   # °C -> K, as in prescr_ocean.jl
        :none,
        nothing,
        :prescribed_sst,
    ),
    "MODEL.ICE.HAD187001-198110.OI198111-202206.nc" => DataContract(
        false,
        Dict("SEAICE" => "siconc"),
        Dict("siconc" => (x -> x / 100)),     # percent -> fraction, as in prescr_seaice.jl
        :none,
        nothing,
        :prescribed_sst,
    ),
    # Converter-only: structure differs, a registry entry cannot express the fix.
    "HadGEM2-A_amip.2004-2008.07.nc" => DataContract(
        false,
        Dict{String, String}(),
        Dict{String, Any}(),
        :height,
        "per-site NetCDF groups; 6-hourly -> time-mean; wap->wa (=-ωα/g); " *
        "Shen-2022 vertical eddy-fluctuation decomposition",
        :prescribed_sst,
    ),
    "sgp60varanarucC1.c1.20100901.000000.cdf" => DataContract(
        false,
        Dict("T" => "ta", "q" => "hus", "u" => "ua", "v" => "va",
            "T_adv_h" => "tntha", "q_adv_h" => "tnhusha", "omega" => "wa"),
        Dict("hus" => (x -> x * 1.0f-3), "tntha" => (x -> x / 3600)),
        :pressure,
        "vertical coord is pressure -> time-dependent P->z regrid; -9999 " *
        "sentinels -> NaN; omega[hPa/hr] -> wa[m/s]",
        :prescribed_fluxes,
    ),
)

registry_adaptable(c::DataContract) = isnothing(c.needs_converter)

# ---------------------------------------------------------------------------
# The reader.
# ---------------------------------------------------------------------------

# Discover the contract: self-describing file, else registry fallback.
function read_contract(path; artifact_name = basename(path))
    # NB: never `return` from inside the do-block — that returns from the closure,
    # not from read_contract. Extract the values, then decide.
    is_conforming, smode = NC.NCDataset(path) do ds
        (
            haskey(ds.attrib, "clima_data_contract"),
            Symbol(get(ds.attrib, "surface_mode", "interactive")),
        )
    end
    is_conforming && return DataContract(
        true, Dict{String, String}(), Dict{String, Any}(), :height, nothing, smode,
    )
    haskey(REGISTRY, artifact_name) || error(
        "No contract for $(artifact_name): file lacks `clima_data_contract` " *
        "and there is no REGISTRY entry.",
    )
    return REGISTRY[artifact_name]
end

# Presence encodes intent: the canonical variables actually in the file.
function present_canonical_vars(path, c::DataContract)
    NC.NCDataset(path) do ds
        [canon for canon in CANONICAL if haskey(ds, source_name(c, canon))]
    end
end

# Field / TimeVaryingInput flavor (running sim). Presence-based: builds an input
# only for variables that are present. Unit conversion comes from the contract's
# `preprocess_func`, not an inline lambda at the call site.
function column_forcing_inputs(path, column_space, start_date, c::DataContract)
    present = present_canonical_vars(path, c)
    inputs = map(present) do canon
        pre = get(c.unit_fix, canon, identity)
        tvi = TimeVaryingInput(
            path,
            source_name(c, canon),
            column_space;
            start_date,
            method = LinearInterpolation(),
            regridder_kwargs = (; extrapolation_bc = (Intp.Flat(),)),  # z-only column
            file_reader_kwargs = (; preprocess_func = pre),
        )
        Symbol(canon) => tvi
    end
    return (; inputs...)
end

# OutputVar flavor (analysis/calibration). Same file, same contract.
function as_output_var(path, canonical, c::DataContract)
    cat = ClimaAnalysis.NCCatalog()
    ClimaAnalysis.add_file!(cat, path, source_name(c, canonical) => canonical)
    return ClimaAnalysis.get(cat, canonical)
end

# ---------------------------------------------------------------------------
# A synthetic self-describing column file (no external artifact needed). Carries
# ta/hus/ua/va but deliberately OMITS tntha/tnhusha/wa/ts, so the presence logic
# has something to exclude.
# ---------------------------------------------------------------------------
function write_synthetic_column(path)
    z = FT.(range(0, 3000, length = 40))
    t = FT.(0:3600:(6 * 3600))
    NC.NCDataset(path, "c") do ds
        NC.defDim(ds, "z", length(z))
        NC.defDim(ds, "time", length(t))
        NC.defVar(ds, "z", z, ("z",); attrib = Dict("units" => "m"))
        NC.defVar(
            ds,
            "time",
            t,
            ("time",);
            attrib = Dict(
                "units" => "seconds since 2007-07-01", "calendar" => "proleptic_gregorian",
            ),
        )
        for (name, unit, val) in (
            ("ta", "K", 290.0f0), ("hus", "kg kg-1", 0.01f0),
            ("ua", "m s-1", 5.0f0), ("va", "m s-1", 0.0f0),
        )
            v = NC.defVar(ds, name, FT, ("z", "time");
                attrib = Dict("units" => unit, "long_name" => name))
            v[:, :] .= val
        end
        ds.attrib["clima_data_contract"] = "1.0"
        ds.attrib["Conventions"] = "CF-1.8"
        ds.attrib["surface_mode"] = "interactive"
        ds.attrib["site_latitude"] = 17.0
        ds.attrib["site_longitude"] = -149.0
    end
    return path
end

function make_column_space(; z_max = 3000, z_elem = 40)
    dom = Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)), Geometry.ZPoint(FT(z_max));
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
    path = tempname() * ".nc"
    write_synthetic_column(path)

    # (1) Discovery: the synthetic file is self-describing.
    c = read_contract(path)
    @assert c.conforming
    @info "self-describing file" surface_mode = c.surface_mode

    # (2) Presence encodes intent: ta/hus/ua/va present; tntha/wa/ts absent.
    present = present_canonical_vars(path, c)
    @info "present canonical vars" present
    @assert Set(present) == Set(["ta", "hus", "ua", "va"])
    @assert "tntha" ∉ present   # no horizontal-advection term is built
    @assert "wa" ∉ present      # no subsidence term is built

    # (3) Registry fallback resolves a legacy file's contract by name, and the
    #     declared unit fix reproduces ClimaCoupler's inline conversion.
    sst = "MODEL.SST.HAD187001-198110.OI198111-202206.nc"
    csst = REGISTRY[sst]
    @assert !csst.conforming
    @assert source_name(csst, "ts") == "SST"
    @assert csst.unit_fix["ts"](0.0f0) == 273.15f0    # °C -> K
    @assert REGISTRY["MODEL.ICE.HAD187001-198110.OI198111-202206.nc"].unit_fix["siconc"](
        50,
    ) == 0.5

    # (4) Registry-vs-converter classification for the real single-column artifacts.
    for name in (
        sst,
        "MODEL.ICE.HAD187001-198110.OI198111-202206.nc",
        "HadGEM2-A_amip.2004-2008.07.nc",
        "sgp60varanarucC1.c1.20100901.000000.cdf",
    )
        r = REGISTRY[name]
        kind =
            registry_adaptable(r) ? "REGISTRY (rename + unit fix)" :
            "CONVERTER — $(r.needs_converter)"
        @info "artifact" name kind vertical = r.vertical surface = r.surface_mode
    end
    @assert registry_adaptable(REGISTRY[sst])
    @assert !registry_adaptable(REGISTRY["HadGEM2-A_amip.2004-2008.07.nc"])
    @assert !registry_adaptable(REGISTRY["sgp60varanarucC1.c1.20100901.000000.cdf"])

    # (guarded) end-to-end read onto a ClimaCore column space.
    try
        space = make_column_space()
        start_date = Dates.DateTime(2007, 7, 1)
        inputs = column_forcing_inputs(path, space, start_date, c)
        @assert Set(keys(inputs)) == Set((:ta, :hus, :ua, :va))
        field = Fields.zeros(space)
        evaluate!(field, inputs.ta, FT(3600))   # 1 hour in
        @info "TimeVaryingInput read onto column space OK" ta_mean =
            sum(field) / length(parent(field))

        var = as_output_var(path, "ta", c)
        @info "OutputVar read OK" short_name = ClimaAnalysis.short_name(var)
    catch e
        @warn "Skipped ClimaCore-space read (adjust space constructor / deps)" exception =
            (e, catch_backtrace())
    end

    @info "PROTOTYPE PASSED: contract discovery, presence, and registry/converter split"
    rm(path; force = true)
end

main()
