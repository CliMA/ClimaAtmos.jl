# Shared helpers for variance_adjustments (included by model_interface, observation_map, drivers).
import LinearAlgebra
import TOML
import YAML
import ClimaComms
import ClimaAtmos as CA

"""
Subdirectory of the experiment root holding **named calibration experiment YAMLs** (`experiment_config_*_*.yml`;
each file is one independent EKI configuration / sweep row, not a different binary).
The **default** YAML when none is selected is **`config/experiment_config.yml`** (see [`va_experiment_config_path`](@ref)).
"""
const VA_EXPERIMENT_CONFIGS_DIR = "experiment_configs"

"""Default experiment YAML relative to the experiment directory (canonical layout)."""
const VA_DEFAULT_EXPERIMENT_CONFIG_RELPATH = joinpath("config", "experiment_config.yml")

function _va_default_experiment_config_path(experiment_dir::AbstractString)::String
    p = joinpath(experiment_dir, VA_DEFAULT_EXPERIMENT_CONFIG_RELPATH)
    isfile(p) && return p
    leg = joinpath(experiment_dir, "experiment_config.yml")
    return isfile(leg) ? leg : p
end

"""
    va_experiment_config_path(experiment_dir[, explicit]) -> String

Path to the active experiment YAML. If `explicit` is set (non-empty), it is interpreted relative to
`experiment_dir` unless absolute. Else if **`VA_EXPERIMENT_CONFIG`** is set in the environment, use that
(same relative/absolute rules). Else **`experiment_dir/config/experiment_config.yml`** (if that file exists),
otherwise **`experiment_dir/experiment_config.yml`** at the experiment root if present, else the canonical
**`config/experiment_config.yml`** path (load may error until the file exists).

Named experiment YAMLs (per-case / per-parameter EKI configs) live under **`experiment_dir/$(VA_EXPERIMENT_CONFIGS_DIR)/`**
(e.g. `experiment_configs/experiment_config_googleles_01_N3_varfix_off.yml`).
"""
function va_experiment_config_path(
    experiment_dir::AbstractString,
    explicit::Union{Nothing, AbstractString} = nothing,
)
    if explicit !== nothing && !isempty(strip(string(explicit)))
        p = String(strip(string(explicit)))
        return isabspath(p) ? p : joinpath(experiment_dir, p)
    end
    env = strip(get(ENV, "VA_EXPERIMENT_CONFIG", ""))
    if !isempty(env)
        return isabspath(env) ? env : joinpath(experiment_dir, env)
    end
    return _va_default_experiment_config_path(experiment_dir)
end

"""Load the experiment YAML (same resolution rules as [`va_experiment_config_path`](@ref))."""
function va_load_experiment_config(
    experiment_dir::AbstractString,
    explicit::Union{Nothing, AbstractString} = nothing,
)
    return YAML.load_file(va_experiment_config_path(experiment_dir, explicit))
end

"""True if `sgs_distribution` selects the vertical half-cell layer-profile SGS path (varfix “on”)."""
function va_is_vertical_profile_sgs_distribution(d::AbstractString)
    return occursin(r"^(gaussian|lognormal)_vertical_profile", String(strip(d)))
end

"""
    va_base_to_vertical_profile_sgs_distribution(base_dist) -> String

For sweeps that toggle “varfix”: map a base `sgs_distribution` string to the default
vertical-profile counterpart (`gaussian` → `gaussian_vertical_profile`, etc.).

**Forward sweep override:** if merged case YAML defines `sgs_distribution_varfix_on`, `sweep_forward_core.jl`
uses that full string for `varfix_on` instead of this mapping (then deletes the key before `AtmosConfig`).
"""
function va_base_to_vertical_profile_sgs_distribution(base_dist::AbstractString)
    b = String(base_dist)
    b == "gaussian" && return "gaussian_vertical_profile"
    b == "lognormal" && return "lognormal_vertical_profile"
    return b
end

"""
    va_resolve_sgs_distribution_for_atmos_config(config_dict, expc) -> String

Effective `sgs_distribution` for `AtmosConfig`: `expc["sgs_distribution"]` if set,
otherwise the case YAML `sgs_distribution` (default `lognormal`).
"""
function va_resolve_sgs_distribution_for_atmos_config(config_dict, expc)
    base = string(get(config_dict, "sgs_distribution", "lognormal"))
    if haskey(expc, "sgs_distribution") && expc["sgs_distribution"] !== nothing
        return string(expc["sgs_distribution"])
    end
    return base
end

"""
    va_varfix_tag(expc, merged_case_cfg = nothing) -> String

`"varfix_on"` if the effective `sgs_distribution` uses the vertical-profile family.
Uses `expc["sgs_distribution"]` when set; otherwise `merged_case_cfg["sgs_distribution"]` when
`merged_case_cfg` is provided (merged case YAML).
"""
function va_varfix_tag(expc, merged_case_cfg = nothing)
    d = ""
    if haskey(expc, "sgs_distribution") && expc["sgs_distribution"] !== nothing
        s = string(expc["sgs_distribution"])
        isempty(strip(s)) || (d = s)
    end
    if isempty(d) && merged_case_cfg !== nothing
        d = string(get(merged_case_cfg, "sgs_distribution", ""))
    end
    if !isempty(d) && va_is_vertical_profile_sgs_distribution(d)
        return "varfix_on"
    end
    return "varfix_off"
end

"""
    va_reference_output_dir(experiment_dir, expc) -> String

Directory for the **reference truth** ClimaAtmos run (`output_active` lives underneath). If the YAML
key **`reference_output_dir`** is set, it is used (relative to `experiment_dir` unless absolute).
Otherwise: `simulation_output/<case_name>/N_<quadrature_order>/<varfix_tag>/reference`.
"""
function va_reference_output_dir(experiment_dir::AbstractString, expc)
    if haskey(expc, "reference_output_dir") && expc["reference_output_dir"] !== nothing
        relp = expc["reference_output_dir"]
        s = string(relp)
        !isempty(strip(s)) || error("reference_output_dir is empty")
        return isabspath(s) ? String(s) : joinpath(experiment_dir, s)
    end
    merged_case = va_load_merged_case_yaml_dict(experiment_dir, expc["model_config_path"])
    case = string(expc["case_name"])
    n = Int(expc["quadrature_order"])
    tag = va_varfix_tag(expc, merged_case)
    return joinpath(experiment_dir, "simulation_output", case, "N_$(n)", tag, "reference")
end

"""
    va_default_figure_dir(experiment_dir, expc) -> String

Subfolder under `analysis/figures/` for post-run plots (avoids clobbering between calibration cases).
"""
function va_default_figure_dir(experiment_dir::AbstractString, expc)
    case = replace(string(get(expc, "case_name", "case")), r"[^\w\.\-]+" => "_")
    n = Int(get(expc, "quadrature_order", 0))
    merged_case = va_load_merged_case_yaml_dict(experiment_dir, expc["model_config_path"])
    vf = va_varfix_tag(expc, merged_case)
    mode = replace(string(get(expc, "calibration_mode", "run")), r"[^\w\.\-]+" => "_")
    return joinpath(experiment_dir, "analysis", "figures", "$(case)_N$(n)_$(vf)_$(mode)")
end

"""Absolute path to `observations_path` in `expc`."""
function va_observations_abs_path(experiment_dir::AbstractString, expc)
    p = expc["observations_path"]
    return isabspath(p) ? String(p) : joinpath(experiment_dir, p)
end

"""Root directory of EKI outputs for this experiment YAML (`output_dir` from YAML, resolved relative to `experiment_dir`)."""
function va_eki_output_root(experiment_dir::AbstractString, expc)
    out = expc["output_dir"]
    return isabspath(out) ? String(out) : joinpath(experiment_dir, out)
end

"""
    va_field_specs(expc) -> Vector{Dict}

Diagnostics used to build the observation vector. Default: single `thetaa` profile.
Override with `observation_fields` in `experiment_config.yml` (list of `short_name`, optional `reduction`, `period`).

Special composite: `short_name: clw_plus_cli` builds one `z_elem` block as **cloud liquid + cloud ice** mass fractions
(`clw` + `cli` at each level). It does **not** include rain/snow (`husra` / `hussn`). Requires both `clw` and `cli`
in the case YAML `diagnostics`.

`reduction` must match ClimaAnalysis `SimDir` keys: YAML diagnostics without `reduction_time` are stored as `inst`;
use `average` only if the case YAML sets `reduction_time: average` for that variable.

`period` must match the **SimDir** label (often abbreviated, e.g. `"10m"` for YAML `10mins`), not necessarily the YAML string verbatim.
"""
function va_field_specs(expc)
    if haskey(expc, "observation_fields")
        return map(expc["observation_fields"]) do f
            Dict{String, Any}(
                "short_name" => f["short_name"],
                "reduction" => get(f, "reduction", "inst"),
                "period" => get(f, "period", "10m"),
            )
        end
    else
        return [
            Dict{String, Any}(
                "short_name" => "thetaa",
                "reduction" => "inst",
                "period" => "10m",
            ),
        ]
    end
end

function va_z_elem(experiment_dir::AbstractString, expc)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, expc["model_config_path"])
    return Int(cfg["z_elem"])
end

"""
    va_z_centers_column(experiment_dir, expc) -> Vector{Float64}

Vertical cell-center heights (m) for the column in **`model_config_path`**, for interpolating LES onto the
SCM grid (see `les_truth_build.jl`). Optional **`les_truth.z_max`** in the experiment YAML trims levels above
`z_max` (same idea as `gcm_driven_scm` `z_max`).
"""
function va_z_centers_column(experiment_dir::AbstractString, expc)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, expc["model_config_path"])
    atmos_config = CA.AtmosConfig(cfg)
    params = CA.ClimaAtmosParameters(atmos_config)
    grid = CA.get_grid(atmos_config.parsed_args, params, atmos_config.comms_ctx)
    spaces = CA.get_spaces(grid, atmos_config.comms_ctx)
    coord = CA.Fields.coordinate_field(spaces.center_space)
    z_vec = convert(Vector{Float64}, parent(coord.z)[:])
    lt = get(expc, "les_truth", nothing)
    if lt isa AbstractDict && haskey(lt, "z_max")
        zm = Float64(lt["z_max"])
        z_vec = filter(z -> z <= zm, z_vec)
    end
    return z_vec
end

function va_expected_obs_length(experiment_dir, expc)
    va_z_elem(experiment_dir, expc) * length(va_field_specs(expc))
end

"""
    va_model_diagnostic_shortname_period_pairs(experiment_dir) -> Vector{Tuple{String,String}}

Expand the case YAML `diagnostics` section (`experiment_config.yml` → `model_config_path`): for each
block, pair every `short_name` (scalar or vector) with that block’s `period` string (e.g. `10mins`).
Used for post-run profile plots of **all** configured diagnostics, not only `observation_fields`.
"""
function va_model_diagnostic_shortname_period_pairs(
    experiment_dir::AbstractString,
    config_path::Union{Nothing, AbstractString} = nothing,
)
    expc = va_load_experiment_config(experiment_dir, config_path)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, expc["model_config_path"])
    pairs = Tuple{String, String}[]
    !haskey(cfg, "diagnostics") && return pairs
    for block in cfg["diagnostics"]
        period = string(get(block, "period", "10m"))
        sn = block["short_name"]
        if sn isa AbstractVector || sn isa Tuple
            for x in sn
                push!(pairs, (string(x), period))
            end
        else
            push!(pairs, (string(sn), period))
        end
    end
    return pairs
end

"""Diagnostics list from merged case YAML (`model_config_path` master + overlay), for forward-sweep plots."""
function va_case_yaml_diagnostic_shortname_period_pairs(
    experiment_dir::AbstractString,
    model_yaml_spec,
)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, model_yaml_spec)
    pairs = Tuple{String, String}[]
    !haskey(cfg, "diagnostics") && return pairs
    for block in cfg["diagnostics"]
        period = string(get(block, "period", "10m"))
        sn = block["short_name"]
        if sn isa AbstractVector || sn isa Tuple
            for x in sn
                push!(pairs, (string(x), period))
            end
        else
            push!(pairs, (string(sn), period))
        end
    end
    return pairs
end

function va_z_elem_from_case_yaml(experiment_dir::AbstractString, model_yaml_spec)
    cfg = va_load_merged_case_yaml_dict(experiment_dir, model_yaml_spec)
    return Int(cfg["z_elem"])
end

"""Comms context: `CLIMACALIB_DEVICE` env matches `get_comms_context` (`auto`, `CUDADevice`, etc.)."""
function va_comms_ctx()
    dev = get(ENV, "CLIMACALIB_DEVICE", "auto")
    return CA.get_comms_context(Dict{String, Any}("device" => dev))
end

"""
    va_scm_toml_path(experiment_dir, rel::AbstractString) -> String

Resolve **one** layer of `experiment_config.yml` key `scm_toml` (`rel`, e.g. `toml/foo.toml`).
For **master + case overlays**, use a YAML list in `scm_toml` and [`va_scm_toml_spec_layers`](@ref) /
[`va_merge_scm_baseline_dict`](@ref) instead.

Uses `joinpath(experiment_dir, rel)` when that file exists (vendored experiment TOML);
otherwise falls back to `ClimaAtmos.jl/toml/` with `basename(rel)` for shared defaults
(e.g. `prognostic_edmfx.toml` in sweeps).
"""
function va_scm_toml_path(experiment_dir::AbstractString, rel::AbstractString)
    local_path = joinpath(experiment_dir, rel)
    isfile(local_path) && return abspath(local_path)
    atmos_path = joinpath(pkgdir(CA), "toml", basename(rel))
    isfile(atmos_path) || error("scm_toml not found at $local_path or $atmos_path")
    return atmos_path
end

"""
    va_scm_toml_spec_layers(experiment_dir, scm_spec) -> Vector{String}

Resolve `experiment_config.yml` key **`scm_toml`** to an ordered list of absolute paths.

- **String:** one layer (same as [`va_scm_toml_path`](@ref)).
- **Vector / tuple of strings:** master + case overlays; each entry is resolved like a single `scm_toml` path.
  **Later files override** earlier ones at the **top-level** `[name]` table (same rule as
  [`va_write_combined_member_atmos_parameters_toml`](@ref)).
"""
function va_scm_toml_spec_layers(experiment_dir::AbstractString, scm_spec)::Vector{String}
    if scm_spec isa AbstractVector || scm_spec isa Tuple
        length(scm_spec) >= 1 || error("scm_toml list must be non-empty")
        return String[va_scm_toml_path(experiment_dir, string(x)) for x in scm_spec]
    else
        return String[va_scm_toml_path(experiment_dir, string(scm_spec))]
    end
end

"""
    va_model_config_path_layers(raw_spec) -> Vector{String}

Normalize **`model_config_path`** from **`experiment_config.yml`** or a forward-sweep registry row: a **string**
is a single layer; a **non-empty list** loads and shallow-merges **in order** (master first, case overlay last).
The **last** path is the canonical case id ([`va_model_config_yaml_rel`](@ref)).

Nested lists are **flattened** (YAML parsers sometimes yield `[[a, b]]` instead of `[a, b]`).
"""
function _va_flatten_model_config_path_layers!(out::Vector{String}, x)
    if x isa AbstractString
        s = String(strip(string(x)))
        isempty(s) && error("model_config_path entry is empty")
        push!(out, s)
    elseif x isa AbstractVector
        isempty(x) && error("model_config_path list must be non-empty")
        for y in x
            _va_flatten_model_config_path_layers!(out, y)
        end
    else
        error("model_config_path entries must be strings or nested lists of strings; got $(typeof(x))")
    end
    return out
end

function va_model_config_path_layers(raw_spec)::Vector{String}
    if raw_spec isa AbstractString
        s = String(strip(string(raw_spec)))
        isempty(s) && error("model_config_path is empty")
        return String[s]
    elseif raw_spec isa AbstractVector
        out = String[]
        _va_flatten_model_config_path_layers!(out, raw_spec)
        isempty(out) && error("model_config_path list must be non-empty")
        return out
    else
        error("model_config_path must be a string or a non-empty list of strings")
    end
end

"""Last path in [`va_model_config_path_layers`](@ref); used as registry / task `yaml_rel` key."""
function va_model_config_yaml_rel(raw_spec)::String
    layers = va_model_config_path_layers(raw_spec)
    return layers[end]
end

"""
    va_load_merged_case_yaml_dict(experiment_dir, raw_spec) -> Dict{String, Any}

Shallow-merge YAML dicts for each relative path in [`va_model_config_path_layers`](@ref); **later** files
override **earlier** keys (same as multi-file [`CA.AtmosConfig`](@ref)).
"""
function va_load_merged_case_yaml_dict(experiment_dir::AbstractString, raw_spec)::Dict{String, Any}
    layers = va_model_config_path_layers(raw_spec)
    merged = Dict{String, Any}()
    for rel in layers
        path = joinpath(experiment_dir, rel)
        isfile(path) || error("Case YAML not found: $path")
        merge!(merged, YAML.load_file(path))
    end
    return merged
end

"""Merge SCM TOML layers (see [`va_scm_toml_spec_layers`](@ref)) into one `Dict` for ClimaParams-style TOML."""
function va_merge_scm_baseline_dict(experiment_dir::AbstractString, scm_spec)
    merged = Dict{String, Any}()
    for p in va_scm_toml_spec_layers(experiment_dir, scm_spec)
        d = TOML.parsefile(p)
        for (k, v) in d
            merged[k] = v
        end
    end
    return merged
end

"""Basename for a single-file merge of layered `scm_toml` (reference runs and forward-sweep baselines)."""
const VA_MERGED_SCM_BASELINE_BASENAME = "_scm_merged_baseline.toml"

function va_write_merged_scm_baseline_file!(
    experiment_dir::AbstractString,
    scm_spec,
    out_path::AbstractString,
)
    d = va_merge_scm_baseline_dict(experiment_dir, scm_spec)
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        TOML.print(io, d)
    end
    return out_path
end

"""
    VA_COMBINED_MEMBER_ATMOS_PARAMETERS_BASENAME

Basename of the per-member ClimaParams TOML produced by
[`va_write_combined_member_atmos_parameters_toml`](@ref). This is the **only** path passed in
`config["toml"]` for `AtmosConfig` in this experiment (see `build_atmos_config_dict` in
`model_interface.jl`).
"""
const VA_COMBINED_MEMBER_ATMOS_PARAMETERS_BASENAME = "_atmos_merged_parameters.toml"

"""
    va_write_combined_member_atmos_parameters_toml(scm_baseline_path, eki_member_parameters_path, out_path) -> out_path

Write one ClimaParams-style TOML that combines the **SCM case baseline** with this ensemble member’s
ClimaCalibrate **`parameters.toml`**, for use as the sole `config["toml"]` entry.

# Why this exists

The usual ClimaAtmos calibration `forward_model` pattern is: YAML lists a case baseline TOML, then
`push!(config["toml"], member_path/parameters.toml)`. `ClimaParams.merge_toml_files` (inside
`AtmosConfig`) is called with `override=false` by default, so **duplicate top-level `[name]` tables
across those files are an error**.

Many calibrations **avoid** that by choosing a **small** baseline TOML and a prior whose drawn names
**do not overlap** those tables. This experiment intentionally uses a **dense** vendored SCM baseline
([`va_scm_toml_path`](@ref)) that **already defines** the same parameter names EKP perturbs, so a
two-file `toml` vector would collide.

Ways to handle overlap include: strip overlapping names from a copy of the baseline (disjoint pair),
`merge_toml_files(...; override=true)` (can trip ClimaParams’ overwrite `@warn` when entries lack
`type`/`value`), or **pre-merge to one file** as done here—same top-level override semantics as
`override=true`, without that code path.

# Behavior

1. Load the SCM baseline as a `Dict` — either from a **single** TOML path, or from **`scm_toml` layers**
   in the experiment YAML (see [`va_merge_scm_baseline_dict`](@ref)).
2. Load `eki_member_parameters_path` (`parameters.toml` for this member).
3. For each **top-level** key in the EKI file, **replace** the baseline entry (whole table). Keys
   only in the baseline are unchanged. This matches `merge_toml_files` across two files with
   later-file wins on name clashes; it does **not** deep-merge inside a table.

`AtmosConfig` still calls `merge_toml_files` on this path and `create_toml_dict` still merges against
package defaults; this helper only collapses **two user-provided sources** into one.

# Arguments

- `scm_baseline`: parsed baseline (`Dict`, e.g. from [`va_merge_scm_baseline_dict`](@ref)), **or** `scm_baseline_path`
  as a single TOML file path (string).
- `eki_member_parameters_path`: `joinpath(member_path, "parameters.toml")`.
- `out_path`: where to write (typically `joinpath(member_path, VA_COMBINED_MEMBER_ATMOS_PARAMETERS_BASENAME)`).
"""
function va_write_combined_member_atmos_parameters_toml(
    scm_baseline::AbstractDict,
    eki_member_parameters_path::AbstractString,
    out_path::AbstractString,
)
    combined = Dict{String, Any}(scm_baseline)
    for (k, v) in TOML.parsefile(eki_member_parameters_path)
        combined[k] = v
    end
    open(out_path, "w") do io
        TOML.print(io, combined)
    end
    return out_path
end

function va_write_combined_member_atmos_parameters_toml(
    scm_baseline_path::AbstractString,
    eki_member_parameters_path::AbstractString,
    out_path::AbstractString,
)
    return va_write_combined_member_atmos_parameters_toml(
        Dict{String, Any}(TOML.parsefile(scm_baseline_path)),
        eki_member_parameters_path,
        out_path,
    )
end

function va_job_id_reference(expc)
    jid = get(ENV, "JOB_ID", "")
    !isempty(jid) && return string(jid, "_ref")
    case = get(expc, "case_name", "case")
    mode = get(expc, "calibration_mode", "ref")
    return "va_$(case)_$(mode)_reference"
end

function va_job_id_member(expc, iteration::Integer, member::Integer)
    jid = get(ENV, "JOB_ID", "")
    !isempty(jid) && return string(jid, "_i", iteration, "_m", member)
    case = get(expc, "case_name", "case")
    return string(
        "va_",
        case,
        "_iter",
        lpad(iteration, 3, '0'),
        "_mem",
        lpad(member, 3, '0'),
    )
end

"""
    va_reference_output_active(experiment_dir[, config_path]) -> String

`output_active` under [`va_reference_output_dir`](@ref) for the active experiment YAML.
"""
function va_reference_output_active(
    experiment_dir::AbstractString,
    config_path::Union{Nothing, AbstractString} = nothing,
)
    expc = va_load_experiment_config(experiment_dir, config_path)
    return joinpath(va_reference_output_dir(experiment_dir, expc), "output_active")
end

"""
    va_latest_eki_jld2_path(experiment_dir) -> Union{String,Nothing}

Path to the newest `eki_file.jld2` under the active experiment YAML’s `output_dir` (lexicographically
last `iteration_*` that contains that file). Optional `config_path` overrides which YAML is read (same as
[`va_experiment_config_path`](@ref)).
"""
function va_latest_eki_jld2_path(
    experiment_dir::AbstractString,
    config_path::Union{Nothing, AbstractString} = nothing,
)
    expc = va_load_experiment_config(experiment_dir, config_path)
    out = expc["output_dir"]
    root = isabspath(out) ? String(out) : joinpath(experiment_dir, out)
    isdir(root) || return nothing
    names = sort!(filter(x -> startswith(x, "iteration_"), readdir(root)))
    for name in Iterators.reverse(names)
        p = joinpath(root, name, "eki_file.jld2")
        isfile(p) && return p
    end
    return nothing
end

"""
    va_latest_eki_iteration_number(experiment_dir[, config_path]) -> Int

Largest `iteration_NNN` index present under the YAML’s `output_dir` (same layout as ClimaCalibrate).
"""
function va_latest_eki_iteration_number(
    experiment_dir::AbstractString,
    config_path::Union{Nothing, AbstractString} = nothing,
)
    expc = va_load_experiment_config(experiment_dir, config_path)
    out = expc["output_dir"]
    root = isabspath(out) ? String(out) : joinpath(experiment_dir, out)
    isdir(root) || error("EKI output root does not exist: $root")
    best = 0
    for name in readdir(root)
        m = match(r"^iteration_(\d+)$", name)
        m === nothing && continue
        isdir(joinpath(root, name)) || continue
        best = max(best, parse(Int, m.captures[1]))
    end
    best >= 1 || error("No iteration_* directories under $root")
    return best
end

"""`output_active` for the naive varfix-on forward (baseline `z_elem`/`dt` from the source case YAML)."""
function va_naive_forward_output_active(
    experiment_dir::AbstractString,
    config_path::AbstractString,
)
    expc = va_load_experiment_config(experiment_dir, config_path)
    case = string(expc["case_name"])
    n = Int(expc["quadrature_order"])
    return joinpath(
        experiment_dir,
        "simulation_output",
        case,
        "N_$(n)",
        "varfix_on",
        "naive_from_varfix_off",
        "forward_only",
        "output_active",
    )
end

"""Figure folder for naive-track profile overlays (reference + naive forward vs varfix-off EKI losses)."""
function va_naive_post_analysis_figure_dir(
    experiment_dir::AbstractString,
    expc,
)
    case = replace(string(expc["case_name"]), r"[^\w\.\-]+" => "_")
    n = Int(expc["quadrature_order"])
    off_mode = replace(string(get(expc, "calibration_mode", "varfix_off")), r"[^\w\.\-]+" => "_")
    return joinpath(
        experiment_dir,
        "analysis",
        "figures",
        "$(case)_N$(n)_naive_varfix_on_from_$(off_mode)",
    )
end

"""
    va_prior_abs_path(experiment_dir[, config_path]) -> String

Absolute path to `prior_path` from the active experiment YAML. Each calibration case can use a different file; the same
parameter **name** can therefore have a different prior distribution in another case’s `prior*.toml`.
"""
function va_prior_abs_path(
    experiment_dir::AbstractString,
    config_path::Union{Nothing, AbstractString} = nothing,
)
    expc = va_load_experiment_config(experiment_dir, config_path)
    p = expc["prior_path"]
    return isabspath(p) ? String(p) : joinpath(experiment_dir, p)
end

"""
    va_build_noise_matrix(observations, expc, experiment_dir)

Diagonal observation noise covariance. `observation_noise_std` may be a scalar (all levels) or
a vector of length `n_fields` (one std per stacked field block of size `z_elem`).
"""
function va_build_noise_matrix(
    observations::AbstractVector,
    expc,
    experiment_dir::AbstractString,
)
    z = va_z_elem(experiment_dir, expc)
    specs = va_field_specs(expc)
    n_fields = length(specs)
    n_y = length(observations)
    expected = z * n_fields
    n_y == expected || error(
        "Observation length $n_y != z_elem * n_fields ($z * $n_fields = $expected). Regenerate the file at `observations_path` after changing observation_fields.",
    )
    σ_raw = expc["observation_noise_std"]
    if σ_raw isa AbstractVector || σ_raw isa Tuple
        σs = Float64.(collect(σ_raw))
        length(σs) == n_fields || error(
            "observation_noise_std must have length $n_fields (one per observation field), got $(length(σs))",
        )
        d = zeros(Float64, n_y)
        o = 1
        for i in 1:n_fields
            d[o:(o + z - 1)] .= σs[i]^2
            o += z
        end
        return LinearAlgebra.Diagonal(d)
    else
        σ = Float64(σ_raw)
        return (σ^2) * LinearAlgebra.I(n_y)
    end
end

"""
    va_worker_julia_exeflags(project_dir::AbstractString, worker_threads::Int) -> Cmd

`exeflags` for [`Distributed.addprocs`](@ref): each worker uses `--project=project_dir` and **`-t worker_threads`**.

Typical pattern: start the driver with **`julia -t 20`** (threads for the main process / threaded forward sweep),
set **`VARIANCE_CALIB_WORKER_THREADS=1`** (EKI) and/or **`VA_FORWARD_SWEEP_WORKER_THREADS=1`** (distributed forward
sweep) so ensemble workers stay **one thread per SCM** and do not oversubscribe the machine.
"""
function va_worker_julia_exeflags(project_dir::AbstractString, worker_threads::Int)
    worker_threads >= 1 || error("worker_threads must be >= 1, got $worker_threads")
    return `--project=$project_dir -t $worker_threads`
end

"""Parse `VARIANCE_CALIB_BACKEND`-style strings to `:worker` or `:julia` (single boundary for env vs structs)."""
function va_parse_eki_calibration_backend(s::AbstractString)::Symbol
    s = lowercase(String(strip(s)))
    if s in ("worker", "distributed", "workers")
        return :worker
    elseif s in ("julia", "serial", "sequential", "main")
        return :julia
    else
        error("VARIANCE_CALIB_BACKEND must be worker or julia; got $(repr(s)).")
    end
end
