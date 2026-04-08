# Build `observations.jld2` from GCM-forced LES `Stats*.nc` using the same vertical interpolation and
# time-window logic as `calibration/experiments/gcm_driven_scm` (`get_profile` in `helper_funcs.jl`).
#
# Include after `experiment_common.jl`. Requires extra deps in `Project.toml` (NCDatasets, Interpolations, Glob).
#
# Call from `build_observations_from_les.jl` after `Pkg.activate`.

include(joinpath(@__DIR__, "experiment_common.jl"))

import JLD2

const _GCM_DIR = joinpath(@__DIR__, "..", "..", "gcm_driven_scm")
include(joinpath(_GCM_DIR, "helper_funcs.jl"))
include(joinpath(_GCM_DIR, "get_les_metadata.jl"))

"""
    _va_caltech_hpc_allowed() -> Bool

`true` when env **`VA_CALTECH_HPC`** is set to a truthy value (`1`, `true`, `yes`).
Required to resolve **`les_truth.source: gcm_forced_cfsite`** via **`GCMFORCEDLES_ROOT`** (cluster layout).
Not required when **`LES_STATS_FILE`** or **`les_truth.stats_file`** points to a local `Stats*.nc`.
"""
function _va_caltech_hpc_allowed()
    v = strip(get(ENV, "VA_CALTECH_HPC", ""))
    return v in ("1", "true", "yes")
end

"""
Same relative layout as `get_cfsite_les_dir` in `gcm_driven_scm/get_les_metadata.jl`, but **`GCMFORCEDLES_ROOT` is required**:
`variance_adjustments` does **not** fall back to a hardcoded cluster path. On systems where data live under
`/resnick/.../GCMForcedLES`, set e.g. `GCMFORCEDLES_ROOT=/resnick/groups/esm/zhaoyi/GCMForcedLES` explicitly.
"""
function _va_cfsite_les_dir(
    cfsite::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",
)
    root = strip(get(ENV, "GCMFORCEDLES_ROOT", ""))
    if isempty(root)
        error(
            "`GCMFORCEDLES_ROOT` is not set. This experiment does not default to any cluster filesystem. " *
                "Set **`GCMFORCEDLES_ROOT`** to the directory that contains **`cfsite/<MM>/<model>/<exp>/...`** " *
                "(see `gcm_driven_scm/get_les_metadata.jl`), or use **`LES_STATS_FILE`** / **`les_truth.stats_file`** " *
                "with a concrete `Stats*.nc` path.",
        )
    end
    month_s = string(month, pad = 2)
    cfsite_s = string(cfsite)
    root_dir = joinpath(root, "cfsite", month_s, forcing_model, experiment)
    rel_dir = join(
        ["Output.cfsite$cfsite_s", forcing_model, experiment, "2004-2008.$month_s.4x"],
        "_",
    )
    return joinpath(root_dir, rel_dir)
end

function _les_mean_var_name_for_scm_short(sn::AbstractString)
    sn == "thetaa" && return "theta_mean"
    sn == "hus" && return "qt_mean"
    sn == "clw" && return "ql_mean"
    sn == "cli" && return "qi_mean"
    error("Unknown observation short_name for LES: $(repr(sn))")
end

"""
    va_resolve_les_stats_path(experiment_dir, expc) -> String

Resolve path to LES **Stats** NetCDF:

1. If **`LES_STATS_FILE`** is set in the environment, use it (absolute or relative to `experiment_dir`).
2. Else if **`les_truth.source: gcm_forced_cfsite`**, resolve via **`GCMFORCEDLES_ROOT`** + `get_stats_path`
   (same relative layout as `gcm_driven_scm/get_les_metadata.jl`) from `(cfsite, forcing_model, month, experiment)`.
   **`GCMFORCEDLES_ROOT` is mandatory** for this branch (no implicit `/resnick` default).
3. Else require non-empty **`les_truth.stats_file`** in the experiment YAML.
"""
function va_resolve_les_stats_path(experiment_dir::AbstractString, expc)
    envp = strip(get(ENV, "LES_STATS_FILE", ""))
    if !isempty(envp)
        p = envp
        return (isabspath(p) ? String(p) : joinpath(experiment_dir, p), "LES_STATS_FILE")
    end
    lt = get(expc, "les_truth", nothing)
    lt === nothing && error(
        "experiment YAML needs a `les_truth` block (or set environment variable LES_STATS_FILE). See README.",
    )
    lt isa AbstractDict || error("les_truth must be a mapping")
    sf = get(lt, "stats_file", "")
    if sf isa AbstractString && !isempty(strip(sf))
        p = strip(string(sf))
        return (isabspath(p) ? String(p) : joinpath(experiment_dir, p), "les_truth.stats_file")
    end
    src = string(get(lt, "source", ""))
    if src == "gcm_forced_cfsite"
        haskey(lt, "cfsite") || error("les_truth.source=gcm_forced_cfsite requires les_truth.cfsite")
        if !_va_caltech_hpc_allowed()
            error(
                "les_truth.source=gcm_forced_cfsite with cluster path resolution requires **`VA_CALTECH_HPC=1`** " *
                    "(Caltech HPC). Off-cluster, use **GoogleLES** experiment YAMLs (`les_truth.source: googleles_cloudbench`, default track) " *
                    "or set **`LES_STATS_FILE`** / **`les_truth.stats_file`** to a local `Stats*.nc`.",
            )
        end
        cfsite = Int(lt["cfsite"])
        fm = string(get(lt, "forcing_model", "HadGEM2-A"))
        mo = Int(get(lt, "month", 7))
        ex = string(get(lt, "experiment", "amip"))
        try
            les_dir = _va_cfsite_les_dir(cfsite; forcing_model = fm, month = mo, experiment = ex)
            return (get_stats_path(les_dir), "les_truth.source=gcm_forced_cfsite")
        catch err
            error(
                "Could not load GCM-forced LES stats for cfsite=$(cfsite), forcing_model=$(fm), month=$(mo), experiment=$(ex). " *
                    "Check **`GCMFORCEDLES_ROOT`** and that `Output.cfsite*/...` contains a unique `Stats*.nc`, or set **`LES_STATS_FILE`** / **`les_truth.stats_file`**, " *
                    "or **`scripts/run_full_study.jl --skip-les-observations`** if **`observations.jld2`** is already staged. " *
                    "Underlying error: $err",
            )
        end
    end
    error(
        "les_truth.stats_file is empty. Set it explicitly, set LES_STATS_FILE, or use les_truth.source=gcm_forced_cfsite or googleles_cloudbench.",
    )
end

include(joinpath(@__DIR__, "googleles_truth_build.jl"))
include(joinpath(@__DIR__, "googleles_forcing_build.jl"))

"""Build observation vector **y** (physical units, SCM grid) from LES stats; matches `process_member_column` layout."""
function va_build_y_vector_from_les(nc_path::AbstractString, experiment_dir::AbstractString, expc)
    lt = get(expc, "les_truth", Dict())
    ti = Float64(get(lt, "y_t_start_sec", 475200.0))
    tf = Float64(get(lt, "y_t_end_sec", 518400.0))
    z_vec = va_z_centers_column(experiment_dir, expc)
    specs = va_field_specs(expc)
    y = Float64[]
    for spec in specs
        sn = string(spec["short_name"])
        if sn == "clw_plus_cli"
            p1 = get_profile(nc_path, ["ql_mean"]; ti = ti, tf = tf, z_scm = z_vec)
            p2 = get_profile(nc_path, ["qi_mean"]; ti = ti, tf = tf, z_scm = z_vec)
            length(p1) == length(p2) || error("ql_mean / qi_mean length mismatch")
            append!(y, p1 .+ p2)
        else
            lesn = _les_mean_var_name_for_scm_short(sn)
            append!(y, get_profile(nc_path, [lesn]; ti = ti, tf = tf, z_scm = z_vec))
        end
    end
    expected = va_expected_obs_length(experiment_dir, expc)
    length(y) == expected || error(
        "LES observation length $(length(y)) != expected $expected (check observation_fields vs column z_elem).",
    )
    return y
end

"""Write **`observations_path`** from LES using **`les_truth`** in the experiment YAML (`config_rel` or default / `VA_EXPERIMENT_CONFIG`)."""
function va_write_les_observations_jld2!(
    experiment_dir::AbstractString,
    config_rel::Union{Nothing, AbstractString} = nothing,
)
    expc = va_load_experiment_config(experiment_dir, config_rel)
    lt = get(expc, "les_truth", nothing)
    src = lt isa AbstractDict ? string(get(lt, "source", "")) : ""
    if src == "googleles_cloudbench"
        y = va_build_y_vector_from_googleles_zarr(experiment_dir, expc)
        out = va_observations_abs_path(experiment_dir, expc)
        mkpath(dirname(out))
        JLD2.save_object(out, y)
        @info "Wrote GoogleLES CloudBench observations (physical units on SCM grid)" out length(y)
        va_ensure_googleles_shen_forcing!(experiment_dir, expc)
        return out
    end
    nc_path, source = va_resolve_les_stats_path(experiment_dir, expc)
    isfile(nc_path) || error("LES stats file not found: $nc_path")
    y = va_build_y_vector_from_les(nc_path, experiment_dir, expc)
    out = va_observations_abs_path(experiment_dir, expc)
    mkpath(dirname(out))
    JLD2.save_object(out, y)
    @info "Wrote LES observations (physical units on SCM grid)" out length(y) nc_path source
    return out
end

"""
    va_ensure_les_observations_for_calibration_sweep!(experiment_dir, config_yamls; force=false)

For each experiment YAML path in `config_yamls` (relative to `experiment_dir` unless absolute), ensure **`observations.jld2`**
exists at **`observations_path`**. If the file is already present and `force` is `false`, skip. Set env
**`VA_FORCE_LES_OBSERVATIONS=1`** to always rebuild.

Called by **`scripts/run_full_study.jl`** before the EKI sweep (typically `va_calibration_sweep_configs()`), unless **`--skip-les-observations`**
or env **`VA_SKIP_LES_OBSERVATIONS_BUILD=1`**.
"""
function va_ensure_les_observations_for_calibration_sweep!(
    experiment_dir::AbstractString,
    config_yamls::Vector{String};
    force::Bool = strip(get(ENV, "VA_FORCE_LES_OBSERVATIONS", "")) in ("1", "true", "yes"),
)
    for rel in config_yamls
        expc = va_load_experiment_config(experiment_dir, rel)
        out = va_observations_abs_path(experiment_dir, expc)
        if !force && isfile(out)
            @info "LES observations already present; skipping build" config = rel path = out
            continue
        end
        @info "Building LES observations from experiment YAML" config = rel
        va_write_les_observations_jld2!(experiment_dir, rel)
    end
    return nothing
end
