# Generic vertical-resolution ladder for forward sweeps (any column `model_config` YAML).
#
# **Uniform** (`z_stretch: false`): coarsen by reducing `z_elem` geometrically (see `VALadderParams` / CLI `--ladder-*`).
#   Timestep stays the case YAML value (coarsening does not increase `dt`).
#
# **Stretched** (`z_stretch: true`, hyperbolic-tangent stretching): coarsen primarily by scaling
#   `dz_bottom` (raises the effective minimum spacing; already-coarse aloft stays relatively smooth).
#   If the mesh constructor rejects a `dz_bottom` step, we reduce `z_elem` and reset `dz_bottom` to the
#   YAML baseline, then continue (see `va_resolution_tiers_from_case_dict`).
#
# Tuning: pass [`VALadderParams`](@ref) into [`va_resolution_tiers_from_case_dict`](@ref), or set forward-sweep
# CLI flags `--ladder-n-tiers`, `--ladder-coarsen-ratio`, `--ladder-z-elem-min`, `--ladder-min-dz-factor`.
#
# Requires `ClimaCore` (same mesh as ClimaAtmos `DefaultZMesh`).

"""Parameters for generic vertical coarsening (no environment reads)."""
struct VALadderParams
    n_tiers::Int
    coarsen_ratio::Float64
    z_elem_min::Int
    min_dz_factor::Float64
end

function VALadderParams()
    return VALadderParams(4, 2.0, 4, 2.0)
end

import ClimaCore as CC
import ClimaCore.CommonGrids as CG
import ClimaCore.Meshes as Meshes

"""One forward-sweep vertical discretization step (`dt` is always the case baseline string here)."""
struct VAResolutionTier
    z_elem::Int
    dt_str::String
    """If set, written to `config["dz_bottom"]`; if `nothing`, YAML value is left in effect."""
    dz_bottom_written::Union{Nothing, Float64}
end

function _ft_for_mesh()
    return Float64
end

function _stretching(::Type{FT}, z_stretch::Bool, dz_bottom::Float64) where {FT}
    return z_stretch ? Meshes.HyperbolicTangentStretching{FT}(FT(dz_bottom)) : Meshes.Uniform()
end

"""Try building the same interval mesh ClimaAtmos uses for columns; return `true` if successful."""
function va_column_mesh_builds(
    z_max::Float64,
    z_elem::Int,
    z_stretch::Bool,
    dz_bottom::Float64,
)
    FT = _ft_for_mesh()
    try
        CG.DefaultZMesh(
            FT;
            z_min = FT(0),
            z_max = FT(z_max),
            z_elem,
            stretch = _stretching(FT, z_stretch, dz_bottom),
        )
        return true
    catch
        return false
    end
end

function _yaml_bool(v, default::Bool)
    v === nothing && return default
    return Bool(v)
end

function _yaml_float(v, default::Float64)
    v === nothing && return default
    return Float64(v)
end

"""
    va_resolution_tiers_from_case_dict(case_dict[, ladder_params]) -> Vector{VAResolutionTier}

`case_dict` is the merged column model YAML (must include `z_elem`, `z_max`, `dt`). Baseline is tier 1.
"""
function va_resolution_tiers_from_case_dict(
    case_dict::AbstractDict,
    ladder::VALadderParams = VALadderParams(),
)
    z_base = Int(case_dict["z_elem"])
    z_max = Float64(case_dict["z_max"])
    dt_str = string(case_dict["dt"])
    z_stretch = _yaml_bool(get(case_dict, "z_stretch", true), true)
    dz_yaml = _yaml_float(get(case_dict, "dz_bottom", nothing), 500.0)

    n_tiers = ladder.n_tiers
    ratio = ladder.coarsen_ratio
    z_min = ladder.z_elem_min
    dz_mult = ladder.min_dz_factor

    n_tiers >= 1 || error("ladder n_tiers must be >= 1")

    tiers = VAResolutionTier[]

    if !z_stretch
        z = z_base
        for _ in 1:n_tiers
            push!(tiers, VAResolutionTier(z, dt_str, nothing))
            z_new = max(z_min, round(Int, z / ratio))
            z_new == z && break
            z = z_new
        end
    else
        z = z_base
        db = dz_yaml
        db0 = dz_yaml
        for _ in 1:n_tiers
            va_column_mesh_builds(z_max, z, z_stretch, db) ||
                error("Baseline mesh invalid: z_elem=$z dz_bottom=$db z_stretch=true")
            dz_write = isapprox(db, db0; atol = 1e-3, rtol = 0) ? nothing : db
            push!(tiers, VAResolutionTier(z, dt_str, dz_write))
            length(tiers) >= n_tiers && break
            db_try = db * dz_mult
            if va_column_mesh_builds(z_max, z, z_stretch, db_try)
                db = db_try
            else
                z_new = max(z_min, round(Int, z / ratio))
                z_new == z && break
                z = z_new
                db = db0
            end
        end
    end

    # Deduplicate consecutive identical tiers (can happen with aggressive z_min)
    out = VAResolutionTier[]
    for t in tiers
        isempty(out) || t != last(out) || continue
        push!(out, t)
    end
    return out
end

"""Effective `dz_bottom` for path slugs (stretched grids need disambiguation when `z_elem` repeats)."""
function va_effective_dz_bottom(tier::VAResolutionTier, yaml_dz_bottom::Float64)
    return something(tier.dz_bottom_written, yaml_dz_bottom)
end

function va_dt_path_slug(dt_str::AbstractString)
    s = replace(lowercase(string(dt_str)), "secs" => "s", "sec" => "s")
    return replace(s, r"[^0-9a-z]+" => "")
end

function va_tier_path_segment(
    tier::VAResolutionTier,
    z_stretch::Bool,
    yaml_dz_bottom::Float64,
)
    dt_slug = va_dt_path_slug(tier.dt_str)
    seg = string("z", tier.z_elem, "_dt", dt_slug)
    if z_stretch
        db = va_effective_dz_bottom(tier, yaml_dz_bottom)
        seg = string(seg, "_dzb", round(Int, db))
    end
    return seg
end

"""Directory slug under `simulation_output/` (defaults to `initial_condition`)."""
function va_forward_sweep_case_slug(case_dict::AbstractDict)
    if haskey(case_dict, "forward_sweep_case_slug") && case_dict["forward_sweep_case_slug"] !== nothing
        s = string(case_dict["forward_sweep_case_slug"])
        !isempty(strip(s)) && return replace(s, r"[^\w\.\-]+" => "_")
    end
    return replace(string(case_dict["initial_condition"]), r"[^\w\.\-]+" => "_")
end
