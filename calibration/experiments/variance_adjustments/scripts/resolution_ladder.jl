# Generic vertical-resolution ladder for forward sweeps (any column `model_config` YAML).
#
# **Uniform** (`z_stretch: false`): coarsen by reducing `z_elem` geometrically (see `VALadderParams` / CLI `--ladder-*`).
#   Timestep stays the case YAML value (coarsening does not increase `dt`).
#
# **Stretched** (`z_stretch: true`, hyperbolic-tangent stretching): each tier after the baseline multiplies
#   `dz_bottom` by `min_dz_factor` and **searches** integer `z_elem` so the **top face spacing** (thickness of the
#   top vertical element from `DefaultZMesh`) matches the **baseline tier’s** top spacing. That keeps upper-column
#   resolution approximately fixed while coarsening the surface spacing — the discrete `z_elem` is the unknown that
#   satisfies the ClimaCore stretch, not `z_elem / coarsen_ratio`. No alternate-`dz_bottom` fallbacks.
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

import ClimaCore.CommonGrids as CG
import ClimaCore.Geometry as Geometry
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

"""Thickness of the top vertical element (top face minus next face down), same definition as ClimaCore `IntervalMesh` faces."""
function va_htstretch_top_face_dz(
    z_max::Float64,
    z_elem::Int,
    dz_bottom::Float64,
)::Float64
    FT = _ft_for_mesh()
    mesh = CG.DefaultZMesh(
        FT;
        z_min = FT(0),
        z_max = FT(z_max),
        z_elem,
        stretch = Meshes.HyperbolicTangentStretching{FT}(FT(dz_bottom)),
    )
    faces = mesh.faces
    n = length(faces)
    z_lo = Geometry.component(faces[n - 1], 1)
    z_hi = Geometry.component(faces[n], 1)
    return Float64(z_hi - z_lo)
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

"""
    va_find_z_elem_for_target_top_dz(z_max, dz_bottom, target_top_dz, nelem_lo, nelem_hi) -> Int

Among integers `nelem_lo:nelem_hi` where `DefaultZMesh` builds, choose `z_elem` that minimizes
`|top_face_dz(z_elem, dz_bottom) - target_top_dz|`. Tie-break: smaller `z_elem` (fewer DOFs).
"""
function va_find_z_elem_for_target_top_dz(
    z_max::Float64,
    dz_bottom::Float64,
    target_top_dz::Float64,
    nelem_lo::Int,
    nelem_hi::Int,
)::Int
    nelem_lo >= 2 || error("va_find_z_elem_for_target_top_dz: nelem_lo must be >= 2")
    nelem_hi >= nelem_lo || error("va_find_z_elem_for_target_top_dz: nelem_hi must be >= nelem_lo")
    best_n = -1
    best_err = Inf
    for n in nelem_lo:nelem_hi
        va_column_mesh_builds(z_max, n, true, dz_bottom) || continue
        td = va_htstretch_top_face_dz(z_max, n, dz_bottom)
        err = abs(td - target_top_dz)
        same_err = isapprox(err, best_err; rtol = 1e-12, atol = 1e-9)
        if err < best_err || (same_err && (best_n < 0 || n < best_n))
            best_err = err
            best_n = n
        end
    end
    best_n > 0 || error(
        "No valid stretched mesh in z_elem ∈ [$nelem_lo, $nelem_hi] for dz_bottom=$dz_bottom, z_max=$z_max.",
    )
    return best_n
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
    va_resolution_tiers_from_case_dict(case_dict[, ladder_params]) -> Vector{VAResolutionTier]

`case_dict` is the merged column model YAML (must include `z_elem`, `z_max`, `dt`). Baseline is tier 1.
Stretched grids: later tiers increase `dz_bottom` by `min_dz_factor` and pick `z_elem` by scanning
[`z_elem_min`, `max(512, 32 * baseline z_elem)`] to match the baseline **top-face** Δz (see file header).
"""
function va_resolution_tiers_from_case_dict(
    case_dict::AbstractDict,
    ladder::VALadderParams = VALadderParams(),
)
    z_elem_base = Int(case_dict["z_elem"])
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
        z_elem = z_elem_base
        for _ in 1:n_tiers
            push!(tiers, VAResolutionTier(z_elem, dt_str, nothing))
            z_elem_new = max(z_min, round(Int, z_elem / ratio))
            z_elem_new == z_elem && break
            z_elem = z_elem_new
        end
    else
        va_column_mesh_builds(z_max, z_elem_base, z_stretch, dz_yaml) ||
            error("Baseline mesh invalid: z_elem=$z_elem_base dz_bottom=$dz_yaml z_stretch=true")
        target_top_dz = va_htstretch_top_face_dz(z_max, z_elem_base, dz_yaml)
        nelem_hi = max(512, 32 * z_elem_base)
        dz0 = dz_yaml
        db = dz_yaml
        for tier_i in 1:n_tiers
            dz_write = isapprox(db, dz0; atol = 1e-3, rtol = 0) ? nothing : db
            z_use = tier_i == 1 ? z_elem_base :
                va_find_z_elem_for_target_top_dz(z_max, db, target_top_dz, z_min, nelem_hi)
            va_column_mesh_builds(z_max, z_use, z_stretch, db) ||
                error("Stretched mesh invalid: z_elem=$z_use dz_bottom=$db z_stretch=true")
            push!(tiers, VAResolutionTier(z_use, dt_str, dz_write))
            tier_i >= n_tiers && break
            db = db * dz_mult
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
