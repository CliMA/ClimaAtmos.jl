# CONTROLLED Beres-C vs Beres-G band-resolved deposition on ONE frozen state.
#
# Question (manuscript, resolution of the "more slow flux, slower QBO" pivot):
# band_deposition.jl showed where each band of the BERES-C spectrum deposits;
# the C-vs-G drag contrast so far rests on two verification windows at
# DIFFERENT oscillation phases (uncontrolled). This script removes that gap:
# both vertical-shape formulations are evaluated on the IDENTICAL frozen
# state, so every difference in deposited drag is attributable to the shape
# factor alone (same convection, same winds, same envelope/Q0 — the shape
# enters only through the launched spectrum).
#
# Machinery = band_deposition.jl (restart + B0 band mask + unchanged
# deposition kernel) + the same-type cache swap of calibrate_beres_replacement
# (here flipping beres_shape_general, a plain Bool field of BeresSourceParams,
# instead of beres_scale_factor — typeof(p) is asserted unchanged, so no
# fresh-cache-type recompile; the Val(beres_shape_general) inner dispatch
# compiles the sine-transform branch once on the first Beres-G evaluation).
#
# Per formulation F in (C = half-sine, G = sine transform):
#   1. compute_tendency! with F's params rebuilds B0 and gives the FULL drag;
#   2. per band: rebuild B0, zero bins outside the band (CPU-side), re-run the
#      deposition kernel with F's cache;
#   3. band drag = eval(band-masked) − eval(all-zeroed = exact AD99 baseline).
# Gates: per-formulation band additivity, AND the two AD99 baselines must be
# bit-close (identical state ⇒ identical background).
#
# Usage (repo root of the rc/beres-g checkout, GPU node):
#   export CLIMACOMMS_DEVICE=CUDA
#   julia --project=.buildkite test/parameterized_tendencies/gravity_wave/\
#     non_orographic_gravity_wave/band_deposition_cg.jl [restart_override.yml]
# Default override = beres_calibration_restart_override.yml (day 1020); run
# also with the _day870 / _day1170 variants for QBO-phase coverage
# (/scratch/chew/beres_band_deposition_cg/submit_band_deposition_cg.sbatch).
#
# Output: beres_manuscript/derived/band_deposition_cg_<jobid>.json
#   (tropical-mean profiles: wind, per-formulation full and per-band drag,
#    shared AD99 baseline, z levels, gates).

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Geometry as Geometry
import JSON
using Dates
using Statistics

const LAT_BAND = 10.0   # deep tropics, matching band_deposition.jl
const NC_TOL = 1e-6     # |c| below this is the stationary bin

# --- 1. frozen state: restart + 1-day continuation (calibration recipe) ---
override_yml =
    isempty(ARGS) ? joinpath(@__DIR__, "beres_calibration_restart_override.yml") :
    abspath(ARGS[1])
config_files = [
    "config/common_configs/numerics_sphere_he16ze63.yml",
    "config/longrun_configs/longrun_aquaplanet_allsky_progedmf_0M.yml",
    override_yml,
]
stem = splitext(basename(override_yml))[1]
job_id = "band_deposition_cg" * chopprefix(stem, "beres_calibration_restart_override")
comms_ctx = ClimaComms.SingletonCommsContext()
config = CA.AtmosConfig(config_files; job_id, comms_ctx)
simulation = CA.get_simulation(config)
CA.solve_atmos!(simulation)

integrator = simulation.integrator
p = integrator.p
Y = integrator.u
gw_model = p.atmos.non_orographic_gravity_wave
gw = p.non_orographic_gravity_wave
FT = Spaces.undertype(axes(Y.c))

# --- 2. the same-type shape swap (calibrate_beres_replacement pattern) ---
bs = gw.gw_beres_source
@assert !isnothing(bs) "this override must run with nogw_beres_source: true"
bs_fields = (; (n => getfield(bs, n) for n in fieldnames(typeof(bs)))...)
make_bs(general::Bool) = CA.BeresSourceParams{FT}(;
    merge(bs_fields, (; beres_shape_general = general))...,
)
function swap_beres_cache(p, beres_source)
    gw2 = merge(gw, (; gw_beres_source = beres_source))
    args = map(
        n -> n === :non_orographic_gravity_wave ? gw2 : getfield(p, n),
        fieldnames(typeof(p)),
    )
    return CA.AtmosCache{map(typeof, args)...}(args...)
end
@assert typeof(swap_beres_cache(p, make_bs(true))) === typeof(p)
println(
    "trajectory beres_shape_general = $(bs.beres_shape_general); " *
    "evaluating both shapes on the frozen state",
)

# --- 3. tropical-mean profile extraction (as in band_deposition.jl) ---
ᶜlat = Fields.coordinate_field(Y.c).lat
ᶜtropmask = @. ifelse(abs(ᶜlat) <= FT(LAT_BAND), FT(1), FT(0))
const NLEV = Spaces.nlevels(axes(Y.c))
mask_area = [sum(Fields.level(ᶜtropmask, k)) for k in 1:NLEV]

function trop_profile(field)
    masked = field .* ᶜtropmask
    [sum(Fields.level(masked, k)) / mask_area[k] for k in 1:NLEV]
end

zprof = trop_profile(Fields.coordinate_field(Y.c).z)
uprof = trop_profile(Geometry.UVVector.(Y.c.uₕ).components.data.:1)

# --- 4. phase-speed bands on the launch grid ---
cgrid = collect(FT, gw.gw_c)
nc = length(cgrid)
bands = Dict(
    "c0" => [abs(c) < NC_TOL for c in cgrid],
    "slowE" => [NC_TOL <= c <= 15 for c in cgrid],
    "slowW" => [-15 <= c <= -NC_TOL for c in cgrid],
    "fastE" => [c > 15 for c in cgrid],
    "fastW" => [c < -15 for c in cgrid],
)
@assert sum(sum.(values(bands))) == nc "bands must partition the c-grid"

function mask_B0!(field, keep)
    pa = parent(field)
    dims = findall(==(length(keep)), collect(size(pa)))
    length(dims) == 1 ||
        error("cannot identify the spectrum axis in parent size $(size(pa))")
    d = dims[1]
    a = Array(pa)
    idx = ntuple(i -> i == d ? findall(!, keep) : Colon(), ndims(a))
    a[idx...] .= 0
    copyto!(pa, a)
    nothing
end

# --- 5. forcing-only re-evaluation (band_deposition.jl, with p threaded) ---
function run_forcing!(p_eval)
    # compute_tendency! overwrites the cache field in place with the floored N
    # (not N^2), so it is passed to the forcing as is. Taking the square root
    # again here would feed sqrt(N) to the deposition kernel.
    ᶜbf = gw.ᶜbuoyancy_frequency
    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2
    src = gw.source_p_ρ_z_u_v_level
    gw.uforcing .= 0
    gw.vforcing .= 0
    CA.non_orographic_gravity_wave_forcing(
        ᶜu, ᶜv, ᶜbf, Y.c.ρ, Fields.coordinate_field(Y.c).z, gw.ᶜlevel,
        src.:6, gw.damp_level, src.:2, src.:4, src.:5,
        gw.uforcing, gw.vforcing, gw.gw_ncval,
        gw.u_waveforcing, gw.v_waveforcing, p_eval,
    )
    nothing
end

# The swapped caches share every array with `gw`; only gw_beres_source differs,
# so B0 rebuilt under p_eval is visible through gw.gw_beres_B0_* for masking.
function band_drag(p_eval, keep)
    CA.non_orographic_gravity_wave_compute_tendency!(Y, p_eval, gw_model)
    if !isnothing(keep)
        mask_B0!(gw.gw_beres_B0_u, keep)
        mask_B0!(gw.gw_beres_B0_v, keep)
        run_forcing!(p_eval)
    end
    trop_profile(gw.uforcing)
end

# --- 6. both formulations on the identical state ---
out = Dict{String, Any}(
    "job_id" => job_id,
    "restart_override" => override_yml,
    "lat_band" => LAT_BAND,
    "trajectory_shape_general" => bs.beres_shape_general,
    "z" => zprof,
    "u" => uprof,
    "timestamp" => string(Dates.now()),
)
ad99_by_tag = Dict{String, Vector{Float64}}()
for (tag, general) in (("c", false), ("g", true))
    p_eval = swap_beres_cache(p, make_bs(general))
    println(
        "evaluating $(uppercase(tag)) (beres_shape_general = $general): " *
        "full, AD99 baseline, and $(length(bands)) bands...",
    )
    full = band_drag(p_eval, nothing)
    ad99 = band_drag(p_eval, falses(nc))
    band_profiles = Dict(name => band_drag(p_eval, keep) for (name, keep) in bands)

    beres_full = full .- ad99
    beres_sum = reduce(.+, [prof .- ad99 for prof in values(band_profiles)])
    resid = maximum(abs.(beres_sum .- beres_full))
    scale = maximum(abs.(beres_full))
    println(
        "  additivity gate ($tag): max |Σbands − full| = $resid " *
        "(full-scale $scale) → " * (resid <= 1e-4 * scale ? "PASS" : "FAIL"),
    )

    ad99_by_tag[tag] = ad99
    out["utend_full_$tag"] = full
    out["utend_band_$tag"] =
        Dict(name => prof .- ad99 for (name, prof) in band_profiles)
    out["additivity_resid_$tag"] = resid
end

# --- 7. identical-baseline gate: AD99 must not care about the Beres shape ---
base_diff = maximum(abs.(ad99_by_tag["c"] .- ad99_by_tag["g"]))
base_scale = maximum(abs.(ad99_by_tag["c"]))
println(
    "baseline gate: max |AD99_c − AD99_g| = $base_diff (scale $base_scale) → " *
    (base_diff <= 1e-10 * base_scale ? "PASS" : "FAIL"),
)
out["utend_ad99"] = ad99_by_tag["c"]
out["ad99_baseline_diff"] = base_diff

# --- 8. output ---
dest = normpath(
    joinpath(
        @__DIR__, "..", "..", "..", "..", "beres_manuscript", "derived",
        "band_deposition_cg_$(job_id).json"),
)
mkpath(dirname(dest))
open(dest, "w") do io
    JSON.print(io, out, 1)
end
println("wrote $dest")
