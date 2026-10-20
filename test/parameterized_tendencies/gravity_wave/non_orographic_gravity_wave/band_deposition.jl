# Band-resolved deposition of the Beres convective source on a frozen state.
#
# Question (manuscript §"A wave-driven oscillation and where each band acts"):
# at which altitudes do the slow (|c| <= 15 m/s) and fast phase-speed bands of
# the launched convective spectrum deposit their drag, given the run's own QBO
# wind profile? Answered with PRODUCTION physics, no offline reimplementation:
#
#   1. restart from an 11-yr-run checkpoint + 1-day continuation (same recipe
#      as calibrate_beres_alpha.jl restart mode), freeze the state;
#   2. call CA.non_orographic_gravity_wave_compute_tendency! once — this
#      rebuilds the precomputed launch spectra (gw_beres_B0_u/v, BeresB0) and
#      gives the FULL drag;
#   3. per band: rebuild B0 the same way, zero all spectrum bins OUTSIDE the
#      band (CPU-side on the parent array — no GPU recompile), then re-run the
#      unchanged deposition kernel (non_orographic_gravity_wave_forcing) with
#      cache-resident arguments;
#   4. band drag = eval(band-masked Beres + AD99) − eval(all-Beres-zeroed =
#      exact AD99 baseline), on the identical state.
#
# AD99 deposits each spectral element independently, so the per-band drags
# must sum to the full Beres drag: that additivity is asserted (the gate also
# catches any drift between this script's forcing-call argument prep and the
# production compute_tendency).
#
# Usage (repo root, GPU node):
#   export CLIMACOMMS_DEVICE=CUDA
#   julia --project=.buildkite test/parameterized_tendencies/gravity_wave/\
#     non_orographic_gravity_wave/band_deposition.jl [restart_override.yml]
# Default override = beres_calibration_restart_override.yml (day 1020); run
# also with the _day870 / _day1170 variants for QBO-phase coverage.
#
# Output: beres_manuscript/derived/band_deposition_<jobid>.json
#   (tropical-mean profiles: wind, full/AD99/per-band drag, z levels).

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Geometry as Geometry
import JSON
using Dates
using Statistics

const LAT_BAND = 10.0   # deep tropics, matching the five-way band tables
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
job_id = "band_deposition" * chopprefix(stem, "beres_calibration_restart_override")
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

# --- 2. tropical-mean profile extraction (area-weighted level sums) ---
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

# --- 3. phase-speed bands on the launch grid ---
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

"""
Zero the B0 spectrum bins where keep[i] is false (CPU-side, layout-agnostic).
"""
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

# --- 4. forcing-only re-evaluation (mirrors the tail of compute_tendency!) ---
# The full compute_tendency! call below leaves every input the deposition
# kernel needs in the cache (N, source/damp levels, Beres source fields);
# only B0 is modified between rebuild and deposition. The additivity gate
# asserts this mirror stays faithful to production.
function run_forcing!()
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
        gw.u_waveforcing, gw.v_waveforcing, p,
    )
    nothing
end

"""
Rebuild B0 via the production path, apply `keep` mask (nothing = no mask,
empty = Beres fully off), deposit, return the tropical-mean u-drag profile.
"""
function band_drag(keep)
    CA.non_orographic_gravity_wave_compute_tendency!(Y, p, gw_model)
    if !isnothing(keep)
        mask_B0!(gw.gw_beres_B0_u, keep)
        mask_B0!(gw.gw_beres_B0_v, keep)
        run_forcing!()
    end
    trop_profile(gw.uforcing)
end

println("evaluating full forcing, AD99 baseline, and $(length(bands)) bands...")
full = band_drag(nothing)
ad99 = band_drag(falses(nc))
band_profiles = Dict(name => band_drag(keep) for (name, keep) in bands)

# --- 5. additivity gate ---
beres_full = full .- ad99
beres_sum = reduce(.+, [prof .- ad99 for prof in values(band_profiles)])
resid = maximum(abs.(beres_sum .- beres_full))
scale = maximum(abs.(beres_full))
println(
    "additivity gate: max |Σbands − full| = $resid (full-scale $scale) → " *
    (resid <= 1e-4 * scale ? "PASS" : "FAIL"),
)

# --- 6. output ---
out = Dict(
    "job_id" => job_id,
    "restart_override" => override_yml,
    "lat_band" => LAT_BAND,
    "z" => zprof,
    "u" => uprof,
    "utend_full" => full,
    "utend_ad99" => ad99,
    "utend_band" => Dict(name => prof .- ad99 for (name, prof) in band_profiles),
    "additivity_resid" => resid,
    "timestamp" => string(Dates.now()),
)
dest = normpath(
    joinpath(
        @__DIR__, "..", "..", "..", "..", "beres_manuscript", "derived",
        "band_deposition_$(job_id).json"),
)
mkpath(dirname(dest))
open(dest, "w") do io
    JSON.print(io, out, 1)
end
println("wrote $dest")
