# post_processing/run_all_scm_intercomparisons.jl
#
# Driver that runs edmf_scm_intercomparison.jl for every registered SCM case
# and combines the results into a single joint PDF using pdfunite (Poppler_jll).
#
# All dependencies live in the shared .buildkite environment; always use
# --project=.buildkite (the CI Initialize step pre-installs it).
#
# Usage (from the repo root):
#   julia --project=.buildkite post_processing/run_all_scm_intercomparisons.jl \
#       [--sim_root <dir>]   # root that contains <job_id>/output_active/nc_files/
#                            # defaults to the current working directory
#       [--ref_root <dir>]   # directory containing the PyCLES NC files;
#                            # omit to use Julia artifacts (Artifacts.toml)
#       [--outdir <dir>]     # directory for PDFs; default: scm_intercomparison_plots/
#       [--cases A,B,C]      # comma-separated subset of cases to run (optional)
#
# Output:
#   <outdir>/scm_intercomparison.pdf  – joint multi-page PDF (all cases)
#   Individual per-case PDFs are removed after merging.

import ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using Poppler_jll: pdfunite

# ── CLI ───────────────────────────────────────────────────────────────────────
s = ArgParseSettings()
#! format: off
@add_arg_table! s begin
    "--sim_root"
        help = "Root directory containing <job_id>/output_active/nc_files/ subdirs"
        default = "."
    "--ref_root"
        help = "Directory with PyCLES .nc files (omit to use Julia artifacts)"
        default = ""
    "--outdir"
        help = "Output directory for PDFs"
        default = "scm_intercomparison_plots"
    "--cases"
        help = "Comma-separated list of cases to run (default: all)"
        default = ""
end
#! format: on
args = parse_args(s)

sim_root = args["sim_root"]
ref_root = args["ref_root"]
outdir = args["outdir"]
only_cases = isempty(args["cases"]) ? String[] : split(args["cases"], ",")

# ── Case registry ─────────────────────────────────────────────────────────────
# Order matches the Buildkite pipeline: Soares, GABLS, BOMEX, DYCOMS RF01/RF02, Rico, TRMM.
const CASES = [
    (case = "Soares", job_id = "prognostic_edmfx_soares_column", ref_file = "Soares.nc"),
    (case = "GABLS", job_id = "prognostic_edmfx_gabls_column", ref_file = "GABLS.nc"),
    (case = "Bomex", job_id = "prognostic_edmfx_bomex_column", ref_file = "Bomex.nc"),
    (
        case = "DYCOMS_RF01",
        job_id = "prognostic_edmfx_dycoms_rf01_column",
        ref_file = "DYCOMS_RF01.nc",
    ),
    (
        case = "DYCOMS_RF02",
        job_id = "prognostic_edmfx_dycoms_rf02_column",
        ref_file = "DYCOMS_RF02.nc",
    ),
    (case = "Rico", job_id = "prognostic_edmfx_rico_column", ref_file = "Rico.nc"),
    (case = "TRMM_LBA", job_id = "prognostic_edmfx_trmm_column", ref_file = "TRMM_LBA.nc"),
]

# ── Setup ─────────────────────────────────────────────────────────────────────
mkpath(outdir)

plotter = joinpath(@__DIR__, "edmf_scm_intercomparison.jl")

# ── Run each case ─────────────────────────────────────────────────────────────
let n_ok = 0, n_err = 0, page_pdfs = String[]

    for (; case, job_id, ref_file) in CASES
        # Apply --cases filter if given
        if !isempty(only_cases) && !(case in only_cases)
            println("Skipping $case (not in --cases filter)")
            continue
        end

        println("\n═══════════════════════════════════════════")
        println("  Case: $case  (job: $job_id)")
        println("═══════════════════════════════════════════")

        # Locate simulation nc_files directory
        sim_dir = joinpath(sim_root, job_id, "output_active", "nc_files")
        if !isdir(sim_dir)
            @warn "nc_files dir not found, plotting reference only: $sim_dir"
            sim_dir = ""
        end

        # Locate reference NC file
        ref_nc = isempty(ref_root) ? "" : joinpath(ref_root, ref_file)
        if !isempty(ref_nc) && !isfile(ref_nc)
            @warn "Reference file not found, will try artifact fallback: $ref_nc"
            ref_nc = ""
        end

        # Per-case PDF (intermediate; merged below)
        out_pdf = joinpath(outdir, "$(lowercase(case))_intercomparison.pdf")

        # Build the argument list for the plotting script
        script_args = String[
            "--case", case,
            "--output", out_pdf,
        ]
        !isempty(sim_dir) && append!(script_args, ["--sim_dir", sim_dir])
        !isempty(ref_nc) && append!(script_args, ["--ref_nc", ref_nc])

        # Invoke the plotting script in a fresh julia process to keep environments
        # isolated and avoid any state leakage between cases.
        cmd = `$(Base.julia_cmd()) --color=yes --project=$(joinpath(@__DIR__, "..", ".buildkite")) $plotter $script_args`

        try
            run(cmd)
            println("✓ $case → $out_pdf")
            push!(page_pdfs, out_pdf)
            n_ok += 1
        catch e
            @error "✗ $case failed: $e"
            n_err += 1
        end
    end

    # ── Merge into single joint PDF ───────────────────────────────────────────────
    joint_pdf = joinpath(outdir, "scm_intercomparison.pdf")
    if !isempty(page_pdfs)
        run(`$(pdfunite()) $(Cmd(page_pdfs)) $joint_pdf`)
        println("\nJoint PDF → $joint_pdf  ($(length(page_pdfs)) pages)")
        # Remove individual per-case PDFs now that they are merged
        rm.(page_pdfs; force = true)
    end

    # ── Summary ───────────────────────────────────────────────────────────────────
    println("\n═══════════════════════════════════════════")
    println("  Done: $n_ok succeeded, $n_err failed")
    println("  Joint PDF: $(abspath(joint_pdf))")
    println("═══════════════════════════════════════════")

    n_err > 0 && exit(1)

end # let
