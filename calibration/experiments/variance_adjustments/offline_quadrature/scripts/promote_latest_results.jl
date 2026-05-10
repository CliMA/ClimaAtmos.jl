# Promote timestamped quadrature results to deterministic "latest" names per case
# Usage (from repo root):
#   julia --project=calibration/experiments/variance_adjustments calibration/experiments/variance_adjustments/offline_quadrature/scripts/promote_latest_results.jl

using Dates

outputs_root = joinpath(@__DIR__, "..", "outputs")
if !isdir(outputs_root)
    println("No outputs directory found at ", outputs_root)
    exit(0)
end

for case_dir in filter(isdir, readdir(outputs_root; join=true))
    try
        # find jld2 files not already named latest
        jld2_files = filter(f -> endswith(f, ".jld2") && !occursin("_latest.jld2", f), readdir(case_dir; join=true))
        if !isempty(jld2_files)
            sort!(jld2_files)
            latest = last(jld2_files)
            dest = joinpath(case_dir, "quadrature_results_latest.jld2")
            try
                cp(latest, dest; force=true)
                println("Promoted JLD2 for case ", basename(case_dir), ": ", basename(latest), " -> ", basename(dest))
            catch err
                @warn "Failed to copy JLD2" src=latest dest=dest error=(err, catch_backtrace())
            end
        else
            println("No JLD2 files to promote for case ", basename(case_dir))
        end

        # PNGs
        png_files = filter(f -> endswith(f, ".png") && !occursin("_latest.png", f), readdir(case_dir; join=true))
        if !isempty(png_files)
            sort!(png_files)
            latest_png = last(png_files)
            dest_png = joinpath(case_dir, "quadrature_summary_latest.png")
            try
                cp(latest_png, dest_png; force=true)
                println("Promoted PNG for case ", basename(case_dir), ": ", basename(latest_png), " -> ", basename(dest_png))
            catch err
                @warn "Failed to copy PNG" src=latest_png dest=dest_png error=(err, catch_backtrace())
            end
        else
            println("No PNG files to promote for case ", basename(case_dir))
        end
    catch err
        @warn "Error processing case directory" case=case_dir error=(err, catch_backtrace())
    end
end

println("Promotion complete at ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
