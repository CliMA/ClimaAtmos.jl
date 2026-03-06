# reload plotting definitions
include("../post_processing/ci_plots.jl")

# Visualize the solution
if ClimaComms.iamroot(config.comms_ctx)
    include(
        joinpath(
            pkgdir(CA),
            "reproducibility_tests",
            "reproducibility_utils.jl",
        ),
    )
    @info "Plotting"
    paths = latest_comparable_dirs() # __build__ path (not job path)
    if isempty(paths)
        make_plots(Val(Symbol(reference_job_id)), simulation.output_dir)
    else
        main_job_path = joinpath(first(paths), reference_job_id)
        nc_dir = joinpath(main_job_path, "nc_files")
        if ispath(nc_dir)
            @info "nc_dir exists"
        else
            mkpath(nc_dir)
            # Try to extract nc files from tarball:
            @info "Comparing against $(readdir(nc_dir))"
        end
        if isempty(readdir(nc_dir))
            if isfile(joinpath(main_job_path, "nc_files.tar"))
                Tar.extract(joinpath(main_job_path, "nc_files.tar"), nc_dir)
            else
                @warn "No nc_files found"
            end
        else
            @info "Files already extracted"
        end

        paths = if isempty(readdir(nc_dir))
            simulation.output_dir
        else
            [simulation.output_dir, nc_dir]
        end
        make_plots(Val(Symbol(reference_job_id)), paths)
    end
    @info "Plotting done"

end