using JLD2
function get_jld2data(output_dir, job_id, s)
    FT = Float64
    nprocs_clima_atmos = Int[]
    ncols_per_process = Int[]
    walltime_clima_atmos = FT[]
    found = false
    for foldername in readdir(output_dir)
        if occursin(job_id, foldername) && occursin(s, foldername)
            nprocs_string = split(split(foldername, s)[end], "process")[1]
            file = joinpath(
                output_dir,
                foldername,
                "output_active",
                "scaling_data_$(nprocs_string)_processes.jld2",
            )
            if !isfile(file)
                @show readdir(output_dir)
                @show readdir(dirname(file))
            end
            dict = load(file)
            push!(nprocs_clima_atmos, Int(dict["nprocs"]))
            push!(ncols_per_process, Int(dict["ncols_per_process"]))
            push!(walltime_clima_atmos, FT(dict["walltime"]))
        else
            @show occursin(job_id, foldername)
            @show occursin(s, foldername)
        end
    end
    if !found
        @show readdir(output_dir)
    end
    @show nprocs_clima_atmos
    @show ncols_per_process
    @show walltime_clima_atmos
    return (; nprocs_clima_atmos, ncols_per_process, walltime_clima_atmos)
end
