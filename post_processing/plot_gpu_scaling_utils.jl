using JLD2
import PrettyTables as PT

function get_jld2data(output_dir, job_id, t_int_days, s)
    secs_per_day = 60 * 60 * 24
    secs_per_hour = 60 * 60
    days_per_year = 8760 / 24
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
            found = true
        end
    end
    if !found
        @show readdir(output_dir)
        for foldername in readdir(output_dir)
            @show occursin(job_id, foldername)
            @show occursin(s, foldername)
        end
    end
    order = sortperm(nprocs_clima_atmos)
    nprocs_clima_atmos, ncols_per_process, walltime_clima_atmos =
        nprocs_clima_atmos[order],
        ncols_per_process[order],
        walltime_clima_atmos[order]

    # simulated years per day
    sypd_clima_atmos =
        (secs_per_day ./ walltime_clima_atmos) * t_int_days ./ days_per_year

    # GPU hours
    gpu_hours_clima_atmos =
        nprocs_clima_atmos .* walltime_clima_atmos / secs_per_hour

    data = hcat(
        nprocs_clima_atmos,
        ncols_per_process,
        walltime_clima_atmos,
        sypd_clima_atmos,
    )
    PT.pretty_table(
        data;
        header = ["N procs", "Ncols per process", "walltime (seconds)", "SYPD"],
        alignment = :l,
    )
    return (;
        nprocs_clima_atmos,
        ncols_per_process,
        walltime_clima_atmos,
        sypd_clima_atmos,
        gpu_hours_clima_atmos,
    )
end
