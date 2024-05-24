using Test
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import ClimaAtmos as CA

include(joinpath("..", "perf", "common.jl"))

function extract_job_ids(folder_path)
    job_id_dict = Dict()
    for (root, _, files) in walkdir(folder_path)
        for file in files
            filepath = joinpath(root, file)
            data = YAML.load_file(filepath)
            if haskey(data, "job_id")
                job_id_dict[filepath] = data["job_id"]
            end
        end
    end
    return job_id_dict
end

file_to_job_id = extract_job_ids("config")
# Check that all jobs have a unique job_id
value_to_keys = Dict()
for (key, value) in file_to_job_id
    if haskey(value_to_keys, value)
        push!(value_to_keys[value], key)
    else
        value_to_keys[value] = [key]
    end
end
# Filter the keys that have more than one associated key
repeated_job_ids = filter(kv -> length(kv[2]) > 1, value_to_keys)

@test isempty(repeated_job_ids)
