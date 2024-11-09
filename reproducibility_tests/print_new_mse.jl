import OrderedCollections
import JSON

# Get cases from JobIDs in mse_tables file:
include(joinpath(@__DIR__, "latest_comparable_paths.jl"))
paths = latest_comparable_paths()

include(joinpath(@__DIR__, "mse_tables.jl"))
job_ids = reproducibility_test_job_ids

computed_mse = OrderedCollections.OrderedDict()
files_skipped = OrderedCollections.OrderedDict()
is_mse_file(x) = startswith(basename(x), "computed_mse") && endswith(x, ".json")
for job_id in job_ids
    files_skipped[job_id] = false
end

@info "length(job_ids) = $(length(job_ids))"
for job_id in job_ids
    all_filenames = readdir(joinpath(job_id, "output_active"); join = true)
    mse_filenames = filter(is_mse_file, all_filenames)
    isempty(mse_filenames) || @info "mse_filenames: $mse_filenames"
    for filename in mse_filenames
        if !isfile(filename)
            @warn "File $filename skipped"
            files_skipped[job_id] = true
            continue
        end
        if !haskey(computed_mse, job_id)
            jsonfile = JSON.parsefile(
                filename;
                dicttype = OrderedCollections.OrderedDict,
            )
            computed_mse[job_id] = jsonfile
        end
    end
end

println("################################# Computed MSEs")
println("#! format: off")
println("#")

for job_id in keys(computed_mse)
    for var in keys(computed_mse[job_id])
        if computed_mse[job_id][var] == "NA"
            println(
                "mse_dict[\"$job_id\"][$(var)] = \"$(computed_mse[job_id][var])\"",
            )
        else
            # It's easier to update the reference counter, rather than updating
            # the mse tables, so let's always print zeros:
            computed_mse[job_id][var] = 0
            println(
                "mse_dict[\"$job_id\"][$(var)] = $(computed_mse[job_id][var])",
            )
        end
    end
    println("#")
end
println("#! format: on")

println("#################################")

isempty(paths) && @warn string("No comparable references.")

# Cleanup
for job_id in job_ids
    all_files = readdir(job_id)
    mse_filenames = filter(is_mse_file, all_files)
    for f in mse_filenames
        rm(f; force = true)
    end
end

println("-- DO NOT COPY --")

if any(values(files_skipped))
    @info "Skipped files:"
    for key in keys(files_skipped)
        @info "     job_id:`$key`, file:`$(files_skipped[key])`"
    end
    error("Some MSE files where skipped due to missing files")
end
