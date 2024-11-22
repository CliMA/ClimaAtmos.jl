import OrderedCollections
import JSON

# Get cases from JobIDs in mse_tables file:
include(joinpath(@__DIR__, "reproducibility_utils.jl"))
paths = latest_comparable_paths()

all_lines = readlines(joinpath(@__DIR__, "mse_tables.jl"))
lines = deepcopy(all_lines)
filter!(x -> occursin("] = OrderedCollections", x), lines)
job_ids = getindex.(split.(lines, "\""), 2)
@assert count(x -> occursin("OrderedDict", x), all_lines) == length(job_ids) + 1
@assert length(job_ids) ≠ 0 # safety net

include(joinpath(@__DIR__, "mse_tables.jl"))

computed_mse = OrderedCollections.OrderedDict()
files_skipped = OrderedCollections.OrderedDict()
is_mse_file(x) = startswith(basename(x), "computed_mse") && endswith(x, ".json")
for job_id in job_ids
    files_skipped[job_id] = false
end

for job_id in job_ids
    all_filenames = readdir(joinpath(job_id, "output_active"); join = true)
    mse_filenames = filter(is_mse_file, all_filenames)
    @info "mse_filenames: $mse_filenames"
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

println("#################################")
println("################################# MSE tables")
println("#################################")
println("#! format: off")
println("#")

println("all_best_mse = OrderedCollections.OrderedDict()\n#")
for job_id in keys(computed_mse)
    println("all_best_mse[\"$job_id\"] = OrderedCollections.OrderedDict()")
    for var in keys(computed_mse[job_id])
        if computed_mse[job_id][var] == "NA"
            println(
                "all_best_mse[\"$job_id\"][$(var)] = \"$(computed_mse[job_id][var])\"",
            )
        else
            # It's easier to update the reference counter, rather than updating
            # the mse tables, so let's always print zeros:
            computed_mse[job_id][var] = 0
            println(
                "all_best_mse[\"$job_id\"][$(var)] = $(computed_mse[job_id][var])",
            )
        end
    end
    println("#")
end
println("#! format: on")

println("#################################")
println("#################################")
println("#################################")

if isempty(paths)
    @warn string(
        "The printed `all_best_mse` values have",
        "been set to zero, due to no comparable references,",
        "for copy-paste convenience.",
    )
end

# Cleanup
for job_id in job_ids
    all_files = readdir(job_id)
    mse_filenames = filter(is_mse_file, all_files)
    for f in mse_filenames
        rm(f; force = true)
    end
end

println("-- DO NOT COPY --")

for job_id in keys(computed_mse)
    for var in keys(computed_mse[job_id])
        if haskey(all_best_mse[job_id], var)
            all_best_mse[job_id][var] isa Real || continue # skip if "NA"
            computed_mse[job_id][var] isa Real || continue # skip if "NA"
        end
    end
end

if any(values(files_skipped))
    @info "Skipped files:"
    for key in keys(files_skipped)
        @info "     job_id:`$key`, file:`$(files_skipped[key])`"
    end
    error("Some MSE files where skipped due to missing files")
end
