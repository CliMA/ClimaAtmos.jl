import OrderedCollections
import JSON

# Get cases from JobIDs in mse_tables file:
all_lines = readlines(joinpath(@__DIR__, "mse_tables.jl"))
lines = deepcopy(all_lines)
filter!(x -> occursin("] = OrderedCollections", x), lines)
job_ids = getindex.(split.(lines, "\""), 2)
@assert count(x -> occursin("OrderedDict", x), all_lines) == length(job_ids) + 1
@assert length(job_ids) ≠ 0 # safety net

include(joinpath(@__DIR__, "mse_tables.jl"))

percent_reduction_mse = Dict()

computed_mse = OrderedCollections.OrderedDict()
files_skipped = OrderedCollections.OrderedDict()
for job_id in job_ids
    filename = joinpath(job_id, "computed_mse.json")
    if !isfile(filename)
        @warn "File $filename skipped"
        files_skipped[job_id] = true
        continue
    end
    jsonfile =
        JSON.parsefile(filename; dicttype = OrderedCollections.OrderedDict)
    files_skipped[job_id] = false
    computed_mse[job_id] = jsonfile
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
            println("all_best_mse[\"$job_id\"][$(var)] = \"$(computed_mse[job_id][var])\"")
        else
            println("all_best_mse[\"$job_id\"][$(var)] = $(computed_mse[job_id][var])")
        end
    end
    println("#")
end
println("#! format: on")

println("#################################")
println("#################################")
println("#################################")

# Cleanup
for job_id in job_ids
    rm(joinpath(job_id, "computed_mse.json"); force = true)
end

#####
##### min percentage reduction of mse across cases
#####

println("-- DO NOT COPY --")

for job_id in keys(computed_mse)
    percent_reduction_mse[job_id] = 0
    for var in keys(computed_mse[job_id])
        if haskey(all_best_mse[job_id], var)
            all_best_mse[job_id][var] isa Real || continue # skip if "NA"
            computed_mse[job_id][var] isa Real || continue # skip if "NA"
            percent_reduction_mse[job_id] = min(
                percent_reduction_mse[job_id],
                (all_best_mse[job_id][var] - computed_mse[job_id][var]) /
                all_best_mse[job_id][var] * 100,
            )
        else
            percent_reduction_mse[job_id] = "NA"
        end
    end
end

for job_id in keys(percent_reduction_mse)
    @info "percent_reduction_mse[$job_id] = $(percent_reduction_mse[job_id])"
end
@info "min mse reduction (%) over all cases = $(min(values(percent_reduction_mse)...))"

if any(values(files_skipped))
    @show files_skipped
    error("Some MSE files where skipped due to missing files")
end
