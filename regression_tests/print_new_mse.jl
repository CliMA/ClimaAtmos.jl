import OrderedCollections
import JSON

# Get cases from JobIDs in mse_tables file:
include(joinpath(@__DIR__, "self_reference_or_path.jl"))
self_reference = self_reference_or_path() == :self_reference

all_lines = readlines(joinpath(@__DIR__, "mse_tables.jl"))
lines = deepcopy(all_lines)
filter!(x -> occursin("] = OrderedCollections", x), lines)
test_ids = getindex.(split.(lines, "\""), 2)
@assert count(x -> occursin("OrderedDict", x), all_lines) ==
        length(test_ids) + 1
@assert length(test_ids) ≠ 0 # safety net

include(joinpath(@__DIR__, "mse_tables.jl"))

percent_reduction_mse = Dict()

computed_mse = OrderedCollections.OrderedDict()
files_skipped = OrderedCollections.OrderedDict()
for test_id in test_ids
    filename = joinpath(test_id, "output_active/computed_mse.json")
    if !isfile(filename)
        @warn "File $filename skipped"
        files_skipped[test_id] = true
        continue
    end
    jsonfile =
        JSON.parsefile(filename; dicttype = OrderedCollections.OrderedDict)
    files_skipped[test_id] = false
    computed_mse[test_id] = jsonfile
end

println("#################################")
println("################################# MSE tables")
println("#################################")
println("#! format: off")
println("#")

println("all_best_mse = OrderedCollections.OrderedDict()\n#")
for test_id in keys(computed_mse)
    println("all_best_mse[\"$test_id\"] = OrderedCollections.OrderedDict()")
    for var in keys(computed_mse[test_id])
        if computed_mse[test_id][var] == "NA"
            println(
                "all_best_mse[\"$test_id\"][$(var)] = \"$(computed_mse[test_id][var])\"",
            )
        else
            self_reference && (computed_mse[test_id][var] = 0)
            println(
                "all_best_mse[\"$test_id\"][$(var)] = $(computed_mse[test_id][var])",
            )
        end
    end
    println("#")
end
println("#! format: on")

println("#################################")
println("#################################")
println("#################################")

if self_reference
    @warn string(
        "The printed `all_best_mse` values have",
        "been set to zero, due to self-reference,",
        "for copy-paste convenience.",
    )
end

# Cleanup
for test_id in test_ids
    rm(joinpath(test_id, "computed_mse.json"); force = true)
end

#####
##### min percentage reduction of mse across cases
#####

println("-- DO NOT COPY --")

for test_id in keys(computed_mse)
    percent_reduction_mse[test_id] = 0
    for var in keys(computed_mse[test_id])
        if haskey(all_best_mse[test_id], var)
            all_best_mse[test_id][var] isa Real || continue # skip if "NA"
            computed_mse[test_id][var] isa Real || continue # skip if "NA"
            percent_reduction_mse[test_id] = min(
                percent_reduction_mse[test_id],
                (all_best_mse[test_id][var] - computed_mse[test_id][var]) /
                all_best_mse[test_id][var] * 100,
            )
        else
            percent_reduction_mse[test_id] = "NA"
        end
    end
end

for test_id in keys(percent_reduction_mse)
    @info "percent_reduction_mse[$test_id] = $(percent_reduction_mse[test_id])"
end
if !isempty(percent_reduction_mse)
    @info "min mse reduction (%) over all cases = $(min(values(percent_reduction_mse)...))"
end

if any(values(files_skipped))
    @info "Skipped files:"
    for key in keys(files_skipped)
        @info "     test_id:`$key`, file:`$(files_skipped[key])`"
    end
    error("Some MSE files where skipped due to missing files")
end
