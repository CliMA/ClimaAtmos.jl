import OrderedCollections

# Get cases from JobIDs in mse_tables file:
include(joinpath(@__DIR__, "latest_comparable_paths.jl"))
paths = latest_comparable_paths()
include(joinpath(@__DIR__, "mse_tables.jl"))

#### Test that mse values are all zero if ref counter is incremented
mse_vals = collect(Iterators.flatten(map(x -> values(x), values(all_best_mse))))
if isempty(paths) && !all(mse_vals .== 0)
    error(
        "All mse values in `reproducibility_tests/mse_tables.jl` must be set to zero when the reference counter is incremented",
    )
end
