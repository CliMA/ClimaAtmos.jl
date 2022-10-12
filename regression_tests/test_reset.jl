import OrderedCollections

# Get cases from JobIDs in mse_tables file:
include(joinpath(@__DIR__, "self_reference_or_path.jl"))
self_reference = self_reference_or_path() == :self_reference
include(joinpath(@__DIR__, "mse_tables.jl"))

#### Test that mse values are all zero if ref counter is incremented
mse_vals = collect(Iterators.flatten(map(x -> values(x), values(all_best_mse))))
if self_reference && !all(mse_vals .== 0)
    error(
        "All mse values in `regression_tests/mse_tables.jl` must be set to zero when the reference counter is incremented",
    )
end
