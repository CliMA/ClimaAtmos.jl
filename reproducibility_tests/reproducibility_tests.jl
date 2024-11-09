import JSON
import ClimaCore.Fields
import ClimaCore.InputOutput
include(joinpath(@__DIR__, "compute_mse.jl"))

function export_reproducibility_results(
    comms_ctx,
    job_id::String,
    Y_last::Fields.FieldVector,
    output_dir::String,
)
    # This is helpful for starting up new tables
    @info "Comparing variables:"
    for prop_chain in Fields.property_chains(Y_last)
        println("   $prop_chain")
    end

    ds_filename_computed = joinpath(output_dir, "prog_state.hdf5")

    hdfwriter = InputOutput.HDF5Writer(ds_filename_computed, comms_ctx)
    InputOutput.write!(hdfwriter, Y_last, "Y")
    Base.close(hdfwriter)

    (computed_mses, paths) =
        reproducibility_results(config.comms_ctx; job_id, ds_filename_computed)

    for (computed_mse, path) in zip(computed_mses, paths)
        commit_hash = basename(path)
        computed_mse_filename =
            joinpath(output_dir, "computed_mse_$commit_hash.json")

        open(computed_mse_filename, "w") do io
            JSON.print(io, computed_mse)
        end
    end
end
