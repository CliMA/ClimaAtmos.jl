import JSON
import ClimaCore.Fields as Fields
include(joinpath(@__DIR__, "compute_mse.jl"))

function perform_reproducibility_tests(
    job_id::String,
    Y_last::Fields.FieldVector,
    all_best_mse::AbstractDict,
    output_dir::String,
)
    # This is helpful for starting up new tables
    @info "Job-specific MSE table format:"
    println("all_best_mse[\"$job_id\"] = OrderedCollections.OrderedDict()")
    for prop_chain in Fields.property_chains(Y_last)
        println("all_best_mse[\"$job_id\"][$prop_chain] = 0.0")
    end
    # Extract best mse for this job:
    best_mse = all_best_mse[job_id]

    ds_filename_computed = joinpath(output_dir, "prog_state.nc")

    function process_name(s::AbstractString)
        # "c_ρ", "c_ρe", "c_uₕ_1", "c_uₕ_2", "f_w_1", "c_sgs⁰_ρatke"
        s = replace(s, "components_data_" => "")
        s = replace(s, "ₕ" => "_h")
        s = replace(s, "ρ" => "rho")
        s = replace(s, "⁰" => "_0")
        return s
    end
    varname(pc::Tuple) = process_name(join(pc, "_"))

    export_nc(Y_last; nc_filename = ds_filename_computed, varname)
    (computed_mses, paths) = reproducibility_test(;
        job_id,
        reference_mse = best_mse,
        ds_filename_computed,
        varname,
    )

    for (computed_mse, path) in zip(computed_mses, paths)
        commit_hash = basename(path)
        computed_mse_filename =
            joinpath(output_dir, "computed_mse_$commit_hash.json")

        open(computed_mse_filename, "w") do io
            JSON.print(io, computed_mse)
        end
    end
end
