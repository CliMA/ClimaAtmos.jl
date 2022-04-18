include("NCRegressionTests.jl")
import .NCRegressionTests
import NCDatasets
import Dates
import ClimaCoreTempestRemap
const CCTR = ClimaCoreTempestRemap

function find_latest_dataset_folder(; dir = pwd())
    matching_paths = String[]
    for file in readdir(dir)
        !ispath(joinpath(dir, file)) && continue
        push!(matching_paths, joinpath(dir, file))
    end
    isempty(matching_paths) && return ""
    # sort by timestamp
    sorted_paths =
        sort(matching_paths; by = f -> Dates.unix2datetime(stat(f).mtime))
    return pop!(sorted_paths)
end

"""
    regression_test(;
        job_id,
        best_mse,
        ds_filename_computed,
        ds_filename_reference = nothing,
        varname,
    )

Returns a `Dict` of mean-squared errors between
`NCDataset`s `ds_filename_computed` and
`ds_filename_reference` for all keys in `best_mse`.
Keys in `best_mse` may directly map to keys in
the `NCDataset`s, or they may be mapped to the keys
via `varname`.

If running on buildkite, we get `ds_filename_reference`
from the latest merged dataset on Caltech central.
"""
function regression_test(;
    job_id,
    best_mse,
    ds_filename_computed,
    ds_filename_reference = nothing,
    varname,
)

    # # Kickstart CI: do not uncomment.
    # return best_mse

    # Note: cluster_data_prefix is also defined in move_output.jl
    if haskey(ENV, "BUILDKITE_COMMIT") && isnothing(ds_filename_reference)
        cluster_data_prefix = "/central/scratch/esm/slurm-buildkite/climaatmos-main"
        path = find_latest_dataset_folder(; dir = cluster_data_prefix)

        # To fix the reference dataset, use, for example:
        #     path = joinpath(cluster_data_prefix, "992d070")
        # where `992d070` is the commit sha to fix our reference to.

        # TODO: make this more robust in case folder/file changes
        main_files = readdir(path)
        @info "Files on main:"
        for file_on_main in main_files
            println("   file:$file_on_main, basename: $(basename(file_on_main))")
        end
        println("ds_filename_computed: $ds_filename_computed")
        ds_filename_reference = joinpath(path, ds_filename_computed)
    end
    println("ClimaAtmos.jl main dataset: $ds_filename_reference")

    computed_mse = NCRegressionTests.compute_mse(;
        job_name = string(job_id),
        best_mse = best_mse,
        ds_filename_computed = ds_filename_computed,
        ds_filename_reference = ds_filename_reference,
        varname = varname,
    )
    return computed_mse

end


##### TODO: move below functions to ClimaCore

function first_center_space(fv::Fields.FieldVector)
    for prop_chain in Fields.property_chains(fv)
        f = Fields.single_field(fv, prop_chain)
        space = axes(f)
        if space isa Spaces.CenterExtrudedFiniteDifferenceSpace
            return space
        end
    end
    error("Unfound space")
end

function first_face_space(fv::Fields.FieldVector)
    for prop_chain in Fields.property_chains(fv)
        f = Fields.single_field(fv, prop_chain)
        space = axes(f)
        if space isa Spaces.FaceExtrudedFiniteDifferenceSpace
            return space
        end
    end
    error("Unfound space")
end

function export_nc(
    Y::Fields.FieldVector;
    nc_filename,
    t_now = 0.0,
    center_space = first_center_space,
    face_space = first_face_space,
    filter_prop_chain = pn -> true, # use all fields
    varname::Function,
)
    prop_chains = Fields.property_chains(Y)
    filter!(filter_prop_chain, prop_chains)
    cspace = center_space(Y)
    fspace = face_space(Y)
    # create a temporary dir for intermediate data
    FT = eltype(Y)
    NCDatasets.NCDataset(nc_filename, "c") do nc
        # defines the appropriate dimensions and variables for a space coordinate
        # defines the appropriate dimensions and variables for a time coordinate (by default, unlimited size)
        nc_time = CCTR.def_time_coord(nc)
        CCTR.def_space_coord(nc, cspace, type = "cgll")
        CCTR.def_space_coord(nc, fspace, type = "cgll")
        # define variables for the prognostic states
        for prop_chain in Fields.property_chains(Y)
            f = Fields.single_field(Y, prop_chain)
            space = axes(f)
            nc_var = CCTR.defVar(nc, varname(prop_chain), FT, space, ("time",))
            nc_var[:, 1] = f
        end
        # TODO: interpolate w onto center space and save it the same way as the other vars
        nc_time[1] = t_now
    end
    return nothing
end
