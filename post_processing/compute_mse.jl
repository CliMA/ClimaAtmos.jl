import NCRegressionTests
import NCDatasets
import ClimaCoreTempestRemap as CCTR

include("self_reference_or_path.jl")

"""
    regression_test(;
        job_id,
        reference_mse,
        ds_filename_computed,
        ds_filename_reference = nothing,
        varname,
    )

Returns a `Dict` of mean-squared errors between
`NCDataset`s `ds_filename_computed` and
`ds_filename_reference` for all keys in `reference_mse`.
Keys in `reference_mse` may directly map to keys in
the `NCDataset`s, or they may be mapped to the keys
via `varname`.

If running on buildkite, we get `ds_filename_reference`
from the latest merged dataset on Caltech central.
"""
function regression_test(; job_id, reference_mse, ds_filename_computed, varname)
    local ds_filename_reference

    if haskey(ENV, "BUILDKITE_COMMIT")
        path = self_reference_or_path()
        path == :self_reference && return reference_mse
        ds_filename_reference = joinpath(path, ds_filename_computed)
        @info "`ds_filename_computed`: `$ds_filename_computed`"
        @info "`ds_filename_reference`: `$ds_filename_reference`"
        if !isfile(ds_filename_reference)
            msg = "\n\n"
            msg *= "Pull request author:\n"
            msg *= "    It seems that a new dataset,\n"
            msg *= "\n"
            msg *= "dataset file:`$(ds_filename_computed)`,"
            msg *= "\n"
            msg *= "    was created, or the name of the dataset\n"
            msg *= "    has changed. Please increment the reference\n"
            msg *= "    counter in `post_processing/ref_counter.jl`.\n"
            msg *= "\n"
            msg *= "    If this is not the case, then please\n"
            msg *= "    open an issue with a link pointing to this\n"
            msg *= "    PR and build.\n"
            msg *= "\n"
            msg *= "For more information, please find\n"
            msg *= "`post_processing/README.md` and read the section\n\n"
            msg *= "  `How to merge pull requests (PR) that get approved\n"
            msg *= "   but *break* regression tests`\n\n"
            msg *= "for how to merge this PR."
            error(msg)
        end
    else
        @warn "Buildkite not detected. Skipping regression tests."
        @info "Please review output results before merging."
        return reference_mse
    end

    local computed_mse
    try
        computed_mse = NCRegressionTests.compute_mse(;
            job_name = string(job_id),
            reference_mse = reference_mse,
            ds_filename_computed = ds_filename_computed,
            ds_filename_reference = ds_filename_reference,
            varname = varname,
        )
    catch err
        msg = ""
        msg *= "The regression test broke. Please find\n"
        msg *= "`post_processing/README.md` and read the section\n\n"
        msg *= "  `How to merge pull requests (PR) that get approved but *break* regression tests`\n\n"
        msg *= "for how to merge this PR."
        @info msg
        rethrow(err.error)
    end
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
