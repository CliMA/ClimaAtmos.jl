import ClimaReproducibilityTests as CRT
import NCDatasets
import Tar
import ClimaCoreTempestRemap as CCTR

include("reproducibility_utils.jl")

function get_nc_data(ds, var::String)
    if haskey(ds, var)
        return ds[var]
    else
        for key in keys(ds.group)
            if haskey(ds.group[key], var)
                return ds.group[key][var]
            end
        end
    end
    error("No key $var for mse computation.")
    return nothing
end

"""
    to_dict(nc_filename::String, reference_keys::Vector{String})

Convert an NCDatasets file to a `Dict`.
"""
function to_dict(nc_filename::String, reference_keys::Vector{String})
    dict = Dict{String, AbstractArray}()
    NCDatasets.Dataset(nc_filename, "r") do ds
        for key in reference_keys
            dict[key] = vec(Array(get_nc_data(ds, key)))
        end
    end
    return dict
end

"""
    reproducibility_test(;
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
function reproducibility_test(;
    job_id,
    reference_mse,
    ds_filename_computed,
    varname,
)
    local ds_filename_reference
    reference_keys = map(k -> varname(k), collect(keys(reference_mse)))
    paths = String[] # initialize for later handling

    if haskey(ENV, "BUILDKITE_COMMIT")
        paths = latest_comparable_paths(; n = 10)
        isempty(paths) && return (reference_mse, paths)
        @info "`ds_filename_computed`: `$ds_filename_computed`"
        ds_filename_references =
            map(p -> joinpath(p, ds_filename_computed), paths)
        for ds_filename_reference in ds_filename_references
            @info "`ds_filename_reference`: `$ds_filename_reference`"
            job_dir = dirname(ds_filename_reference)
            nc_tar = joinpath(job_dir, "nc_files.tar")
            # We may have converted to tarball, try to
            # extract nc files from tarball first:
            if !isfile(ds_filename_reference)
                if isfile(nc_tar)
                    mktempdir(joinpath(job_dir, tempdir())) do tdir
                        # We must extract to an empty folder, let's
                        # move it back to job_dir after.
                        Tar.extract(nc_tar, tdir) do hdr
                            basename(hdr.path) ==
                            basename(ds_filename_reference)
                        end
                        mv(
                            joinpath(tdir, basename(ds_filename_reference)),
                            joinpath(job_dir, basename(ds_filename_reference));
                            force = true,
                        )
                    end
                else
                    @warn "There is no reference dataset, and no NC tar file."
                end
            end
            if !isfile(ds_filename_reference)
                msg = "\n\n"
                msg *= "Pull request author:\n"
                msg *= "    It seems that a new dataset,\n"
                msg *= "\n"
                msg *= "dataset file:`$(ds_filename_computed)`,"
                msg *= "\n"
                msg *= "    was created, or the name of the dataset\n"
                msg *= "    has changed. Please increment the reference\n"
                msg *= "    counter in `reproducibility_tests/ref_counter.jl`.\n"
                msg *= "\n"
                msg *= "    If this is not the case, then please\n"
                msg *= "    open an issue with a link pointing to this\n"
                msg *= "    PR and build.\n"
                msg *= "\n"
                msg *= "For more information, please find\n"
                msg *= "`reproducibility_tests/README.md` and read the section\n\n"
                msg *= "  `How to merge pull requests (PR) that get approved\n"
                msg *= "   but *break* reproducibility tests`\n\n"
                msg *= "for how to merge this PR."
                error(msg)
            end
        end
    else
        @warn "Buildkite not detected. Skipping reproducibility tests."
        @info "Please review output results before merging."
        return (reference_mse, paths)
    end

    local computed_mse
    @info "Prescribed reference keys $reference_keys"
    dict_computed = to_dict(ds_filename_computed, reference_keys)
    dict_references =
        map(ds -> to_dict(ds, reference_keys), ds_filename_references)
    @info "Computed keys $(collect(keys(dict_computed)))"
    @info "Reference keys $(collect(keys(first(dict_references))))"
    if all(dr -> keys(dict_computed) == keys(dr), dict_references) && all(
        dr -> typeof(values(dict_computed)) == typeof(values(dr)),
        dict_references,
    )
        computed_mses = map(dict_references) do dict_reference
            CRT.compute_mse(;
                job_name = string(job_id),
                reference_keys = reference_keys,
                dict_computed,
                dict_reference,
            )
        end
    else
        msg = ""
        msg *= "The reproducibility test broke. Please find\n"
        msg *= "`reproducibility_tests/README.md` and read the section\n\n"
        msg *= "  `How to merge pull requests (PR) that get approved but *break* reproducibility tests`\n\n"
        msg *= "for how to merge this PR."
        error(msg)
    end
    return (computed_mses, paths)

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
