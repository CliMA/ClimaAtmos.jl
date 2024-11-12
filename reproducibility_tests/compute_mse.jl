import ClimaReproducibilityTests as CRT
import NCDatasets
import Tar
import ClimaCoreTempestRemap as CCTR

include("latest_comparable_paths.jl")

"""
    to_dict(filename::String, comms_ctx)

Convert the HDF5 file containing the
prognostic field `Y` into a `Dict`
using ClimaCore's `property_chains` and
`single_field` functions.
"""
function to_dict(filename::String, comms_ctx)
    dict = Dict{Any, AbstractArray}()
    reader = InputOutput.HDF5Reader(filename, comms_ctx)
    Y = InputOutput.read_field(reader, "Y")
    Base.close(reader)
    for prop_chain in Fields.property_chains(Y)
        dict[prop_chain] =
            vec(Array(parent(Fields.single_field(Y, prop_chain))))
    end
    return dict
end

"""
    zero_dict(filename::String, comms_ctx)

Return a dict of zeros for all `ClimaCore.Fields.property_chains`
in the fieldvector `Y` contained in the HDF5 file `filename`.
"""
function zero_dict(filename::String, comms_ctx)
    dict = Dict{Any, AbstractArray}()
    reader = InputOutput.HDF5Reader(filename, comms_ctx)
    Y = InputOutput.read_field(reader, "Y")
    Base.close(reader)
    for prop_chain in Fields.property_chains(Y)
        dict[prop_chain] =
            vec(Array(parent(Fields.single_field(Y, prop_chain)))) .* 0
    end
    return dict
end

"""
    reproducibility_results(
        comms_ctx;
        job_id,
        ds_filename_computed,
        ds_filename_reference = nothing,
    )

Returns a `Dict` of mean-squared errors between
datasets `ds_filename_computed` and
`ds_filename_reference` for all variables.

If running on buildkite, we get `ds_filename_reference`
from the latest merged dataset on Caltech central.
"""
function reproducibility_results(comms_ctx; job_id, ds_filename_computed)
    local ds_filename_reference
    paths = String[] # initialize for later handling

    if haskey(ENV, "BUILDKITE_COMMIT")
        paths = latest_comparable_paths(10)
        isempty(paths) &&
            return (zero_dict(ds_filename_computed, comms_ctx), paths)
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
        end
        non_existent_files = filter(x -> !isfile(x), ds_filename_references)
        if !isempty(non_existent_files)
            msg = "\n\n"
            msg *= "Pull request author:\n"
            msg *= "    It seems that a new dataset,\n"
            msg *= "\n"
            msg *= "dataset file(s):`$(non_existent_files)`,"
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
    else
        @warn "Buildkite not detected. Skipping reproducibility tests."
        @info "Please review output results before merging."
        return (zero_dict(ds_filename_computed, comms_ctx), paths)
    end

    local computed_mse
    dict_computed = to_dict(ds_filename_computed, comms_ctx)
    dict_references = map(ds -> to_dict(ds, comms_ctx), ds_filename_references)
    reference_keys = keys(first(dict_references))
    @info "Reference keys $reference_keys"
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
