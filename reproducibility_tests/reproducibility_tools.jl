import ClimaReproducibilityTests as CRT
import ClimaCore: InputOutput, Fields
import ClimaComms
import JSON

include("reproducibility_utils.jl")

function error_if_dissimilar_dicts(dicts, dict)
    if !similar_dicts(dicts, dict)
        println("Dictionaries assembled in the reproducibility tests are ")
        println("not similar, and cannot be compared.")
        foreach(dicts) do d
            if !similar_dicts(d, dict)
                @show keys(dict)
                @show keys(d)
                @show typeof(collect(values(dict)))
                @show typeof(collect(values(d)))
                @show size(collect(values(dict)))
                @show size(collect(values(d)))
            end
        end
        msg = "\nPlease find\n"
        msg *= "`reproducibility_tests/README.md` and read the section\n\n"
        msg *= "  `How to merge pull requests (PR) that get approved but *break* reproducibility tests`\n\n"
        msg *= "for how to merge this PR."
        error(msg)
    end
end

function no_comparison_error(non_existent_files)
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

"""
    to_dict(file::String, name, comms_ctx)

Convert the HDF5 file containing the prognostic field with name `name` into a
`Dict` using ClimaCore's `property_chains` and `single_field` functions.
"""
function to_dict(file::String, name, comms_ctx)
    dict = Dict{String, AbstractArray}()
    reader = InputOutput.HDF5Reader(file, comms_ctx)
    Y = InputOutput.read_field(reader, name)
    Base.close(reader)
    for prop_chain in Fields.property_chains(Y)
        dict[string(prop_chain)] =
            Array(vec(parent(Fields.single_field(Y, prop_chain))))
    end
    return dict
end

"""
    zero_dict(file::String, name, comms_ctx)

Return a dict of zeros for all `ClimaCore.Fields.property_chains` in the
fieldvector contained in the HDF5 file `file`.
"""
function zero_dict(file::String, name, comms_ctx)
    dict = Dict{String, AbstractArray}()
    reader = InputOutput.HDF5Reader(file, comms_ctx)
    Y = InputOutput.read_field(reader, name)
    Base.close(reader)
    for prop_chain in Fields.property_chains(Y)
        arr = vec(Array(parent(Fields.single_field(Y, prop_chain))))
        fill!(arr, 0)
        dict[string(prop_chain)] = arr
    end
    return dict
end

function maybe_extract(file)
    job_dir = dirname(file)
    nc_tar = joinpath(job_dir, "nc_files.tar")
    # We may have converted to tarball, try to
    # extract nc files from tarball first:
    isfile(file) && return nothing
    if !isfile(nc_tar)
        @warn "There is no reference dataset, and no NC tar file."
        return nothing
    end
    mktempdir(joinpath(job_dir, tempdir())) do tdir
        # We must extract to an empty folder, let's
        # move it back to job_dir after.
        Tar.extract(nc_tar, tdir) do hdr
            basename(hdr.path) == basename(file)
        end
        mv(
            joinpath(tdir, basename(file)),
            joinpath(job_dir, basename(file));
            force = true,
        )
    end
end

"""
    similar_dicts(dict::Dict, dict::Dict)
    similar_dicts(vec_of_dicts::Vector{<:Dict}, dict::Dict)

Returns `true` if dicts have the same keys and same value _types_ as `dict`, and
false otherwise.
"""
similar_dicts(v::Vector{<:Dict}, dict::Dict) =
    all(d -> similar_dicts(d, dict), v)
function similar_dicts(a::Dict, b::Dict)
    keys(a) == keys(b) || return false
    typeof(values(a)) == typeof(values(b)) || return false
    return true
end

"""
    reproducibility_results(
        comms_ctx;
        job_id,
        data_file_computed,
        skip = !haskey(ENV, "BUILDKITE_COMMIT")
    )

Returns a tuple containing:
 - the paths used for the comparison
 - a vector of `Dict`s of mean-squared errors between datasets
   `data_file_computed` and `data_file_reference` for all variables.
 - a symbol indicating how results were returned (for unit testing)

If running on buildkite, we get `data_file_reference` from the latest merged
dataset on Caltech central.
"""
function reproducibility_results(
    comms_ctx;
    job_id::String,
    data_file_computed::String,
    n::Int = 10,
    name::String,
    save_dir::String = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
    ref_counter_PR::Int = read_ref_counter(
        joinpath(@__DIR__, "ref_counter.jl"),
    ),
    reference_filename::String = "prog_state.hdf5",
    skip::Bool = !haskey(ENV, "BUILDKITE_COMMIT"),
)
    dirs = String[] # initialize for later handling

    # Skip if we're not on buildkite:
    skip && return (
        dirs,
        [zero_dict(data_file_computed, name, comms_ctx)],
        :skipped,
    )

    dirs =
        latest_comparable_dirs(; n, root_dir = save_dir, ref_counter_PR, skip) # should we pass / expose more kwargs?
    isempty(dirs) && return (
        dirs,
        [zero_dict(data_file_computed, name, comms_ctx)],
        :no_comparable_dirs,
    )

    data_file_references = map(p -> joinpath(p, reference_filename), dirs)

    # foreach(x->maybe_extract(x), data_file_references)

    non_existent_files = filter(x -> !isfile(x), data_file_references)
    isempty(non_existent_files) || no_comparison_error(non_existent_files)

    dict_computed_solution = to_dict(data_file_computed, name, comms_ctx)
    dict_reference_solutions =
        map(ds -> to_dict(ds, name, comms_ctx), data_file_references)
    reference_keys = keys(first(dict_reference_solutions))

    error_if_dissimilar_dicts(dict_reference_solutions, dict_computed_solution)

    computed_mses =
        map(zip(dirs, dict_reference_solutions)) do (p, dict_reference_solution)
            CRT.compute_mse(;
                job_name = string(job_id, "_", basename(p)),
                reference_keys = collect(string.(reference_keys)),
                dict_computed = dict_computed_solution,
                dict_reference = dict_reference_solution,
            )
        end
    return (dirs, computed_mses, :successful_comparison)
end


"""
    export_reproducibility_results(
        field_vec::Fields.FieldVector,
        comms_ctx::ClimaComms.AbstractCommsContext;
        job_id::String,
        computed_dir::String,
        save_dir::String = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
        name::String = "Y",
        reference_filename = "prog_state.hdf5",
        computed_filename = reference_filename,
        n::Int = 10,
        ref_counter_PR::Int = read_ref_counter(
            joinpath(@__DIR__, "ref_counter.jl"),
        ),
        skip::Bool = !haskey(ENV, "BUILDKITE_COMMIT"),
    )

This function returns:
    `(data_file_computed, computed_mses, dirs, how)`
 where
 - `data_file_computed` is the computed solution, i.e., the field vector
   `field_vec`
 - `computed_mses` is a vector of dictionaries containing mean-squared errors
   against the reference files, found via `latest_comparable_dirs`.

 - exports the results from field-vector `field_vec`, and saves it into the
   reproducibility folder
 - Compares the computed results against comparable references
 - Writes the dictionary of comparisons to json files in the reproducibility
   folder
"""
function export_reproducibility_results(
    field_vec::Fields.FieldVector,
    comms_ctx::ClimaComms.AbstractCommsContext;
    job_id::String,
    computed_dir::String,
    save_dir::String = "/central/scratch/esm/slurm-buildkite/climaatmos-main",
    name::String = "Y",
    reference_filename = "prog_state.hdf5",
    computed_filename = reference_filename,
    n::Int = 10,
    ref_counter_PR::Int = read_ref_counter(
        joinpath(@__DIR__, "ref_counter.jl"),
    ),
    skip::Bool = !haskey(ENV, "BUILDKITE_COMMIT"),
)
    repro_folder = joinpath(computed_dir, "reproducibility_bundle")
    data_file_computed = joinpath(repro_folder, reference_filename)

    mkpath(dirname(data_file_computed))
    hdfwriter = InputOutput.HDF5Writer(data_file_computed, comms_ctx)
    InputOutput.write!(hdfwriter, field_vec, name)
    Base.close(hdfwriter)

    (dirs, computed_mses, how) = reproducibility_results(
        comms_ctx;
        job_id,
        data_file_computed,
        n,
        name,
        reference_filename,
        save_dir = save_dir,
        ref_counter_PR,
        skip,
    )

    for (computed_mse, dir) in zip(computed_mses, dirs)
        commit_hash = basename(dir)
        computed_mse_file =
            joinpath(repro_folder, "computed_mse_$commit_hash.json")

        open(computed_mse_file, "w") do io
            JSON.print(io, computed_mse)
        end
    end
    return (data_file_computed, computed_mses, dirs, how)
end
