import ClimaReproducibilityTests as CRT
import ClimaCore: InputOutput, Fields
import ClimaComms

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

function no_comparison_error(dirs, non_existent_files)
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
    msg *= "\n\n"
    for dir in dirs
        msg *= "Files in dir $dir\n"
        for file in all_files_in_dir(dir)
            msg *= "     $file\n"
        end
    end
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

    data_file_references =
        map(p -> joinpath(p, job_id, reference_filename), dirs)

    # foreach(x->maybe_extract(x), data_file_references)

    non_existent_files = filter(x -> !isfile(x), data_file_references)
    isempty(non_existent_files) || no_comparison_error(dirs, non_existent_files)

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
    if debug_reproducibility()
        println("------ end of reproducibility_results")
        @show computed_mses
        println("------")
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
 - Writes the dictionary of comparisons to mse files in the reproducibility
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
    computed_filename = "prog_state.hdf5",
    n::Int = 10,
    ref_counter_PR::Int = read_ref_counter(
        joinpath(@__DIR__, "ref_counter.jl"),
    ),
    skip::Bool = !haskey(ENV, "BUILDKITE_COMMIT"),
    repro_folder = "reproducibility_bundle",
)
    repro_dir = joinpath(computed_dir, repro_folder)
    data_file_computed = joinpath(repro_dir, computed_filename)

    mkpath(repro_dir)
    hdfwriter = InputOutput.HDF5Writer(data_file_computed, comms_ctx)
    InputOutput.write!(hdfwriter, field_vec, name)
    Base.close(hdfwriter)
    @info "Reproducibility: File $data_file_computed exported"

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

    commit_shas = readdir(save_dir)
    for (computed_mse, dir) in zip(computed_mses, dirs)
        commit_hash = commit_sha_from_dir(commit_shas, dir)
        computed_mse_file = joinpath(repro_dir, "computed_mse_$commit_hash.dat")

        open(computed_mse_file, "w") do io
            print(io, computed_mse)
        end
    end
    return (data_file_computed, computed_mses, dirs, how)
end

function commit_sha_from_mse_file(file)
    filename = basename(file)
    if startswith(filename, "computed_mse_") && endswith(filename, ".dat")
        return replace(filename, "computed_mse_" => "", ".dat" => "")
    else
        error("File $file does not follow correct format.")
    end
end

import ClimaReproducibilityTests as CRT
using Test
import PrettyTables

"""
    report_reproducibility_results(
        [computed_mses, computed_mse_filenames];
        n_pass_limit = 5,
        test_broken_report_flakiness,
    )

Prints reproducibility summary and returns a symbol indicating that results
are:
 - `:not_yet_reproducible` not yet reproducible against compared references
 - `:now_reproducible` _now_ reproducible against compared references
 - `:reproducible` reproducible against compared references
 - `:not_reproducible` _not_ reproducible against compared references

given:

 - `computed_mses` a dictionary of Mean-Squared-Errors (MSEs) between the
   computed and reference data
 - `n_pass_limit` the minimum number of reproducible times to flag that a test
   is considered reproducible
 - `test_broken_report_flakiness` a Bool indicating to measure a report the
   flakiness while allow passing

The reason for the distinction between

 - `:not_yet_reproducible` vs `:not_reproducible` and
 - `:reproducible` vs `:now_reproducible`

is a matter of expectation. We want users to be able to declare certain types of
states and transitions:
 - I have a simulation that is reproducible, I expect it remains that way
 - I have a simulation that is not reproducible, but please notify me if it
   seems to become reproducible

The returned symbols should allow users to declare tests with `@test` or
`@test_broken` more easily.
"""
function report_reproducibility_results end

cflatten(x) = collect(Base.Iterators.flatten(x))
function get_status_emoji(b, test_broken_report_flakiness)
    if test_broken_report_flakiness
        b ? "🟢" : "🟡"
    else
        b ? "🟢" : "🔴"
    end
end

# Default to stdout
report_reproducibility_results(
    sources::Vector{String},
    computed_mses::Vector{<:AbstractDict};
    n_pass_limit = 5,
    test_broken_report_flakiness,
) = report_reproducibility_results(
    stdout,
    sources,
    computed_mses;
    n_pass_limit,
    test_broken_report_flakiness,
)

function report_reproducibility_results(
    io::IO,
    sources::Vector{String},
    computed_mses::Vector{<:AbstractDict};
    n_pass_limit = 5,
    test_broken_report_flakiness,
)
    # Initialize summary results
    test_broken = false
    statuses = Symbol[]
    status_emoji(x) = get_status_emoji(x, test_broken_report_flakiness)

    for computed_mse in computed_mses
        all_reproducible = true
        if debug_reproducibility()
            println("---- in report_reproducibility_results")
            @show computed_mse
            println("----")
        end
        for (var, reproducible) in CRT.test_mse(; computed_mse)
            if !reproducible
                all_reproducible = false
            end
        end
        strict_status = all_reproducible ? :pass : :fail
        push!(statuses, strict_status)
    end
    n_passes = count(x -> x == :pass, statuses)

    n_times_reproducible = n_passes
    n_times_not_reproducible = length(computed_mses) - n_passes

    header = ["Source", "Variable", "Data scale", "MSE", "Status"]

    computed_mse1 = first(computed_mses)
    variables = collect(keys(computed_mse1))
    data_scales = cflatten(map(computed_mses) do computed_mse
        collect(values(computed_mse))
    end)
    statuses = cflatten(map(computed_mses) do computed_mse
        collect(values(CRT.test_mse(; computed_mse)))
    end)

    computed_data = vcat(map(variables) do var
        ["computed" var "skipped" "" ""] # TODO: export computed data
    end...)
    nv = length(variables)
    ns = length(sources)
    scolumn = cflatten(map(x -> map(y -> x, 1:nv), sources))
    vcolumn = cflatten(map(x -> variables, 1:ns))
    compare_data = hcat(
        scolumn,
        vcolumn,
        data_scales,
        data_scales,
        map(status_emoji, statuses),
    )

    table_data = vcat(computed_data, compare_data)

    PrettyTables.pretty_table(io, table_data; header, crop = :none)


    header = ["Source", "Status"]

    summary_statuses = map(1:nv:(ns * nv)) do i
        all(statuses[i:(i + nv - 1)])
    end
    summary_statuses = map(status_emoji, summary_statuses)
    table_data = hcat(sources, summary_statuses)
    PrettyTables.pretty_table(io, table_data; header, crop = :none)

    n_comparisons = length(computed_mses)
    println(io, "Summary:")
    println(io, "   n_comparisons                = $n_comparisons")
    println(io, "   n_times_reproducible         = $n_times_reproducible")
    println(io, "   n_times_not_reproducible     = $n_times_not_reproducible")
    println(io, "   n_passes                     = $n_passes")
    println(io, "   n_pass_limit                 = $n_pass_limit")
    println(
        io,
        "   test_broken_report_flakiness = $test_broken_report_flakiness",
    )


    # If we successfully compare against 5 other jobs,
    # let's error and tell the user that the job now
    # seems reproducible.
    if test_broken_report_flakiness && n_passes ≥ n_pass_limit
        return :now_reproducible
    elseif test_broken_report_flakiness
        return :not_yet_reproducible
    else
        if n_passes ≥ n_pass_limit || n_passes == n_comparisons
            return :reproducible
        else
            return :not_reproducible
        end
    end
end
