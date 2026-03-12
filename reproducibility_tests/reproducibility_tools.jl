import ClimaCore: InputOutput, Fields
import ClimaComms
import OrderedCollections
import PrettyTables
import Test: @testset, @test, @test_broken

include("reproducibility_utils.jl")

# ────────────────────────────────────────────────────────────────────
# RMS result type alias (used throughout)
# ────────────────────────────────────────────────────────────────────

"""Named-tuple type for per-variable RMS comparison results."""
const RMSResult = @NamedTuple{
    rms_diff::Float64,
    data_scale::Float64,
    relative_rms::Float64,
    n_points::Int,
}

# ────────────────────────────────────────────────────────────────────
# Physical variable metadata (labels + units in one place)
# ────────────────────────────────────────────────────────────────────

"""
    PHYSICAL_VAR_METADATA

Constant mapping ClimaCore property-chain names to `(label, unit)` tuples
for human-readable table output. Add entries here when new prognostic
variables are introduced.
"""
const PHYSICAL_VAR_METADATA =
    Dict{String, NamedTuple{(:label, :unit), Tuple{String, String}}}(
        "c.ρ" => (label = "ρ (density)", unit = "kg/m³"),
        "c.ρe_tot" => (label = "ρe_tot (total energy density)", unit = "J/m³"),
        "c.ρq_tot" => (label = "ρq_tot (total moisture density)", unit = "kg/m³"),
        "c.ρq_liq" => (label = "ρq_liq (liquid water density)", unit = "kg/m³"),
        "c.ρq_ice" => (label = "ρq_ice (ice water density)", unit = "kg/m³"),
        "c.ρq_rai" => (label = "ρq_rai (rain water density)", unit = "kg/m³"),
        "c.ρq_sno" => (label = "ρq_sno (snow water density)", unit = "kg/m³"),
        "c.ρn_liq" => (label = "ρn_liq (cloud droplet number)", unit = "1/m³"),
        "c.ρn_rai" => (label = "ρn_rai (rain droplet number)", unit = "1/m³"),
        "c.uₕ.components.data.:1" =>
            (label = "u₁ (covariant horiz. wind 1)", unit = "m²/s"),
        "c.uₕ.components.data.:2" =>
            (label = "u₂ (covariant horiz. wind 2)", unit = "m²/s"),
        "f.u₃.components.data.:1" => (label = "u₃ (covariant vert. wind)", unit = "m/s"),
        "c.sgs⁰.ρatke" => (label = "ρa·TKE (subgrid TKE)", unit = "kg/(m·s²)"),
        "c.sgsʲs.:1.ρa" => (label = "ρaʲ (updraft ρa)", unit = "kg/m³"),
        "c.sgsʲs.:1.ρae_tot" => (label = "ρae_totʲ (updraft energy)", unit = "J/m³"),
        "c.sgsʲs.:1.ρaq_tot" => (label = "ρaq_totʲ (updraft moisture)", unit = "kg/m³"),
        "f.sgsʲs.:1.u₃.components.data.:1" =>
            (label = "u₃ʲ (updraft vert. wind)", unit = "m/s"),
    )

"""
    prop_chain_key(prop_chain) -> String

Convert a ClimaCore property-chain tuple (e.g. `(:c, :ρq_tot)`) to a
dot-separated string key (e.g. `"c.ρq_tot"`) that matches the format
used in [`PHYSICAL_VAR_METADATA`](@ref).
"""
prop_chain_key(pc) = join(string.(pc), ".")

"""Look up the human-readable label for a property-chain key, falling back to the key itself."""
physical_var_label(key::String) =
    haskey(PHYSICAL_VAR_METADATA, key) ? PHYSICAL_VAR_METADATA[key].label : key

"""Look up the SI unit string for a property-chain key, falling back to `\"\"`."""
physical_var_unit(key::String) =
    haskey(PHYSICAL_VAR_METADATA, key) ? PHYSICAL_VAR_METADATA[key].unit : ""

# ────────────────────────────────────────────────────────────────────
# RMS comparison: computing differences between datasets
# ────────────────────────────────────────────────────────────────────

"""
    compute_rms(;
        job_name::String,
        reference_keys::Vector{String},
        dict_computed::AbstractDict,
        dict_reference::AbstractDict,
    )

Compute the root-mean-square (RMS) difference between `dict_computed` and
`dict_reference` for each key in `reference_keys`.

Returns an `OrderedDict{String, RMSResult}` mapping each key to:
  - `rms_diff`     = √(mean((computed - reference)²))  [same units as field]
  - `data_scale`   = mean(|reference|)                  [same units as field]
  - `relative_rms` = rms_diff / data_scale              [dimensionless]
  - `n_points`     = number of grid points compared
"""
function compute_rms(;
    job_name::String,
    reference_keys::Vector{String},
    dict_computed::AbstractDict,
    dict_reference::AbstractDict,
    # Deprecated kwargs kept for call-site compat; ignored in favor of PHYSICAL_VAR_METADATA
    labels::AbstractDict = Dict{String, String}(),
    units::AbstractDict = Dict{String, String}(),
)
    results = OrderedCollections.OrderedDict{String, RMSResult}()

    n_keys = length(reference_keys)
    display_names = Vector{String}(undef, n_keys)
    display_units = Vector{String}(undef, n_keys)
    rms_diffs = zeros(n_keys)
    data_scales = zeros(n_keys)
    relative_rmss = zeros(n_keys)

    for (i, key) in enumerate(reference_keys)
        computed_arr = dict_computed[key]
        reference_arr = dict_reference[key]
        n = length(reference_arr)

        # Data scale: mean of absolute reference values
        data_scale = sum(abs, reference_arr) / n

        # RMS difference
        rms = sqrt(sum((computed_arr .- reference_arr) .^ 2) / n)

        # Relative RMS (guard against zero data scale)
        rel = iszero(data_scale) ? rms : rms / data_scale

        rms_diffs[i] = rms
        data_scales[i] = data_scale
        relative_rmss[i] = rel
        display_names[i] = physical_var_label(key)
        display_units[i] = physical_var_unit(key)

        results[key] = (;
            rms_diff = rms,
            data_scale = data_scale,
            relative_rms = rel,
            n_points = n,
        )
    end

    _print_rms_table(
        job_name,
        display_names,
        display_units,
        rms_diffs,
        data_scales,
        relative_rmss,
    )
    return results
end

"""
    _print_rms_table(job_name, names, units, rms_diffs, data_scales, relative_rmss)

Print a formatted comparison table to stdout. Non-zero numeric values are
highlighted in red.
"""
function _print_rms_table(
    job_name,
    names,
    units,
    rms_diffs,
    data_scales,
    relative_rmss,
)
    header = ["Variable", "Unit", "RMS Diff", "Data Scale", "Relative RMS"]
    table_data = hcat(names, units, rms_diffs, data_scales, relative_rmss)

    hl_nonzero = PrettyTables.TextHighlighter(
        (data, i, j) -> j >= 3 && data[i, j] isa Number && data[i, j] > 0,
        PrettyTables.crayon"red bold",
    )
    @info "RMS comparison for `$job_name`"
    PrettyTables.pretty_table(
        table_data;
        column_labels = [header],
        formatters = [PrettyTables.fmt__printf("%.6e", 3:5)],
        highlighters = [hl_nonzero],
    )
end

# ────────────────────────────────────────────────────────────────────
# Tolerance and reproducibility testing
# ────────────────────────────────────────────────────────────────────

"""
    default_tolerance(data_scale; eps_factor = 10, FT = Float32)

Return a physically meaningful tolerance for an RMS difference:
`eps_factor * eps(FT) * max(data_scale, 1)`.

The `max(data_scale, 1)` floor prevents the tolerance from collapsing
to zero for fields with very small mean magnitude.
"""
function default_tolerance(data_scale; eps_factor = 10, FT = Float32)
    return eps_factor * eps(FT) * max(data_scale, one(data_scale))
end

"""
    test_reproducibility(;
        rms_results::AbstractDict,
        eps_factor = 10,
        FT = Float32,
    )

Test whether each variable's RMS difference is within tolerance.

Returns an `OrderedDict{String, NamedTuple}` mapping each key to:
    (reproducible::Bool, rms_diff::Float64, tolerance::Float64)
"""
function test_reproducibility(;
    rms_results::AbstractDict,
    eps_factor = 10,
    FT = Float32,
)
    results = OrderedCollections.OrderedDict{
        String,
        @NamedTuple{reproducible::Bool, rms_diff::Float64, tolerance::Float64}
    }()
    for (key, r) in rms_results
        tol = default_tolerance(r.data_scale; eps_factor, FT)
        pass = r.rms_diff ≤ tol
        results[key] =
            (; reproducible = pass, rms_diff = r.rms_diff, tolerance = tol)
    end
    return results
end

# ────────────────────────────────────────────────────────────────────
# Dict conversion from HDF5 FieldVectors
# ────────────────────────────────────────────────────────────────────

"""
    _read_field_dict(file, name, comms_ctx; zero_fill = false)

Read the HDF5 `file` containing a FieldVector stored under `name` and
return a `Dict{String, AbstractArray}` keyed by property-chain strings.

If `zero_fill = true`, all arrays are filled with zeros (used for
placeholder comparisons when no reference exists).
"""
function _read_field_dict(
    file::String,
    name,
    comms_ctx;
    zero_fill::Bool = false,
)
    dict = Dict{String, AbstractArray}()
    reader = InputOutput.HDF5Reader(file, comms_ctx)
    Y = InputOutput.read_field(reader, name)
    Base.close(reader)
    for prop_chain in Fields.property_chains(Y)
        arr = vec(Array(parent(Fields.single_field(Y, prop_chain))))
        zero_fill && fill!(arr, 0)
        dict[prop_chain_key(prop_chain)] = arr
    end
    return dict
end

"""
    to_dict(file, name, comms_ctx)

Convert an HDF5 prognostic-state file into a `Dict{String, AbstractArray}`.
"""
to_dict(file::String, name, comms_ctx) =
    _read_field_dict(file, name, comms_ctx)

"""
    zero_dict(file, name, comms_ctx)

Return a zero-filled dict with the same structure as the field in `file`.
"""
zero_dict(file::String, name, comms_ctx) =
    _read_field_dict(file, name, comms_ctx; zero_fill = true)

"""
    similar_dicts(a::Dict, b::Dict)
    similar_dicts(vec::Vector{<:Dict}, b::Dict)

Return `true` if dicts share the same keys and value lengths.
"""
similar_dicts(v::Vector{<:Dict}, dict::Dict) =
    all(d -> similar_dicts(d, dict), v)
function similar_dicts(a::Dict, b::Dict)
    keys(a) == keys(b) || return false
    all(k -> length(a[k]) == length(b[k]), keys(a)) || return false
    return true
end

"""
    error_if_dissimilar_dicts(dicts, dict)

Error with a helpful message if any dict in `dicts` has different keys or
value dimensions than `dict`.
"""
function error_if_dissimilar_dicts(dicts, dict)
    if !similar_dicts(dicts, dict)
        println("Dictionaries assembled in the reproducibility tests are ")
        println("not similar, and cannot be compared.")
        foreach(dicts) do d
            if !similar_dicts(d, dict)
                println("Reference keys: ", keys(d))
                println("Computed keys:  ", keys(dict))
                if keys(d) == keys(dict)
                    println("Mismatched array lengths detected.")
                end
            end
        end
        msg = "\nPlease find\n"
        msg *= "`reproducibility_tests/README.md` and read the section\n\n"
        msg *= "  `How to merge PRs that break reproducibility tests`\n\n"
        msg *= "for how to merge this PR."
        error(msg)
    end
end

"""
    no_comparison_error(dirs, non_existent_files)

Error with a message guiding the PR author to increment the reference
counter when expected reference files are missing.
"""
function no_comparison_error(dirs, non_existent_files)
    msg = """


    Pull request author:
        It seems that a new dataset,

    dataset file(s):`$(non_existent_files)`,

        was created, or the name of the dataset
        has changed. Please increment the reference
        counter in `reproducibility_tests/ref_counter.jl`.

        If this is not the case, then please
        open an issue with a link pointing to this
        PR and build.

    For more information, please find
    `reproducibility_tests/README.md` and read the section

      `How to merge PRs that break reproducibility tests`

    for how to merge this PR.

    """
    for dir in dirs
        msg *= "Files in dir $dir\n"
        for file in all_files_in_dir(dir)
            msg *= "     $file\n"
        end
    end
    error(msg)
end

# ────────────────────────────────────────────────────────────────────
# Core comparison: computing and exporting RMS differences
# ────────────────────────────────────────────────────────────────────

"""
    reproducibility_results(comms_ctx; job_id, data_file_computed, ...)

Compare the prognostic state in `data_file_computed` against reference
datasets stored on the cluster.

Returns `(dirs, rms_results_vec, how)` where `how` is one of
`:skipped`, `:no_comparable_dirs`, `:successful_comparison`.
"""
function reproducibility_results(
    comms_ctx;
    job_id::String,
    data_file_computed::String,
    n::Int = 10,
    name::String,
    save_dir::String = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
    ref_counter_PR::Int = read_ref_counter(
        joinpath(@__DIR__, "ref_counter.jl"),
    ),
    reference_filename::String = "prog_state.hdf5",
    skip::Bool = !haskey(ENV, "BUILDKITE_COMMIT"),
)
    dirs = String[]

    skip && return (
        dirs,
        [zero_rms_results(data_file_computed, name, comms_ctx)],
        :skipped,
    )

    dirs =
        latest_comparable_dirs(; n, root_dir = save_dir, ref_counter_PR, skip)
    isempty(dirs) && return (
        dirs,
        [zero_rms_results(data_file_computed, name, comms_ctx)],
        :no_comparable_dirs,
    )

    data_file_references =
        map(p -> joinpath(p, job_id, reference_filename), dirs)

    non_existent_files = filter(x -> !isfile(x), data_file_references)
    isempty(non_existent_files) || no_comparison_error(dirs, non_existent_files)

    dict_computed_solution = to_dict(data_file_computed, name, comms_ctx)
    dict_reference_solutions =
        map(ds -> to_dict(ds, name, comms_ctx), data_file_references)
    reference_keys = keys(first(dict_reference_solutions))

    error_if_dissimilar_dicts(dict_reference_solutions, dict_computed_solution)

    rms_results_vec =
        map(zip(dirs, dict_reference_solutions)) do (p, dict_ref)
            compute_rms(;
                job_name = string(job_id, "_", basename(p)),
                reference_keys = collect(string.(reference_keys)),
                dict_computed = dict_computed_solution,
                dict_reference = dict_ref,
            )
        end
    return (dirs, rms_results_vec, :successful_comparison)
end

"""
    zero_rms_results(file, name, comms_ctx)

Return a zero-valued RMS result dict for all property chains in `file`.
Used as a placeholder when no comparison is possible.
"""
function zero_rms_results(file, name, comms_ctx)
    zd = zero_dict(file, name, comms_ctx)
    results = OrderedCollections.OrderedDict{String, RMSResult}()
    for key in keys(zd)
        results[key] = (;
            rms_diff = 0.0,
            data_scale = 0.0,
            relative_rms = 0.0,
            n_points = length(zd[key]),
        )
    end
    return results
end

"""
    export_reproducibility_results(field_vec, comms_ctx; job_id, computed_dir, ...)

Export the prognostic state to HDF5, compare against references, and write
RMS comparison results to `computed_rms_<commit>.dat` files.

Returns `(data_file_computed, rms_results_vec, dirs, how)`.
"""
function export_reproducibility_results(
    field_vec::Fields.FieldVector,
    comms_ctx::ClimaComms.AbstractCommsContext;
    job_id::String,
    computed_dir::String,
    save_dir::String = "/resnick/scratch/esm/slurm-buildkite/climaatmos-main",
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

    (dirs, rms_results_vec, how) = reproducibility_results(
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
    for (rms_results, dir) in zip(rms_results_vec, dirs)
        commit_hash = commit_sha_from_dir(commit_shas, dir)
        rms_file = joinpath(repro_dir, "computed_rms_$commit_hash.dat")
        open(rms_file, "w") do io
            print(io, rms_results)
        end
    end
    return (data_file_computed, rms_results_vec, dirs, how)
end

"""
    commit_sha_from_rms_file(file)

Extract the commit SHA from a filename of the form `computed_rms_<sha>.dat`.
"""
function commit_sha_from_rms_file(file)
    filename = basename(file)
    if startswith(filename, "computed_rms_") && endswith(filename, ".dat")
        return replace(filename, "computed_rms_" => "", ".dat" => "")
    else
        error("File $file does not follow correct format (expected computed_rms_*.dat).")
    end
end

# ────────────────────────────────────────────────────────────────────
# Reporting: printing results and determining pass/fail
# ────────────────────────────────────────────────────────────────────

"""
    report_reproducibility_results(
        sources, rms_results_vec;
        n_pass_limit = 5,
        test_broken_report_flakiness,
        FT = Float32,
        eps_factor = 10,
    )

Print a per-reference table of RMS differences and a summary table,
then determine reproducibility status.

Returns one of:
  - `:reproducible`            — passes tolerance for enough references
  - `:not_reproducible`        — fails tolerance
  - `:not_yet_reproducible`    — flaky/broken test that still fails
  - `:now_reproducible`        — flaky test that now passes
"""
function report_reproducibility_results end

# Default to stdout
report_reproducibility_results(
    sources::Vector{String},
    rms_results_vec::Vector;
    n_pass_limit = 5,
    test_broken_report_flakiness,
    FT = Float32,
    eps_factor = 10,
) = report_reproducibility_results(
    stdout,
    sources,
    rms_results_vec;
    n_pass_limit,
    test_broken_report_flakiness,
    FT,
    eps_factor,
)

function report_reproducibility_results(
    io::IO,
    sources::Vector{String},
    rms_results_vec::Vector;
    n_pass_limit = 5,
    test_broken_report_flakiness,
    FT = Float32,
    eps_factor = 10,
)
    status_emoji(pass) =
        if test_broken_report_flakiness
            pass ? "🟢" : "🟡"
        else
            pass ? "🟢" : "🔴"
        end

    n_passes = 0
    per_source_pass = Vector{Bool}(undef, length(sources))

    for (idx, (source, rms_results)) in
        enumerate(zip(sources, rms_results_vec))
        repro = test_reproducibility(; rms_results, eps_factor, FT)

        all_pass = all(r -> r.reproducible, values(repro))
        per_source_pass[idx] = all_pass
        all_pass && (n_passes += 1)

        # Per-variable detail table
        variables = collect(keys(rms_results))
        nv = length(variables)

        header = [
            "Variable",
            "Unit",
            "RMS Diff",
            "Data Scale",
            "Tolerance",
            "Status",
        ]
        table_data = Matrix{Any}(undef, nv, 6)
        for (i, var) in enumerate(variables)
            r = rms_results[var]
            t = repro[var]
            table_data[i, 1] = physical_var_label(var)
            table_data[i, 2] = physical_var_unit(var)
            table_data[i, 3] = r.rms_diff
            table_data[i, 4] = r.data_scale
            table_data[i, 5] = t.tolerance
            table_data[i, 6] = status_emoji(t.reproducible)
        end

        println(io, "\n── Reference: $(basename(source)) ──")
        PrettyTables.pretty_table(
            io,
            table_data;
            column_labels = [header],
            formatters = [PrettyTables.fmt__printf("%.4e", 3:5)],
        )
    end

    # Summary table (reuses cached per_source_pass — no recomputation)
    n_comparisons = length(rms_results_vec)
    n_not = n_comparisons - n_passes

    summary_data = hcat(
        map(basename, sources),
        map(status_emoji, per_source_pass),
    )
    println(io, "\n── Summary ──")
    PrettyTables.pretty_table(
        io,
        summary_data;
        column_labels = [["Source", "Status"]],
    )

    println(io, "   n_comparisons               = $n_comparisons")
    println(io, "   n_times_reproducible        = $n_passes")
    println(io, "   n_times_not_reproducible    = $n_not")
    println(io, "   n_pass_limit                = $n_pass_limit")
    println(io, "   FT                          = $FT")
    println(io, "   eps_factor                  = $eps_factor")
    println(
        io,
        "   test_broken_report_flakiness = $test_broken_report_flakiness",
    )

    # Determine overall result
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
