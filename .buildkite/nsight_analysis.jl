# nsys stats --report cuda_gpu_trace report.nsys-rep --output . --format csv

@info "Starting nsight analysis"

using VegaLite, UnicodePlots, CSV, DataFrames, ArgParse

function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table! s begin
        "--out_dir"
        help = "Output data directory"
        arg_type = String
    end
    return ArgParse.parse_args(ARGS, s)
end

function get_params()
    parsed_args = parse_commandline()
    return parsed_args["out_dir"]
end

output_dir = get_params()

@time "Load CSV file" begin
    if !@isdefined(data_and_init)
        data_and_init = cd(output_dir) do
            CSV.read("_cuda_gpu_trace.csv", DataFrame)
        end
    end
end

"""
	filter_out_initialization(data;
		keep_n_minimum_kernels = 1000,
		gap_percent_threshold = 10
	)

We do not want to include initialization kernels in our analysis,
since they are not representative of our runtime performance. Therefore,

We iterate using a heuristic to filter out initialization:
 - from the start to halfway, find the largest gap between kernel calls, and filter
   out from the start to that point.
 - If the next gap is within some percentage (`gap_percent_threshold`), terminate
 - If trimming results in fewer than `keep_n_minimum_kernels`, terminate
"""
function filter_out_initialization(
    data;
    keep_n_minimum_kernels = 1000,
    gap_percent_threshold = 10,
)
    t_start = data[1, "Start (ns)"]
    t_end = data[end, "Start (ns)"]

    # filter until maximum kernel duration is in the
    # distribution of the remaining kernels:
    halfway(x) = Int(round(length(x[!, "Name"]) / 2))
    continue_trimming = true
    max_gaps = Int[]
    function maximum_gap(data)
        R = 1:halfway(data)
        (max_gap, i_max) = findmax(identity, diff(data[R, "Start (ns)"]))
        i_next_start = i_max + 1
        return (max_gap, i_next_start)
    end
    i_iter = 0
    (next_max_gap, i_next_start) = maximum_gap(data)
    exit_reason = 0
    while continue_trimming
        @info "Trimming initialization data. Iteration $i_iter"
        push!(max_gaps, next_max_gap)
        # i_longest_remaining_kernel = findfirst(x -> x == max_gaps[end], data[1:halfway(data), "Duration (ns)"])
        new_data = data[i_next_start:end, :]
        if length(new_data[!, "Name"]) < keep_n_minimum_kernels
            exit_reason = "trimming more kernels results in fewer than $keep_n_minimum_kernels kernels left"
            @warn "New data length would have been too short: $(length(new_data[!, "Name"]))"
            continue_trimming = false
        else
            data = new_data
            # If the kernel we're filtering out now is within some
            # percentage (gap_percent_threshold) of the largest
            # one that remains, then stop filtering
            (next_max_gap, i_next_start) = maximum_gap(data)
            if (max_gaps[end] - next_max_gap) / max_gaps[end] * 100 ≤
               gap_percent_threshold
                continue_trimming = false
                exit_reason = "next gap between kernels is similar to previously filtered one"
            end
        end
        i_iter += 1
        i_iter > 10^6 && error("Too many iterations")
    end

    # Now, let's trim the end by 10%
    N = length(data[!, "Name"])
    N_end = Int(round(N * 0.9))
    data = data[1:N_end, :]
    t_start_new = data[1, "Start (ns)"]
    @info "Original start time (s)      : $(t_start / 10^9)"
    @info "New start time (s)           : $(t_start_new / 10^9)"
    @info "Fraction of simulation trimmed: $((t_start_new-t_start)/(t_end-t_start))"
    @info "exit_reason                   : $(exit_reason)"

    return data
end

@time "Filter CSV" begin
    data = filter_out_initialization(data_and_init)
end

const logged_uncaught_cases = String[]

function group_name(s)
    transform_name = Dict()
    transform_name["copyto_per_field_"] = "fieldvector"
    transform_name["knl_copyto_"] = "copyto"
    transform_name["copyto_stencil_kernel"] = "stencil"
    transform_name["CUDA memcpy"] = "CUDA memcpy"
    transform_name["knl_fill_"] = "fill"
    transform_name["CUDA memset"] = "CUDA memset"
    transform_name["CuKernelContext"] = "CuKernelContext"
    transform_name["knl_fused_copyto"] = "fused_copyto"
    transform_name["knl_fused_copyto_linear"] = "fused_copyto_linear"
    transform_name["multiple_field_solve_kernel_"] = "multiple_field_solve"
    transform_name["single_field_solve_kernel"] = "single_field_solve_kernel"
    transform_name["copyto_spectral_kernel_"] = "spectral"
    transform_name["bycolumn_kernel"] = "bycolumn_reduce"
    transform_name["dss_load_perimeter_data_kernel"] = "dss_load"
    transform_name["dss_unload_perimeter_data_kernel"] = "dss_unload"
    transform_name["dss_local_kernel"] = "dss_local"
    transform_name["dss_transform_kernel"] = "dss_transform"
    transform_name["dss_untransform_kernel"] = "dss_untransform"
    transform_name["dss_local_ghost_kernel"] = "dss_local_ghost"
    transform_name["fill_send_buffer_kernel"] = "dss_fill_send_buffer"
    transform_name["load_from_recv_buffer_kernel"] = "dss_load_from_recv"
    transform_name["dss_ghost_kernel"] = "dss_ghost"
    transform_name["rte_sw_2stream_solve"] = "RRTMGP_RTE_sw"
    transform_name["rte_lw_2stream_solve"] = "RRTMGP_RTE_lw"
    transform_name["compute_col_gas_CUDA"] = "RRTMGP_col_gas"
    transform_name["set_interpolated_values_kernel"] = "remapping"
    if s in values(transform_name)
        return s # already grouped
    else
        for k in keys(transform_name)
            occursin(k, s) && return transform_name[k]
        end
    end
    if !(s in logged_uncaught_cases)
        @warn "Uncaught case for $s"
        push!(logged_uncaught_cases, s)
    end
    return "Unknown"
end

function vega_pie_chart(data)
    data[:, "Name"] .= group_name.(data[:, "Name"])
    sort!(data, order("Duration (ns)", by = identity))

    data_duration = DataFrame(
        duration = data[!, "Duration (ns)"] / 10^3,
        name = data[!, "Name"],
    )
    data_duration |>
    @vlplot(
        :arc,
        theta = :duration,
        color = "name:n",
        view = {stroke = nothing}
    ) |>
    save("pie_chart.png")
end

function sorted_barplot(x₀, y₀; title)
    x = deepcopy(x₀)
    y = deepcopy(y₀)
    perm = sortperm(y)
    permute!(x, perm)
    permute!(y, perm)
    bp = UnicodePlots.barplot(x, y; title)
    println(bp)
end


function unicode_barchart(data)
    data[:, "Name"] .= group_name.(data[:, "Name"])
    names₀ = collect(Set(data[!, "Name"]))
    duration_sum = sum(data[!, "Duration (ns)"])
    bar_data = Float64[]
    average_kernel_cost = Float64[]
    n_kernels = Int[]
    for name in names₀
        df_name = filter(row -> group_name(row.Name) == name, data; view = true)
        nk = length(df_name[!, "Duration (ns)"])
        s = sum(df_name[!, "Duration (ns)"])
        push!(bar_data, s / duration_sum * 100)
        push!(average_kernel_cost, s / nk / 10^3)
        push!(n_kernels, nk)
    end
    N = length(data[:, "Name"])
    @info "Statistics across $N total kernels"

    sorted_barplot(names₀, bar_data; title = "Kernel duration percentage")
    sorted_barplot(names₀, n_kernels; title = "Number of kernels")
    sorted_barplot(
        names₀,
        average_kernel_cost;
        title = "Average kernel duration (μs)",
    )

    for name in names₀
        df_name = filter(row -> group_name(row.Name) == name, data; view = true)
        duration_ms = df_name[!, "Duration (ns)"] ./ 10^9 .* 10^3
        h = UnicodePlots.histogram(
            duration_ms;
            title = "$name duration distribution (ms)",
        )
        println(h)
    end
end

@time "Make unicode bar chart" unicode_barchart(data)
# @time "Make vega pie chart" vega_pie_chart(data)
