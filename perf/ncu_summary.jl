# Temporary (not for merging): sum Nsight Compute metrics per NVTX range for each CSV
# written by `ncu --csv --page raw --print-units base --nvtx`, and compare the first CSV
# with the others.
#   julia perf/ncu_summary.jl <label>=<csv> [<label>=<csv> ...]
using Printf

const METRICS = (
    ("gpu__time_duration.sum", "time (µs)", 1e-3),
    ("smsp__inst_executed_op_global_ld.sum", "global load instructions × 32 (G)", 32e-9),
    ("dram__bytes_read.sum", "DRAM read (GB)", 1e-9),
    ("dram__bytes_write.sum", "DRAM write (GB)", 1e-9),
    ("lts__t_bytes.sum", "L2 traffic (GB)", 1e-9),
)
const TIME = "gpu__time_duration.sum"
const OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"

function csv_fields(line)
    fields = String[]
    buf = IOBuffer()
    quoted = false
    for c in line
        if c == '"'
            quoted = !quoted
        elseif c == ',' && !quoted
            push!(fields, String(take!(buf)))
        else
            write(buf, c)
        end
    end
    push!(fields, String(take!(buf)))
    return fields
end
number(s) = something(tryparse(Float64, replace(s, "," => "")), 0.0)

function load(path)
    lines = readlines(path)
    start = findfirst(l -> startswith(l, "\"ID\""), lines)
    isnothing(start) && error("no CSV header in $path")
    hdr = csv_fields(lines[start])
    col = Dict(h => i for (i, h) in enumerate(hdr))
    nvtx = findall(h -> occursin("Push/Pop", h) || occursin("NVTX", uppercase(h)), hdr)
    missing = [m for m in (first.(METRICS)..., OCC) if !haskey(col, m)]
    println(path, ": NVTX columns ", hdr[nvtx], isempty(missing) ? "" : "; missing $missing")
    order = String[]
    groups = Dict{String, Dict{String, Float64}}()
    for line in lines[(start + 1):end]
        r = csv_fields(line)
        id = strip(r[1])
        (length(r) == length(hdr) && !isempty(id) && all(isdigit, id)) || continue
        # The NVTX stack is "<process id> <domain>:<range>:...:<message>"; keep the message.
        stack = join(filter(!isempty, strip.(r[nvtx])), " / ")
        key = isempty(stack) ? "(no NVTX range)" : String(last(split(stack, ':')))
        g = get!(groups, key) do
            push!(order, key)
            Dict{String, Float64}()
        end
        g["kernels"] = get(g, "kernels", 0.0) + 1
        for (m, _, _) in METRICS
            haskey(col, m) && (g[m] = get(g, m, 0.0) + number(r[col[m]]))
        end
        if haskey(col, OCC) && haskey(col, TIME)
            g["occ_time"] =
                get(g, "occ_time", 0.0) + number(r[col[OCC]]) * number(r[col[TIME]])
        end
    end
    return order, groups
end

function values_of(g)
    t = get(g, TIME, 0.0)
    occ = t > 0 ? get(g, "occ_time", 0.0) / t : 0.0
    return [get(g, "kernels", 0.0); [get(g, m, 0.0) * s for (m, _, s) in METRICS]; occ]
end

const HEADS = ["kernels"; [h for (_, h, _) in METRICS]; "achieved occupancy (%)"]
header() = println("| range | ", join(HEADS, " | "), " |\n", repeat("|---", length(HEADS) + 1), "|")

labels = String[]
tables = []
for arg in ARGS
    label, path = split(arg, "="; limit = 2)
    push!(labels, label)
    push!(tables, load(path))
end
for (label, (order, groups)) in zip(labels, tables)
    println("\n### ", label, "\n")
    header()
    total = Dict{String, Float64}()
    for key in order
        mergewith!(+, total, groups[key])
        println("| ", key, " | ", join((@sprintf("%.4g", v) for v in values_of(groups[key])), " | "), " |")
    end
    println("| **total** | ", join((@sprintf("%.4g", v) for v in values_of(total)), " | "), " |")
end
# Calls per timestep of ARS222 with one Newton iteration (perf/benchmark_components.jl items)
const CALLS = Dict("set_implicit_precomputed_quantities! (total)" => 6)
per_step(groups) = mergewith(+, (Dict(k => v * get(CALLS, key, 2) for (k, v) in g) for (key, g) in groups)...)
println("\n### per timestep (6 calls of set_implicit_precomputed_quantities!, 2 of every other component)\n")
header()
for (label, (_, groups)) in zip(labels, tables)
    println("| ", label, " | ", join((@sprintf("%.4g", v) for v in values_of(per_step(groups))), " | "), " |")
end
base_order, base = tables[1]
for (label, (_, groups)) in zip(labels[2:end], tables[2:end])
    println("\n### ", label, " vs ", labels[1], " (relative change)\n")
    header()
    for key in base_order
        haskey(groups, key) || continue
        a, b = values_of(base[key]), values_of(groups[key])
        cells = [
            i == 1 ? @sprintf("%d vs %d", b[i], a[i]) :
            a[i] == 0 ? "-" : @sprintf("%+.1f%%", 100 * (b[i] / a[i] - 1)) for
            i in eachindex(a)
        ]
        println("| ", key, " | ", join(cells, " | "), " |")
    end
end
