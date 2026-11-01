"""
    FlameDiff

Diff two [ProfileCanvas](https://github.com/pfitzseb/ProfileCanvas.jl) flame
graphs saved as HTML files (for example, the artifacts a CI flame-graph job
uploads, or files written locally with `ProfileCanvas.html_file`).

Self-sample counts are directly comparable between two profiles of the same
wall-clock duration; total counts and fractions are not (the total shifts).
Deltas below about 5 samples are noise.

# As a script

    julia flame_diff.jl BASELINE.html CANDIDATE.html [TOP_N]

prints the total sample counts for both files followed by the `TOP_N` (default
25) frames with the largest self-sample increases and decreases (candidate
minus baseline), keyed by `function@file:line`.

# From Julia (e.g. alongside `test_compilation.jl`)

    include("flame_diff.jl")
    using .FlameDiff
    rows = flame_diff("baseline.html", "candidate.html")   # prints and returns
    _, self_counts, total_counts = aggregate_flame(load_flame("candidate.html"))

Produce the input files with ProfileCanvas:

    import Profile, ProfileCanvas
    Profile.@profile <workload>
    ProfileCanvas.html_file("flame.html")
"""
module FlameDiff

import ProfileCanvas: JSON  # JSON is a dependency of ProfileCanvas

export load_flame, aggregate_flame, flame_diff

# ─── Loading and aggregating ─────────────────────────────────────────────────

"""
    load_flame(path) -> Dict

Extract and parse the profile tree that ProfileCanvas embeds in an HTML file.
The result maps the string `"1"` to the root node; each node is a `Dict` with
keys `"func"`, `"file"`, `"line"`, `"count"`, and `"children"`.
"""
function load_flame(path)
    text = read(path, String)
    marker = findfirst("new ProfileCanvas.ProfileViewer(", text)
    isnothing(marker) && error("$path: not a ProfileCanvas HTML file")
    range = findnext(", {", text, last(marker))
    isnothing(range) && error("$path: could not find embedded profile data")
    # last(range) indexes the '{' of ", {"; slice out the balanced object that
    # follows it (braces only ever appear inside strings that JSON escapes, so
    # a plain depth count over the bytes is safe) and hand it to JSON.
    bytes = codeunits(text)
    depth = 0
    start = last(range)
    for stop in start:lastindex(bytes)
        bytes[stop] == UInt8('{') && (depth += 1)
        bytes[stop] == UInt8('}') && (depth -= 1) == 0 &&
            return JSON.parse(text[start:stop])
    end
    error("$path: unbalanced braces in embedded profile data")
end

"""
    aggregate_flame(tree) -> (root_count, self_counts, total_counts)

Sum self and total sample counts per `function@file:line` frame over the tree
returned by [`load_flame`](@ref). `self_counts[frame]` is a node's own count
minus its children's; `total_counts[frame]` counts each frame once per root-to-
node path that first reaches it, so recursive frames are not double-counted.
"""
function aggregate_flame(tree)
    root = tree["1"]
    self_counts = Dict{String, Int}()
    total_counts = Dict{String, Int}()
    function visit(node, seen)
        key = string(node["func"], "@", node["file"], ":", node["line"])
        children = node["children"]
        child_sum = isempty(children) ? 0 : sum(child -> child["count"], children)
        self_counts[key] = get(self_counts, key, 0) + node["count"] - child_sum
        if !(key in seen)
            total_counts[key] = get(total_counts, key, 0) + node["count"]
        end
        deeper = push!(copy(seen), key)
        for child in children
            visit(child, deeper)
        end
    end
    visit(root, Set{String}())
    return root["count"], self_counts, total_counts
end

# ─── Diffing and reporting ───────────────────────────────────────────────────

const HEADER = string(
    rpad("frame", 70),
    " ",
    lpad("cand", 6),
    " ",
    lpad("base", 6),
    " ",
    lpad("ctot", 6),
    " ",
    lpad("btot", 6),
)

function print_rows(io, rows)
    println(io, HEADER)
    for row in rows
        println(
            io,
            rpad(first(row.frame, 70), 70),
            " ",
            lpad(row.cand, 6),
            " ",
            lpad(row.base, 6),
            " ",
            lpad(row.cand_total, 6),
            " ",
            lpad(row.base_total, 6),
        )
    end
end

"""
    flame_diff(baseline_path, candidate_path; top_n = 25, io = stdout)

Print the total sample counts for both flame graphs and the `top_n` frames
with the largest self-sample increases and decreases (candidate minus
baseline). Return the full list of per-frame rows (`NamedTuple`s with fields
`frame`, `cand`, `base`, `cand_total`, `base_total`, `delta`) sorted by `delta`
descending, for programmatic use.
"""
function flame_diff(baseline_path, candidate_path; top_n = 25, io = stdout)
    base_root, base_self, base_total = aggregate_flame(load_flame(baseline_path))
    cand_root, cand_self, cand_total = aggregate_flame(load_flame(candidate_path))
    println(io, "baseline root samples:  $base_root  ($baseline_path)")
    println(io, "candidate root samples: $cand_root  ($candidate_path)")
    base_root > 0 &&
        println(io, "ratio: ", round(cand_root / base_root; digits = 2))

    frames = union(keys(base_self), keys(cand_self))
    rows = map(collect(frames)) do frame
        cand = get(cand_self, frame, 0)
        base = get(base_self, frame, 0)
        (;
            frame,
            cand,
            base,
            cand_total = get(cand_total, frame, 0),
            base_total = get(base_total, frame, 0),
            delta = cand - base,
        )
    end
    sort!(rows; by = row -> row.delta, rev = true)

    println(io, "\n=== top $top_n self-sample increases (candidate - baseline) ===")
    print_rows(io, Iterators.filter(row -> row.delta > 0, first(rows, top_n)))
    # Decreases: the last top_n rows (most negative) shown most-negative first.
    println(io, "\n=== top $top_n self-sample decreases ===")
    print_rows(io, Iterators.filter(row -> row.delta < 0, reverse(last(rows, top_n))))
    return rows
end

function main(args)
    if length(args) < 2
        println(stderr, "usage: julia flame_diff.jl BASELINE.html CANDIDATE.html [TOP_N]")
        return
    end
    top_n = length(args) >= 3 ? parse(Int, args[3]) : 25
    flame_diff(args[1], args[2]; top_n)
    return
end

end # module FlameDiff

if abspath(PROGRAM_FILE) == @__FILE__
    FlameDiff.main(ARGS)
end
