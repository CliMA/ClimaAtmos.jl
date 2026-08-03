# Unit tests for flame_diff.jl. Run these to completion before trusting any
# flame_diff output (see SKILL.md). They use synthetic ProfileCanvas-style
# HTML fixtures, so no profiling is required.

using Test

include(joinpath(@__DIR__, "flame_diff.jl"))
using .FlameDiff

# Build a minimal ProfileCanvas-style HTML file around a JSON profile tree.
function fake_flame_file(tree_json)
    path = tempname() * ".html"
    write(
        path,
        """
        <html><body><div id="profiler-container-1"></div><script>
        const viewer = new ProfileCanvas.ProfileViewer("#profiler-container-1", $tree_json, "Profile");
        </script></body></html>
        """,
    )
    return path
end

node(func, file, line, count, children = "[]") = """
    {"func":"$func","file":"$file","path":"/x/$file","line":$line,
     "count":$count,"countLabel":null,"flags":0,"children":$children}"""

# root(10) -> work!(6) -> helper(2); root(10) -> ∫apply!(3)
const BASELINE_TREE = """{"1": $(node("root", "task.jl", 1, 10, "[" *
    node("work!", "a.jl", 5, 6, "[" * node("helper", "b.jl", 9, 2) * "]") *
    "," * node("∫apply!", "c.jl", 3, 3) * "]"))}"""

# Same shape, but work! got slower and recursive: work! -> work! -> helper.
const CANDIDATE_TREE = """{"1": $(node("root", "task.jl", 1, 20, "[" *
    node("work!", "a.jl", 5, 16, "[" *
        node("work!", "a.jl", 5, 8, "[" * node("helper", "b.jl", 9, 2) * "]") *
    "]") * "," * node("∫apply!", "c.jl", 3, 3) * "]"))}"""

@testset "FlameDiff" begin
    baseline_path = fake_flame_file(BASELINE_TREE)
    candidate_path = fake_flame_file(CANDIDATE_TREE)

    @testset "load_flame extracts the embedded tree" begin
        tree = load_flame(baseline_path)
        @test tree["1"]["func"] == "root"
        @test tree["1"]["count"] == 10
        @test length(tree["1"]["children"]) == 2
        @test_throws ErrorException load_flame(@__FILE__) # not ProfileCanvas
    end

    @testset "aggregate_flame computes self and total counts" begin
        root_count, self_counts, total_counts = aggregate_flame(load_flame(baseline_path))
        @test root_count == 10
        @test self_counts["root@task.jl:1"] == 10 - 6 - 3
        @test self_counts["work!@a.jl:5"] == 6 - 2
        @test self_counts["helper@b.jl:9"] == 2
        @test self_counts["∫apply!@c.jl:3"] == 3 # UTF-8 frame names survive
        @test total_counts["work!@a.jl:5"] == 6
    end

    @testset "recursive frames are not double-counted in totals" begin
        _, self_counts, total_counts = aggregate_flame(load_flame(candidate_path))
        @test total_counts["work!@a.jl:5"] == 16 # not 16 + 8
        @test self_counts["work!@a.jl:5"] == (16 - 8) + (8 - 2)
    end

    @testset "flame_diff ranks self-sample deltas" begin
        io = IOBuffer()
        rows = flame_diff(baseline_path, candidate_path; top_n = 3, io)
        output = String(take!(io))
        @test occursin("baseline root samples:  10", output)
        @test occursin("candidate root samples: 20", output)
        @test occursin("ratio: 2.0", output)
        @test issorted(rows; by = row -> row.delta, rev = true)
        biggest = rows[1]
        @test biggest.frame == "work!@a.jl:5"
        @test biggest.delta == 14 - 4
        @test biggest.base_total == 6 && biggest.cand_total == 16
        @test rows[end].delta <= 0 # unchanged frames (∫apply!, helper) sort last
    end
end
