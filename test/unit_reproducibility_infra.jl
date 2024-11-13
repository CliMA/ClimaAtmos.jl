#=
using Revise; include("test/unit_reproducibility_infra.jl")
=#
using Test
using Dates

include(joinpath("..", "reproducibility_tests/latest_comparable_paths.jl"))

function make_ref_file_counter(dir, pathname, i)
    d = mkdir(pathname)
    open(io -> println(io, i), joinpath(d, "ref_counter.jl"), "w")
    return joinpath(dir, d)
end

@testset "Reproducibility infrastructure: latest_comparable_paths" begin
    # No paths at all
    mktempdir() do path
        cd(path) do
            paths =
                latest_comparable_paths(; root_path = path, ref_counter_PR = 2)
            @test paths == []
        end
    end

    # No paths with ref counters
    mktempdir() do path
        cd(path) do
            p1 = mkdir("d1")
            paths =
                latest_comparable_paths(; root_path = path, ref_counter_PR = 2)
            @test paths == []
        end
    end

    # No paths with matching ref counters
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            paths =
                latest_comparable_paths(; root_path = path, ref_counter_PR = 2)
            @test paths == []
        end
    end

    # 1 matching ref counter
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            p3 = make_ref_file_counter(path, "d3", 3)
            paths =
                latest_comparable_paths(; root_path = path, ref_counter_PR = 2)
            @test paths == [p2]
        end
    end

    # multiple matching ref counters
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            p3 = make_ref_file_counter(path, "d3", 3)
            p4 = make_ref_file_counter(path, "d4", 3)
            p5 = make_ref_file_counter(path, "d5", 3)
            p6 = make_ref_file_counter(path, "d6", 3)
            paths =
                latest_comparable_paths(; root_path = path, ref_counter_PR = 3)
            @test paths == [p6, p5, p4, p3] # p6 is most recent
        end
    end

    # matching ref counters that exceed n
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            p3 = make_ref_file_counter(path, "d3", 3)
            p4 = make_ref_file_counter(path, "d4", 3)
            p5 = make_ref_file_counter(path, "d5", 3)
            p6 = make_ref_file_counter(path, "d6", 3)
            paths = latest_comparable_paths(;
                n = 2,
                root_path = path,
                ref_counter_PR = 3,
            )
            @test paths == [p6, p5] # p6 is most recent
        end
    end
end
