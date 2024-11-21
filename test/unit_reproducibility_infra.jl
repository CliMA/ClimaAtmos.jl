#=
using Revise; include("test/unit_reproducibility_infra.jl")
=#
using Test
import Dates
import Logging

include(joinpath("..", "reproducibility_tests/reproducibility_utils.jl"))

quiet_latest_comparable_paths(args...; kwargs...) =
    Logging.with_logger(Logging.NullLogger()) do
        latest_comparable_paths(args...; kwargs...)
    end
basenames(x) = map(basename, x) # for debugging
function make_path(dir, pathname)
    d = mkdir(pathname)
    return joinpath(dir, d)
end

function make_ref_file_counter(dir, pathname, i)
    d = mkdir(pathname)
    open(io -> println(io, i), joinpath(d, "ref_counter.jl"), "w")
    return joinpath(dir, d)
end

@testset "Reproducibility infrastructure: latest_comparable_paths" begin
    # No paths at all
    mktempdir() do path
        cd(path) do
            paths = quiet_latest_comparable_paths(;
                root_path = path,
                ref_counter_PR = 2,
                skip = false,
            )
            @test paths == []
        end
    end

    # No paths with ref counters
    mktempdir() do path
        cd(path) do
            p1 = mkdir("d1")
            paths = quiet_latest_comparable_paths(;
                root_path = path,
                ref_counter_PR = 2,
                skip = false,
            )
            @test paths == []
        end
    end

    # No paths with matching ref counters
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            paths = quiet_latest_comparable_paths(;
                root_path = path,
                ref_counter_PR = 2,
                skip = false,
            )
            @test paths == []
        end
    end

    # 1 matching ref counter
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            p3 = make_ref_file_counter(path, "d3", 3)
            paths = quiet_latest_comparable_paths(;
                root_path = path,
                ref_counter_PR = 2,
                skip = false,
            )
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
            paths = quiet_latest_comparable_paths(;
                root_path = path,
                ref_counter_PR = 3,
                skip = false,
            )
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
            paths = quiet_latest_comparable_paths(;
                n = 2,
                root_path = path,
                ref_counter_PR = 3,
                skip = false,
            )
            @test paths == [p6, p5] # p6 is most recent
        end
    end

    # reverted commits examples
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            p3 = make_ref_file_counter(path, "d3", 3)
            p4 = make_ref_file_counter(path, "d4", 4)
            p5 = make_ref_file_counter(path, "d5", 5)
            p6 = make_ref_file_counter(path, "d6", 3)
            paths = quiet_latest_comparable_paths(;
                n = 2,
                root_path = path,
                ref_counter_PR = 3,
                skip = false,
            )
            @test paths == [p6]
        end
    end

    # appending to p7 now, confusingly, removes p3:
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            p3 = make_ref_file_counter(path, "d3", 3)
            p4 = make_ref_file_counter(path, "d4", 4)
            p5 = make_ref_file_counter(path, "d5", 5)
            p6 = make_ref_file_counter(path, "d6", 3)
            p7 = make_ref_file_counter(path, "d7", 3)
            paths = quiet_latest_comparable_paths(;
                n = 2,
                root_path = path,
                ref_counter_PR = 3,
                skip = false,
            )
            @test paths == [p7, p6]
        end
    end
end

@testset "Reproducibility infrastructure: validate_reference_folders" begin
    # No paths at all
    mktempdir() do path
        cd(path) do
            @test invalid_reference_folders(; root_path = path) == []
        end
    end

    # 1 path without ref counter
    mktempdir() do path
        cd(path) do
            p1 = make_path(path, "d1")
            @test invalid_reference_folders(; root_path = path) == [p1]
        end
    end

    # mix
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            r1 = make_path(path, "r1")
            p2 = make_ref_file_counter(path, "d2", 2)
            r2 = make_path(path, "r2")
            p3 = make_ref_file_counter(path, "d3", 3)
            r3 = make_path(path, "r3")
            @test invalid_reference_folders(; root_path = path) == [r1, r2, r3]
        end
    end
end

@testset "Reproducibility infrastructure: compute_bins" begin
    # No paths at all
    mktempdir() do path
        cd(path) do
            @test compute_bins(path) == []
        end
    end

    # 1 ref counter
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            @test compute_bins(path) == [[p1]]
        end
    end

    # 2 ref counter
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            @test compute_bins(path) == [[p2], [p1]]
        end
    end

    # 4 ref counter
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            p3 = make_ref_file_counter(path, "d3", 3)
            p4 = make_ref_file_counter(path, "d4", 3)
            p5 = make_ref_file_counter(path, "d5", 5)
            p6 = make_ref_file_counter(path, "d6", 5)
            p7 = make_ref_file_counter(path, "d7", 6)
            @test compute_bins(path) == [[p7], [p6, p5], [p4, p3], [p2], [p1]]
        end
    end

    # simulating reverted PR
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            p2 = make_ref_file_counter(path, "d2", 2)
            p3 = make_ref_file_counter(path, "d3", 3)
            p4 = make_ref_file_counter(path, "d4", 4)
            p5 = make_ref_file_counter(path, "d5", 3)
            p6 = make_ref_file_counter(path, "d6", 4)
            p7 = make_ref_file_counter(path, "d7", 5)
            @test compute_bins(path) ==
                  [[p7], [p6], [p5], [p4], [p3], [p2], [p1]]
        end
    end
end

@testset "Reproducibility infrastructure: get_reference_paths_to_delete" begin
    # No paths at all
    mktempdir() do path
        cd(path) do
            paths = get_reference_paths_to_delete(; root_path = path)
            @test paths == []
        end
    end

    # Paths without ref counters (error)
    mktempdir() do path
        cd(path) do
            p1 = mkdir("d1")
            @test_throws AssertionError get_reference_paths_to_delete(;
                root_path = path,
            )
        end
    end

    # keep everything case
    mktempdir() do path
        cd(path) do
            p1 = make_ref_file_counter(path, "d1", 1)
            paths = get_reference_paths_to_delete(; root_path = path)
            @test paths == []
        end
    end

    #=
    # typical example, consider:

    keep_n_comparable_states
             |    <---- keep_n_bins_back    | oldest
             |                              |
             |  B01 B02 B03 B04 B05 B06 B07 |
             |                              |
             |  p01 p02 p05 p06 p08 p09 p11 |
             |      p03     p07     p10     |
             v      p04                     v newest
    =#
    mktempdir() do path
        cd(path) do
            p01 = make_ref_file_counter(path, "01", 1)
            p02 = make_ref_file_counter(path, "02", 2)
            p03 = make_ref_file_counter(path, "03", 2)
            p04 = make_ref_file_counter(path, "04", 2)
            p05 = make_ref_file_counter(path, "05", 3)
            p06 = make_ref_file_counter(path, "06", 4)
            p07 = make_ref_file_counter(path, "07", 4)
            p08 = make_ref_file_counter(path, "08", 5)
            p09 = make_ref_file_counter(path, "09", 6)
            p10 = make_ref_file_counter(path, "10", 6)
            p11 = make_ref_file_counter(path, "11", 7)
            paths = get_reference_paths_to_delete(;
                root_path = path,
                keep_n_comparable_states = 1,
                keep_n_bins_back = 5,
            )
            @test paths == reverse([p01, p02, p03, p04, p06, p09])
            paths = get_reference_paths_to_delete(;
                root_path = path,
                keep_n_comparable_states = 4,
                keep_n_bins_back = 3,
            )
            @test paths == reverse([p01, p02, p03, p04, p05, p06, p07])
        end
    end

    #=
    # Reverted commits example, consider:

    keep_n_comparable_states
             |    <---- keep_n_bins_back    | oldest
             |                              |
             |  B01 B02 B03 B01 B02 B03 B04 |
             |                              |
             |  p01 p02 p05 p06 p08 p09 p11 |
             |      p03     p07     p10     |
             v      p04                     v newest
    =#

    mktempdir() do path
        cd(path) do
            p01 = make_ref_file_counter(path, "01", 1)
            p02 = make_ref_file_counter(path, "02", 2)
            p03 = make_ref_file_counter(path, "03", 2)
            p04 = make_ref_file_counter(path, "04", 2)
            p05 = make_ref_file_counter(path, "05", 3)
            p06 = make_ref_file_counter(path, "06", 1)
            p07 = make_ref_file_counter(path, "07", 1)
            p08 = make_ref_file_counter(path, "08", 2)
            p09 = make_ref_file_counter(path, "09", 3)
            p10 = make_ref_file_counter(path, "10", 3)
            p11 = make_ref_file_counter(path, "11", 4)
            paths = get_reference_paths_to_delete(;
                root_path = path,
                keep_n_comparable_states = 1,
                keep_n_bins_back = 5,
            )
            @test paths == reverse([p01, p02, p03, p04, p06, p09])
            paths = get_reference_paths_to_delete(;
                root_path = path,
                keep_n_comparable_states = 4,
                keep_n_bins_back = 3,
            )
            @test paths == reverse([p01, p02, p03, p04, p05, p06, p07])
        end
    end
end
