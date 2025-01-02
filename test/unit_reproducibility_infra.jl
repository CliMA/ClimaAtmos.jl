#=
julia --project=examples
using Revise; include("test/unit_reproducibility_infra.jl")
=#
using Test
import Dates
import Logging
import ClimaUtilities.OutputPathGenerator

# this also includes reproducibility_utils.jl
include(joinpath("..", "reproducibility_tests/reproducibility_tools.jl"))

quiet_latest_comparable_dirs(args...; kwargs...) =
    Logging.with_logger(Logging.NullLogger()) do
        latest_comparable_dirs(args...; kwargs...)
    end
basenames(x) = map(basename, x) # for debugging
function make_dir(dir, dirname)
    d = mkpath(dirname)
    return joinpath(dir, d)
end

function make_and_cd(f)
    mktempdir() do dir
        cd(dir) do
            f(dir)
        end
    end
end

function make_ref_file_counter(i, dir...)
    d = mkpath(joinpath(dir...))
    open(io -> println(io, i), joinpath(d, "ref_counter.jl"), "w")
    return d
end

function make_json(
    i,
    dir...;
    subfolder = "output_active",
    jsonfile = "computed_mse.json",
)
    d = mkpath(joinpath(dir..., subfolder))
    open(io -> println(io, i), joinpath(d, jsonfile), "w")
    return d
end
rbundle(p) = joinpath(p, "reproducibility_bundle")
function mktempdir2_cd_computed(f) # make two temporary dirs
    mktempdir() do save_dir
        mktempdir() do computed_dir
            cd(computed_dir) do
                f((save_dir, computed_dir))
            end
        end
    end
end

function put_data_file(
    dir,
    fv,
    comms_ctx,
    name = "Y";
    filename = "my_prog_state.hdf5",
)
    mkpath(dir)
    file = joinpath(dir, filename)
    hdfwriter = InputOutput.HDF5Writer(file, comms_ctx)
    InputOutput.write!(hdfwriter, fv, name)
    Base.close(hdfwriter)
end

@testset "Reproducibility infrastructure: latest_comparable_dirs" begin
    # No dirs at all
    make_and_cd() do dir
        dirs = quiet_latest_comparable_dirs(;
            root_dir = dir,
            ref_counter_PR = 2,
            skip = false,
        )
        @test dirs == []
    end

    # No dirs with ref counters
    make_and_cd() do dir
        p1 = mkdir("d1")
        dirs = quiet_latest_comparable_dirs(;
            root_dir = dir,
            ref_counter_PR = 2,
            skip = false,
        )
        @test dirs == []
    end

    # No dirs with matching ref counters
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        dirs = quiet_latest_comparable_dirs(;
            root_dir = dir,
            ref_counter_PR = 2,
            skip = false,
        )
        @test dirs == []
    end

    # 1 matching ref counter
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        d2 = make_ref_file_counter(2, dir, "d2")
        d3 = make_ref_file_counter(3, dir, "d3")
        dirs = quiet_latest_comparable_dirs(;
            root_dir = dir,
            ref_counter_PR = 2,
            skip = false,
        )
        @test dirs == [d2]
    end

    # multiple matching ref counters
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        d2 = make_ref_file_counter(2, dir, "d2")
        d3 = make_ref_file_counter(3, dir, "d3")
        d4 = make_ref_file_counter(3, dir, "d4")
        d5 = make_ref_file_counter(3, dir, "d5")
        d6 = make_ref_file_counter(3, dir, "d6")
        dirs = quiet_latest_comparable_dirs(;
            root_dir = dir,
            ref_counter_PR = 3,
            skip = false,
        )
        @test dirs == [d6, d5, d4, d3] # d6 is most recent
    end

    # matching ref counters that exceed n
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        d2 = make_ref_file_counter(2, dir, "d2")
        d3 = make_ref_file_counter(3, dir, "d3")
        d4 = make_ref_file_counter(3, dir, "d4")
        d5 = make_ref_file_counter(3, dir, "d5")
        d6 = make_ref_file_counter(3, dir, "d6")
        dirs = quiet_latest_comparable_dirs(;
            n = 2,
            root_dir = dir,
            ref_counter_PR = 3,
            skip = false,
        )
        @test dirs == [d6, d5] # d6 is most recent
    end

    # folders modified out of chronological order examples
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        d2 = make_ref_file_counter(2, dir, "d2")
        d3 = make_ref_file_counter(3, dir, "d3")
        d4 = make_ref_file_counter(4, dir, "d4")
        d5 = make_ref_file_counter(5, dir, "d5")
        d6 = make_ref_file_counter(3, dir, "d6")
        dirs = quiet_latest_comparable_dirs(;
            n = 2,
            root_dir = dir,
            ref_counter_PR = 3,
            skip = false,
        )
        @test dirs == [d6, d3]
    end

    # appending to p7 now, confusingly, removes p3:
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        d2 = make_ref_file_counter(2, dir, "d2")
        d3 = make_ref_file_counter(3, dir, "d3")
        d4 = make_ref_file_counter(4, dir, "d4")
        d5 = make_ref_file_counter(5, dir, "d5")
        d6 = make_ref_file_counter(3, dir, "d6")
        d7 = make_ref_file_counter(3, dir, "d7")
        dirs = quiet_latest_comparable_dirs(;
            n = 2,
            root_dir = dir,
            ref_counter_PR = 3,
            skip = false,
        )
        @test dirs == [d7, d6]
    end
end

@testset "Reproducibility infrastructure: validate_reference_folders" begin
    # No dirs at all
    make_and_cd() do dir
        @test invalid_reference_folders(
            sorted_dirs_with_matched_files(; dir),
        ) == []
    end

    # 1 dir without ref counter
    make_and_cd() do dir
        p1 = make_dir(dir, "d1")
        @test invalid_reference_folders(
            sorted_dirs_with_matched_files(; dir),
        ) == []
    end

    # mix
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        r1 = make_dir(dir, "r1")
        d2 = make_ref_file_counter(2, dir, "d2")
        r2 = make_dir(dir, "r2")
        d3 = make_ref_file_counter(3, dir, "d3")
        r3 = make_dir(dir, "r3")
        @test invalid_reference_folders(
            sorted_dirs_with_matched_files(; dir),
        ) == []
    end
end

@testset "Reproducibility infrastructure: compute_bins" begin
    # No dirs at all
    make_and_cd() do dir
        @test compute_bins(dir) == []
    end

    # 1 ref counter
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        @test compute_bins(dir) == [[d1]]
    end

    # 2 ref counter
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        d2 = make_ref_file_counter(2, dir, "d2")
        @test compute_bins(dir) == [[d2], [d1]]
    end

    # 4 ref counter
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        d2 = make_ref_file_counter(2, dir, "d2")
        d3 = make_ref_file_counter(3, dir, "d3")
        d4 = make_ref_file_counter(3, dir, "d4")
        d5 = make_ref_file_counter(5, dir, "d5")
        d6 = make_ref_file_counter(5, dir, "d6")
        d7 = make_ref_file_counter(6, dir, "d7")
        @test compute_bins(dir) == [[d7], [d6, d5], [d4, d3], [d2], [d1]]
        @test occursin(
            "(State 1, ref_counter):",
            string_bins(compute_bins(dir)),
        )
    end

    # simulating folders modified out of chronological order
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        d2 = make_ref_file_counter(2, dir, "d2")
        d3 = make_ref_file_counter(3, dir, "d3")
        d4 = make_ref_file_counter(4, dir, "d4")
        d5 = make_ref_file_counter(3, dir, "d5")
        d6 = make_ref_file_counter(4, dir, "d6")
        d7 = make_ref_file_counter(5, dir, "d7")
        @test compute_bins(dir) == [[d7], [d6, d4], [d5, d3], [d2], [d1]]
    end
end

@testset "Reproducibility infrastructure: get_reference_dirs_to_delete" begin
    # No dirs at all
    make_and_cd() do dir
        dirs = get_reference_dirs_to_delete(; root_dir = dir)
        @test dirs == []
    end

    # dirs without ref counters (assume this isn't reproducibility data)
    make_and_cd() do dir
        d1 = mkdir("d1")
        dirs = get_reference_dirs_to_delete(; root_dir = dir)
        @test dirs == []
    end

    # keep everything case
    make_and_cd() do dir
        d1 = make_ref_file_counter(1, dir, "d1")
        dirs = get_reference_dirs_to_delete(; root_dir = dir)
        @test dirs == []
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
    make_and_cd() do dir
        d01 = make_ref_file_counter(1, dir, "01")
        d02 = make_ref_file_counter(2, dir, "02")
        d03 = make_ref_file_counter(2, dir, "03")
        d04 = make_ref_file_counter(2, dir, "04")
        d05 = make_ref_file_counter(3, dir, "05")
        d06 = make_ref_file_counter(4, dir, "06")
        d07 = make_ref_file_counter(4, dir, "07")
        d08 = make_ref_file_counter(5, dir, "08")
        d09 = make_ref_file_counter(6, dir, "09")
        d10 = make_ref_file_counter(6, dir, "10")
        d11 = make_ref_file_counter(7, dir, "11")
        dirs = get_reference_dirs_to_delete(;
            root_dir = dir,
            keep_n_comparable_states = 1,
            keep_n_bins_back = 5,
        )
        @test dirs == reverse([d01, d02, d03, d04, d06, d09])
        dirs = get_reference_dirs_to_delete(;
            root_dir = dir,
            keep_n_comparable_states = 4,
            keep_n_bins_back = 3,
        )
        @test dirs == reverse([d01, d02, d03, d04, d05, d06, d07])
    end
end

function make_file_with_contents(dir, filename, contents)
    mkpath(dir)
    f = joinpath(dir, filename)
    open(io -> println(io, contents), f, "w")
end
@testset "Reproducibility infrastructure: source_checksum" begin
    mktempdir2_cd_computed() do (dir_A, dir_B)
        make_file_with_contents(dir_A, "file_x.jl", "abc")
        make_file_with_contents(dir_A, "file_y.jl", "abc")
        make_file_with_contents(dir_A, "file_z.jl", "abc")

        make_file_with_contents(dir_B, "file_x.jl", "abc")
        make_file_with_contents(dir_B, "file_y.jl", "abc")
        make_file_with_contents(dir_B, "file_z.jl", "abc")
        @test source_checksum(dir_A) == source_checksum(dir_B)
    end

    mktempdir2_cd_computed() do (dir_A, dir_B)
        make_file_with_contents(dir_A, "file_x.jl", "abc")
        make_file_with_contents(dir_A, "file_y.jl", "abc")
        make_file_with_contents(dir_A, "file_z.jl", "abc")

        make_file_with_contents(dir_B, "file_x.jl", "xyz")
        make_file_with_contents(dir_B, "file_y.jl", "abc")
        make_file_with_contents(dir_B, "file_z.jl", "abc")
        @test source_checksum(dir_A) ≠ source_checksum(dir_B)
    end
end

@testset "Reproducibility infrastructure: source_has_changed" begin
    mktempdir2_cd_computed() do (dir_A, dir_B)
        make_file_with_contents(dir_A, "file_x.jl", "abc")
        make_file_with_contents(dir_A, "file_y.jl", "abc")
        make_file_with_contents(dir_A, "file_z.jl", "abc")
        d_A = make_ref_file_counter(3, dir_A, "d_A")
        make_file_with_contents(
            d_A,
            "source_checksum.dat",
            source_checksum(dir_A),
        )

        make_file_with_contents(dir_B, "file_x.jl", "abc")
        make_file_with_contents(dir_B, "file_y.jl", "abc")
        make_file_with_contents(dir_B, "file_z.jl", "abc")
        d_B = make_ref_file_counter(3, dir_B, "d_B")
        make_file_with_contents(
            d_B,
            "source_checksum.dat",
            source_checksum(dir_B),
        )

        @test source_has_changed(;
            n = 0, # force no comparable reference, source code
            root_dir = dir_A,
            ref_counter_PR = 3,
            skip = false,
            src_dir = dir_B,
        )
    end

    mktempdir2_cd_computed() do (dir_A, dir_B)
        make_file_with_contents(dir_A, "file_x.jl", "abc")
        make_file_with_contents(dir_A, "file_y.jl", "abc")
        make_file_with_contents(dir_A, "file_z.jl", "abc")
        d_A = make_ref_file_counter(3, dir_A, "d_A")
        make_file_with_contents(
            d_A,
            "source_checksum.dat",
            source_checksum(dir_A),
        )

        make_file_with_contents(dir_B, "file_x.jl", "abc")
        make_file_with_contents(dir_B, "file_y.jl", "abc")
        make_file_with_contents(dir_B, "file_z.jl", "abc")
        d_B = make_ref_file_counter(3, dir_B, "d_B")
        make_file_with_contents(
            d_B,
            "source_checksum.dat",
            source_checksum(dir_B),
        )

        @test !source_has_changed(;
            n = 5,
            root_dir = dir_A,
            ref_counter_PR = 3,
            skip = false,
            src_dir = dir_B,
        )
    end

    mktempdir2_cd_computed() do (dir_A, dir_B)
        make_file_with_contents(dir_A, "file_x.jl", "abc")
        make_file_with_contents(dir_A, "file_y.jl", "abc")
        make_file_with_contents(dir_A, "file_z.jl", "abc")
        d_A = make_ref_file_counter(3, dir_A, "d_A")
        make_file_with_contents(
            d_A,
            "source_checksum.dat",
            source_checksum(dir_A),
        )

        make_file_with_contents(dir_B, "file_x.jl", "abc")
        make_file_with_contents(dir_B, "file_y.jl", "abc")
        make_file_with_contents(dir_B, "file_z.jl", "xyz")
        d_B = make_ref_file_counter(3, dir_B, "d_B")
        make_file_with_contents(
            d_B,
            "source_checksum.dat",
            source_checksum(dir_B),
        )

        @test source_has_changed(;
            n = 5,
            root_dir = dir_A,
            ref_counter_PR = 3,
            skip = false,
            src_dir = dir_B,
        )
    end
end


import OrderedCollections: OrderedDict
import JSON

@testset "Reproducibility infrastructure: report_reproducibility_results - filename" begin
    make_and_cd() do dir
        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 1, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        d1 = make_json(mses1, dir, "d1")
        d2 = make_json(mses2, dir, "d2")
        d3 = make_json(mses3, dir, "d3")
        paths = [d1, d2, d3]
        computed_mse_filenames = map(paths) do p
            joinpath(p, "computed_mse.json")
        end
        io = IOBuffer()
        @test report_reproducibility_results(
            io,
            map(x -> basename(dirname(dirname(x))), computed_mse_filenames),
            map(x -> parse_file(x), computed_mse_filenames);
            n_pass_limit = 2,
            test_broken_report_flakiness = true,
        ) == :now_reproducible
    end
end


@testset "Reproducibility infrastructure: report_reproducibility_results - dict, flaky" begin
    make_and_cd() do dir
        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 1, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        io = IOBuffer()
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 2,
            test_broken_report_flakiness = true,
        ) == :now_reproducible

        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 0, "b" => 1) # only partly passing
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 2,
            test_broken_report_flakiness = true,
        ) == :now_reproducible

        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 1, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 5,
            test_broken_report_flakiness = true,
        ) == :not_yet_reproducible

        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 0, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 2,
            test_broken_report_flakiness = true,
        ) == :now_reproducible

        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 0, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 3,
            test_broken_report_flakiness = true,
        ) == :not_yet_reproducible
    end
end

@testset "Reproducibility infrastructure: report_reproducibility_results - dict, strict" begin
    make_and_cd() do dir
        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 1, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        io = IOBuffer()
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 2,
            test_broken_report_flakiness = false,
        ) == :reproducible

        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 0, "b" => 1) # only partly passing
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 2,
            test_broken_report_flakiness = false,
        ) == :reproducible

        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 1, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 5,
            test_broken_report_flakiness = false,
        ) == :not_reproducible

        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 0, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 2,
            test_broken_report_flakiness = false,
        ) == :reproducible

        mses1 = OrderedDict("a" => 0, "b" => 0)
        mses2 = OrderedDict("a" => 0, "b" => 1)
        mses3 = OrderedDict("a" => 0, "b" => 0)
        mses = [mses1, mses2, mses3]
        @test report_reproducibility_results(
            io,
            ["S1", "S2", "S3"],
            mses;
            n_pass_limit = 3,
            test_broken_report_flakiness = false,
        ) == :not_reproducible
    end
end

@testset "Reproducibility infrastructure: mse summary" begin
    make_and_cd() do dir
        mses1 = OrderedDict("a" => 1, "b" => 1)
        mses2 = OrderedDict("a" => 2, "b" => 2)
        mses3 = OrderedDict("a" => 3, "b" => 3)
        d1 = make_json(mses1, dir, "d1")
        d2 = make_json(mses2, dir, "d2")
        d3 = make_json(mses3, dir, "d3")

        job_ids = ["d1", "d2", "d3"]
        computed_mses = get_computed_mses(;
            job_ids,
            subfolder = "output_active",
            is_mse_file = default_is_mse_file,
        )
        io = IOBuffer()
        print_mse_summary(io; mses = computed_mses)
        s = String(take!(io))
        @test s == "################################# Computed MSEs
MSEs[\"d1\"][a] = 1
MSEs[\"d1\"][b] = 1
MSEs[\"d2\"][a] = 2
MSEs[\"d2\"][b] = 2
MSEs[\"d3\"][a] = 3
MSEs[\"d3\"][b] = 3
#################################
"

        any_skipped = print_skipped_jobs(; mses = computed_mses)
        @test !any_skipped
    end
end

@testset "Reproducibility infrastructure: mse summary" begin
    make_and_cd() do dir
        mses1 = OrderedDict("a" => 1, "b" => 1)
        mses2 = OrderedDict("a" => 2, "b" => 2)
        mses3 = OrderedDict("a" => 3, "b" => 3)
        d1 = make_json(mses1, dir, "d1")
        d2 = make_json(mses2, dir, "d2"; jsonfile = "comuted_mse.json") # intentional typo
        d3 = make_json(mses3, dir, "d3")

        job_ids = ["d1", "d2", "d3"]
        computed_mses = get_computed_mses(;
            job_ids,
            subfolder = "output_active",
            is_mse_file = default_is_mse_file,
        )
        io = IOBuffer()
        print_mse_summary(io; mses = computed_mses)
        s = String(take!(io))
        @test s == "################################# Computed MSEs
MSEs[\"d1\"][a] = 1
MSEs[\"d1\"][b] = 1
MSEs[\"d3\"][a] = 3
MSEs[\"d3\"][b] = 3
#################################
"

        io = IOBuffer()
        any_skipped = print_skipped_jobs(io; mses = computed_mses)
        s = String(take!(io))
        @test any_skipped
        @test s == "Skipped files:
     job_id:`d1`, file:`OrderedDict(\"a\" => 1, \"b\" => 1)`
     job_id:`d3`, file:`OrderedDict(\"a\" => 3, \"b\" => 3)`
" || s == "Skipped files:
     job_id:`d1`, file:`OrderedCollections.OrderedDict(\"a\" => 1, \"b\" => 1)`
     job_id:`d3`, file:`OrderedCollections.OrderedDict(\"a\" => 3, \"b\" => 3)`
"
    end
end

@testset "all_files_in_dir with generate_output_path" begin
    make_and_cd() do dir
        # Tests that symlink directories are not
        # returned in all_files_in_dir
        output_dir = OutputPathGenerator.generate_output_path(dir)
        @test all_files_in_dir(dir) == String[]
    end
end

@testset "Reproducibility infrastructure: save_dir_transform" begin
    make_and_cd() do dir
        job_id = "job_id"
        commit = "commit_sha"
        n_hash_characters = 10
        output = "output_active"
        strip_folder = output
        repro_folder = "rbundle"
        src = joinpath("$job_id", "$output", "$repro_folder", "prog_state.hdf5")
        dst = joinpath("$commit", "$repro_folder", "$job_id", "prog_state.hdf5")
        @test save_dir_transform(
            src;
            dest_root = dir,
            job_id,
            commit,
            n_hash_characters,
            repro_folder,
            strip_folder,
        ) == joinpath(dir, dst)

        job_id = "job_id"
        commit = "commit_sha"
        n_hash_characters = 10
        output = "output_active"
        strip_folder = output
        repro_folder = "rbundle"
        src = joinpath("$job_id", "$output", "prog_state.hdf5")
        dst = joinpath("$commit", "$repro_folder", "$job_id", "prog_state.hdf5")
        @test save_dir_transform(
            src;
            dest_root = dir,
            job_id,
            commit,
            n_hash_characters,
            repro_folder,
            strip_folder,
        ) == joinpath(dir, dst)
    end
end

@testset "Reproducibility infrastructure: strip_output_active_path" begin
    @test strip_output_active_path(joinpath("a", "b", "c")) ==
          joinpath("a", "b", "c")
    @test strip_output_active_path(joinpath("a", "output_active", "c")) ==
          joinpath("a", "c")
    @test strip_output_active_path(joinpath("a", "output_1234", "c")) ==
          joinpath("a", "c")
    @test strip_output_active_path(joinpath("a", "output_1A34", "c")) ==
          joinpath("a", "output_1A34", "c")
end

@testset "Reproducibility infrastructure: save_dir_in_out_list" begin
    mktempdir2_cd_computed() do (save_dir, computed_dir)
        hash1 = joinpath(save_dir, "hash1")
        hash2 = joinpath(save_dir, "hash2")
        make_file_with_contents(hash1, "file_x.jl", "abc")
        make_file_with_contents(hash1, "file_y.jl", "abc")
        make_file_with_contents(hash1, "file_z.jl", "abc")
        make_ref_file_counter(3, hash1, "repro_bundle")

        make_file_with_contents(hash2, "file_x.jl", "abc")
        make_file_with_contents(hash2, "file_y.jl", "abc")
        make_file_with_contents(hash2, "file_z.jl", "abc")
        make_ref_file_counter(3, hash2, "repro_bundle")

        make_file_with_contents(computed_dir, "file_x.jl", "abc")
        make_file_with_contents(computed_dir, "file_y.jl", "abc")
        make_file_with_contents(computed_dir, "file_z.jl", "abc")
        ref_counter_file_dir =
            make_ref_file_counter(3, computed_dir, "repro_bundle")
        job_id_1 = joinpath(computed_dir, "repro_bundle", "job_id_1")
        job_id_2 = joinpath(computed_dir, "repro_bundle", "job_id_2")

        mkpath(joinpath(job_id_1, "output_active"))
        file = joinpath(job_id_1, "output_active", "ref_prog_state.dat")
        open(io -> println(io, 1), file, "w")

        mkpath(joinpath(job_id_2, "output_active"))
        file = joinpath(job_id_2, "output_active", "ref_prog_state.dat")
        open(io -> println(io, 1), file, "w")

        @test source_checksum(hash1) == source_checksum(computed_dir)
        @test source_checksum(hash2) == source_checksum(computed_dir)

        repro_folder = "repro_bundle"
        (; files_src, files_dest) = save_dir_in_out_list(;
            dirs_src = [job_id_1, job_id_2],
            dest_root = save_dir,
            commit = "commit_sha",
            n_hash_characters = 10,
            repro_folder,
            strip_folder = "output_active",
        )

        @test files_src[1] == joinpath(
            computed_dir,
            "repro_bundle",
            "job_id_1",
            "output_active",
            "ref_prog_state.dat",
        )
        @test files_src[2] == joinpath(
            computed_dir,
            "repro_bundle",
            "job_id_2",
            "output_active",
            "ref_prog_state.dat",
        )
        @test files_dest[1] == joinpath(
            save_dir,
            "commit_sha",
            "repro_bundle",
            "job_id_1",
            "ref_prog_state.dat",
        )
        @test files_dest[1] == joinpath(
            save_dir,
            "commit_sha",
            "repro_bundle",
            "job_id_1",
            "ref_prog_state.dat",
        )

    end
end

@testset "Reproducibility infrastructure: move_data_to_save_dir legacy folder structure" begin
    mktempdir2_cd_computed() do (save_dir, computed_dir)
        hash1 = joinpath(save_dir, "hash1")
        hash2 = joinpath(save_dir, "hash2")
        make_file_with_contents(hash1, "file_x.jl", "abc")
        make_file_with_contents(hash1, "file_y.jl", "abc")
        make_file_with_contents(hash1, "file_z.jl", "abc")
        make_ref_file_counter(3, hash1)

        make_file_with_contents(hash2, "file_x.jl", "abc")
        make_file_with_contents(hash2, "file_y.jl", "abc")
        make_file_with_contents(hash2, "file_z.jl", "abc")
        make_ref_file_counter(3, hash2)

        make_file_with_contents(computed_dir, "file_x.jl", "abc")
        make_file_with_contents(computed_dir, "file_y.jl", "abc")
        make_file_with_contents(computed_dir, "file_z.jl", "abc")
        ref_counter_file_dir = make_ref_file_counter(3, computed_dir)
        job_id_1 = joinpath(computed_dir, "job_id_1")
        job_id_2 = joinpath(computed_dir, "job_id_2")

        mkpath(joinpath(job_id_1, "output_active"))
        file = joinpath(job_id_1, "output_active", "ref_prog_state.dat")
        open(io -> println(io, 1), file, "w")

        mkpath(joinpath(job_id_2, "output_active"))
        file = joinpath(job_id_2, "output_active", "ref_prog_state.dat")
        open(io -> println(io, 1), file, "w")

        @test source_checksum(hash1) == source_checksum(computed_dir)
        @test source_checksum(hash2) == source_checksum(computed_dir)

        repro_folder = "repro_bundle"
        repro_dir = joinpath(save_dir, "hash_new", repro_folder)
        move_data_to_save_dir(;
            dest_root = save_dir,
            buildkite_ci = true,
            commit = "hash_new",
            n_hash_characters = length("hash_new"),
            branch = "unit_test_move_data_to_save_dir",
            in_merge_queue = true,
            dirs_src = [job_id_1, job_id_2],
            ref_counter_file_PR = joinpath(
                ref_counter_file_dir,
                "ref_counter.jl",
            ),
            ref_counter_PR = 3,
            repro_folder,
            skip = false,
        )
        @test isfile(joinpath(repro_dir, "job_id_1", "ref_prog_state.dat"))
        @test isfile(joinpath(repro_dir, "job_id_2", "ref_prog_state.dat"))
        @test isfile(joinpath(repro_dir, "ref_counter.jl"))
    end
end

@testset "Reproducibility infrastructure: move_data_to_save_dir" begin
    mktempdir2_cd_computed() do (save_dir, computed_dir)
        hash1 = joinpath(save_dir, "hash1")
        hash2 = joinpath(save_dir, "hash2")
        make_file_with_contents(hash1, "file_x.jl", "abc")
        make_file_with_contents(hash1, "file_y.jl", "abc")
        make_file_with_contents(hash1, "file_z.jl", "abc")
        make_ref_file_counter(3, hash1, "repro_bundle")

        make_file_with_contents(hash2, "file_x.jl", "abc")
        make_file_with_contents(hash2, "file_y.jl", "abc")
        make_file_with_contents(hash2, "file_z.jl", "abc")
        make_ref_file_counter(3, hash2, "repro_bundle")

        make_file_with_contents(computed_dir, "file_x.jl", "abc")
        make_file_with_contents(computed_dir, "file_y.jl", "abc")
        make_file_with_contents(computed_dir, "file_z.jl", "abc")
        ref_counter_file_dir =
            make_ref_file_counter(3, computed_dir, "repro_bundle")
        job_id_1 = joinpath(computed_dir, "repro_bundle", "job_id_1")
        job_id_2 = joinpath(computed_dir, "repro_bundle", "job_id_2")


        mkpath(joinpath(job_id_1, "output_active"))
        file = joinpath(job_id_1, "output_active", "ref_prog_state.dat")
        open(io -> println(io, 1), file, "w")

        mkpath(joinpath(job_id_2, "output_active"))
        file = joinpath(job_id_2, "output_active", "ref_prog_state.dat")
        open(io -> println(io, 1), file, "w")

        @test source_checksum(hash1) == source_checksum(computed_dir)
        @test source_checksum(hash2) == source_checksum(computed_dir)

        repro_folder = "repro_bundle"
        move_data_to_save_dir(;
            strip_folder = "output_active",
            dest_root = save_dir,
            buildkite_ci = true,
            commit = "hash_new",
            n_hash_characters = length("hash_new"),
            branch = "unit_test_move_data_to_save_dir",
            in_merge_queue = true,
            dirs_src = [job_id_1, job_id_2],
            ref_counter_file_PR = joinpath(
                ref_counter_file_dir,
                "ref_counter.jl",
            ),
            repro_folder,
            ref_counter_PR = 3,
            skip = false,
        )
        repro_dir = joinpath(save_dir, "hash_new", "repro_bundle")
        @test isfile(joinpath(repro_dir, "job_id_1", "ref_prog_state.dat"))
        @test isfile(joinpath(repro_dir, "job_id_2", "ref_prog_state.dat"))
        @test isfile(joinpath(repro_dir, "ref_counter.jl"))
    end
end

@testset "Reproducibility infrastructure: move_data_to_save_dir with symlinks" begin
    mktempdir2_cd_computed() do (save_dir, computed_dir)
        hash1 = joinpath(save_dir, "hash1")
        hash2 = joinpath(save_dir, "hash2")
        make_file_with_contents(hash1, "file_x.jl", "abc")
        make_file_with_contents(hash1, "file_y.jl", "abc")
        make_file_with_contents(hash1, "file_z.jl", "abc")
        make_ref_file_counter(3, hash1, "repro_bundle")

        make_file_with_contents(hash2, "file_x.jl", "abc")
        make_file_with_contents(hash2, "file_y.jl", "abc")
        make_file_with_contents(hash2, "file_z.jl", "abc")
        make_ref_file_counter(3, hash2, "repro_bundle")

        make_file_with_contents(computed_dir, "file_x.jl", "abc")
        make_file_with_contents(computed_dir, "file_y.jl", "abc")
        make_file_with_contents(computed_dir, "file_z.jl", "abc")
        ref_counter_file_dir =
            make_ref_file_counter(3, computed_dir, "repro_bundle")
        job_id_1 = joinpath(computed_dir, "repro_bundle", "job_id_1")
        job_id_2 = joinpath(computed_dir, "repro_bundle", "job_id_2")


        job_id_1_sym_dir = OutputPathGenerator.generate_output_path(job_id_1)
        file = joinpath(job_id_1_sym_dir, "ref_prog_state.dat")
        open(io -> println(io, 1), file, "w")

        job_id_2_sym_dir = OutputPathGenerator.generate_output_path(job_id_2)
        file = joinpath(job_id_2_sym_dir, "ref_prog_state.dat")
        open(io -> println(io, 1), file, "w")

        @test source_checksum(hash1) == source_checksum(computed_dir)
        @test source_checksum(hash2) == source_checksum(computed_dir)

        repro_folder = "repro_bundle"
        move_data_to_save_dir(;
            strip_folder = "output_active",
            dest_root = save_dir,
            buildkite_ci = true,
            commit = "hash_new",
            n_hash_characters = length("hash_new"),
            branch = "unit_test_move_data_to_save_dir",
            in_merge_queue = true,
            dirs_src = [job_id_1, job_id_2],
            ref_counter_file_PR = joinpath(
                ref_counter_file_dir,
                "ref_counter.jl",
            ),
            repro_folder,
            ref_counter_PR = 3,
            skip = false,
        )
        repro_dir = joinpath(save_dir, "hash_new", "repro_bundle")
        @test isfile(joinpath(repro_dir, "job_id_1", "ref_prog_state.dat"))
        @test isfile(joinpath(repro_dir, "job_id_2", "ref_prog_state.dat"))
        @test isfile(joinpath(repro_dir, "ref_counter.jl"))
    end
end

using ClimaComms
using ClimaCore: Spaces, Fields, Grids, InputOutput
using ClimaCore
if pkgversion(ClimaCore) ≥ v"0.14.18"

    using ClimaCore.CommonGrids
    @testset "Reproducibility infrastructure: to_dict" begin
        make_and_cd() do dir
            grid = ExtrudedCubedSphereGrid(;
                z_elem = 10,
                z_min = 0,
                z_max = 1,
                radius = 10,
                h_elem = 10,
                n_quad_points = 4,
            )
            space =
                Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
            comms_ctx = ClimaComms.context(space)

            fv = Fields.FieldVector(; x = ones(space), y = ones(space))
            file = joinpath(dir, "fv.hdf5")
            hdfwriter = InputOutput.HDF5Writer(file, comms_ctx)
            InputOutput.write!(hdfwriter, fv, "fv")
            Base.close(hdfwriter)
            dict = to_dict(file, "fv", comms_ctx)
            @test dict["(:x,)"] isa Vector{Float64}
            @test dict["(:y,)"] isa Vector{Float64}
            zdict = zero_dict(file, "fv", comms_ctx)
            @test zdict["(:x,)"] isa Vector{Float64}
            @test zdict["(:y,)"] isa Vector{Float64}
            @test all(x -> iszero(x), zdict["(:x,)"])
            @test all(x -> iszero(x), zdict["(:y,)"])
        end
    end

    @testset "Reproducibility infrastructure: reproducibility_results - legacy folder structure" begin
        make_and_cd() do dir
            grid = ExtrudedCubedSphereGrid(;
                z_elem = 5,
                z_min = 0,
                z_max = 1,
                radius = 10,
                h_elem = 5,
                n_quad_points = 2,
            )
            space =
                Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
            comms_ctx = ClimaComms.context(space)

            fv = Fields.FieldVector(; x = ones(space), y = ones(space))
            file = joinpath(dir, "my_prog_state.hdf5")
            hdfwriter = InputOutput.HDF5Writer(file, comms_ctx)
            InputOutput.write!(hdfwriter, fv, "Y")
            Base.close(hdfwriter)

            # Not on buildkite
            job_id = "unit_test"
            (d, v, how) = reproducibility_results(
                comms_ctx;
                job_id,
                name = "Y",
                save_dir = dir,
                ref_counter_PR = 1,
                reference_filename = "my_prog_state.hdf5",
                data_file_computed = file,
                skip = true,
            )
            @test length(v) == 1
            @test v[1]["(:x,)"] isa Vector{Float64}
            @test v[1]["(:y,)"] isa Vector{Float64}
            @test all(x -> iszero(x), v[1]["(:x,)"])
            @test all(x -> iszero(x), v[1]["(:y,)"])

            @test isempty(d)
            @test how == :skipped

            # Empty comparable dirs
            job_id = "unit_test"
            (d, v, how) = reproducibility_results(
                comms_ctx;
                job_id,
                name = "Y",
                save_dir = dir,
                reference_filename = "my_prog_state.hdf5",
                ref_counter_PR = 1,
                data_file_computed = file,
                skip = false,
            )
            @test length(v) == 1
            @test v[1]["(:x,)"] isa Vector{Float64}
            @test v[1]["(:y,)"] isa Vector{Float64}
            @test all(x -> iszero(x), v[1]["(:x,)"])
            @test all(x -> iszero(x), v[1]["(:y,)"])

            @test isempty(d)
            @test how == :no_comparable_dirs

            # Successful comparison

            d01 = make_ref_file_counter(1, dir, "01")
            d02 = make_ref_file_counter(2, dir, "02")
            d03 = make_ref_file_counter(3, dir, "03")
            d04 = make_ref_file_counter(3, dir, "04")
            d05 = make_ref_file_counter(3, dir, "05")

            put_data_file(d01, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d02, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d03, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d04, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            fv.x .= 200
            fv.y .= 300
            put_data_file(d05, fv, comms_ctx; filename = "ref_prog_state.hdf5")

            job_id = "unit_test"
            (d, v, how) = reproducibility_results(
                comms_ctx;
                job_id,
                name = "Y",
                save_dir = dir,
                ref_counter_PR = 3,
                reference_filename = "ref_prog_state.hdf5",
                data_file_computed = file,
                skip = false,
            )
            # The first we compare against is most recent,
            # And we set `fv.x .= 200` and `fv.y .= 300` for
            # that dataset.
            @test v[1]["(:x,)"] == 2970.075
            @test v[1]["(:y,)"] == 2980.0333333333333
            @test v[2]["(:x,)"] == 0.0
            @test v[2]["(:y,)"] == 0.0
            @test v[3]["(:x,)"] == 0.0
            @test v[3]["(:y,)"] == 0.0

            @test d == [d05, d04, d03]
            @test how == :successful_comparison
        end
    end

    @testset "Reproducibility infrastructure: reproducibility_results" begin
        mktempdir2_cd_computed() do (save_dir, computed_dir)
            grid = ExtrudedCubedSphereGrid(;
                z_elem = 5,
                z_min = 0,
                z_max = 1,
                radius = 10,
                h_elem = 5,
                n_quad_points = 2,
            )
            space =
                Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
            comms_ctx = ClimaComms.context(space)

            fv = Fields.FieldVector(; x = ones(space), y = ones(space))
            file = joinpath(
                computed_dir,
                "reproducibility_bundle",
                "computed_prog_state.hdf5",
            )
            mkpath(dirname(file))
            hdfwriter = InputOutput.HDF5Writer(file, comms_ctx)
            InputOutput.write!(hdfwriter, fv, "Y")
            Base.close(hdfwriter)

            # Not on buildkite
            job_id = "unit_test"
            (d, v, how) = reproducibility_results(
                comms_ctx;
                job_id,
                name = "Y",
                save_dir = save_dir,
                ref_counter_PR = 1,
                reference_filename = "ref_prog_state.hdf5",
                data_file_computed = file,
                skip = true,
            )
            @test length(v) == 1
            @test v[1]["(:x,)"] isa Vector{Float64}
            @test v[1]["(:y,)"] isa Vector{Float64}
            @test all(x -> iszero(x), v[1]["(:x,)"])
            @test all(x -> iszero(x), v[1]["(:y,)"])

            @test isempty(d)
            @test how == :skipped

            # Empty comparable dirs
            job_id = "unit_test"
            (d, v, how) = reproducibility_results(
                comms_ctx;
                job_id,
                name = "Y",
                save_dir = save_dir,
                reference_filename = "ref_prog_state.hdf5",
                ref_counter_PR = 1,
                data_file_computed = file,
                skip = false,
            )
            @test length(v) == 1
            @test v[1]["(:x,)"] isa Vector{Float64}
            @test v[1]["(:y,)"] isa Vector{Float64}
            @test all(x -> iszero(x), v[1]["(:x,)"])
            @test all(x -> iszero(x), v[1]["(:y,)"])

            @test isempty(d)
            @test how == :no_comparable_dirs

            # Successful comparison

            d01 = make_ref_file_counter(1, save_dir, rbundle("01"))
            d02 = make_ref_file_counter(2, save_dir, rbundle("02"))
            d03 = make_ref_file_counter(3, save_dir, rbundle("03"))
            d04 = make_ref_file_counter(3, save_dir, rbundle("04"))
            d05 = make_ref_file_counter(3, save_dir, rbundle("05"))

            put_data_file(d01, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d02, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d03, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d04, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            fv.x .= 200
            fv.y .= 300
            put_data_file(d05, fv, comms_ctx; filename = "ref_prog_state.hdf5")

            job_id = "unit_test"
            (d, v, how) = reproducibility_results(
                comms_ctx;
                job_id,
                name = "Y",
                save_dir = save_dir,
                ref_counter_PR = 3,
                reference_filename = "ref_prog_state.hdf5",
                data_file_computed = file,
                skip = false,
            )
            # The first we compare against is most recent,
            # And we set `fv.x .= 200` and `fv.y .= 300` for
            # that dataset.
            @test v[1]["(:x,)"] == 2970.075
            @test v[1]["(:y,)"] == 2980.0333333333333
            @test v[2]["(:x,)"] == 0.0
            @test v[2]["(:y,)"] == 0.0
            @test v[3]["(:x,)"] == 0.0
            @test v[3]["(:y,)"] == 0.0

            @test d == [d05, d04, d03]
            @test how == :successful_comparison
        end
    end

    @testset "Reproducibility infrastructure: export_reproducibility_results, legacy folder structure" begin
        mktempdir2_cd_computed() do (save_dir, computed_dir)
            grid = ExtrudedCubedSphereGrid(;
                z_elem = 5,
                z_min = 0,
                z_max = 1,
                radius = 10,
                h_elem = 5,
                n_quad_points = 2,
            )
            space =
                Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
            comms_ctx = ClimaComms.context(space)

            job_id = "unit_test_export_reproducibility_results"

            fv = Fields.FieldVector(; x = ones(space), y = ones(space))

            # Test skipped case
            (data_file_computed, computed_mses, dirs, how) =
                export_reproducibility_results(
                    fv,
                    comms_ctx;
                    job_id,
                    save_dir = save_dir,
                    computed_dir = computed_dir,
                    name = "Y",
                    n = 10,
                    ref_counter_PR = 1,
                    skip = true,
                )
            @test how == :skipped
            @test isempty(dirs)

            # Test no comparable dirs
            (data_file_computed, computed_mses, dirs, how) =
                export_reproducibility_results(
                    fv,
                    comms_ctx;
                    job_id,
                    save_dir = save_dir,
                    computed_dir = computed_dir,
                    name = "Y",
                    n = 10,
                    ref_counter_PR = 1,
                    skip = false,
                )
            @test how == :no_comparable_dirs
            @test isempty(dirs)

            # Successful comparisons, legacy path configuration
            d01 = make_ref_file_counter(1, save_dir, "01")
            d02 = make_ref_file_counter(2, save_dir, "02")
            d03 = make_ref_file_counter(3, save_dir, "03")
            d04 = make_ref_file_counter(3, save_dir, "04")
            d05 = make_ref_file_counter(3, save_dir, "05")

            put_data_file(d01, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d02, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d03, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d04, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            fv.x .= 200
            fv.y .= 300
            put_data_file(d05, fv, comms_ctx; filename = "ref_prog_state.hdf5")

            @test isfile(joinpath(d01, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d02, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d03, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d04, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d05, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d01, "ref_counter.jl"))
            @test isfile(joinpath(d02, "ref_counter.jl"))
            @test isfile(joinpath(d03, "ref_counter.jl"))
            @test isfile(joinpath(d04, "ref_counter.jl"))
            @test isfile(joinpath(d05, "ref_counter.jl"))

            (data_file_computed, computed_mses, dirs, how) =
                export_reproducibility_results(
                    fv,
                    comms_ctx;
                    job_id,
                    save_dir = save_dir,
                    computed_dir = computed_dir,
                    name = "Y",
                    reference_filename = "ref_prog_state.hdf5",
                    computed_filename = "computed_prog_state.hdf5",
                    n = 10,
                    ref_counter_PR = 3,
                    skip = false,
                )
            @test how == :successful_comparison
            @test dirs == [d05, d04, d03]
        end
    end

    @testset "Reproducibility infrastructure: export_reproducibility_results" begin
        mktempdir2_cd_computed() do (save_dir, computed_dir)
            grid = ExtrudedCubedSphereGrid(;
                z_elem = 5,
                z_min = 0,
                z_max = 1,
                radius = 10,
                h_elem = 5,
                n_quad_points = 2,
            )
            space =
                Spaces.ExtrudedFiniteDifferenceSpace(grid, Grids.CellCenter())
            comms_ctx = ClimaComms.context(space)

            job_id = "unit_test_export_reproducibility_results"

            fv = Fields.FieldVector(; x = ones(space), y = ones(space))

            # Test skipped case
            (data_file_computed, computed_mses, dirs, how) =
                export_reproducibility_results(
                    fv,
                    comms_ctx;
                    job_id,
                    save_dir = save_dir,
                    computed_dir = computed_dir,
                    reference_filename = "ref_prog_state.hdf5",
                    computed_filename = "computed_prog_state.hdf5",
                    name = "Y",
                    n = 10,
                    ref_counter_PR = 1,
                    skip = true,
                )
            @test how == :skipped
            @test isempty(dirs)

            # Test no comparable dirs
            (data_file_computed, computed_mses, dirs, how) =
                export_reproducibility_results(
                    fv,
                    comms_ctx;
                    job_id,
                    save_dir = save_dir,
                    computed_dir = computed_dir,
                    reference_filename = "ref_prog_state.hdf5",
                    computed_filename = "computed_prog_state.hdf5",
                    name = "Y",
                    n = 10,
                    ref_counter_PR = 1,
                    skip = false,
                )
            @test how == :no_comparable_dirs
            @test isempty(dirs)

            # Successful comparisons, legacy path configuration
            d01 = make_ref_file_counter(1, save_dir, rbundle("CH01"))
            d02 = make_ref_file_counter(2, save_dir, rbundle("CH02"))
            d03 = make_ref_file_counter(3, save_dir, rbundle("CH03"))
            d04 = make_ref_file_counter(3, save_dir, rbundle("CH04"))
            d05 = make_ref_file_counter(3, save_dir, rbundle("CH05"))

            put_data_file(d01, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d02, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d03, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            put_data_file(d04, fv, comms_ctx; filename = "ref_prog_state.hdf5")
            fv.x .= 200
            fv.y .= 300
            put_data_file(d05, fv, comms_ctx; filename = "ref_prog_state.hdf5")

            @test isfile(joinpath(d01, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d02, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d03, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d04, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d05, "ref_prog_state.hdf5"))
            @test isfile(joinpath(d01, "ref_counter.jl"))
            @test isfile(joinpath(d02, "ref_counter.jl"))
            @test isfile(joinpath(d03, "ref_counter.jl"))
            @test isfile(joinpath(d04, "ref_counter.jl"))
            @test isfile(joinpath(d05, "ref_counter.jl"))

            (data_file_computed, computed_mses, dirs, how) =
                export_reproducibility_results(
                    fv,
                    comms_ctx;
                    job_id,
                    save_dir = save_dir,
                    computed_dir = computed_dir,
                    name = "Y",
                    reference_filename = "ref_prog_state.hdf5",
                    computed_filename = "computed_prog_state.hdf5",
                    n = 10,
                    ref_counter_PR = 3,
                    skip = false,
                )
            @test how == :successful_comparison
            @test dirs == [d05, d04, d03]
            repro_dir = joinpath(computed_dir, "reproducibility_bundle")
            @test isfile(joinpath(repro_dir, "computed_prog_state.hdf5"))
            @test isfile(joinpath(repro_dir, "computed_mse_CH05.json"))
            @test isfile(joinpath(repro_dir, "computed_mse_CH04.json"))
            @test isfile(joinpath(repro_dir, "computed_mse_CH03.json"))
        end
    end
end
