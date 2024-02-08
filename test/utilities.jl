using Test
using Random
Random.seed!(1234)
import ClimaAtmos as CA

@testset "sort_hdf5_files" begin
    day_sec(t) =
        (floor(Int, t / (60 * 60 * 24)), floor(Int, t % (60 * 60 * 24)))
    filenames(d, s) = "day$d.$s.hdf5"
    filenames(t) = filenames(day_sec(t)...)
    t = map(i -> rand(1:(10^6)), 1:100)
    t_sorted = sort(t)
    fns = filenames.(t)
    sort!(fns)
    @test CA.sort_files_by_time(fns) == filenames.(t_sorted)
end

@testset "gaussian_smooth" begin

    # No smooth on constant
    @test CA.gaussian_smooth(3.0 * ones(132, 157)) ≈ 3.0 * ones(132, 157)

    randy = rand(123, 145)

    smoothed = CA.gaussian_smooth(randy)

    # min
    @test extrema(randy)[1] <= smoothed[1]

    # max
    @test extrema(randy)[2] >= smoothed[2]
end
