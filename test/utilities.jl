using Test
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
