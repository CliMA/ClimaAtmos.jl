using Test
using LinearAlgebra
using ClimaAtmos: thomas_algorithm!

@testset "Thomas algorithm tests for Float32 and Float64" begin
    FT = Float64
    for FT in (Float32, Float64)
#! format: off
        dl = FT.([0.0, 1825.3503, 645.0884, 329.23483, 201.36221, 136.82605, 99.648796, 76.03936, 59.650623, 0.0])
        d =  FT.([-1.0, -5813.132, -1572.7207, -718.2993, -409.75415, -262.26147, -180.3288, -130.9273, -99.37381, -77.86787, -1.0])
        du = FT.([0.0, 924.0845, 386.9652, 207.68958, 125.552765, 80.648224, 53.839355, 37.18377, 26.495516, 0.0])
        b = FT.([0.0, -3.813298, -19.938122, -51.053738, -99.55792, -201.72447, -450.88504, -944.41315, -1691.7865, -2629.067, 0.0])
#! format: off
        A = Tridiagonal(dl, d, du)
        Ac = deepcopy(A)
        bc = deepcopy(b)
        thomas_algorithm!(Ac, bc)
        ldiv!(lu!(A), b)
        relative_err = norm(b .- bc) / norm(b)
        @test relative_err ≤ sqrt(eps(FT))
    end
end
