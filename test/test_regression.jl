@testset "Bickley jet 2D plane" begin
    include("test_cases/run_bickley_jet_2d_plane.jl")
    for FT in float_types
        run_bickley_jet_2d_plane(FT, mode = :regression)
    end
end

@testset "Ekman column 1D" begin
    include("test_cases/run_ekman_column_1d.jl")
    for FT in float_types
        run_ekman_column_1d(FT, mode = :regression)
    end
end

@testset "Dry rising bubble 2D" begin
    include("test_cases/run_dry_rising_bubble_2d.jl")
    for FT in float_types
        run_dry_rising_bubble_2d(FT, mode = :regression)
    end
end
