include("test_cases/run_bickley_jet_2d_plane.jl")
include("test_cases/run_ekman_column_1d.jl")
include("test_cases/run_dry_rising_bubble_2d.jl")

function test_simulations(mode)
    @testset "Bickley jet 2D plane" begin
        for FT in float_types
            run_bickley_jet_2d_plane(FT; mode)
        end
    end

    @testset "Ekman column 1D" begin
        for FT in float_types
            run_ekman_column_1d(FT; mode)
        end
    end

    @testset "Dry rising bubble 2D" begin
        for FT in float_types
            run_dry_rising_bubble_2d(FT; mode)
        end
    end
end
