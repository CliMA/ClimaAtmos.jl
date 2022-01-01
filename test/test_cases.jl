include("test_cases/run_ekman_column_1d.jl")
include("test_cases/run_dry_rising_bubble_2d.jl")

function test_cases(mode)
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
