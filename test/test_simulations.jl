@testset "Simulations" begin
    @info "Testing ClimaAtmos.Simulations..."
    for FT in float_types
        @testset "Bickley jet 2D simulation" begin
            include("test_cases/run_bickley_jet_2d_plane.jl")
            run_bickley_jet_2d_plane(FT, mode = :unit)
        end
        @testset "Dry rising bubble 2D simulation" begin
            include("test_cases/run_dry_rising_bubble_2d.jl")
            run_dry_rising_bubble_2d(FT, mode = :unit)
        end
    end
end
