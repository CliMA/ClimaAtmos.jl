# Julia ecosystem
using OrdinaryDiffEq: SSPRK33

# Clima ecosystem
using ClimaAtmos
using ClimaAtmos.Domains: PeriodicPlane
using ClimaAtmos.ShallowWaterModels: ShallowWaterModel
using ClimaAtmos.Simulations: Simulation, step!, run!
using ClimaCore: Geometry

@testset "Simulations" begin
    @info "Testing ClimaAtmos.Simulations..."
    for FT in float_types
        @testset "Bickley jet 2D simulation" begin
            include("test_cases/run_bickley_jet_2d_plane.jl")
            run_bickley_jet_2d_plane(FT, mode = :unit)
        end
    end
end
