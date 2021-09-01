# Julia ecosystem
using OrdinaryDiffEq: SSPRK33

# Clima ecosystem
using ClimaAtmos
using ClimaAtmos.BoundaryConditions: NoFluxCondition, DragLawCondition
using ClimaAtmos.Domains: PeriodicPlane, Column
using ClimaAtmos.ShallowWaterModels: ShallowWaterModel
using ClimaAtmos.SingleColumnModels: SingleColumnModel
using ClimaAtmos.Simulations: Simulation, step!, run!
using ClimaCore: Geometry

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
