# Julia ecosystem
using Test
using UnPack: @unpack
using JLD2
using OrdinaryDiffEq: SSPRK33
using DiffEqBase
using Plots

# Clima ecosystem
using ClimaAtmos
using ClimaAtmos.BoundaryConditions: NoFluxCondition, DragLawCondition
using ClimaAtmos.Domains: Plane, PeriodicPlane, Column, HybridPlane
using ClimaAtmos.Models.ShallowWaterModels: ShallowWaterModel
using ClimaAtmos.Models.SingleColumnModels: SingleColumnModel
using ClimaAtmos.Simulations: Simulation, set!, step!, run!
using ClimaAtmos.Callbacks
using ClimaCore
using ClimaCore: Geometry, Fields

float_types = (Float32, Float64)

#####
##### Run tests!
#####

group = get(ENV, "TEST_GROUP", :all) |> Symbol

@testset "ClimaAtmos" begin
    if group == :unit || group == :all
        @testset "Unit tests" begin
            include("test_domains.jl")
            include("test_simulations.jl")
            include("test_callbacks.jl")
        end
    end

    if group == :regression || group == :all
        @info "Regression tests..."
        @testset "Regression" begin
            include("test_regression.jl")
            include("test_callbacks.jl")
        end
    end

    if group == :validation || group == :all
        @info "Validation tests..."
        @testset "Validation" begin
            include("test_validation.jl")
        end
    end
end

nothing
