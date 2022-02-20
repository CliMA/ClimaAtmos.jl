if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end
using Test

using OrdinaryDiffEq: SSPRK33
using ClimaCorePlots, Plots
using UnPack

using ClimaCoreVTK
using ClimaCore: Geometry
using CLIMAParameters
using ClimaAtmos.Utils.InitialConditions: init_3d_rising_bubble
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models
using ClimaAtmos.Models.Nonhydrostatic3DModels
using ClimaAtmos.Simulations

# Set up parameters
struct Bubble3DParameters <: CLIMAParameters.AbstractEarthParameterSet end
CLIMAParameters.Planet.Omega(::Bubble3DParameters) = 0.0 # Bubble isn't rotating

function run_3d_rising_bubble(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (5, 5, 10),
    npolynomial = 5,
    dt = 0.02,
    callbacks = (),
    test_mode = :regression,
) where {FT}

    params = Bubble3DParameters()

    domain = HybridBox(
        FT,
        xlim = (-5e2, 5e2),
        ylim = (-5e2, 5e2),
        zlim = (0.0, 1e3),
        nelements = nelements,
        npolynomial = npolynomial,
    )

    boundary_conditions = (;
        base = (;
            ρ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
            uh = (top = NoFluxCondition(), bottom = NoFluxCondition()),
            w = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        ),
        thermodynamics = (;
            ρe_tot = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        ),
        moisture = (;
            ρq_tot = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        ),
    )
    model = Nonhydrostatic3DModel(
        domain = domain,
        boundary_conditions = boundary_conditions,
        parameters = params,
        moisture = EquilibriumMoisture(),
        hyperdiffusivity = FT(100),
    )

    boundary_conditions = (;
        base = (;
            ρ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
            uh = (top = NoFluxCondition(), bottom = NoFluxCondition()),
            w = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        ),
        thermodynamics = (;
            ρθ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        ),
    )
    model_pottemp = Nonhydrostatic3DModel(
        domain = domain,
        boundary_conditions = boundary_conditions,
        thermodynamics = PotentialTemperature(),
        parameters = params,
        hyperdiffusivity = FT(100),
    )

    # execute differently depending on testing mode
    if test_mode == :regression
        # Compares variables against current reference results. 
        # Compares conservation properties of prognostic variables. 
        # TODO: Upon reasonable time-to-solution increase test time to 700s.
        @testset "Regression: Potential Temperature Model" begin
            simulation =
                Simulation(model_pottemp, stepper, dt = dt, tspan = (0.0, 1.0))
            @test simulation isa Simulation

            @unpack ρ, uh, w, ρθ = init_3d_rising_bubble(
                FT,
                params,
                thermo_style = model_pottemp.thermodynamics,
                moist_style = model_pottemp.moisture,
            )
            set!(simulation, :base, ρ = ρ, uh = uh, w = w)
            set!(simulation, :thermodynamics, ρθ = ρθ)
            u = simulation.integrator.u
            ∫ρ_0 = sum(u.base.ρ)
            ∫ρθ_0 = sum(u.thermodynamics.ρθ)
            step!(simulation)
            u = simulation.integrator.u

            # Current θ
            current_min = 299.9999999523305
            current_max = 300.468563373248
            θ = u.thermodynamics.ρθ ./ u.base.ρ

            @test minimum(parent(u.thermodynamics.ρθ ./ u.base.ρ)) ≈ current_min atol =
                1e-2
            @test maximum(parent(u.thermodynamics.ρθ ./ u.base.ρ)) ≈ current_max atol =
                1e-2
            u_end = simulation.integrator.u
            ∫ρ_e = sum(u_end.base.ρ)
            ∫ρθ_e = sum(u_end.thermodynamics.ρθ)
            Δρ = (∫ρ_e - ∫ρ_0) ./ ∫ρ_0 * 100
            Δρθ = (∫ρθ_e - ∫ρθ_0) ./ ∫ρθ_0 * 100
            @test abs(Δρ) < 1e-12
            @test abs(Δρθ) < 1e-5
        end
        @testset "Regression: Total Energy Model" begin
            simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
            @unpack ρ, uh, w, ρe_tot, ρq_tot = init_3d_rising_bubble(
                FT,
                params,
                thermo_style = model.thermodynamics,
                moist_style = model.moisture,
            )
            set!(simulation, :base, ρ = ρ, uh = uh, w = w)
            set!(simulation, :thermodynamics, ρe_tot = ρe_tot)
            set!(simulation, :moisture, ρq_tot = ρq_tot)
            u = simulation.integrator.u
            ∫ρ_0 = sum(u.base.ρ)
            ∫ρe_tot_0 = sum(u.thermodynamics.ρe_tot)
            ∫ρq_tot_0 = sum(u.moisture.ρq_tot)

            step!(simulation)

            # Current ρe_tot
            current_min = 237082.14581933746
            current_max = 252441.54599695574

            u = simulation.integrator.u

            @test minimum(parent(u.thermodynamics.ρe_tot)) ≈ current_min atol =
                1e-2
            @test maximum(parent(u.thermodynamics.ρe_tot)) ≈ current_max atol =
                1e-2
            # perform regression check
            u = simulation.integrator.u
            ∫ρ_e = sum(u.base.ρ)
            ∫ρe_tot_e = sum(u.thermodynamics.ρe_tot)
            ∫ρq_tot_e = sum(u.moisture.ρq_tot)
            Δρ = (∫ρ_e - ∫ρ_0) ./ ∫ρ_0 * 100
            Δρe_tot = (∫ρe_tot_e - ∫ρe_tot_0) ./ ∫ρe_tot_0 * 100
            Δρq_tot = (∫ρq_tot_e - ∫ρq_tot_0) ./ ∫ρq_tot_0 * 100
            if FT == Float32
                @test abs(Δρ) < 3e-5
                @test abs(Δρe_tot) < 1e-5
                @test abs(Δρq_tot) < 1e-3
            else
                @test abs(Δρ) < 1e-12
                @test abs(Δρe_tot) < 1e-5
                @test abs(Δρq_tot) < 1e-3
            end
        end
    elseif test_mode == :validation
        # Produces VTK output plots for visual inspection of results
        # Periodic Output related to saveat kwarg issue below.

        # 1. sort out saveat kwarg for Simulation
        @testset "Validation: Potential Temperature Model" begin
            simulation =
                Simulation(model_pottemp, stepper, dt = dt, tspan = (0.0, 1.0))
            @unpack ρ, uh, w, ρθ = init_3d_rising_bubble(
                FT,
                params,
                thermo_style = model_pottemp.thermodynamics,
                moist_style = model_pottemp.moisture,
            )
            set!(simulation, :base, ρ = ρ, uh = uh, w = w)
            set!(simulation, :thermodynamics, ρθ = ρθ)

            # Initial values. Get domain integrated quantity
            u_start = simulation.integrator.u
            ∫ρ_0 = sum(u_start.base.ρ)
            ∫ρθ_0 = sum(u_start.thermodynamics.ρθ)
            run!(simulation)

            u_end = simulation.integrator.u

            ∫ρ_e = sum(u_end.base.ρ)
            ∫ρθ_e = sum(u_end.thermodynamics.ρθ)
            Δρ = (∫ρ_e - ∫ρ_0) ./ ∫ρ_0 * 100
            Δρθ = (∫ρθ_e - ∫ρθ_0) ./ ∫ρθ_0 * 100

            θ = u_end.thermodynamics.ρθ ./ u_end.base.ρ

            # post-processing
            # ENV["GKSwstype"] = "nul"
            # Plots.GRBackend()
            # # make output directory
            # path = joinpath(@__DIR__, "output_validation")
            # mkpath(path)
            # ClimaCoreVTK.writevtk(joinpath(path, "test"), θ)
            @test true # check is visual
        end
        # Total Energy Prognostic
        @testset "Validation: Total Energy Model" begin
            simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
            @unpack ρ, uh, w, ρe_tot, ρq_tot = init_3d_rising_bubble(
                FT,
                params,
                thermo_style = model.thermodynamics,
                moist_style = model.moisture,
            )
            set!(simulation, :base, ρ = ρ, uh = uh, w = w)
            set!(simulation, :thermodynamics, ρe_tot = ρe_tot)
            set!(simulation, :moisture, ρq_tot = ρq_tot)

            # Initial values. Get domain integrated quantity
            u_start = simulation.integrator.u
            ∫ρ_0 = sum(u_start.base.ρ)
            ∫ρetot_0 = sum(u_start.thermodynamics.ρe_tot)
            ∫ρqtot_0 = sum(u_start.moisture.ρq_tot)
            run!(simulation)

            u_end = simulation.integrator.u
            ∫ρ_e = sum(u_end.base.ρ)
            ∫ρetot_e = sum(u_end.thermodynamics.ρe_tot)
            ∫ρqtot_e = sum(u_end.moisture.ρq_tot)
            Δρ = (∫ρ_e - ∫ρ_0) ./ ∫ρ_0 * 100
            Δρetot = (∫ρetot_e - ∫ρetot_0) ./ ∫ρetot_0 * 100
            Δρqtot = (∫ρqtot_e - ∫ρqtot_0) ./ ∫ρqtot_0 * 100
            println("Relative error at end of simulation:")
            println("Δρ = $Δρ %")
            println("Δρe_tot = $Δρetot %")
            println("Δρq_tot = $Δρqtot %")

            e_tot = u_end.thermodynamics.ρe_tot ./ u_end.base.ρ

            # post-processing
            # ENV["GKSwstype"] = "nul"
            # Plots.GRBackend()
            # # make output directory
            # path = joinpath(@__DIR__, "output_validation")
            # mkpath(path)
            # ClimaCoreVTK.writevtk(joinpath(path, "test"), e_tot)
            # #TODO: Additional thermodynamics diagnostic vars
            @test true # check is visual
        end
    else
        throw(ArgumentError("$test_mode incompatible with test case."))
    end

    nothing
end

@testset "3D rising bubble" begin
    for FT in (Float32, Float64)
        run_3d_rising_bubble(FT)
    end
end
