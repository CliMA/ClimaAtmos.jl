include("initial_conditions/dry_rising_bubble_3d.jl")

# Set up parameters
using CLIMAParameters
struct Bubble3DParameters <: CLIMAParameters.AbstractEarthParameterSet end
CLIMAParameters.Planet.Omega(::Bubble3DParameters) = 0.0 # Bubble isn't rotating

function run_dry_rising_bubble_3d(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (10, 10, 50),
    npolynomial = 4,
    dt = 0.02,
    callbacks = (),
    mode = :regression,
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

    model = Nonhydrostatic3DModel(
        domain = domain,
        boundary_conditions = nothing,
        parameters = params,
        hyperdiffusivity = FT(100),
    )

    # execute differently depending on testing mode
    if mode == :integration
        # TODO!: run with input callbacks = ...
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
        @test simulation isa Simulation

        # test set function
        @unpack ρ, uh, w, ρe_tot =
            init_dry_rising_bubble_3d(FT, params, :ρe_tot)
        set!(simulation, :base, ρ = ρ, uh = uh, w = w)
        set!(simulation, :thermodynamics, ρe_tot = ρe_tot)

        # test error handling
        @test_throws ArgumentError set!(simulation, quack = ρ)
        @test_throws ArgumentError set!(simulation, ρ = "quack")

        # test successful integration
        @test step!(simulation) isa Nothing # either error or integration runs
    elseif mode == :regression
        # TODO!: Implement meaningful(!) regression test
    elseif mode == :validation
        # TODO!: Implement the rest plots and analyses
        # 1. sort out saveat kwarg for Simulation
        # 2. create animation for a rising bubble; timeseries of total energy
    else
        throw(ArgumentError("$mode incompatible with test case."))
    end

    nothing
end
