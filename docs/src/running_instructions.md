# Running the Test Suite

We store conventional benchmark simulation examples in the test folder, where they have access to standardized
initial conditions. These benchmarks are used and reused in several tests ranging from unit, over regression, to complex
validation tests. This guarantees that the same code piece can be efficiently reused.

Run all the test cases with:

```
$ julia --project test/runtests.jl
```

## Running Simulations

We create a `Simulation` object which contains information about the `model` configuration (including prognostic, auxiliary variables, initial conditions and parameters). In addition, we supply a time-stepping method (currently from a choice of solvers available within OrdinaryDiffEq.jl), and a list of callbacks (functions which are executed when pre-defined conditions are met, or at user specified time intervals). Simulation data may be stored via one such output callback, with the option to `restart` at an advanced timestep from available output files. Initial conditions are stored in the `Utils` directory. 

Here we "unwrap" one of the conventional benchmarks (3D baroclinic wave) to present an example of a simulation assembled in the Julia REPL. Modules are loaded when necessary.  

- Set up parameters and get the initial conditions
```
julia> using CLIMAParameters
julia> using ClimaAtmos.Utils.InitialConditions: init_3d_baroclinic_wave
julia> struct DryBaroclinicWaveParameters <: CLIMAParameters.AbstractEarthParameterSet end
julia> params = DryBaroclinicWaveParameters();
julia> FT = Float64;
```
- Set up the problem domain, assume we want 6 horizontal of polynomial order 3, and 10 vertical elements. ClimaAtmos is tested with `Float32`, `Float64`; if something doesn't work in `Float32` please raise an issue! 
```
julia> using ClimaAtmos.Domains 
julia> domain = SphericalShell(FT, 
                               radius = CLIMAParameters.Planet.planet_radius(params), 
                               height = FT(30.0e3), 
                               nelements = (6,10),
                               npolynomial = 3)
```
- For details on domain properties, see the `Domains` API documentation. To explore the properties of an existing `domain`, use 
`show(domain)`. With the domain set up, we assemble our `model`, which contains information on discretization, boundary conditions, and problem specific parameters. 

```
julia> using ClimaAtmos.Models.Nonhydrostatic3DModels
julia> using ClimaAtmos.BoundaryConditions
julia> model = Nonhydrostatic3DModel(
        domain = domain,
        boundary_conditions = nothing,
        parameters = params,
        hyperdiffusivity = FT(1e16),
    )
julia> 
```
- For details on model properties, see the `Models` API documentation. To explore the properties of an existing `model`, use `show(model)`. We now select an integration method, SSPRK33 (available in the `OrdinaryDiffEq.jl` package) and construct the `simulation` object as follows:

```
julia> using ClimaAtmos.Simulations
julia> using UnPack
julia> using OrdinaryDiffEq: SSPRK33
julia> simulation = Simulation(model, SSPRK33(), dt = 0.02, tspan = (0.0, 1.0)) # Run for 1.0s in 0.02s intervals
julia> @unpack ρ, uh, w, ρe_tot = init_3d_baroclinic_wave(FT, params); # Unpack from the standard initial condition function
julia> Y = simulation.integrator.u
julia> set!(Y.base.ρ, ρ) # This depends on your model choice!
julia> set!(Y.base.uh, uh) # This depends on your model choice!
julia> set!(Y.base.w, w) # This depends on your model choice!
julia> set!(Y.thermodynamics.ρe_tot, ρe_tot) # This depends on your model choice!
```
- For details on model properties, see the `Simulations` API documentation. Tracers, EDMF variables will have similar assignments using symbol identifiers. `simulation.integrator.u` now contains the initial state. e.g. `simulation.integrator.base.uh` contains horizontal velocities, and so on. `show(simulation)` to view information on an existing `simulation`. To begin time integration, we either `step!` or `run!` the simulation. 

```
julia> step!(simulation) # Advance simulation by one time-step
julia> run!(simulation)  # Advance simulation to completion
```
`simulation.integrator.u` now contains the updated state. For `integrator` object properties, see the OrdinaryDiffEq documentation. 
