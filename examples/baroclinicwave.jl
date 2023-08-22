using ClimaAtmos

# set global parameters
set_params(DefaultParameters())

# parameters are queried by default constructors
space = CubedSphereSpace()
# analogous to Oceananigans Grid?
# at the moment, just returns a ClimaCore Space
# but could be a more complicated object?

# at the moment ClimaAtmos doesn't contain the state, unlike Oceananigans
# we could change this to include state?
model = AtmosModel(space,
    moisture = Dry(),
)

# set initial conditions (and construct state if need be)
state = atmos_state(DryBaroclinicWave(), model, space)

# set up time steppper and callbacks
simulation = Simulation(
    model,
    state;
    dt = 580,
    time_end = 10days,
    callbacks=(
        Checkpoint("sphere_baroclinic_wave_rhoe", 2days),
        Diagnostics(...)
    )
)

run!(simulation)
