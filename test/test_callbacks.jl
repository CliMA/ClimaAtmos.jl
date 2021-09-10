# Generate simple model using shallow water equations for bickley jet problem
include("test_cases/initial_conditions/bickley_jet_2d_plane.jl");
FT = Float64;
npolynomial = 4;
nelements = (2, 2);
params = map(FT, (
    g = 9.8,  # gravitational constant
    D₄ = 1e-4,  # hyperdiffusion constant
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5,  # Gaussian width
    k = 0.5,  # sinusoidal wavenumber
    h₀ = 1.0,  # reference density
));

@unpack h, u, c = init_bickley_jet_2d_plane(params);
domain = PeriodicPlane(
    FT,
    xlim = (-2π, 2π),
    ylim = (-2π, 2π),
    nelements = nelements,
    npolynomial = npolynomial,
);
model = ShallowWaterModel(domain = domain, parameters = params);

@info "Testing ClimaAtmos.Callbacks..."
# Begin Tests

# Populate Callback Containers
filepath = joinpath(@__DIR__, "output_validation")
cb_1 = JLD2Output(model, filepath, "TestFilename1", 1);
cb_2 = JLD2Output(model, filepath, "TestFilename2", 2);

# Generate CallbackSet 
cb_set =
    DiffEqBase.CallbackSet(generate_callback(cb_1), generate_callback(cb_2))

# Type Checks
@test generate_callback(cb_1) isa DiffEqBase.DiscreteCallback
@test generate_callback(cb_2) isa DiffEqBase.DiscreteCallback

# Generate simple simulation data for test

simulation = Simulation(
    model,
    SSPRK33(),
    dt = 0.01,
    tspan = (0.0, 0.02),
    callbacks = cb_set,
)
run!(simulation)

@test isfile(joinpath(cb_1.filedir, cb_1.filename * ".jld2")) == true
@test isfile(joinpath(cb_2.filedir, cb_2.filename * ".jld2")) == true

rm(joinpath(cb_1.filedir, cb_1.filename * ".jld2"))
rm(joinpath(cb_2.filedir, cb_2.filename * ".jld2"))
