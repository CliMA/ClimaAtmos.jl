# Generate simple model using shallow water equations for bickley jet problem
include("test_cases/initial_conditions/ekman_column_1d.jl");
FT = Float64;
nelements = 30;
params = (
    MSLP = FT(1e5), # mean sea level pressure
    grav = FT(9.8), # gravitational constant
    R_d = FT(287.058), # R dry (gas constant / mol mass dry air)
    C_p = FT(287.058 * 7 / 2), # heat capacity at constant pressure
    C_v = FT(287.058 * 5 / 2), # heat capacity at constant volume
    R_m = FT(287.058), # moist R, assumed to be dry
    f = FT(5e-5), # Coriolis parameters
    ν = FT(0.01),
    Cd = FT(0.01 / (2e2 / 30.0)),
    Ch = FT(0.01 / (2e2 / 30.0)),
    uvg = Geometry.UVVector(FT(1.0), FT(0.0)),
    T_surf = FT(300.0),
    T_min_ref = FT(230.0),
    u0 = FT(1.0),
    v0 = FT(0.0),
    w0 = FT(0.0),
);

# Domain
domain = Column(FT, zlim = (0.0, 2e2), nelements = nelements)

# Bcs
coefficients = (Cd = params.Cd, Ch = params.Ch)
boundary_conditions = (
    ρ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
    uv = (top = nothing, bottom = DragLawCondition(coefficients)),
    w = (top = NoFluxCondition(), bottom = NoFluxCondition()),
    ρθ = (
        top = NoFluxCondition(),
        bottom = BulkFormulaCondition(coefficients, params.T_surf),
    ),
)

# Model
model = SingleColumnModel(
    domain = domain,
    boundary_conditions = boundary_conditions,
    parameters = params,
)

# Populate Callback Containers
temp_filepath = joinpath(@__DIR__, "callback_tests")
mkpath(temp_filepath)
cb_1 = JLD2Output(model, temp_filepath, "TestFilename1", 0.01);
cb_2 = JLD2Output(model, temp_filepath, "TestFilename2", 0.02);

# Generate CallbackSet 
cb_set = CallbackSet(generate_callback(cb_1), generate_callback(cb_2))

# Type Checks
@test generate_callback(cb_1) isa DiscreteCallback
@test generate_callback(cb_2) isa DiscreteCallback

# Generate simple simulation data for test
simulation = Simulation(
    model,
    SSPRK33(),
    dt = 0.01,
    tspan = (0.0, 0.03),
    callbacks = cb_set,
)
@unpack ρ, uv, w, ρθ = init_ekman_column_1d(params)
set!(simulation, :scm, ρ = ρ, uv = uv, w = w, ρθ = ρθ)
run!(simulation)

# Test simulation restart
simulation = Simulation(
    model,
    SSPRK33(),
    dt = 0.01,
    tspan = (0.0, 0.03),
    callbacks = cb_set,
    restart = Restart(
        restartfile = joinpath(cb_1.filedir, cb_1.filename * "_0.02.jld2"),
        end_time = 0.05,
    ),
)
set!(simulation, :scm, ρ = ρ, uv = uv, w = w, ρθ = ρθ)
run!(simulation)
@test simulation.integrator.t == 0.05

# Delete test output files
@test isfile(joinpath(cb_1.filedir, cb_1.filename * "_0.01" * ".jld2")) == true
@test isfile(joinpath(cb_2.filedir, cb_2.filename * "_0.02" * ".jld2")) == true

rm(temp_filepath, recursive = true)
