push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using LinearAlgebra, IntervalSets, RecursiveArrayTools
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

# set up boilerplate
#=
include("../src/interface/WIP_domains.jl")
include("../src/interface/WIP_models.jl")
include("../src/interface/WIP_timesteppers.jl")
include("../src/backends/backends.jl")
include("../src/backends/climacore/function_spaces.jl")
include("../src/backends/climacore/initial_conditions.jl")
include("../src/backends/climacore/ode_problems.jl")
include("../src/backends/climacore/tendencies.jl")
include("../src/interface/WIP_boundary_conditions.jl")
include("../src/interface/WIP_simulations.jl")
=#

using ClimaAtmos
using ClimaAtmos.Utils
using ClimaAtmos.Interface
using ClimaAtmos.Backends

# explicit imports from Interface
import ClimaAtmos.Interface: PeriodicRectangle, SingleColumn
import ClimaAtmos.Interface: BarotropicFluidModel, HydrostaticModel
import ClimaAtmos.Interface: TimeStepper
import ClimaAtmos.Interface: DirichletBC, Simulation

# explicit imports from Backends
import ClimaAtmos.Backends: create_ode_problem, evolve

# External Stuff
using IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

# set up parameters
parameters = (
    T_surf = 300.0,
    T_min_ref = 230.0,
    MSLP = 1e5, # mean sea level pressure
    grav = 9.8, # gravitational constant
    R_d = 287.058, # R dry (gas constant / mol mass dry air)
    C_p = 287.058 * 1.4 / (1.4 - 1), # heat capacity at constant pressure
    C_v = 287.058 / (1.4 - 1), # heat capacity at constant volume
    R_m = 87.058, # moist R, assumed to be dry
    f = 5e-5, # Coriolis parameters
    ν = .01,
    Cd = 0.01 / (2e2 / 30.0),
    ug = 1.0,
    vg = 0.0,
    d = sqrt(2.0 * 0.01 / 5e-5),
)

# set up domain
domain = SingleColumn(zlim = 0.0..2e2, nelements = 30)

# set up initial conditions
function init_centers(zc, parameters)
    UnPack.@unpack T_surf, T_min_ref, grav, C_p, MSLP, R_d = parameters

    # temperature
    Γ = grav / C_p
    T= max(T_surf - Γ * zc, T_min_ref)

    # pressure
    p = MSLP * (T / T_surf)^(grav / (R_d * Γ))
    if T == T_min_ref
        z_top = (T_surf - T_min_ref) / Γ
        H_min = R_d * T_min_ref / grav
        p *= exp(-(zc - z_top) / H_min)
    end

    # potential temperature
    θ = T_surf

    # density
    ρ = p / (R_d * θ * (p / MSLP)^(R_d / C_p))

    # velocties
    u = 1.0
    v = 0.0

    return (ρ = ρ, u = u, v = v, ρθ = ρ * θ)
end

function init_faces(zf, parameters)
    return (; w = 0.0 .* zf)
end

# set up model
model = HydrostaticModel( 
    domain = domain, 
    boundary_conditions = nothing, 
    initial_conditions = (centers = init_centers, faces = init_faces), 
    parameters = parameters
)

# set up timestepper
timestepper = TimeStepper(
    method = SSPRK33(),
    dt = 1e-2,
    tspan = (0.0, 3600.0),
    saveat = 600,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

# set up simulation
simulation = Simulation(
    ClimaCoreBackend(),
    model = model, 
    timestepper = timestepper,
    callbacks = nothing,
)

# run simulation
sol = evolve(simulation)

# post-processing
ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "hydrostatic_ekman"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

# get cell center and face locations for plotting
u_init = sol.u[1]
z_centers = parent(Fields.coordinate_field(axes(u_init.x[1])))
z_faces = parent(Fields.coordinate_field(axes(u_init.x[2])))

function ekman_plot(u, parameters; title = "", size = (1024, 600))
    UnPack.@unpack ug, d, vg = parameters

    u_ref =
        ug .-
        exp.(-z_centers / d) .*
        (ug * cos.(z_centers / d) + vg * sin.(z_centers / d))
    sub_plt1 = Plots.plot(
        u_ref,
        z_centers,
        marker = :circle,
        xlabel = "u",
        label = "Ref",
    )
    sub_plt1 =
        Plots.plot!(sub_plt1, parent(u.x[1].u), z_centers, label = "Comp")

    v_ref =
        vg .+
        exp.(-z_centers / d) .*
        (ug * sin.(z_centers / d) - vg * cos.(z_centers / d))
    sub_plt2 = Plots.plot(
        v_ref,
        z_centers,
        marker = :circle,
        xlabel = "v",
        label = "Ref",
    )
    sub_plt2 =
        Plots.plot!(sub_plt2, parent(u.x[1].v), z_centers, label = "Comp")


    return Plots.plot(
        sub_plt1,
        sub_plt2,
        title = title,
        layout = (1, 2),
        size = size,
    )
end

anim = Plots.@animate for (i, u) in enumerate(sol.u)
    ekman_plot(u, parameters, title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "hydrostatic_ekman.mp4"), fps = 10)
Plots.png(ekman_plot(sol[end], parameters), joinpath(path, "hydrostatic_ekman_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/hydrostatic_ekman_end.png", "ekman end")