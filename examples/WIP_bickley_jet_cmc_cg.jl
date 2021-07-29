# set up boilerplate
push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# set up parameters
const parameters = (
    ϵ = 0.1,   # perturbation size for initial condition
    l = 0.5,   # Gaussian width
    k = 0.5,   # Sinusoidal wavenumber
    ρ₀ = 1.0,  # reference density
    c = 2,
    g = 10,
    D₄ = 1e-4, # hyperdiffusion coefficient
)

# set up grid
n1, n2 = 16, 16
Nq = 4
domain = Domains.RectangleDomain(
    -2π..2π,
    -2π..2π,
    x1periodic = true,
    x2periodic = true,
)
mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
grid_topology = Topologies.GridTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

# set initial condition
function init_state(x, p)
    @unpack x1, x2 = x

    # set initial density field
    ρ = p.ρ₀

    # set initial velocity field
    # Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x1) * cos(p.k * x2)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    U₁ = cosh(x2)^(-2)
    gaussian = exp(-(x2 + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (x2 + p.l / 10) / p.l^2 * cos(p.k * x1) * cos(p.k * x2)
    u₁′ += p.k * gaussian * cos(p.k * x1) * sin(p.k * x2)
    u₂′ = -p.k * gaussian * sin(p.k * x1) * cos(p.k * x2)
    u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    
    # set initial tracer field
    θ = sin(p.k * x2)

    return (ρ = ρ, u = u, ρθ = ρ * θ)
end

# set up model
function rhs!(dydt, y, _, t)
    @unpack D₄, g = parameters

    grad = Operators.Divergence()
    wdiv  = Operators.WeakDivergence()
    grad  = Operators.Gradient()
    wgrad = Operators.WeakGradient()
    curl  = Operators.Curl()
    wcurl = Operators.WeakCurl()

    # compute hyperviscosity first
    @. dydt.u =
        wgrad(sdiv(y.u)) -
        Cartesian12Vector(wcurl(Geometry.Covariant3Vector(curl(y.u))))
    @. dydt.ρθ = wdiv(grad(y.ρθ))
    Spaces.weighted_dss!(dydt)
    @. dydt.u =
        -D₄ * (
            wgrad(sdiv(dydt.u)) -
            Cartesian12Vector(wcurl(Geometry.Covariant3Vector(curl(dydt.u))))
        )
    @. dydt.ρθ = -D₄ * wdiv(grad(dydt.ρθ))

    # Euler components
    space = axes(y)
    J = Fields.Field(space.local_geometry.J, space)
    @. begin
        dydt.ρ = -wdiv(y.ρ * y.u)
        dydt.u +=
            -grad(g * y.ρ + norm(y.u)^2 / 2) +
            Cartesian12Vector(J * (y.u × curl(y.u)))
        dydt.ρθ += -wdiv(y.ρθ * y.u)
    end
    Spaces.weighted_dss!(dydt)

    return dydt
end

# set up simulation
x = Fields.coordinate_field(space)
y0 = init_state.(x, Ref(parameters))
dydt = similar(y0)
rhs!(dydt, y0, nothing, 0.0)
prob = ODEProblem(rhs!, y0, (0.0, 200.0))

# run simulation
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.04,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)