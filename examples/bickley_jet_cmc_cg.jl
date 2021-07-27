push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: slab, Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore: slab
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

const parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρ₀ = 1.0, # reference density
    c = 2,
    g = 10,
    D₄ = 1e-3, # hyperdiffusion coefficient
)

domain = Domains.RectangleDomain(
    -2π..2π,
    -2π..2π,
    x1periodic = true,
    x2periodic = true,
)

n1, n2 = 16, 16
Nq = 4
mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
grid_topology = Topologies.GridTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

const J = Fields.Field(space.local_geometry.J, space)

function init_state(x, p)
    @unpack x1, x2 = x
    # set initial state
    ρ = p.ρ₀

    # set initial velocity
    U₁ = cosh(x2)^(-2)

    # Ψ′ = exp(-(x2 + p.l / 10)^2 / 2p.l^2) * cos(p.k * x1) * cos(p.k * x2)
    # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
    gaussian = exp(-(x2 + p.l / 10)^2 / 2p.l^2)
    u₁′ = gaussian * (x2 + p.l / 10) / p.l^2 * cos(p.k * x1) * cos(p.k * x2)
    u₁′ += p.k * gaussian * cos(p.k * x1) * sin(p.k * x2)
    u₂′ = -p.k * gaussian * sin(p.k * x1) * cos(p.k * x2)


    u = Cartesian12Vector(U₁ + p.ϵ * u₁′, p.ϵ * u₂′)
    # set initial tracer
    θ = sin(p.k * x2)

    return (ρ = ρ, u = u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function energy(state, p)
    @unpack ρ, u = state
    return ρ * (u.u1^2 + u.u2^2) / 2 + p.g * ρ^2 / 2
end

function total_energy(y, parameters)
    sum(state -> energy(state, parameters), y)
end

function rhs!(dydt, y, _, t)

    @unpack D₄, g = parameters

    sdiv = Operators.Divergence()
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    wgrad = Operators.WeakGradient()
    curl = Operators.Curl()
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

    # add in pieces
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

dydt = similar(y0)
rhs!(dydt, y0, nothing, 0.0)

# Solve the ODE operator
prob = ODEProblem(rhs!, y0, (0.0, 200.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.04,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "cg_invariant_hypervisc"
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]
Plots.png(Plots.plot(sol.t, Es ./ Es[1] .* 100.0, xlabel="Time (s)", ylabel="Relative total energy (%)"), joinpath(path, "energy.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/energy.png", "Total Energy")

# # interpolate
# n_interp = 4
# L = domain.x1max - domain.x1min
# vec_u = Operators.matrix_interpolate(sol.u[end].u.u1, n_interp);
# vec_v = Operators.matrix_interpolate(sol.u[end].u.u2, n_interp);
# trac_ρθ = Operators.matrix_interpolate(sol.u[end].ρθ, n_interp);

# # save to JLD2
# using JLD2
# save("./output/cg_d4_1e-3.jld2", "total_energy", Es, "time", sol.t, "u", vec_u, "v", vec_v, "L", L, "N", n1*n_interp, "tracer", trac_ρθ)