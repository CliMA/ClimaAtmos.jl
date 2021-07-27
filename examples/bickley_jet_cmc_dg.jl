push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using ClimaCore.Geometry, LinearAlgebra, UnPack
import ClimaCore: Fields, Domains, Topologies, Meshes, Spaces
import ClimaCore: slab
import ClimaCore.Operators
import ClimaCore.Geometry
using LinearAlgebra, IntervalSets
using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

using ClimaCore.RecursiveApply
using ClimaCore.RecursiveApply: rdiv, rmap


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
)

numflux_name = get(ARGS, 1, "roe")
boundary_name = get(ARGS, 2, "")

domain = Domains.RectangleDomain(
    -2π..2π,
    -2π..2π,
    x1periodic = true,
    x2periodic = boundary_name != "noslip",
    x2boundary = boundary_name != "noslip" ? nothing : (:south, :north),
)

n1, n2 = 16, 16
Nq = 4
Nqh = 7
mesh = Meshes.EquispacedRectangleMesh(domain, n1, n2)
grid_topology = Topologies.GridTopology(mesh)
quad = Spaces.Quadratures.GLL{Nq}()
space = Spaces.SpectralElementSpace2D(grid_topology, quad)

Iquad = Spaces.Quadratures.GLL{Nqh}()
Ispace = Spaces.SpectralElementSpace2D(grid_topology, Iquad)

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

    return (ρ = ρ, ρu = ρ * u, ρθ = ρ * θ)
end

y0 = init_state.(Fields.coordinate_field(space), Ref(parameters))

function flux(state, p)
    @unpack ρ, ρu, ρθ = state
    u = ρu ./ ρ
    return (ρ = ρu, ρu = ((ρu ⊗ u) + (p.g * ρ^2 / 2) * I), ρθ = ρθ .* u)
end

function energy(state, p)
    @unpack ρ, ρu = state
    u = ρu ./ ρ
    return ρ * (u.u1^2 + u.u2^2) / 2 + p.g * ρ^2 / 2
end

function total_energy(y, parameters)
    sum(state -> energy(state, parameters), y)
end

# numerical fluxes
wavespeed(y, parameters) = sqrt(parameters.g)

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function roeflux(n, (y⁻, parameters⁻), (y⁺, parameters⁺))
    Favg = rdiv(flux(y⁻, parameters⁻) ⊞ flux(y⁺, parameters⁺), 2)

    λ = sqrt(parameters⁻.g)

    ρ⁻, ρu⁻, ρθ⁻ = y⁻.ρ, y⁻.ρu, y⁻.ρθ
    ρ⁺, ρu⁺, ρθ⁺ = y⁺.ρ, y⁺.ρu, y⁺.ρθ

    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n

    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n

    # in general thermodynamics, (pressure, soundspeed)
    p⁻ = (λ * ρ⁻)^2 * 0.5
    c⁻ = λ * sqrt(ρ⁻)

    p⁺ = (λ * ρ⁺)^2 * 0.5
    c⁺ = λ * sqrt(ρ⁺)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!

    fluxᵀn_ρ = (w1 + w2 + w3) * 0.5
    fluxᵀn_ρu =
        (w1 * (u - c * n) + w2 * (u + c * n) + w3 * u + w4 * (Δu - Δuₙ * n)) *
        0.5
    fluxᵀn_ρθ = ((w1 + w2) * θ + w5) * 0.5

    Δf = (ρ = -fluxᵀn_ρ, ρu = -fluxᵀn_ρu, ρθ = -fluxᵀn_ρθ)
    rmap(f -> f' * n, Favg) ⊞ Δf
end


numflux = if numflux_name == "central"
    Operators.CentralNumericalFlux(flux)
elseif numflux_name == "rusanov"
    Operators.RusanovNumericalFlux(flux, wavespeed)
elseif numflux_name == "roe"
    roeflux
end

function rhs!(dydt, y, (parameters, numflux), t)

    # ϕ' K' W J K dydt =  -ϕ' K' I' [DH' WH JH flux.(I K y)]
    #  =>   K dydt = - K inv(K' WJ K) K' I' [DH' WH JH flux.(I K y)]

    # where:
    #  ϕ = test function
    #  K = DSS scatter (i.e. duplicates points at element boundaries)
    #  K y = stored input vector (with duplicated values)
    #  I = interpolation to higher-order space
    #  D = derivative operator
    #  H = suffix for higher-order space operations
    #  W = Quadrature weights
    #  J = Jacobian determinant of the transformation `ξ` to `x`
    #
    Nh = Topologies.nlocalelems(y)

    F = flux.(y, Ref(parameters))
    dydt .= Operators.slab_weak_divergence(F)

    Operators.add_numerical_flux_internal!(numflux, dydt, y, parameters)

    Operators.add_numerical_flux_boundary!(
        dydt,
        y,
        parameters,
    ) do normal, (y⁻, parameters)
        y⁺ = (ρ = y⁻.ρ, ρu = y⁻.ρu .- dot(y⁻.ρu, normal) .* normal, ρθ = y⁻.ρθ)
        numflux(normal, (y⁻, parameters), (y⁺, parameters))
    end

    # 6. Solve for final result
    dydt_data = Fields.field_values(dydt)
    dydt_data .= rdiv.(dydt_data, space.local_geometry.WJ)

    M = Spaces.Quadratures.cutoff_filter_matrix(
        Float64,
        space.quadrature_style,
        3,
    )
    Operators.tensor_product!(dydt_data, M)

    return dydt
end

dydt = Fields.Field(similar(Fields.field_values(y0)), space)
rhs!(dydt, y0, (parameters, numflux), 0.0);

# Solve the ODE operator
prob = ODEProblem(rhs!, y0, (0.0, 200.0), (parameters, numflux))
sol = solve(
    prob,
    SSPRK33(),
    dt = 0.02,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
)

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dirname = "dg_$(numflux_name)"
if boundary_name != ""
    dirname = "$(dirname)_$(boundary_name)"
end
path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

anim = Plots.@animate for u in sol.u
    Plots.plot(u.ρθ, clim = (-1, 1))
end
Plots.mp4(anim, joinpath(path, "tracer.mp4"), fps = 10)

Es = [total_energy(u, parameters) for u in sol.u]
Plots.png(Plots.plot(Es), joinpath(path, "energy.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig("output/$(dirname)/energy.png", "Total Energy")

# interpolate
n_interp = 4
L = domain.x1max - domain.x1min
vec_u = Operators.matrix_interpolate(sol.u[end].ρu.u1./sol.u[end].ρ, n_interp);
vec_v = Operators.matrix_interpolate(sol.u[end].ρu.u2./sol.u[end].ρ, n_interp);
trac_ρθ = Operators.matrix_interpolate(sol.u[end].ρθ, n_interp);

# save to JLD2
using JLD2
save("./output/dg_roe.jld2", "total_energy", Es, "time", sol.t, "u", vec_u, "v", vec_v, "L", L, "N", n1*n_interp, "tracer", trac_ρθ)