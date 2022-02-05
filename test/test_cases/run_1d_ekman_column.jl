if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end

import UnPack
using LinearAlgebra
import ClimaCore:
    ClimaCore, Fields, Domains, Meshes, Operators, Geometry, Spaces
const CC = ClimaCore

import OrdinaryDiffEq
const ODE = OrdinaryDiffEq

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

# https://github.com/CliMA/Thermodynamics.jl/blob/main/src/TemperatureProfiles.jl#L115-L155
# https://clima.github.io/Thermodynamics.jl/dev/TemperatureProfiles/#DecayingTemperatureProfile
function adiabatic_temperature_profile(
    z,
    ug,
    vg;
    T_surf = 300.0,
    T_min_ref = 230.0,
)
    u = ug
    v = vg
    return (u = u, v = v)
end

function tendency!(
    dY::FV,
    Y::FV,
    cache,
    t::Real,
) where {FV <: CC.Fields.FieldVector}
    UnPack.@unpack f, ν, Cd, ug, vg, d = cache
    Yc = Y.Yc
    w = Y.w

    dYc = dY.Yc
    dw = dY.w

    u = Yc.u
    v = Yc.v

    du = dYc.u
    dv = dYc.v

    # S 4.4.1: potential temperature density
    # Mass conservation

    u_1 = parent(u)[1]
    v_1 = parent(v)[1]
    u_wind = sqrt(u_1^2 + v_1^2)
    A = Operators.AdvectionC2C(
        bottom = Operators.SetValue(0.0),
        top = Operators.SetValue(0.0),
    )

    # u-momentum
    bcs_bottom = Operators.SetValue(Geometry.WVector(Cd * u_wind * u_1))  # Eq. 4.16
    bcs_top = Operators.SetValue(ug)  # Eq. 4.18
    gradc2f = Operators.GradientC2F(top = bcs_top)
    divf2c = Operators.DivergenceF2C(bottom = bcs_bottom)
    @. du = divf2c(ν * gradc2f(u)) + f * (v - vg) - A(w, u)   # Eq. 4.8

    # v-momentum
    bcs_bottom = Operators.SetValue(Geometry.WVector(Cd * u_wind * v_1))  # Eq. 4.17
    bcs_top = Operators.SetValue(vg)  # Eq. 4.19
    gradc2f = Operators.GradientC2F(top = bcs_top)
    divf2c = Operators.DivergenceF2C(bottom = bcs_bottom)
    @. dv = divf2c(ν * gradc2f(v)) - f * (u - ug) - A(w, v)   # Eq. 4.9
    return nothing
end

function ode_integrator(::Type{FT}) where {FT}

    f::FT = 5e-5
    ν::FT = 0.01
    L::FT = 2e2
    nelems = 30
    Cd::FT = ν / (L / nelems)
    ug::FT = 1.0
    vg::FT = 0.0
    d::FT = sqrt(2 * ν / f)

    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0.0),
        Geometry.ZPoint{FT}(L);
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain; nelems = nelems)

    cspace = Spaces.CenterFiniteDifferenceSpace(mesh)
    fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

    zc = Fields.coordinate_field(cspace)
    Yc = adiabatic_temperature_profile.(zc.z, Ref(ug), Ref(vg))
    w = Geometry.WVector.(zeros(Float64, fspace))

    Y = Fields.FieldVector(Yc = Yc, w = w)
    cache = (; f, ν, Cd, ug, vg, d, cspace, fspace)

    Δt = 2.0
    ndays = 0
    # Solve the ODE operator
    prob = ODE.ODEProblem(tendency!, Y, (0.0, 60 * 60 * 50), cache)
    integrator = ODE.init(
        prob,
        ODE.SSPRK33(),
        dt = Δt,
        saveat = 600, # save 10 min
        progress = true,
        progress_message = (dt, u, p, t) -> t,
    )
    return integrator, cache
end

integrator, cache = ode_integrator(Float64)

sol = ODE.solve!(integrator)

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

path = joinpath(@__DIR__, first(split(basename(@__FILE__), ".jl")))
mkpath(path)


function ekman_plot(u, cache; title = "", size = (1024, 600))
    UnPack.@unpack cspace, fspace, d, ug, vg = cache
    z_centers = parent(Fields.coordinate_field(cspace))
    z_faces = parent(Fields.coordinate_field(fspace))
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
    sub_plt1 = Plots.plot!(sub_plt1, parent(u.Yc.u), z_centers, label = "Comp")

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
    sub_plt2 = Plots.plot!(sub_plt2, parent(u.Yc.v), z_centers, label = "Comp")

    return Plots.plot(
        sub_plt1,
        sub_plt2,
        title = title,
        layout = (1, 2),
        size = size,
    )
end

anim = Plots.@animate for (i, u) in enumerate(sol.u)
    ekman_plot(u, cache, title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "ekman.mp4"), fps = 10)

Plots.png(ekman_plot(sol[end], cache), joinpath(path, "ekman_end.png"))

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

linkfig(
    relpath(joinpath(path, "ekman_end.png"), joinpath(@__DIR__, "../..")),
    "Ekman End",
)
