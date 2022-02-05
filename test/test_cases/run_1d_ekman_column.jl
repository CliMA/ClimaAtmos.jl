if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end

using LinearAlgebra
import ClimaCore:
    Fields,
    Domains,
    Topologies,
    Meshes,
    DataLayouts,
    Operators,
    Geometry,
    Spaces

import OrdinaryDiffEq
const ODE = OrdinaryDiffEq

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const FT = Float64

# https://github.com/CliMA/CLIMAParameters.jl/blob/master/src/Planet/planet_parameters.jl#L5
const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacit at constant volume
const R_m = R_d # moist R, assumed to be dry

const f = 5e-5
const ν = 0.01
const L = 2e2
const nelems = 30
const Cd = ν / (L / nelems)
const ug = 1.0
const vg = 0.0
const d = sqrt(2 * ν / f)
domain = Domains.IntervalDomain(
    Geometry.ZPoint{FT}(0.0),
    Geometry.ZPoint{FT}(L);
    boundary_names = (:bottom, :top),
)
mesh = Meshes.IntervalMesh(domain; nelems = nelems)

cspace = Spaces.CenterFiniteDifferenceSpace(mesh)
fspace = Spaces.FaceFiniteDifferenceSpace(cspace)

# https://github.com/CliMA/Thermodynamics.jl/blob/main/src/TemperatureProfiles.jl#L115-L155
# https://clima.github.io/Thermodynamics.jl/dev/TemperatureProfiles/#DecayingTemperatureProfile
function adiabatic_temperature_profile(z; T_surf = 300.0, T_min_ref = 230.0)
    u = FT(ug)
    v = FT(vg)
    return (u = u, v = v)
end

zc = Fields.coordinate_field(cspace)
Yc = adiabatic_temperature_profile.(zc.z)
w = Geometry.WVector.(zeros(Float64, fspace))

Y_init = copy(Yc)
w_init = copy(w)

function tendency!(dY, Y, _, t)
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
    bcs_top = Operators.SetValue(FT(ug))  # Eq. 4.18
    gradc2f = Operators.GradientC2F(top = bcs_top)
    divf2c = Operators.DivergenceF2C(bottom = bcs_bottom)
    @. du = divf2c(ν * gradc2f(u)) + f * (v - vg) - A(w, u)   # Eq. 4.8

    # v-momentum
    bcs_bottom = Operators.SetValue(Geometry.WVector(Cd * u_wind * v_1))  # Eq. 4.17
    bcs_top = Operators.SetValue(FT(vg))  # Eq. 4.19
    gradc2f = Operators.GradientC2F(top = bcs_top)
    divf2c = Operators.DivergenceF2C(bottom = bcs_bottom)
    @. dv = divf2c(ν * gradc2f(v)) - f * (u - ug) - A(w, v)   # Eq. 4.9
    return nothing
end


Y = Fields.FieldVector(Yc = Yc, w = w)

Δt = 2.0
ndays = 0
# Solve the ODE operator
prob = ODE.ODEProblem(tendency!, Y, (0.0, 60 * 60 * 50))
sol = ODE.solve(
    prob,
    ODE.SSPRK33(),
    dt = Δt,
    saveat = 600, # save 10 min
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

path = joinpath(@__DIR__, first(split(basename(@__FILE__), ".jl")))
mkpath(path)

z_centers = parent(Fields.coordinate_field(cspace))
z_faces = parent(Fields.coordinate_field(fspace))

function ekman_plot(u; title = "", size = (1024, 600))
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
    ekman_plot(u, title = "Hour $(i)")
end
Plots.mp4(anim, joinpath(path, "ekman.mp4"), fps = 10)

Plots.png(ekman_plot(sol[end]), joinpath(path, "ekman_end.png"))

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
