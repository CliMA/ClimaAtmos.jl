if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end

import UnPack
import ClimaCore
const CC = ClimaCore
const CCO = CC.Operators

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

function ∑tendencies!(
    dY::FV,
    Y::FV,
    cache,
    t::Real,
) where {FV <: CC.Fields.FieldVector}
    UnPack.@unpack f, ν, Cd, ug, vg, d = cache
    u = Y.Yc.u
    v = Y.Yc.v
    w = Y.w

    du = dY.Yc.u
    dv = dY.Yc.v
    dw = dY.w

    # S 4.4.1: potential temperature density
    # Mass conservation
    wvec = CC.Geometry.WVector

    u_1 = parent(u)[1]
    v_1 = parent(v)[1]
    u_wind = sqrt(u_1^2 + v_1^2)
    A = CCO.AdvectionC2C(bottom = CCO.SetValue(0.0), top = CCO.SetValue(0.0))

    # u-momentum
    bcs_bottom = CCO.SetValue(wvec(Cd * u_wind * u_1))  # Eq. 4.16
    bcs_top = CCO.SetValue(ug)  # Eq. 4.18
    ugradc2f = CCO.GradientC2F(top = bcs_top)
    udivf2c = CCO.DivergenceF2C(bottom = bcs_bottom)
    @. du = udivf2c(ν * ugradc2f(u)) + f * (v - vg) - A(w, u)   # Eq. 4.8

    # v-momentum
    bcs_bottom = CCO.SetValue(wvec(Cd * u_wind * v_1))  # Eq. 4.17
    bcs_top = CCO.SetValue(vg)  # Eq. 4.19
    vgradc2f = CCO.GradientC2F(top = bcs_top)
    vdivf2c = CCO.DivergenceF2C(bottom = bcs_bottom)
    @. dv = vdivf2c(ν * vgradc2f(v)) - f * (u - ug) - A(w, v)   # Eq. 4.9
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

    domain = CC.Domains.IntervalDomain(
        CC.Geometry.ZPoint{FT}(0.0),
        CC.Geometry.ZPoint{FT}(L);
        boundary_names = (:bottom, :top),
    )
    mesh = CC.Meshes.IntervalMesh(domain; nelems = nelems)

    cspace = CC.Spaces.CenterFiniteDifferenceSpace(mesh)
    fspace = CC.Spaces.FaceFiniteDifferenceSpace(cspace)

    zc = CC.Fields.coordinate_field(cspace)
    Yc = adiabatic_temperature_profile.(zc.z, Ref(ug), Ref(vg))
    w = CC.Geometry.WVector.(zeros(Float64, fspace))

    Y = CC.Fields.FieldVector(Yc = Yc, w = w)
    cache = (; f, ν, Cd, ug, vg, d)

    Δt = 2.0
    ndays = 0
    # Solve the ODE operator
    prob = ODE.ODEProblem(∑tendencies!, Y, (0.0, 60 * 60 * 50), cache)
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
import ClimaCorePlots, Plots
Plots.GRBackend()

path = joinpath(@__DIR__, first(split(basename(@__FILE__), ".jl")))
mkpath(path)


function ekman_plot(u, cache; title = "", size = (1024, 600))
    UnPack.@unpack d, ug, vg = cache
    cspace = axes(u.Yc.u)
    z_centers = parent(CC.Fields.coordinate_field(cspace))
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
