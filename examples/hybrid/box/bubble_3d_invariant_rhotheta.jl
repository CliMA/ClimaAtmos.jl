using Test
using LinearAlgebra

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators
using ClimaCore.Geometry

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())
function hvspace_3D(
    xlim = (-π, π),
    ylim = (-π, π),
    zlim = (0, 4π),
    xelem = 4,
    yelem = 4,
    zelem = 16,
    npoly = 3,
)
    FT = Float64

    xdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    ydomain = Domains.IntervalDomain(
        Geometry.YPoint{FT}(ylim[1]),
        Geometry.YPoint{FT}(ylim[2]),
        periodic = true,
    )

    horzdomain = Domains.RectangleDomain(xdomain, ydomain)
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelem, yelem)
    horztopology = Topologies.Topology2D(horzmesh)

    zdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(zdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

# set up 3D domain - doubly periodic box
hv_center_space, hv_face_space = hvspace_3D((-500, 500), (-500, 500), (0, 1000))

const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const R_m = R_d # moist R, assumed to be dry

function pressure(ρθ)
    if ρθ >= 0
        return MSLP * (R_d * ρθ / MSLP)^γ
    else
        return NaN
    end
end

Φ(z) = grav * z

# Reference: https://journals.ametsoc.org/view/journals/mwre/140/4/mwr-d-10-05073.1.xml, Section 5a
function init_dry_rising_bubble_3d(x, y, z)
    x_c = 0.0
    y_c = 0.0
    z_c = 350.0
    r_c = 250.0
    θ_b = 300.0
    θ_c = 0.5
    cp_d = C_p
    cv_d = C_v
    p_0 = MSLP
    g = grav

    # auxiliary quantities
    r = sqrt((x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - g * z / cp_d / θ # exner function
    T = π_exn * θ # temperature
    p = p_0 * π_exn^(cp_d / R_d) # pressure
    ρ = p / R_d / T # density
    ρθ = ρ * θ # potential temperature density

    return (ρ = ρ, ρθ = ρθ)
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_dry_rising_bubble_3d(coord.x, coord.y, coord.z), coords)
uₕ = map(_ -> Geometry.Covariant12Vector(0.0, 0.0), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

function energy(ρ, ρθ, ρu, z)
    u = ρu ./ ρ
    kinetic = ρ .* norm(u)^2 ./ 2
    potential = z .* grav .* ρ
    internal = C_v .* pressure.(ρθ) ./ R_d
    return kinetic .+ potential .+ internal
end

function combine_momentum(ρuₕ, ρw)
    Geometry.transform(Geometry.Covariant123Axis(), ρuₕ) +
    Geometry.transform(Geometry.Covariant123Axis(), ρw)
end

function center_momentum(Y)
    If2c = Operators.InterpolateF2C()
    ρ = Y.Yc.ρ
    ρuₕ = ρ .* Y.uₕ
    ρw = ρ .* If2c.(Y.w)
    combine_momentum.(ρuₕ, ρw)
end

function total_energy(Y)
    ρ = Y.Yc.ρ
    ρu = center_momentum(Y)
    ρθ = Y.Yc.ρθ
    z = Fields.coordinate_field(axes(ρ)).z
    sum(energy(ρ, ρθ, ρu, z))
end

energy_0 = total_energy(Y)
mass_0 = sum(Yc.ρ) # Computes ∫ρ∂Ω such that quadrature weighting is accounted for.
theta_0 = sum(Yc.ρθ) # Computes ∫ρ∂Ω such that quadrature weighting is accounted for.

function rhs_invariant!(dY, Y, _, t)

    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρθ = Y.Yc.ρθ

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρθ = dY.Yc.ρθ


    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.WeakCurl()

    dρ .= 0 .* cρ

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients

    χθ = @. dρθ = hwdiv(hgrad(cρθ / cρ)) # we store χθ in dρθ
    χuₕ = @. duₕ =
        hwgrad(hdiv(cuₕ)) -
        Geometry.Covariant12Vector(hwcurl(Geometry.Covariant3Vector(hcurl(
            cuₕ,
        ))),)

    Spaces.weighted_dss!(dρθ)
    Spaces.weighted_dss!(duₕ)

    κ₄ = 100.0 # m^4/s
    @. dρθ = -κ₄ * hwdiv(cρ * hgrad(χθ))
    @. duₕ =
        -κ₄ * (
            hwgrad(hdiv(χuₕ)) -
            Geometry.Covariant12Vector(hwcurl(Geometry.Covariant3Vector(hcurl(
                χuₕ,
            ))),)
        )

    # 1) Mass conservation
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cw = If2c.(fw)
    fuₕ = Ic2f.(cuₕ)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuvw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    dρ .-= vdivf2c.(Ic2f.(cρ .* cuₕ))
    # implicit part
    dρ .-= vdivf2c.(Ic2f.(cρ) .* fw)

    # 2) Momentum equation

    # curl term
    hcurl = Operators.Curl()
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    )
    cω³ = hcurl.(cuₕ) # Contravariant3Vector
    fω¹² = hcurl.(fw) # Contravariant12Vector
    fω¹² .+= vcurlc2f.(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² =
        Geometry.Contravariant12Vector.(Geometry.Covariant123Vector.(Ic2f.(
            cuₕ,
        )),) # Contravariant12Vector in 3D
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant123Vector.(fw))
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    # Needed for 3D:
    @. duₕ -=
        cω³ × Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    cp = @. pressure(cρθ)
    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. (norm(cuvw)^2) / 2 + Φ(coords.z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) potential temperature

    @. dρθ -= hdiv(cuvw * cρθ)
    @. dρθ -= vdivf2c(fw * Ic2f(cρθ))
    @. dρθ -= vdivf2c(Ic2f(cuₕ * cρθ))

    fcc = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    @. dρ += fcc(fw, cρ)
    @. dρθ += fcc(fw, cρθ)
    # dYc.ρuₕ += fcc(w, Yc.ρuₕ)

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)


    return dY
end

dYdt = similar(Y);
rhs_invariant!(dYdt, Y, nothing, 0.0);

# run!
using OrdinaryDiffEq
Δt = 0.050
prob = ODEProblem(rhs_invariant!, Y, (0.0, 1.0))
sol = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 1.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

dir = "bubble_3d_invariant_rhotheta"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

# slice along the center XZ axis
Plots.png(
    Plots.plot(
        sol.u[end].Yc.ρθ ./ sol.u[end].Yc.ρ,
        slice = (:, 0.0, :),
        clim = (300.0, 300.8),
    ),
    joinpath(path, "theta_end.png"),
)

# post-processing
Es = [total_energy(u) for u in sol.u]
Mass = [sum(u.Yc.ρ) for u in sol.u]
Theta = [sum(u.Yc.ρθ) for u in sol.u]

Plots.png(
    Plots.plot((Es .- energy_0) ./ energy_0),
    joinpath(path, "energy.png"),
)

Plots.png(
    Plots.plot((Theta .- theta_0) ./ theta_0),
    joinpath(path, "rho_theta.png"),
)
Plots.png(Plots.plot((Mass .- mass_0) ./ mass_0), joinpath(path, "mass.png"))

linkfig(
    relpath(joinpath(path, "theta_end.png"), joinpath(@__DIR__, "../..")),
    "Theta end",
)
linkfig(
    relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
    "Total Energy",
)
linkfig(
    relpath(joinpath(path, "rho_theta.png"), joinpath(@__DIR__, "../..")),
    "Potential Temperature",
)
linkfig(
    relpath(joinpath(path, "mass.png"), joinpath(@__DIR__, "../..")),
    "Mass",
)
