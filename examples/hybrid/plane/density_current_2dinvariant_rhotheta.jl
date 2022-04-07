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
# set up function space
function hvspace_2D(
    xlim = (-π, π),
    zlim = (0, 4π),
    helem = 64,
    velem = 32,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = velem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(xlim[1]),
        Geometry.XPoint{FT}(xlim[2]),
        periodic = true,
    )
    horzmesh = Meshes.IntervalMesh(horzdomain; nelems = helem)
    horztopology = Topologies.IntervalTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

hv_center_space, hv_face_space = hvspace_2D((-25600, 25600), (0, 6400))

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
# Prognostic thermodynamic variable: Total Energy 
function init_dry_density_current_2d(x, z)
    x_c = 0.0
    z_c = 3000.0
    r_c = 1.0
    x_r = 4000.0
    z_r = 2000.0
    θ_b = 300.0
    θ_c = -15.0
    cp_d = C_p
    cv_d = C_v
    p_0 = MSLP
    g = grav

    # auxiliary quantities
    r = sqrt((x - x_c)^2 / x_r^2 + (z - z_c)^2 / z_r^2)
    θ_p = r < r_c ? 0.5 * θ_c * (1.0 + cospi(r / r_c)) : 0.0 # potential temperature perturbation

    θ = θ_b + θ_p # potential temperature
    π_exn = 1.0 - Φ(z) / cp_d / θ # exner function
    T = π_exn * θ # temperature
    p = p_0 * π_exn^(cp_d / R_d) # pressure
    ρ = p / R_d / T # density
    ρθ = ρ * θ  # total energy

    return (ρ = ρ, ρθ = ρθ)
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_dry_density_current_2d(coord.x, coord.z), coords)
uₕ = map(_ -> Geometry.Covariant1Vector(0.0), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

theta_0 = sum(Y.Yc.ρθ)
mass_0 = sum(Y.Yc.ρ)

function rhs_invariant!(dY, Y, _, t)

    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant1Vector on centers
    cρθ = Y.Yc.ρθ

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρθ = dY.Yc.ρθ
    ρθ = Yc.ρθ
    z = coords.z


    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()

    dρ .= 0 .* cρ

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients

    χθ = @. dρθ = hwdiv(hgrad(cρθ / cρ)) # we store χθ in dρθ
    χuₕ = @. duₕ = hwgrad(hdiv(cuₕ))
    Spaces.weighted_dss!(dρθ)
    Spaces.weighted_dss!(duₕ)

    κ₄ = 0.0 # m^4/s
    @. dρθ = -κ₄ * hwdiv(cρ * hgrad(χθ))
    @. duₕ = -κ₄ * hwgrad(hdiv(χuₕ))

    # 1) Mass conservation
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cw = If2c.(fw)
    fuₕ = Ic2f.(cuₕ)
    cuw = Geometry.Covariant13Vector.(cuₕ) .+ Geometry.Covariant13Vector.(cw)

    dw .= fw .* 0

    # 1.a) horizontal divergence
    dρ .-= hdiv.(cρ .* (cuw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    vdivc2f = Operators.DivergenceC2F(
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
        bottom = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
        top = Operators.SetCurl(Geometry.Contravariant2Vector(0.0)),
    )
    #cω³ = hcurl.(cu) # zero because we're only in 1D: we can leave this off for the bubble
    fω¹² = hcurl.(fw) # Contravariant2Vector / Contravariant12Vector
    fω¹² .+= vcurlc2f.(cuₕ) # Contravariant2Vector / Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    fu¹² =
        Geometry.Contravariant1Vector.(Geometry.Covariant13Vector.(Ic2f.(cuₕ))) # Contravariant12Vector in 3D
    fu³ = Geometry.Contravariant3Vector.(Geometry.Covariant13Vector.(fw))
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    # Needed for 3D:
    #cu¹² = Contravariant12Vector.(cu)
    #@. du += cω³ × cu¹²

    cp = @. pressure(cρθ)
    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. (norm(cuw)^2) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) potential temperature

    @. dρθ -= hdiv(cuw * ρθ)
    @. dρθ -= vdivf2c(fw * Ic2f(cρθ))
    @. dρθ -= vdivf2c(Ic2f(cuₕ * cρθ))

    # Order 2 Diffusion 
    κ₂ = 75.0
    θ = @. cρθ / cρ
    fρ = @. Ic2f(cρ)
    ∂c = Operators.GradientF2C()
    dfw = dY.w.components.data.:1
    dcu = dY.uₕ.components.data.:1

    ᶠ∇ᵥuₕ = @. vgradc2f(cuₕ.components.data.:1)
    ᶜ∇ᵥw = @. ∂c(fw.components.data.:1)
    ᶠ∇ᵥθ = @. vgradc2f(θ)

    ᶜ∇ₕuₕ = @. hgrad(cuₕ.components.data.:1)
    ᶠ∇ₕw = @. hgrad(fw.components.data.:1)
    ᶜ∇ₕθ = @. hgrad(θ)

    # Laplacian Diffusion (Uniform)
    hκ₂∇²uₕ = @. hwdiv(κ₂ * ᶜ∇ₕuₕ)
    vκ₂∇²uₕ = @. vdivf2c(κ₂ * ᶠ∇ᵥuₕ)
    hκ₂∇²w = @. hwdiv(κ₂ * ᶠ∇ₕw)
    vκ₂∇²w = @. vdivc2f(κ₂ * ᶜ∇ᵥw)
    hκ₂∇²θ = @. hwdiv(cρ * κ₂ * ᶜ∇ₕθ)
    vκ₂∇²θ = @. vdivf2c(fρ * κ₂ * ᶠ∇ᵥθ)

    @. dcu += hκ₂∇²uₕ
    @. dcu += vκ₂∇²uₕ
    @. dfw += hκ₂∇²w
    @. dfw += vκ₂∇²w
    @. dρθ += hκ₂∇²θ
    @. dρθ += vκ₂∇²θ

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY
end

dYdt = similar(Y);
rhs_invariant!(dYdt, Y, nothing, 0.0);

# run!
using OrdinaryDiffEq
Δt = 0.3
prob = ODEProblem(rhs_invariant!, Y, (0.0, 900.0))
sol_invariant = solve(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 5.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dir = "dc_rhotheta"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

# post-processing
anim = Plots.@animate for u in sol_invariant.u
    Plots.plot(u.Yc.ρθ ./ u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "theta.mp4"), fps = 20)

anim = Plots.@animate for u in sol_invariant.u
    Plots.plot(u.Yc.ρ)
end
Plots.mp4(anim, joinpath(path, "rho.mp4"), fps = 20)


If2c = Operators.InterpolateF2C()
anim = Plots.@animate for u in sol_invariant.u
    Plots.plot(Geometry.WVector.(Geometry.Covariant13Vector.(If2c.(u.w))))
end
Plots.mp4(anim, joinpath(path, "vel_w.mp4"), fps = 20)

anim = Plots.@animate for u in sol_invariant.u
    Plots.plot(Geometry.UVector.(Geometry.Covariant13Vector.(u.uₕ)))
end
Plots.mp4(anim, joinpath(path, "vel_u.mp4"), fps = 20)

θs = [sum(u.Yc.ρθ) for u in sol_invariant.u]
Mass = [sum(u.Yc.ρ) for u in sol_invariant.u]

Plots.png(
    Plots.plot((θs .- theta_0) ./ theta_0),
    joinpath(path, "energy_cons.png"),
)
Plots.png(
    Plots.plot((Mass .- mass_0) ./ mass_0),
    joinpath(path, "mass_cons.png"),
)
