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
import ClimaCore.Utilities: half
import UnPack

using OrdinaryDiffEq: ODEProblem, solve, SSPRK33

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

const n_vert = 10
const n_horz = 4
const p_horz = 4

const R = 6.4e6 # radius
const Ω = 7.2921e-5 # Earth rotation (radians / sec)
const z_top = 3.0e4 # height position of the model top
const grav = 9.8 # gravitational constant
const p_0 = 1e5 # mean sea level pressure

const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const T_tri = 273.16 # triple point temperature
const γ = 1.4 # heat capacity ratio
const cv_d = R_d / (γ - 1)
const cp_d = R_d + cv_d

const T_0 = 300 # isothermal atmospheric temperature
const H = R_d * T_0 / grav # scale height

# set up function space
function sphere_3D(
    R = 6.4e6,
    zlim = (0, 30.0e3),
    helem = 4,
    zelem = 15,
    npoly = 5,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space)
end

# geopotential
Φ(z) = grav * z

# pressure
function pressure(ρ, e, normuvw, z)
    I = e - Φ(z) - normuvw^2 / 2
    T = I / cv_d + T_tri
    return ρ * R_d * T
end

# initial conditions for density and total energy
function init_sbr_thermo(z)

    p = p_0 * exp(-z / H)
    ρ = 1 / R_d / T_0 * p

    e = cv_d * (T_0 - T_tri) + Φ(z)
    ρe = ρ * e

    return (ρ = ρ, ρe = ρe)
end

function rhs!(dY, Y, parameters, t)
    UnPack.@unpack c_coords, cuvw, cw, cω³, fω¹², fu¹², fu³, cp, cE = parameters
    cρ = Y.Yc.ρ # density on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe = Y.Yc.ρe # total energy on centers

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    z = c_coords.z

    # # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33 ????????

    hdiv = Operators.Divergence()
    hgrad = Operators.Gradient()
    hcurl = Operators.Curl()

    @. dρ = 0 * cρ
    @. dw = 0 * fw
    @. duₕ = 0 * cuₕ
    @. dρe = 0 * cρe

    # hyperdiffusion not needed in SBR

    # 1) Mass conservation
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @. cw = If2c(fw)
    @. cuvw =
        Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    # 1.a) horizontal divergence
    @. dρ -= hdiv(cρ * (cuvw))

    # 1.b) vertical divergence
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(0.0)),
    )
    # we want the total u³ at the boundary to be zero: we can either constrain
    # both to be zero, or allow one to be non-zero and set the other to be its
    # negation

    # explicit part
    @. dρ -= vdivf2c(Ic2f(cρ * cuₕ))
    # implicit part
    @. dρ -= vdivf2c(Ic2f(cρ) * fw)

    # 2) Momentum equation

    # curl term
    # effectively a homogeneous Dirichlet condition on u₁ at the boundary
    vcurlc2f = Operators.CurlC2F(
        bottom = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
        top = Operators.SetCurl(Geometry.Contravariant12Vector(0.0, 0.0)),
    )
    @. cω³ = hcurl(cuₕ) # Contravariant3Vector
    @. fω¹² = hcurl(fw) # Contravariant12Vector
    @. fω¹² += vcurlc2f(cuₕ) # Contravariant12Vector

    # cross product
    # convert to contravariant
    # these will need to be modified with topography
    @. fu¹² =
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(Ic2f(cuₕ))) # Contravariant12Vector in 3D
    @. fu³ = Geometry.Contravariant3Vector(Geometry.Covariant123Vector(fw))
    @. dw -= fω¹² × fu¹² # Covariant3Vector on faces
    @. duₕ -= If2c(fω¹² × fu³)

    # Needed for 3D:
    @. duₕ -=
        (f + cω³) ×
        Geometry.Contravariant12Vector(Geometry.Covariant123Vector(cuₕ))

    @. cp = pressure(cρ, cρe / cρ, norm(cuvw), z)

    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    @. cE = (norm(cuvw)^2) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) total energy

    @. dρe -= hdiv(cuvw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    Spaces.weighted_dss!(dY.Yc)
    Spaces.weighted_dss!(dY.uₕ)
    Spaces.weighted_dss!(dY.w)

    return dY

end

# set up 3D spherical domain and coords
hv_center_space, hv_face_space =
    sphere_3D(R, (0, z_top), n_horz, n_vert, p_horz)
c_coords = Fields.coordinate_field(hv_center_space)
f_coords = Fields.coordinate_field(hv_face_space)

# Coriolis
const f = @. Geometry.Contravariant3Vector(Geometry.WVector(
    2 * Ω * sind(c_coords.lat),
),)

# discrete hydrostatic profile
zc_vec = parent(c_coords.z) |> unique

N = length(zc_vec)
ρ = zeros(Float64, N)
p = zeros(Float64, N)
ρe = zeros(Float64, N)

# compute ρ, ρe, p from analytical formula; not discretely balanced
for i in 1:N
    var = init_sbr_thermo(zc_vec[i])
    ρ[i] = var.ρ
    ρe[i] = var.ρe
    p[i] = pressure(ρ[i], ρe[i] / ρ[i], 0.0, zc_vec[i])
end

ρ_ana = copy(ρ) # keep a copy for analytical ρ which will be used in correction ρe

function discrete_hydrostatic_balance!(ρ, p, dz, grav)
    for i in 1:(length(ρ) - 1)
        ρ[i + 1] = -ρ[i] - 2 * (p[i + 1] - p[i]) / dz / grav
    end
end

discrete_hydrostatic_balance!(ρ, p, z_top / n_vert, grav)
# now ρ (after correction) and p (computed from analytical relation) are in discrete hydrostatic balance
# only need to correct ρe without changing ρ and p, i.e., keep ρT unchanged before vs after the correction on ρ
ρe = @. ρe + (ρ - ρ_ana) * Φ(zc_vec) - (ρ - ρ_ana) * cv_d * T_tri

# Note: In princile, ρe = @. cv_d * p /R_d - ρ * cv_d * T_tri + ρ * Φ(zc_vec) should work,
#       however, it is not as accurate as the above correction

# set up initial condition: not discretely balanced; only create a Field as a place holder
Yc = map(coord -> init_sbr_thermo(coord.z), c_coords)
# put the dicretely balanced ρ and ρe into Yc
parent(Yc.ρ) .= ρ  # Yc.ρ is a VIJFH layout
parent(Yc.ρe) .= ρe

# initialize velocity: at rest
uₕ = map(_ -> Geometry.Covariant12Vector(0.0, 0.0), c_coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), f_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

cuvw = Geometry.Covariant123Vector.(Y.uₕ)
If2c = Operators.InterpolateF2C()
hcurl = Operators.Curl()
Ic2f = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

cw = If2c.(Y.w)
cω³ = hcurl.(Y.uₕ)
fω¹² = hcurl.(Y.w)
fu¹² = @. Geometry.Contravariant12Vector(Geometry.Covariant123Vector(Ic2f(Y.uₕ)))
fu³ = @. Geometry.Contravariant3Vector(Geometry.Covariant123Vector(Y.w))
cp = @. pressure(Y.Yc.ρ, Y.Yc.ρe / Y.Yc.ρ, norm(cuvw), c_coords.z)
cE = @. (norm(cuvw)^2) / 2 + Φ(c_coords.z)

parameters = (; c_coords, cuvw, cw, cω³, fω¹², fu¹², fu³, cp, cE)
# initialize tendency
dYdt = similar(Y)
# set up rhs
rhs!(dYdt, Y, parameters, 0.0)

# run!
using OrdinaryDiffEq
# Solve the ODE
T = 3600
dt = 5
prob = ODEProblem(rhs!, Y, (0.0, T), parameters)

integrator = OrdinaryDiffEq.init(
    prob,
    SSPRK33(),
    dt = dt,
    saveat = dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, p, t) -> t,
)

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

# solve ode
sol = @timev OrdinaryDiffEq.solve!(integrator)

uₕ_phy = Geometry.transform.(Ref(Geometry.UVAxis()), sol.u[end].uₕ)
w_phy = Geometry.transform.(Ref(Geometry.WAxis()), sol.u[end].w)

@test maximum(abs.(uₕ_phy.components.data.:1)) ≤ 1e-11
@test maximum(abs.(uₕ_phy.components.data.:2)) ≤ 1e-11
@test maximum(abs.(w_phy |> parent)) ≤ 1e-11
@test norm(sol.u[end].Yc.ρ) ≈ norm(sol.u[1].Yc.ρ) rtol = 1e-2
@test norm(sol.u[end].Yc.ρe) ≈ norm(sol.u[1].Yc.ρe) rtol = 1e-2
