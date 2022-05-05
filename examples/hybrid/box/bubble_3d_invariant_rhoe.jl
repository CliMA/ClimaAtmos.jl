using Test
using StaticArrays, IntervalSets, LinearAlgebra, UnPack

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    DataLayouts,
    Spaces,
    Fields,
    Operators
using ClimaCore.Geometry
using Logging
# ENV["CLIMACORE_DISTRIBUTED"] = "MPI" # TODO: remove before merging
usempi = get(ENV, "CLIMACORE_DISTRIBUTED", "") == "MPI"
if usempi
    using ClimaComms
    using ClimaCommsMPI
    const comms_ctx = ClimaCommsMPI.MPICommsContext()
    const pid, nprocs = ClimaComms.init(comms_ctx)
    if pid == 1
        println("parallel run with $nprocs processes")
    end
    logger_stream = ClimaComms.iamroot(comms_ctx) ? stderr : devnull

    prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
    atexit() do
        global_logger(prev_logger)
    end
else
    using Logging: global_logger
    using TerminalLoggers: TerminalLogger
    global_logger(TerminalLogger())
end

function hvspace_3D(
    xlim = (-π, π),
    ylim = (-π, π),
    zlim = (0, 4π),
    xelem = 4,
    yelem = 4,
    zelem = 16,
    npoly = 3;
    usempi = false,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.RectangleDomain(
        Geometry.XPoint{FT}(xlim[1])..Geometry.XPoint{FT}(xlim[2]),
        Geometry.YPoint{FT}(ylim[1])..Geometry.YPoint{FT}(ylim[2]),
        x1periodic = true,
        x2periodic = true,
    )
    Nv = Meshes.nelements(vertmesh)
    Nf_center, Nf_face = 2, 1 #1 + 3 + 1
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzmesh = Meshes.RectilinearMesh(horzdomain, xelem, yelem)
    horztopology =
        usempi ? Topologies.DistributedTopology2D(comms_ctx, horzmesh) :
        Topologies.Topology2D(horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (hv_center_space, hv_face_space, horztopology)
end
# set up 3D domain - doubly periodic box
hv_center_space, hv_face_space, horztopology =
    hvspace_3D((-500, 500), (-500, 500), (0, 1000), usempi = usempi)

const MSLP = 1e5 # mean sea level pressure
const grav = 9.8 # gravitational constant
const R_d = 287.058 # R dry (gas constant / mol mass dry air)
const γ = 1.4 # heat capacity ratio
const C_p = R_d * γ / (γ - 1) # heat capacity at constant pressure
const C_v = R_d / (γ - 1) # heat capacity at constant volume
const T_0 = 273.16 # triple point temperature

Φ(z) = grav * z

function pressure(ρ, e, uvw, z)
    I = e - Φ(z) - (norm(uvw)^2) / 2
    T = I / C_v + T_0
    return ρ * R_d * T
end

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
    e = cv_d * (T - T_0) + g * z
    ρe = ρ * e # total energy

    return (ρ = ρ, ρe = ρe)
end

# initial conditions
coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

Yc = map(coord -> init_dry_rising_bubble_3d(coord.x, coord.y, coord.z), coords)
uₕ = map(_ -> Geometry.Covariant12Vector(0.0, 0.0), coords)
w = map(_ -> Geometry.Covariant3Vector(0.0), face_coords)
Y = Fields.FieldVector(Yc = Yc, uₕ = uₕ, w = w)

energy_0 = usempi ? ClimaComms.reduce(comms_ctx, sum(Y.Yc.ρe), +) : sum(Y.Yc.ρe)
mass_0 = usempi ? ClimaComms.reduce(comms_ctx, sum(Y.Yc.ρ), +) : sum(Y.Yc.ρ)

if !usempi || (usempi && ClimaComms.iamroot(comms_ctx))
    @show (energy_0, mass_0)
end
ghost_buffer = if usempi
    (;
        dYc = Spaces.create_ghost_buffer(Y, Y.Yc),
        dρe = Spaces.create_ghost_buffer(Y, Y.Yc.ρe),
        duₕ = Spaces.create_ghost_buffer(Y, Y.uₕ),
        dw = Spaces.create_ghost_buffer(Y, Y.w),
    )
else
    nothing
end

function rhs_invariant!(dY, Y, p, t)
    (; ghost_buffer) = p
    cρ = Y.Yc.ρ # scalar on centers
    fw = Y.w # Covariant3Vector on faces
    cuₕ = Y.uₕ # Covariant12Vector on centers
    cρe = Y.Yc.ρe

    dρ = dY.Yc.ρ
    dw = dY.w
    duₕ = dY.uₕ
    dρe = dY.Yc.ρe
    z = coords.z

    # 0) update w at the bottom
    # fw = -g^31 cuₕ/ g^33

    hdiv = Operators.Divergence()
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hwgrad = Operators.WeakGradient()
    hcurl = Operators.Curl()
    hwcurl = Operators.WeakCurl()

    dρ .= 0 .* cρ

    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    cw = If2c.(fw)
    fuₕ = Ic2f.(cuₕ)
    cuvw = Geometry.Covariant123Vector.(cuₕ) .+ Geometry.Covariant123Vector.(cw)

    ce = @. cρe / cρ
    cI = @. ce - Φ(z) - (norm(cuvw)^2) / 2
    cT = @. cI / C_v + T_0
    cp = @. cρ * R_d * cT

    ### HYPERVISCOSITY
    # 1) compute hyperviscosity coefficients
    ch_tot = @. ce + cp / cρ
    χe = @. dρe = hwdiv(hgrad(ch_tot)) # we store χe in dρe
    χuₕ = @. duₕ =
        hwgrad(hdiv(cuₕ)) -
        Geometry.Covariant12Vector(hwcurl(Geometry.Covariant3Vector(hcurl(
            cuₕ,
        ))),)
    Spaces.weighted_dss!(dρe, ghost_buffer.dρe)
    Spaces.weighted_dss!(duₕ, ghost_buffer.duₕ)

    κ₄ = 100.0 # m^4/s
    @. dρe = -κ₄ * hwdiv(cρ * hgrad(χe))
    @. duₕ =
        -κ₄ * (
            hwgrad(hdiv(χuₕ)) -
            Geometry.Covariant12Vector(hwcurl(Geometry.Covariant3Vector(hcurl(
                χuₕ,
            ))),)
        )

    # 1) Mass conservation

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


    @. duₕ -= hgrad(cp) / cρ
    vgradc2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0.0)),
    )
    @. dw -= vgradc2f(cp) / Ic2f(cρ)

    cE = @. (norm(cuvw)^2) / 2 + Φ(z)
    @. duₕ -= hgrad(cE)
    @. dw -= vgradc2f(cE)

    # 3) potential temperature

    @. dρe -= hdiv(cuvw * (cρe + cp))
    @. dρe -= vdivf2c(fw * Ic2f(cρe + cp))
    @. dρe -= vdivf2c(Ic2f(cuₕ * (cρe + cp)))

    fcc = Operators.FluxCorrectionC2C(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    fcf = Operators.FluxCorrectionF2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )

    @. dρ += fcc(fw, cρ)
    @. dρe += fcc(fw, cρe)
    # dYc.ρuₕ += fcc(w, Yc.ρuₕ)

    Spaces.weighted_dss!(dY.Yc, ghost_buffer.dYc)
    Spaces.weighted_dss!(dY.uₕ, ghost_buffer.duₕ)
    Spaces.weighted_dss!(dY.w, ghost_buffer.dw)
    return dY
end

p = (; ghost_buffer)
dYdt = similar(Y);
rhs_invariant!(dYdt, Y, p, 0.0);
# run!
using OrdinaryDiffEq
Δt = 0.050
prob = ODEProblem(rhs_invariant!, Y, (0.0, 700.0), p)
integrator = OrdinaryDiffEq.init(
    prob,
    SSPRK33(),
    dt = Δt,
    saveat = 10.0,
    progress = true,
    progress_message = (dt, u, p, t) -> t,
);

if haskey(ENV, "CI_PERF_SKIP_RUN") # for performance analysis
    throw(:exit_profile)
end

sol_invariant = @timev OrdinaryDiffEq.solve!(integrator)

FT = Float64
Es = FT[]
Mass = FT[]
if usempi
    for sol_step in sol_invariant.u
        Es_step = ClimaComms.reduce(comms_ctx, sum(sol_step.Yc.ρe), +)
        Mass_step = ClimaComms.reduce(comms_ctx, sum(sol_step.Yc.ρ), +)
        if ClimaComms.iamroot(comms_ctx)
            push!(Es, Es_step)
            push!(Mass, Mass_step)
        end
    end
else
    for sol_step in sol_invariant.u
        push!(Es, sum(sol_step.Yc.ρe))
        push!(Mass, sum(sol_step.Yc.ρ))
    end
end

ENV["GKSwstype"] = "nul"
import Plots
Plots.GRBackend()

dir = "bubble_3d_invariant_rhoe"
path = joinpath(@__DIR__, "output", dir)
mkpath(path)

if !usempi || (usempi && ClimaComms.iamroot(comms_ctx))
    # post-processing
    Plots.png(
        Plots.plot((Es .- energy_0) ./ energy_0),
        joinpath(path, "energy.png"),
    )
    Plots.png(
        Plots.plot((Mass .- mass_0) ./ mass_0),
        joinpath(path, "mass.png"),
    )

    function linkfig(figpath, alt = "")
        # buildkite-agent upload figpath
        # link figure in logs if we are running on CI
        if get(ENV, "BUILDKITE", "") == "true"
            artifact_url = "artifact://$figpath"
            print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
        end
    end

    linkfig(
        relpath(joinpath(path, "energy.png"), joinpath(@__DIR__, "../..")),
        "Total Energy",
    )
    linkfig(
        relpath(joinpath(path, "mass.png"), joinpath(@__DIR__, "../..")),
        "Mass",
    )
end
