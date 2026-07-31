#=
DGModel: everything the tendency/Jacobian functions need, built once per
problem (the de-globalized replacement for the ClimaCore examples'
include-time module constants). Passed to the integrator as the parameter
`p`, so every RHS/Wfact call receives it.

Port of the setup sections of sphere_dg_fd_model.jl (grid, geopotential,
Coriolis, Cartesian basis fields, sponge profiles, κ₄ cap, FD operators)
plus the flux-form driver's tangential basis projections and vvdivc2f.

DG note: the horizontal space is the SAME SpectralElementSpace2D + GLL
construction CG uses — DG means `weighted_dss!` is never called and the
face coupling comes from `Operators.add_numerical_flux_internal!` /
lifting corrections instead.
=#

struct DGModel{FT <: AbstractFloat, P, PA, S, F, O, M, FLX}
    prob::P
    c::DGConstants{FT}   # scalar constants
    params::PA           # CAP.ClimaAtmosParameters (Setups IC, HS forcing)
    spaces::S            # (; horzspace, hv_center_space, hv_face_space)
    fields::F            # precomputed fields (Φ, Coriolis, basis, sponges)
    ops::O               # FD/DG operator instances
    opmats::M            # MatrixFields operator matrices (Jacobians)
    interface_flux_fn::FLX
    filter_Nc::Int
    Δt::FT
    κ₄::FT
    κ₄_cap::FT
end

float_type(::DGModel{FT}) where {FT} = FT

function dg_spaces(prob, c::DGConstants{FT}) where {FT}
    context = ClimaComms.context()
    device = ClimaComms.device(context)
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(FT)),
        Geometry.ZPoint(prob.zmax);
        boundary_names = (:bottom, :top),
    )
    vertmesh = if prob.zstretch === nothing
        Meshes.IntervalMesh(vertdomain, nelems = prob.zelem)
    else
        dzb, dzt = prob.zstretch
        Meshes.IntervalMesh(
            vertdomain,
            Meshes.GeneralizedExponentialStretching(dzb, dzt);
            nelems = prob.zelem,
        )
    end
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(device, vertmesh)

    horzdomain = Domains.SphereDomain(c.R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, prob.helem)
    horztopology = Topologies.Topology2D(context, horzmesh)
    quad = Quadratures.GLL{prob.npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(
        horzspace,
        vert_center_space,
        dg_hypsography(prob, horzspace, c),
    )
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (; horzspace, hv_center_space, hv_face_space)
end

#=
Earth topography: ETOPO2022 60arcsec ClimaArtifacts orography regridded
onto the GLL nodes with SpaceVaryingInput, then diffusion-smoothed with the
same recipe as ClimaAtmos `hypsography_function_from_topography` (diffusion
Courant number 0.05, iteration count from the damping factor) and warped
with LinearAdaption. The smoothing runs the CG Laplacian (DSS inside) — a
grid-generation preprocessing step, not part of the DG discretization; the
warp needs a single-valued, continuous z_sfc anyway.
=#
function dg_hypsography(prob, horzspace, ::DGConstants{FT}) where {FT}
    prob.topography == :none && return Grids.Flat()
    if prob.topography == :hughes2023
        # Hughes & Jablonowski (2023) analytic double mountain (two
        # zonally separated midlatitude peaks, h₀ = 2 km): pointwise
        # evaluation on the GLL nodes, no smoothing (already smooth).
        z_surface = SpaceVaryingInput(
            CA.topography_function(CA.Hughes2023Topography()),
            horzspace,
        )
        @info "Hughes2023 double-mountain orography" extrema(z_surface)
        return Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
    end
    context = ClimaComms.context(horzspace)
    z_surface = SpaceVaryingInput(
        CA.AtmosArtifacts.earth_orography_file_path(; context),
        "z",
        horzspace,
    )
    diff_courant = FT(0.05)
    Δh_scale = Spaces.node_horizontal_length_scale(horzspace)
    # Explicit-Euler stability is set by the SMALLEST node spacing, not the
    # average: GLL endpoint clustering + cubed-sphere corner distortion make
    # the ClimaAtmos average-based recipe (κ = 0.05·Δh²) unstable at coarse
    # helem (measured: helem=4/npoly=4 diverges after ~12 iterations). Take
    # the per-step κ·dt from the minimum quadrature node area and keep the
    # total smoothing κ·dt·maxiter = log(damping_factor)·Δh² invariant.
    Δx²_min = minimum(Fields.local_geometry_field(horzspace).WJ)
    κ = FT(diff_courant * Δx²_min)
    maxiter = Int(
        round(
            log(prob.topography_damping_factor) * Δh_scale^2 /
            (diff_courant * Δx²_min),
        ),
    )
    Hypsography.diffuse_surface_elevation!(z_surface; κ, dt = FT(1), maxiter)
    @. z_surface = max(z_surface, 0)
    @info "Earth orography (ETOPO2022 60arcsec)" Δh_scale maxiter extrema(
        z_surface,
    )
    return Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
end

function dg_fields(prob, c::DGConstants{FT}, spaces) where {FT}
    ccoords = Fields.coordinate_field(spaces.hv_center_space)
    fcoords = Fields.coordinate_field(spaces.hv_face_space)

    ᶜΦ = @. c.grav * ccoords.z
    ᶜf_cor = @. CT3(Geometry.WVector(2 * c.Ω * sind(ccoords.lat)))

    # Cartesian basis fields (centers): ê_E, ê_N, r̂ from lat/long [deg].
    # Position-dependent but state-independent; velocity components advected
    # as scalars must live in this globally constant frame.
    eE1 = @. -sind(ccoords.long)
    eE2 = @. cosd(ccoords.long)
    eE3 = map(_ -> FT(0), eE1)
    eN1 = @. -sind(ccoords.lat) * cosd(ccoords.long)
    eN2 = @. -sind(ccoords.lat) * sind(ccoords.long)
    eN3 = @. cosd(ccoords.lat)
    eR1 = @. cosd(ccoords.lat) * cosd(ccoords.long)
    eR2 = @. cosd(ccoords.lat) * sind(ccoords.long)
    eR3 = @. sind(ccoords.lat)
    # Tangential projections of the Cartesian unit vectors: (ê_c·ê_E, ê_c·ê_N)
    # as UVVectors — the Cartesian components of ê_E, ê_N (KG pressure flux).
    E1 = @. Geometry.UVVector(eE1, eN1)
    E2 = @. Geometry.UVVector(eE2, eN2)
    E3 = @. Geometry.UVVector(eE3, eN3)

    # Rayleigh sponge over the top z_sponge: absorbing layer for the rigid
    # lid. β profile as in the CG examples, Δt-independent peak rate 1/τ.
    # Applied to ρw always (τ = Inf disables); sponge_uh additionally damps
    # Cartesian horizontal momentum (NOT canonical for the baroclinic wave).
    z_sponge = FT(7.5e3)
    zmax = prob.zmax
    τs = prob.sponge_τ
    ᶠβ_sponge = @. ifelse(
        fcoords.z > zmax - z_sponge,
        1 / τs * sin(FT(π) / 2 * (fcoords.z - (zmax - z_sponge)) / z_sponge)^2,
        FT(0),
    )
    ᶜβ_sponge = @. ifelse(
        ccoords.z > zmax - z_sponge,
        1 / τs * sin(FT(π) / 2 * (ccoords.z - (zmax - z_sponge)) / z_sponge)^2,
        FT(0),
    )

    # Constant surface-temperature level field for the HS forcing's σ
    # computation: on flat topography z_sfc = 0, so σ = p/MSLP and the
    # value is irrelevant — it only enters via exp(-g·z_sfc/(R_d·T_sfc)).
    T_sfc = Fields.level(fcoords.z, Fields.half) .* FT(0) .+ FT(288)

    return (;
        ccoords,
        fcoords,
        T_sfc,
        ᶜΦ,
        ᶜf_cor,
        eE1, eE2, eE3, eN1, eN2, eN3, eR1, eR2, eR3,
        E1, E2, E3,
        ᶠβ_sponge,
        ᶜβ_sponge,
    )
end

function dg_operators(::DGConstants{FT}) where {FT}
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    hcurl = Operators.Curl()
    Ic = Operators.InterpolateF2C()
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    # ρJ-weighted C2F interpolation (CA's ᶠwinterp): REQUIRED for the
    # terrain CT3(uₕ) vertical flux — the plain If version leaves an O(1)
    # cross-discretization free-stream defect at resolved slopes (measured
    # 0.19 s⁻¹ relative ρe tendency at the Hughes2023 mountain flank).
    wIf = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    # CT3-valued flux variant (full contravariant vertical transport
    # ᶠu³ = CT3(uₕ) + CT3(w) over terrain-warped grids); the zero-flux BCs
    # are the terrain-following no-normal-flow condition u³ = 0.
    vdivf2c3 = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    )
    vvdivc2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
        top = Operators.SetDivergence(Geometry.WVector(FT(0))),
    )
    VanLeer = Operators.LinVanLeerC2F(
        bottom = Operators.FirstOrderOneSided(),
        top = Operators.FirstOrderOneSided(),
        constraint = Operators.MonotoneLocalExtrema(),
    )
    ᶠgradᵥ = Operators.GradientC2F(
        bottom = Operators.SetGradient(C3(FT(0))),
        top = Operators.SetGradient(C3(FT(0))),
    )
    ᶠcurlᵥ = Operators.CurlC2F(
        bottom = Operators.SetCurl(CT12(FT(0), FT(0))),
        top = Operators.SetCurl(CT12(FT(0), FT(0))),
    )
    Bw = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(C3(FT(0))),
        top = Operators.SetValue(C3(FT(0))),
    )
    return (;
        hwdiv, hgrad, hcurl,
        Ic, If, wIf, vdivf2c, vdivf2c3, vvdivc2f, VanLeer, ᶠgradᵥ, ᶠcurlᵥ, Bw,
    )
end

interface_flux_fn(s::Symbol) =
    s == :roe ? Operators.kennedy_gruber_roe_cartesian :
    Operators.kennedy_gruber_rusanov_cartesian

function DGModel(prob::DGProblem)
    validate(prob)
    FT = float_type(prob)
    c = DGConstants{FT}(; mode = prob.constants_mode)
    params = dg_params(FT, prob.constants_mode)
    spaces = dg_spaces(prob, c)
    fields = dg_fields(prob, c, spaces)
    ops = dg_operators(c)
    opmats = (;
        ᶜinterp_matrix = MatrixFields.operator_matrix(ops.Ic),
        ᶠinterp_matrix = MatrixFields.operator_matrix(ops.If),
        ᶜdivᵥ_matrix = MatrixFields.operator_matrix(ops.vdivf2c),
        ᶠgradᵥ_matrix = MatrixFields.operator_matrix(ops.ᶠgradᵥ),
    )

    Δt = prob.dt
    # Explicit SIPG biharmonic stability cap (validated on the plane DG-FD
    # cases; the CG value 2e17 is only stable there because DSS makes the
    # first-pass Laplacian continuous). Default κ₄ = cap/10: the SIPG
    # penalty acts on O(truncation) face jumps of the element-local first
    # pass, so cap-level κ₄ measurably forces smooth balanced states.
    κ₄_cap = FT(
        Spaces.node_horizontal_length_scale(spaces.horzspace)^3 /
        ((2 * prob.npoly + 1)^2 * Δt),
    )
    kep_vi = prob isa BaroclinicWaveDG && prob.face_set == :kep
    κ₄ = if prob.κ₄ !== nothing
        prob.κ₄
    elseif kep_vi
        # KEP face set: the advective KE budget closes; run unstabilized
        FT(0)
    else
        min(FT(2e17), κ₄_cap / 10)
    end
    κ₄ > κ₄_cap && @warn "κ₄ exceeds the explicit SIPG stability cap" κ₄ κ₄_cap
    # tendency cutoff filter: REQUIRED default for the legacy (:kg)
    # vector-invariant core (npoly); 0 for the KEP face set; NEVER for
    # FDDG (voids its KEP — not even exposed there)
    filter_Nc = if prob isa BaroclinicWaveDG
        prob.filter_Nc === nothing ? (kep_vi ? 0 : prob.npoly) :
        prob.filter_Nc
    else
        0
    end

    # Eager face-connectivity build: self-caches per space; doing it here
    # surfaces device/topology errors at init instead of in the first step.
    Operators.dg_connectivity(spaces.horzspace)

    return DGModel(
        prob,
        c,
        params,
        spaces,
        fields,
        ops,
        opmats,
        prob isa BaroclinicWaveFDDG ?
        interface_flux_fn(prob.interface_flux) : nothing,
        filter_Nc,
        Δt,
        κ₄,
        κ₄_cap,
    )
end
