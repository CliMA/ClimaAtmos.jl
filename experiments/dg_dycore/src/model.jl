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

    horzmesh = if prob isa MountainWaveDG
        # x-periodic quasi-2D slab: one y element of one-element width
        # (the DG face kernels need a genuine 2D horizontal space)
        Δy = prob.xmax / prob.helem
        horzdomain = Domains.RectangleDomain(
            Domains.IntervalDomain(
                Geometry.XPoint{FT}(-prob.xmax / 2),
                Geometry.XPoint{FT}(prob.xmax / 2);
                periodic = true,
            ),
            Domains.IntervalDomain(
                Geometry.YPoint{FT}(-Δy / 2),
                Geometry.YPoint{FT}(Δy / 2);
                periodic = true,
            ),
        )
        Meshes.RectilinearMesh(horzdomain, prob.helem, 1)
    else
        horzdomain = Domains.SphereDomain(c.R)
        Meshes.EquiangularCubedSphere(horzdomain, prob.helem)
    end
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

# Build the terrain adaption object (LinearAdaption or SLEVEAdaption).
# Called after the surface elevation field has been smoothed.
function terrain_adaption(prob, z_surface, ::DGConstants{FT}) where {FT}
    if prob.terrain_warp == :sleve
        return Hypsography.SLEVEAdaption(
            Geometry.ZPoint.(z_surface),
            FT(prob.sleve_eta_h),
            FT(prob.sleve_s),
        )
    end
    return Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
end

# ETOPO2022 pre-smoothing: Laplacian diffusion on the horizontal CG space
# (grid-generation preprocessing; not part of the DG discretization).
# κ is set from the MINIMUM node area (GLL endpoint + cubed-sphere corner
# clustering) to keep the explicit-Euler step stable at coarse helem.
# Total smoothing κ·dt·maxiter = log(topography_damping_factor)·Δh²_avg.
function smooth_earth_orography!(z_surface, prob, horzspace, ::DGConstants{FT}) where {FT}
    diff_courant = FT(0.05)
    Δh_scale = Spaces.node_horizontal_length_scale(horzspace)
    Δx²_min = minimum(Fields.local_geometry_field(horzspace).WJ)
    κ = FT(diff_courant * Δx²_min)
    maxiter = Int(
        round(
            log(prob.topography_damping_factor) * Δh_scale^2 /
            (diff_courant * Δx²_min),
        ),
    )
    Hypsography.diffuse_surface_elevation!(z_surface; κ, dt = FT(1), maxiter)
    @. z_surface = max(z_surface, FT(0))
    return maxiter
end

# :earth = ETOPO2022 artifact → SpaceVaryingInput → diffusion smoothing →
# LinearAdaption or SLEVEAdaption; :hughes2023 = analytic double mountain,
# evaluated pointwise, no smoothing. MountainWaveDG always uses LinearAdaption
# (Agnesi ridge is analytic and not excessively steep).
function dg_hypsography(
    prob::MountainWaveDG,
    horzspace,
    ::DGConstants{FT},
) where {FT}
    x = Fields.coordinate_field(horzspace).x
    z_surface = @. prob.h₀ / (1 + (x / prob.a)^2)
    @info "Agnesi orography" prob.h₀ prob.a extrema(z_surface)
    return Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
end

function dg_hypsography(prob, horzspace, c::DGConstants{FT}) where {FT}
    prob.topography == :none && return Grids.Flat()
    if prob.topography == :hughes2023
        z_surface = SpaceVaryingInput(
            CA.topography_function(CA.Hughes2023Topography()),
            horzspace,
        )
        @info "Hughes2023 double-mountain orography" extrema(z_surface)
        # analytic mountain — always LinearAdaption (smooth, not steep; validated here)
        return Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
    end
    context = ClimaComms.context(horzspace)
    z_surface = SpaceVaryingInput(
        CA.AtmosArtifacts.earth_orography_file_path(; context),
        "z",
        horzspace,
    )
    maxiter = smooth_earth_orography!(z_surface, prob, horzspace, c)
    @info "Earth orography (ETOPO2022 60arcsec)" prob.terrain_warp maxiter extrema(
        z_surface,
    )
    return terrain_adaption(prob, z_surface, c)
end

# w-only Rayleigh sponge profiles over the top sponge_depth (peak rate
# 1/τ; τ = Inf disables) + the τ-independent sin² shape for ν_vert
function sponge_fields(prob, ::Type{FT}, ccoords, fcoords) where {FT}
    z_sponge = FT(prob.sponge_depth)
    zmax = prob.zmax
    τs = prob.sponge_τ
    shape(z) = ifelse(
        z > zmax - z_sponge,
        sin(FT(π) / 2 * (z - (zmax - z_sponge)) / z_sponge)^2,
        FT(0),
    )
    ᶠβ_sponge = @. shape(fcoords.z) / τs
    ᶜβ_sponge = @. shape(ccoords.z) / τs
    ᶠsponge_shape = @. shape(fcoords.z)
    return (; ᶠβ_sponge, ᶜβ_sponge, ᶠsponge_shape)
end

function dg_fields(prob::MountainWaveDG, c::DGConstants{FT}, spaces) where {FT}
    ccoords = Fields.coordinate_field(spaces.hv_center_space)
    fcoords = Fields.coordinate_field(spaces.hv_face_space)

    ᶜΦ = @. c.grav * ccoords.z
    # f-plane at f = 0 (non-rotating mountain wave)
    ᶜf_cor = @. CT3(Geometry.WVector(0 * ccoords.z))

    # Cartesian basis on the plane: ê_E = x̂, ê_N = ŷ, r̂ = ẑ
    o = @. 0 * ccoords.z + 1
    zf = @. 0 * ccoords.z
    eE1, eE2, eE3 = o, zf, zf
    eN1, eN2, eN3 = zf, o, zf
    eR1, eR2, eR3 = zf, zf, o
    E1 = @. Geometry.UVVector(o, zf)
    E2 = @. Geometry.UVVector(zf, o)
    E3 = @. Geometry.UVVector(zf, zf)

    T_sfc = Fields.level(fcoords.z, Fields.half) .* FT(0) .+ prob.T₀

    return (;
        ccoords,
        fcoords,
        T_sfc,
        ᶜΦ,
        ᶜf_cor,
        eE1, eE2, eE3, eN1, eN2, eN3, eR1, eR2, eR3,
        E1, E2, E3,
        sponge_fields(prob, FT, ccoords, fcoords)...,
    )
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

    # Rayleigh sponge over the top sponge_depth (peak rate 1/τ; τ = Inf
    # disables): applied to ρw/w — the correct gravity-wave absorber.
    # sponge_uh additionally drags the upper-level jet (leave off).
    z_sponge = FT(prob.sponge_depth)
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
    # τ-independent sin² sponge shape (0 → 1) at faces: the ν_vert
    # vertical-diffusion profile
    ᶠsponge_shape = @. ifelse(
        fcoords.z > zmax - z_sponge,
        sin(FT(π) / 2 * (fcoords.z - (zmax - z_sponge)) / z_sponge)^2,
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
        ᶠsponge_shape,
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
    # ρJ-weighted C2F interpolation (CA's ᶠwinterp) for the terrain
    # CT3(uₕ) vertical flux.
    wIf = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    vdivf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    # BC-less F2C divergence for the ν_vert diffusion flux (ᶠgradᵥ's
    # SetGradient(0) BCs already make it a zero-flux at the boundaries)
    vdivf2c0 = Operators.DivergenceF2C()
    # CT3-flux variant for the full contravariant vertical transport;
    # zero-flux BCs = the terrain-following no-normal-flow condition.
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
        Ic, If, wIf, vdivf2c, vdivf2c0, vdivf2c3, vvdivc2f, VanLeer,
        ᶠgradᵥ, ᶠcurlᵥ, Bw,
    )
end

interface_flux_fn(s::Symbol) =
    s == :roe ? Operators.kennedy_gruber_roe_cartesian :
    s == :curvilinear_roe ? Operators.kennedy_gruber_roe_cartesian_curvilinear :
    s == :curvilinear_roe_wb ? kennedy_gruber_roe_cartesian_curvilinear_wb :
    Operators.kennedy_gruber_rusanov_cartesian

function DGModel(prob::DGProblem)
    validate(prob)
    FT = float_type(prob)
    c = DGConstants{FT}(; mode = prob.constants_mode)
    params = dg_params(FT, prob.constants_mode)
    spaces = dg_spaces(prob, c)
    fields = dg_fields(prob, c, spaces)
    # Unperturbed base state at the warped node heights: the terrain-aware
    # κ₄ diffusion reference (full fields carry an O(Δz_warp) terrain
    # signature along coordinate surfaces).
    fields = let
        (; T, p, uE, uN) = reference_values(prob, c, params, fields.ccoords)
        ᶜK_ref = @. (uE^2 + uN^2) / 2
        ᶜh_ref = @. (c.cv_d + c.R_d) * T + ᶜK_ref + fields.ᶜΦ -
                    c.cv_d * c.T_tri
        # Hydrostatically composed reference pressure: the :es2 acoustic
        # channel penalizes [[p′]] = [[p − p_ref]], because along
        # terrain-following faces the raw [[p]] carries an O(1)
        # HYDROSTATIC jump (neighbors at different true altitude) that is
        # not acoustic. Φ-only composition (see discrete_hydrostatic_p!).
        ᶜp_ref = copy(p)
        ᶜρ_ref = @. p / (c.R_d * T)
        discrete_hydrostatic_p!(ᶜp_ref, ᶜρ_ref, T, c.R_d, fields.ᶜΦ)
        # Well-balanced metric-defect correction (calibrated post-build in
        # calibrate_wb_reference!): δ_c = tangential rest-tendency / p_ref, the
        # pressure-scaled horizontal-kernel GCL defect the discrete metric
        # identity fails to cancel over terrain. Zero until calibrated (and on
        # flat grids / the VI core, which never read them).
        ᶜwb_δ1 = zero(ᶜp_ref)
        ᶜwb_δ2 = zero(ᶜp_ref)
        ᶜwb_δ3 = zero(ᶜp_ref)
        (;
            fields...,
            ᶜh_ref,
            ᶜu_ref = uE,
            ᶜv_ref = uN,
            ᶜp_ref,
            ᶜρ_ref,
            ᶜwb_δ1,
            ᶜwb_δ2,
            ᶜwb_δ3,
        )
    end
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
    hls = Spaces.node_horizontal_length_scale(spaces.horzspace)
    κ₄_cap = FT(hls^3 / ((2 * prob.npoly + 1)^2 * Δt))
    # Laplacian-scale divergence damping ν∇ₕ(∇ₕ·uₕ), fraction of the
    # explicit cap Δh²/((2n+1)²Δt) (0 disables).
    ν_div = if prob isa VIProblem
        FT(prob.ν_div_frac * hls^2 / ((2 * prob.npoly + 1)^2 * Δt))
    else
        FT(0)
    end
    fields = (; fields..., ν_div)
    kep_vi = prob isa VIProblem && prob.face_set in (:kep, :es, :es2)
    κ₄ = if prob.κ₄ !== nothing
        prob.κ₄
    elseif prob.κ₄_frac !== nothing
        # resolution/Δt-aware specification (CA-style ν₄ ∝ h³ scaling)
        FT(prob.κ₄_frac) * κ₄_cap
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
    filter_Nc = if prob isa VIProblem
        prob.filter_Nc === nothing ? (kep_vi ? 0 : prob.npoly) :
        prob.filter_Nc
    else
        0
    end

    # Eager face-connectivity build: self-caches per space; doing it here
    # surfaces device/topology errors at init instead of in the first step.
    Operators.dg_connectivity(spaces.horzspace)

    m = DGModel(
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
    # Calibrate the well-balanced metric-defect correction on the hydrostatic
    # rest state (FDDG over terrain only; flat grids have no metric defect).
    prob isa BaroclinicWaveFDDG &&
        prob.topography != :none &&
        calibrate_wb_reference!(m)
    return m
end

"""
    calibrate_wb_reference!(m)

Fill `m.fields.ᶜwb_δ{1,2,3}` with the pressure-scaled horizontal-kernel GCL
metric defect, measured on the hydrostatic rest state (ρ=ρ_ref, p=p_ref, u=0).
At rest the only nonzero momentum tendency is the horizontal pressure-flux
metric defect `p_ref·δ_c` (Coriolis/advection/vertical transport all vanish for
u=0; Held–Suarez momentum drag ∝ u = 0), so `δ_c = tangential(dρu_c)/p_ref` is a
pure, state-independent metric quantity. Subtracting `p·δ_c` in the tendency
(flux_form.jl) then makes rest an exact discrete steady state and removes the
pressure-scaled spurious terrain force off-rest. Runs with `ᶜwb_δ ≡ 0`, so the
measured tendency is the raw defect. See project memory: the literal
`−∂_ξ3(Ja³)` term is spatially misaligned (discrete GCL fails), hence this
reference-subtraction closure.
"""
function calibrate_wb_reference!(m::DGModel{FT}) where {FT}
    c = m.c
    ρ = m.fields.ᶜρ_ref
    p = m.fields.ᶜp_ref
    ᶜΦ = m.fields.ᶜΦ
    (; eR1, eR2, eR3) = m.fields
    ρe = @. c.cv_d * p / c.R_d + ρ * (ᶜΦ - c.cv_d * c.T_tri)
    Yc = map(
        (ρi, ρei) ->
            (; ρ = ρi, ρe = ρei, ρu1 = FT(0), ρu2 = FT(0), ρu3 = FT(0)),
        ρ,
        ρe,
    )
    Yf = map(_ -> (; ρw = C3(FT(0))), m.fields.fcoords)
    Y_ref = Fields.FieldVector(c = Yc, f = Yf)
    dY_ref = similar(Y_ref)
    rhs_fddg!(dY_ref, Y_ref, m, FT(0))
    dr = @. dY_ref.c.ρu1 * eR1 + dY_ref.c.ρu2 * eR2 + dY_ref.c.ρu3 * eR3
    @. m.fields.ᶜwb_δ1 = (dY_ref.c.ρu1 - dr * eR1) / p
    @. m.fields.ᶜwb_δ2 = (dY_ref.c.ρu2 - dr * eR2) / p
    @. m.fields.ᶜwb_δ3 = (dY_ref.c.ρu3 - dr * eR3) / p
    return m
end
