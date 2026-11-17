import ClimaCore:
    Geometry, Hypsography, Fields, Spaces, Meshes, Grids, CommonGrids, Topologies
using ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput
import .AtmosArtifacts as AA
import ClimaComms

export SphereGrid, ColumnGrid, MultiColumnGrid, BoxGrid, PlaneGrid

"""
    SphereGrid(::Type{FT}; kwargs...)

Create an ExtrudedCubedSphereGrid with topography support.

# Arguments

  - `FT`: the floating-point type [`Float32`, `Float64`]

# Keyword Arguments

  - `context = ClimaComms.context()`: the ClimaComms communications context
  - `z_elem = 10`: the number of z-points
  - `z_max = 30000.0`: the domain maximum along the z-direction
  - `z_stretch = false`: whether to stretch the vertical mesh with the two-scale
    rule of [`get_z_mesh`](@ref)
  - `dz_bottom = 500.0`: bottom layer thickness of the two-scale stretching
  - `dz_top = 3000.0`: top layer thickness of the two-scale stretching
  - `z_pbl = 2200.0`: scale height of the boundary-layer refinement of the
    two-scale stretching
  - `z_mesh`: a custom z-mesh; by default, [`get_z_mesh`](@ref) builds it from
    `z_elem`, `z_max`, `z_stretch`, `dz_bottom`, `dz_top`, and `z_pbl`
  - `radius = 6.371229e6`: the radius of the cubed sphere
  - `h_elem = 6`: the number of horizontal elements per side of every panel (6
    panels in total)
  - `nh_poly = 3`: the polynomial order. Note: The number of quadrature points in
    1D within each horizontal element is then `n_quad_points = nh_poly + 1`
  - `bubble = false`: enables the "bubble correction" for more accurate element
    areas when computing the spectral element space
  - `deep_atmosphere = true`: use deep atmosphere equations and metric terms,
    otherwise assume columns are cylindrical (shallow atmosphere)
  - `topography = NoTopography()`: topography type
  - `topography_damping_factor = 5.0`: factor by which smallest resolved
    length-scale is to be damped
  - `mesh_warp_type = SLEVEWarp{FT}()`: mesh warping type ([`SLEVEWarp`](@ref) or
    [`LinearWarp`](@ref))
  - `topo_smoothing = false`: apply topography smoothing
"""
function SphereGrid(
    ::Type{FT};
    context = ClimaComms.context(),
    z_elem = 10,
    z_max = 30000.0,
    z_stretch = false,
    dz_bottom = 500.0,
    dz_top = 3000.0,
    z_pbl = 2200.0,
    radius = 6.371229e6,
    h_elem = 6,
    nh_poly = 3,
    bubble = false,
    deep_atmosphere = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor = 5.0,
    mesh_warp_type::MeshWarpType = SLEVEWarp{FT}(),
    topo_smoothing = false,
    z_mesh = get_z_mesh(FT; z_max, z_elem, z_stretch, dz_bottom, dz_top, z_pbl),
) where {FT}
    n_quad_points = nh_poly + 1
    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type, topo_smoothing,
    )

    global_geometry = if deep_atmosphere
        Geometry.DeepSphericalGlobalGeometry{FT}(radius)
    else
        Geometry.ShallowSphericalGlobalGeometry{FT}(radius)
    end

    h_mesh = Meshes.EquiangularCubedSphere(Domains.SphereDomain{FT}(radius), h_elem)
    h_topology =
        Topologies.Topology2D(context, h_mesh, Topologies.spacefillingcurve(h_mesh))

    grid = CommonGrids.ExtrudedCubedSphereGrid(
        FT;
        z_elem, z_min = 0, z_max, z_mesh, radius, h_elem,
        n_quad_points,
        device = ClimaComms.device(context),
        context,
        hypsography_fun,
        global_geometry,
        enable_bubble = bubble,
        h_topology,
    )

    return grid
end

"""
    ColumnGrid(::Type{FT}; kwargs...)

Create a ColumnGrid.

# Arguments

  - `FT`: the floating-point type [`Float32`, `Float64`]

# Keyword Arguments

  - `context = ClimaComms.context()`: the ClimaComms communications context
  - `z_elem = 10`: the number of z-points
  - `z_max = 30000.0`: the domain maximum along the z-direction
  - `z_stretch = false`: whether to stretch the vertical mesh with the two-scale
    rule of [`get_z_mesh`](@ref)
  - `dz_bottom = 500.0`: bottom layer thickness of the two-scale stretching
  - `dz_top = 3000.0`: top layer thickness of the two-scale stretching
  - `z_pbl = 2200.0`: scale height of the boundary-layer refinement of the
    two-scale stretching
  - `z_mesh`: a custom z-mesh; by default, [`get_z_mesh`](@ref) builds it from
    `z_elem`, `z_max`, `z_stretch`, `dz_bottom`, `dz_top`, and `z_pbl`
"""
function ColumnGrid(
    ::Type{FT};
    context = ClimaComms.context(),
    z_elem = 10,
    z_max = 30000.0,
    z_stretch = false,
    dz_bottom = 500.0,
    dz_top = 3000.0,
    z_pbl = 2200.0,
    z_mesh = get_z_mesh(FT; z_max, z_elem, z_stretch, dz_bottom, dz_top, z_pbl),
) where {FT}
    grid = CommonGrids.ColumnGrid(
        FT;
        z_elem, z_min = 0, z_max, z_mesh,
        device = ClimaComms.device(context),
        context,
    )

    return grid
end

"""
    MultiColumnGrid(::Type{FT}; n_columns, kwargs...)

Create a grid of `n_columns` independent columns, each with the vertical
discretization of a [`ColumnGrid`](@ref).

# Arguments

  - `FT`: the floating-point type [`Float32`, `Float64`]

# Keyword Arguments

  - `n_columns`: the number of columns
  - `context = ClimaComms.context()`: the ClimaComms communications context
  - `z_elem = 10`: the number of z-points
  - `z_max = 30000.0`: the domain maximum along the z-direction
  - `z_stretch = false`: whether to stretch the vertical mesh with the two-scale
    rule of [`get_z_mesh`](@ref)
  - `dz_bottom = 500.0`: bottom layer thickness of the two-scale stretching
  - `dz_top = 3000.0`: top layer thickness of the two-scale stretching
  - `z_pbl = 2200.0`: scale height of the boundary-layer refinement of the
    two-scale stretching
  - `z_mesh`: a custom z-mesh; by default, [`get_z_mesh`](@ref) builds it from
    `z_elem`, `z_max`, `z_stretch`, `dz_bottom`, `dz_top`, and `z_pbl`
"""
function MultiColumnGrid(
    ::Type{FT};
    n_columns,
    context = ClimaComms.context(),
    z_elem = 10,
    z_max = 30000.0,
    z_stretch = false,
    dz_bottom = 500.0,
    dz_top = 3000.0,
    z_pbl = 2200.0,
    z_mesh = get_z_mesh(FT; z_max, z_elem, z_stretch, dz_bottom, dz_top, z_pbl),
) where {FT}
    context isa ClimaComms.SingletonCommsContext ||
        error("Multi-column grids can only be created on Singleton contexts.")
    n_columns ≥ 1 || error("`n_columns` must be at least 1, got $n_columns")
    return CommonGrids.MultiColumnGrid(
        FT;
        # Placeholder coordinates: setups that read latitude or longitude see
        # (0°, 0°)
        points = fill(Geometry.LatLongPoint(FT(0), FT(0)), n_columns),
        z_elem, z_min = 0, z_max, z_mesh,
        device = ClimaComms.device(context),
    )
end

"""
    BoxGrid(::Type{FT}; kwargs...)

Create a Box3DGrid with topography support.

# Arguments

  - `FT`: the floating-point type [`Float32`, `Float64`]

# Keyword Arguments

  - `context = ClimaComms.context()`: the ClimaComms communications context
  - `x_elem = 6`: the number of x-points
  - `x_max = 300000.0`: the domain maximum along the x-direction
  - `y_elem = 6`: the number of y-points
  - `y_max = 300000.0`: the domain maximum along the y-direction
  - `z_elem = 10`: the number of z-points
  - `z_max = 30000.0`: the domain maximum along the z-direction
  - `nh_poly = 3`: the polynomial order. Note: The number of quadrature points in
    1D within each horizontal element is then `n_quad_points = nh_poly + 1`
  - `z_stretch = false`: whether to stretch the vertical mesh with the two-scale
    rule of [`get_z_mesh`](@ref)
  - `dz_bottom = 500.0`: bottom layer thickness for vertical stretching
  - `dz_top = 3000.0`: top layer thickness of the two-scale stretching
  - `z_pbl = 2200.0`: scale height of the boundary-layer refinement of the
    two-scale stretching
  - `z_mesh`: a custom z-mesh; by default, [`get_z_mesh`](@ref) builds it from
    `z_elem`, `z_max`, `z_stretch`, `dz_bottom`, `dz_top`, and `z_pbl`
  - `bubble = false`: enables the "bubble correction" for more accurate element
    areas when computing the spectral element space.
  - `periodic_x = true`: use periodic domain along x-direction
  - `periodic_y = true`: use periodic domain along y-direction
  - `topography = NoTopography()`: topography type
  - `topography_damping_factor = 5.0`: factor by which smallest resolved
    length-scale is to be damped
  - `mesh_warp_type = LinearWarp()`: mesh warping type ([`SLEVEWarp`](@ref) or
    [`LinearWarp`](@ref))
  - `topo_smoothing = false`: apply topography smoothing
"""
function BoxGrid(
    ::Type{FT};
    context = ClimaComms.context(),
    x_elem = 6,
    x_max = 300000.0,
    y_elem = 6,
    y_max = 300000.0,
    z_elem = 10,
    z_max = 30000.0,
    nh_poly = 3,
    z_stretch = false,
    dz_bottom = 500.0,
    dz_top = 3000.0,
    z_pbl = 2200.0,
    bubble = false,
    periodic_x = true,
    periodic_y = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor = 5.0,
    mesh_warp_type::MeshWarpType = LinearWarp(),
    topo_smoothing = false,
    z_mesh = get_z_mesh(FT; z_max, z_elem, z_stretch, dz_bottom, dz_top, z_pbl),
) where {FT}
    n_quad_points = nh_poly + 1
    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type, topo_smoothing,
    )
    grid = CommonGrids.Box3DGrid(
        FT;
        z_elem, x_min = 0, x_max, y_min = 0, y_max, z_min = 0, z_max,
        periodic_x, periodic_y, n_quad_points, x_elem, y_elem,
        device = ClimaComms.device(context),
        context,
        hypsography_fun,
        global_geometry = Geometry.CartesianGlobalGeometry(),
        z_mesh,
        enable_bubble = bubble,
    )

    return grid
end

"""
    PlaneGrid(::Type{FT}; kwargs...)

Create a SliceXZGrid with topography support.

# Arguments

  - `FT`: the floating-point type [`Float32`, `Float64`]

# Keyword Arguments

  - `context = ClimaComms.context()`: the ClimaComms communications context
  - `x_elem = 6`: the number of x-points
  - `x_max = 300000.0`: the domain maximum along the x-direction
  - `z_elem = 10`: the number of z-points
  - `z_max = 30000.0`: the domain maximum along the z-direction
  - `z_mesh`: a custom z-mesh; by default, [`get_z_mesh`](@ref) builds it from
    `z_elem`, `z_max`, `z_stretch`, `dz_bottom`, `dz_top`, and `z_pbl`
  - `nh_poly = 3`: the polynomial order. Note: The number of quadrature points in
    1D within each horizontal element is then `n_quad_points = nh_poly + 1`
  - `z_stretch = false`: whether to stretch the vertical mesh with the two-scale
    rule of [`get_z_mesh`](@ref)
  - `dz_bottom = 500.0`: bottom layer thickness of the two-scale stretching
  - `dz_top = 3000.0`: top layer thickness of the two-scale stretching
  - `z_pbl = 2200.0`: scale height of the boundary-layer refinement of the
    two-scale stretching
  - `periodic_x = true`: use periodic domain along x-direction
  - `topography = NoTopography()`: topography type
  - `topography_damping_factor = 5.0`: factor by which smallest resolved
    length-scale is to be damped
  - `mesh_warp_type = LinearWarp()`: mesh warping type ([`SLEVEWarp`](@ref) or
    [`LinearWarp`](@ref))
  - `topo_smoothing = false`: apply topography smoothing
"""
function PlaneGrid(
    ::Type{FT};
    context = ClimaComms.context(),
    x_elem = 6,
    x_max = 300000.0,
    z_elem = 10,
    z_max = 30000.0,
    nh_poly = 3,
    z_stretch = false,
    dz_bottom = 500.0,
    dz_top = 3000.0,
    z_pbl = 2200.0,
    periodic_x = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor = 5.0,
    mesh_warp_type::MeshWarpType = LinearWarp(),
    topo_smoothing = false,
    z_mesh = get_z_mesh(FT; z_max, z_elem, z_stretch, dz_bottom, dz_top, z_pbl),
) where {FT}
    n_quad_points = nh_poly + 1
    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type, topo_smoothing,
    )


    grid = CommonGrids.SliceXZGrid(
        FT;
        z_elem, x_elem, x_min = 0, x_max, z_min = 0, z_max, z_mesh,
        periodic_x,
        n_quad_points,
        device = ClimaComms.device(context),
        context,
        hypsography_fun,
        global_geometry = Geometry.CartesianGlobalGeometry(),
    )

    return grid
end

"""
    hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type, topo_smoothing)

Create the `(h_grid, z_grid) -> hypsography` function that a ClimaCore extruded grid uses
to warp its levels onto the surface elevation.

The returned closure gives `Grids.Flat()` for `NoTopography`. Otherwise it evaluates the
surface elevation on the horizontal space — read from the orography artifact for
`EarthTopography`, and from the analytic profile of the topography type otherwise — and
smooths it: Earth topography is always diffused, with the number of iterations set by
`topography_damping_factor`, and analytic topography only when `topo_smoothing` is set.
The elevation is then turned into a `SLEVEAdaption` or a `LinearAdaption` according to
`mesh_warp_type`.
"""
function hypsography_function_from_topography(
    ::Type{FT},
    topography::AbstractTopography,
    topography_damping_factor,
    mesh_warp_type::MeshWarpType,
    topo_smoothing,
) where {FT}
    return function hypsography(h_grid, z_grid)
        topography isa NoTopography && return Grids.Flat()

        # Create horizontal space to work with topography
        h_space = if h_grid isa Grids.SpectralElementGrid1D
            Spaces.SpectralElementSpace1D(h_grid)
        elseif h_grid isa Grids.SpectralElementGrid2D
            Spaces.SpectralElementSpace2D(h_grid)
        else
            error("Unsupported horizontal grid type $(typeof(h_grid))")
        end

        # Load topography data
        if topography isa EarthTopography
            context = ClimaComms.context(h_space)
            z_surface = SpaceVaryingInput(
                AA.earth_orography_file_path(; context),
                "z",
                h_space,
            )
        else
            z_surface = SpaceVaryingInput(topography_function(topography), h_space)
            @info "Using $(nameof(typeof(topography))) orography"
        end

        if topography isa EarthTopography
            # Diffuse Earth topography to remove small-scale features, using a
            # diffusion Courant number (CFL = νΔt/Δx²) to control smoothing
            diff_courant = FT(0.05)
            Δh_scale = Spaces.node_horizontal_length_scale(h_space)
            κ = FT(diff_courant * Δh_scale^2)
            maxiter = Int(round(log(topography_damping_factor) / diff_courant))
            # Coefficient for horizontal diffusion may alternatively be
            # determined from the empirical parameters suggested by E3SM v1/v2
            # Topography documentation found here:
            # https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/1456603764/V1+Topography+GLL+grids
            Hypsography.diffuse_surface_elevation!(z_surface; κ, dt = FT(1), maxiter)
            @. z_surface = max(z_surface, 0)
        elseif topo_smoothing
            Hypsography.diffuse_surface_elevation!(z_surface)
        end

        # Create hypsography from mesh warp type
        if mesh_warp_type isa SLEVEWarp
            hypsography = Hypsography.SLEVEAdaption(
                Geometry.ZPoint.(z_surface),
                FT(mesh_warp_type.eta),
                FT(mesh_warp_type.s),
            )
        elseif mesh_warp_type isa LinearWarp
            @info "Linear mesh warp"
            hypsography = Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
        else
            error("Undefined mesh-warping option $(nameof(typeof(mesh_warp_type)))")
        end
        return hypsography
    end
end

"""
    get_z_mesh(::Type{FT}; z_max, z_elem, z_stretch, dz_bottom, dz_top, z_pbl)

Build the vertical mesh of `z_elem` elements between the surface and `z_max` [m]
for [`SphereGrid`](@ref), [`ColumnGrid`](@ref), [`BoxGrid`](@ref), and
[`PlaneGrid`](@ref).

With `z_stretch = false`, the elements are uniform. Otherwise, they follow the
two-scale [`ClimaCore.Meshes.BoundaryLayerHyperbolicTangentStretching`](@extref),
which refines the boundary layer with scale height `z_pbl` [m] while matching
the element spacing `dz_bottom` [m] at the surface and `dz_top` [m] at the top.
The rule admits `z_elem` within a range set by `dz_bottom`, `dz_top`, `z_pbl`,
and `z_max`; the mesh constructor throws an `ArgumentError` that names the
range when `z_elem` lies outside it. The 10-level default grid with
`dz_bottom = 500` m up to 30 km lies below the range, so the grid constructors
default to `z_stretch = false`.
"""
function get_z_mesh(
    ::Type{FT};
    z_max,
    z_elem,
    z_stretch,
    dz_bottom,
    dz_top,
    z_pbl,
) where {FT}
    stretch = if z_stretch
        Meshes.BoundaryLayerHyperbolicTangentStretching{FT}(dz_bottom, dz_top, z_pbl)
    else
        Meshes.Uniform()
    end
    return CommonGrids.DefaultZMesh(FT; z_min = 0, z_max, z_elem, stretch)
end

"""
    z_faces_mesh(::Type{FT}, z_faces, z_max; dz_min = 0.1)

Build a vertical mesh from the explicit face heights `z_faces` [m], which must rise
strictly from `0` to `z_max`, each layer at least `dz_min` [m] thick; one element lies
between each pair of neighbouring faces. Throws an `ArgumentError` naming the first face
that breaks these rules.
"""
function z_faces_mesh(::Type{FT}, z_faces, z_max; dz_min = 0.1) where {FT}
    faces = Float64.(collect(z_faces))
    length(faces) >= 2 ||
        throw(ArgumentError("`z_faces` needs at least two faces; got $(length(faces))"))
    faces[1] == 0 ||
        throw(ArgumentError("`z_faces` must start at 0 m; the first face is $(faces[1]) m"))
    faces[end] == z_max || throw(
        ArgumentError(
            "`z_faces` must end at `z_max` = $(z_max) m; the last face is $(faces[end]) m",
        ),
    )
    k = findfirst(i -> faces[i + 1] <= faces[i], 1:(length(faces) - 1))
    isnothing(k) || throw(
        ArgumentError(
            "`z_faces` must rise strictly; face $(k + 1) ($(faces[k + 1]) m) is not \
             above face $k ($(faces[k]) m)",
        ),
    )
    k = findfirst(i -> faces[i + 1] - faces[i] < dz_min, 1:(length(faces) - 1))
    isnothing(k) || throw(
        ArgumentError(
            "`z_faces` layers must be at least $(dz_min) m thick; faces $k and \
             $(k + 1) ($(faces[k]) m, $(faces[k + 1]) m) are $(faces[k + 1] - faces[k]) m apart",
        ),
    )
    domain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = (:bottom, :top),
    )
    return Meshes.IntervalMesh(domain, Geometry.ZPoint{FT}.(faces))
end

"""
    hybrid_z_faces(; dz_strat = 500, z_strat_bottom = 11000, z_strat_top = 35000,
                   z_max = 60000, dz_max = 3000, max_change = 0.1)

Face heights [m] of a hybrid vertical grid with uniform spacing `dz_strat` in the
stratosphere, for resolving the QBO:

 1. The faces of the 63-level two-scale grid (`z_max` 60 km, `dz_bottom` 30 m,
    `dz_top` 3 km, `z_pbl` 2.2 km, the grid of the 1M aquaplanet Beres runs), up to the
    face from which the spacing, shrinking to `dz_strat` by at most `max_change` per
    level, reaches `dz_strat` closest to `z_strat_bottom`.
 2. That shrinking: a geometric sequence from the last spacing of the 63-level grid to
    `dz_strat` in as few levels as `max_change` allows, so the spacing has no jump.
    (The inherited 63-level faces themselves change by up to 10.6% per level, near
    3.9 km; `max_change` bounds only the shrinking.)
 3. Uniform `dz_strat` up to the last face at or below `z_strat_top`.
 4. Spacing growing geometrically to `z_max`, in the number of levels whose ratio makes
    the top spacing closest to `dz_max`; the ratio is solved so the last face is
    exactly `z_max`.

The 63-level base and the stretching to the top are built for `z_max = 60` km; any other
`z_max` is an `ArgumentError`. The defaults (the `hybrid_0p5km` preset of the `z_faces`
config key) give 106 levels: the
63-level faces up to 8.1 km (37 levels, the last 745 m thick), shrinking by 7.7% per
level to 500 m at 10.6 km (5 levels, the last 500 m thick), 500 m up to 34.6 km (47
more levels), and growing by 11.3% per level to 3.08 km at 60 km (17 levels). From 500 m
at 35 km, a 9% growth would reach only about 2.3 km at 60 km.
"""
function hybrid_z_faces(;
    dz_strat = 500.0,
    z_strat_bottom = 11000.0,
    z_strat_top = 35000.0,
    z_max = 60000.0,
    dz_max = 3000.0,
    max_change = 0.1,
)
    z_max == 60000 || throw(
        ArgumentError(
            "the hybrid faces are built on the 60 km 63-level grid; got z_max = $(z_max) m",
        ),
    )
    base_mesh = CommonGrids.DefaultZMesh(
        Float64;
        z_min = 0,
        z_max = 60000.0,
        z_elem = 63,
        stretch = Meshes.BoundaryLayerHyperbolicTangentStretching{Float64}(
            30.0,
            3000.0,
            2200.0,
        ),
    )
    base = [Geometry.component(f, 1) for f in base_mesh.faces]
    # The spacings shrinking from `dz` to `dz_strat` by at most `max_change` per level.
    taper(dz) =
        dz <= dz_strat ? Float64[] :
        (n = ceil(Int, log(dz / dz_strat) / log(1 + max_change));
            [dz * (dz_strat / dz)^(i / n) for i in 1:n])
    # The shared face from which the shrinking ends closest to z_strat_bottom.
    taper_end(k) = base[k] + sum(taper(base[k] - base[k - 1]); init = 0.0)
    k = argmin(k -> abs(taper_end(k) - z_strat_bottom), 2:length(base))
    faces = base[1:k]
    for dz in taper(base[k] - base[k - 1])
        push!(faces, faces[end] + dz)
    end
    while faces[end] + dz_strat <= z_strat_top + 1e-6
        push!(faces, faces[end] + dz_strat)
    end
    # Geometric growth over m levels from dz_strat, ratio r: dz_strat Σᵢ rⁱ = remaining.
    z_join = faces[end]
    remaining = z_max - z_join
    function ratio(m)
        g(r) = dz_strat * r * (r^m - 1) / (r - 1) - remaining
        lo, hi = 1 + 1e-9, 2.0
        for _ in 1:200
            mid = (lo + hi) / 2
            g(mid) < 0 ? (lo = mid) : (hi = mid)
        end
        return (lo + hi) / 2
    end
    ms = filter(m -> dz_strat * m < remaining, 1:200)
    m = argmin(m -> abs(dz_strat * ratio(m)^m - dz_max), ms)
    r = ratio(m)
    for i in 1:m
        push!(faces, z_join + dz_strat * r * (r^i - 1) / (r - 1))
    end
    faces[end] = z_max
    return faces
end

"""
    z_faces_from_config(z_faces, z_max)

The face heights the `z_faces` config key selects: `nothing` for none, the faces of a
preset for its name (`"hybrid_0p5km"`, see `hybrid_z_faces`), or the list itself.
"""
z_faces_from_config(::Nothing, z_max) = nothing
z_faces_from_config(z_faces::AbstractVector, z_max) = Float64.(z_faces)
function z_faces_from_config(z_faces::AbstractString, z_max)
    if z_faces == "hybrid_0p5km"
        z_max == 60000 || throw(
            ArgumentError(
                "the `z_faces` preset \"hybrid_0p5km\" needs `z_max` = 60000 m; got $(z_max) m",
            ),
        )
        return hybrid_z_faces()
    end
    throw(
        ArgumentError(
            "Unknown `z_faces` preset \"$(z_faces)\"; give a list of face heights or \
             \"hybrid_0p5km\"",
        ),
    )
end

"""
    get_spaces(grid)

Create the center and face spaces of a ClimaCore grid, returned as
`(; center_space, face_space)`.

Supports every grid with a vertical staggering: extruded grids (sphere, box, plane),
single-column finite difference grids, and multi-column grids.
"""
function get_spaces(grid)
    center_space = Spaces.space(grid, Grids.CellCenter())
    face_space = Spaces.space(grid, Grids.CellFace())
    return (; center_space, face_space)
end
