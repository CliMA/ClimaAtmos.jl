import ClimaCore: Geometry, Hypsography, Fields, Spaces, Meshes, Grids, CommonGrids
using ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput
import .AtmosArtifacts as AA
import ClimaComms

export SphereGrid, ColumnGrid, BoxGrid, PlaneGrid

"""
    SphereGrid(::Type{FT}; kwargs...)

Create an ExtrudedCubedSphereGrid with topography support.

# Arguments
- `FT`: the floating-point type [`Float32`, `Float64`]

# Keyword Arguments
- `context = ClimaComms.context()`: the ClimaComms communications context
- `z_elem = 10`: the number of z-points
- `z_max = 30000.0`: the domain maximum along the z-direction
- `z_stretch = true`: whether to use vertical stretching
- `dz_bottom = 500.0`: bottom layer thickness for stretching
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
    z_stretch = true,
    dz_bottom = 500.0,
    radius = 6.371229e6,
    h_elem = 6,
    nh_poly = 3,
    bubble = false,
    deep_atmosphere = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor = 5.0,
    mesh_warp_type::MeshWarpType = SLEVEWarp{FT}(),
    topo_smoothing = false,
) where {FT}
    n_quad_points = nh_poly + 1
    stretch =
        z_stretch ? Meshes.HyperbolicTangentStretching{FT}(dz_bottom) : Meshes.Uniform()
    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type, topo_smoothing,
    )

    global_geometry = if deep_atmosphere
        Geometry.DeepSphericalGlobalGeometry{FT}(radius)
    else
        Geometry.ShallowSphericalGlobalGeometry{FT}(radius)
    end

    grid = CommonGrids.ExtrudedCubedSphereGrid(
        FT;
        z_elem, z_min = 0, z_max, radius, h_elem,
        n_quad_points,
        device = ClimaComms.device(context),
        context,
        stretch,
        hypsography_fun,
        global_geometry,
        enable_bubble = bubble,
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
- `z_stretch = true`: whether to use vertical stretching
- `dz_bottom = 500.0`: bottom layer thickness for stretching
"""
function ColumnGrid(
    ::Type{FT};
    context = ClimaComms.context(),
    z_elem = 10,
    z_max = 30000.0,
    z_stretch = true,
    dz_bottom = 500.0,
) where {FT}
    stretch =
        z_stretch ? Meshes.HyperbolicTangentStretching{FT}(dz_bottom) : Meshes.Uniform()
    z_mesh = CommonGrids.DefaultZMesh(FT; z_min = 0, z_max, z_elem, stretch)
    grid = CommonGrids.ColumnGrid(
        FT;
        z_elem, z_min = 0, z_max, z_mesh,
        device = ClimaComms.device(context),
        context,
        stretch,
    )

    return grid
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
- `z_stretch = true`: whether to use vertical stretching
- `dz_bottom = 500.0`: bottom layer thickness for stretching
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
    z_stretch = true,
    dz_bottom = 500.0,
    bubble = false,
    periodic_x = true,
    periodic_y = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor = 5.0,
    mesh_warp_type::MeshWarpType = LinearWarp(),
    topo_smoothing = false,
) where {FT}
    n_quad_points = nh_poly + 1
    stretch =
        z_stretch ? Meshes.HyperbolicTangentStretching{FT}(dz_bottom) : Meshes.Uniform()
    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type, topo_smoothing,
    )
    z_mesh = CommonGrids.DefaultZMesh(FT; z_min = 0, z_max, z_elem, stretch)
    grid = CommonGrids.Box3DGrid(
        FT;
        z_elem, x_min = 0, x_max, y_min = 0, y_max, z_min = 0, z_max,
        periodic_x, periodic_y, n_quad_points, x_elem, y_elem,
        device = ClimaComms.device(context),
        context,
        stretch,
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
- `nh_poly = 3`: the polynomial order. Note: The number of quadrature points in
  1D within each horizontal element is then `n_quad_points = nh_poly + 1`
- `z_stretch = true`: whether to use vertical stretching
- `dz_bottom = 500.0`: bottom layer thickness for stretching
- `bubble = false`: enables the "bubble correction" for more accurate element
  areas when computing the spectral element space. Note: Currently not supported
  by SliceXZGrid in ClimaCore.
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
    z_stretch = true,
    dz_bottom = 500.0,
    bubble = false,
    periodic_x = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor = 5.0,
    mesh_warp_type::MeshWarpType = LinearWarp(),
    topo_smoothing = false,
) where {FT}
    n_quad_points = nh_poly + 1
    stretch =
        z_stretch ? Meshes.HyperbolicTangentStretching{FT}(dz_bottom) : Meshes.Uniform()
    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type, topo_smoothing,
    )
    z_mesh = CommonGrids.DefaultZMesh(FT; z_min = 0, z_max, z_elem, stretch)

    grid = CommonGrids.SliceXZGrid(
        FT;
        z_elem, x_elem, x_min = 0, x_max, z_min = 0, z_max, z_mesh,
        periodic_x,
        n_quad_points,
        device = ClimaComms.device(context),
        context,
        stretch,
        hypsography_fun,
        global_geometry = Geometry.CartesianGlobalGeometry(),
    )

    return grid
end

"""
    hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type, topo_smoothing)

Create a hypsography function that handles topography integration.
"""
function hypsography_function_from_topography(
    ::Type{FT},
    topography::AbstractTopography,
    topography_damping_factor,
    mesh_warp_type::MeshWarpType,
    topo_smoothing,
) where {FT}
    function hypsography(h_grid, z_grid)
        topography isa NoTopography && return Hypsography.Flat()

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
            @info "Remapping Earth orography from ETOPO2022 data onto horizontal space"
        else
            z_surface = SpaceVaryingInput(topography_function(topography), h_space)
            @info "Using $(nameof(typeof(topography))) orography"
        end

        if topography isa EarthTopography
            # Diffuse Earth topography to remove small-scale features Using a
            # diffusion Courant number (CFL = νΔt/Δx²) to control smoothing
            diff_courant = FT(0.05)
            Δh_scale = Spaces.node_horizontal_length_scale(h_space)
            κ = FT(diff_courant * Δh_scale^2)
            maxiter = Int(round(log(topography_damping_factor) / diff_courant))
            Hypsography.diffuse_surface_elevation!(z_surface; κ, dt = FT(1), maxiter)
            # Coefficient for horizontal diffusion may alternatively be
            # determined from the empirical parameters suggested by E3SM v1/v2
            # Topography documentation found here:
            # https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/1456603764/V1+Topography+GLL+grids
            @. z_surface = max(z_surface, 0)
        elseif topo_smoothing
            # Apply optional smoothing for other topography types
            Hypsography.diffuse_surface_elevation!(z_surface)
        end

        # Create hypsography from mesh warp type
        if mesh_warp_type isa SLEVEWarp
            @info "SLEVE mesh warp (eta=$(mesh_warp_type.eta), s=$(mesh_warp_type.s))"
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
    return hypsography
end
