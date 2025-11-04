using ClimaCore: Geometry, Hypsography, Fields, Spaces, Meshes
using ClimaCore.CommonGrids:
    ExtrudedCubedSphereGrid, ColumnGrid, Box3DGrid, SliceXZGrid, DefaultZMesh
using ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput
import .AtmosArtifacts as AA
import ClimaComms

export SphereGrid, ColGrid, BoxGrid, PlaneGrid

abstract type MeshWarpType end

struct LinearWarp <: MeshWarpType end
struct SLEVEWarp <: MeshWarpType end

"""
    SphereGrid(FT; kwargs...)

Create an ExtrudedCubedSphereGrid with topography support.

# Arguments
- `FT`: Floating point type

# Keyword Arguments
- `context`: Communications context
- `z_elem::Int`: Number of vertical elements
- `z_max::Real`: Maximum height
- `z_stretch::Bool`: Whether to use vertical stretching
- `dz_bottom::Real`: Bottom layer thickness for stretching
- `radius::Real`: Earth radius
- `h_elem::Int`: Number of horizontal elements per panel
- `nh_poly::Int`: Polynomial order
- `bubble::Bool`: Enable bubble correction
- `deep_atmosphere::Bool`: Enable deep atmosphere
- `topography::AbstractTopography`: Topography type
- `topography_damping_factor::Real`: Topography damping
- `mesh_warp_type`: MeshWarpType: Mesh warping type ("SLEVE" or "Linear")
- `sleve_eta::Real`: SLEVE parameter
- `sleve_s::Real`: SLEVE parameter
- `topo_smoothing::Bool`: Apply topography smoothing
"""
function SphereGrid(
    FT;
    context = ClimaComms.context(),
    z_elem::Int = 10,
    z_max::Real = 30000.0,
    z_stretch::Bool = true,
    dz_bottom::Real = 500.0,
    radius::Real = 6.371229e6,
    h_elem::Int = 6,
    nh_poly::Int = 3,
    bubble::Bool = false,
    deep_atmosphere::Bool = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor::Real = 5.0,
    mesh_warp_type::MeshWarpType = SLEVEWarp(),
    sleve_eta::Real = 0.7,
    sleve_s::Real = 10.0,
    topo_smoothing::Bool = false,
)
    n_quad_points = nh_poly + 1

    stretch =
        z_stretch ? Meshes.HyperbolicTangentStretching{FT}(dz_bottom) : Meshes.Uniform()

    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type,
        sleve_eta, sleve_s, topo_smoothing,
    )

    global_geometry = if deep_atmosphere
        Geometry.DeepSphericalGlobalGeometry{FT}(radius)
    else
        Geometry.ShallowSphericalGlobalGeometry{FT}(radius)
    end

    grid = ExtrudedCubedSphereGrid(
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
    ColGrid(FT; kwargs...)

Create a ColumnGrid.

# Arguments
- `FT`: Floating point type

# Keyword Arguments
- `context`: Communications context
- `z_elem::Int`: Number of vertical elements
- `z_max::Real`: Maximum height
- `z_stretch::Bool`: Whether to use vertical stretching
- `dz_bottom::Real`: Bottom layer thickness for stretching
"""
function ColGrid(
    FT;
    context = ClimaComms.context(),
    z_elem::Int = 10,
    z_max::Real = 30000.0,
    z_stretch::Bool = true,
    dz_bottom::Real = 500.0,
)
    stretch =
        z_stretch ? Meshes.HyperbolicTangentStretching{FT}(dz_bottom) : Meshes.Uniform()
    z_mesh = DefaultZMesh(
        FT;
        z_min = 0,
        z_max,
        z_elem,
        stretch,
    )
    grid = ColumnGrid(
        FT;
        z_elem, z_min = 0, z_max, z_mesh,
        device = ClimaComms.device(context),
        context,
        stretch,
    )

    return grid
end

"""
    BoxGrid(FT; kwargs...)

Create a Box3DGrid with topography support.

# Arguments
- `FT`: Floating point type

# Keyword Arguments
- `context`: Communications context
- `x_elem::Int`: Number of x elements
- `x_max::Real`: Maximum x coordinate
- `y_elem::Int`: Number of y elements
- `y_max::Real`: Maximum y coordinate
- `z_elem::Int`: Number of vertical elements
- `z_max::Real`: Maximum height
- `nh_poly::Int`: Polynomial order
- `z_stretch::Bool`: Whether to use vertical stretching
- `dz_bottom::Real`: Bottom layer thickness for stretching
- `bubble::Bool`: Enable bubble correction
- `deep_atmosphere::Bool`: Enable deep atmosphere
- `periodic_x::Bool`: Periodic in x direction
- `periodic_y::Bool`: Periodic in y direction
- `topography::AbstractTopography`: Topography type
- `topography_damping_factor::Real`: Topography damping
- `mesh_warp_type::String`: Mesh warping type ("SLEVE" or "Linear")
- `sleve_eta::Real`: SLEVE parameter
- `sleve_s::Real`: SLEVE parameter
- `topo_smoothing::Bool`: Apply topography smoothing
"""
function BoxGrid(
    FT;
    context = ClimaComms.context(),
    x_elem::Int = 6,
    x_max::Real = 300000.0,
    y_elem::Int = 6,
    y_max::Real = 300000.0,
    z_elem::Int = 10,
    z_max::Real = 30000.0,
    nh_poly::Int = 3,
    z_stretch::Bool = true,
    dz_bottom::Real = 500.0,
    bubble::Bool = false,
    deep_atmosphere::Bool = true,
    periodic_x::Bool = true,
    periodic_y::Bool = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor::Real = 5.0,
    mesh_warp_type::MeshWarpType = LinearWarp(),
    sleve_eta::Real = 0.7,
    sleve_s::Real = 10.0,
    topo_smoothing::Bool = false,
)
    n_quad_points = nh_poly + 1

    stretch =
        z_stretch ? Meshes.HyperbolicTangentStretching{FT}(dz_bottom) : Meshes.Uniform()

    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type,
        sleve_eta, sleve_s, topo_smoothing,
    )
    z_mesh = DefaultZMesh(
        FT;
        z_min = 0,
        z_max,
        z_elem,
        stretch,
    )
    grid = Box3DGrid(
        FT;
        z_elem, x_min = 0, x_max, y_min = 0, y_max, z_min = 0, z_max, z_mesh,
        periodic_x, periodic_y, n_quad_points, x_elem, y_elem,
        device = ClimaComms.device(context),
        context,
        stretch,
        hypsography_fun,
        global_geometry = Geometry.CartesianGlobalGeometry(),
        enable_bubble = bubble,
    )

    return grid
end

"""
    PlaneGrid(FT; kwargs...)

Create a SliceXZGrid with topography support.

# Arguments
- `FT`: Floating point type

# Keyword Arguments
- `context`: Communications context
- `x_elem::Int`: Number of x elements
- `x_max::Real`: Maximum x coordinate
- `z_elem::Int`: Number of vertical elements
- `z_max::Real`: Maximum height
- `nh_poly::Int`: Polynomial order
- `z_stretch::Bool`: Whether to use vertical stretching
- `dz_bottom::Real`: Bottom layer thickness for stretching
- `bubble::Bool`: Enable bubble correction
- `deep_atmosphere::Bool`: Enable deep atmosphere
- `periodic_x::Bool`: Periodic in x direction
- `topography::AbstractTopography`: Topography type
- `topography_damping_factor::Real`: Topography damping
- `mesh_warp_type::String`: Mesh warping type ("SLEVE" or "Linear")
- `sleve_eta::Real`: SLEVE parameter
- `sleve_s::Real`: SLEVE parameter
- `topo_smoothing::Bool`: Apply topography smoothing
"""
function PlaneGrid(
    FT;
    context = ClimaComms.context(),
    x_elem::Int = 6,
    x_max::Real = 300000.0,
    z_elem::Int = 10,
    z_max::Real = 30000.0,
    nh_poly::Int = 3,
    z_stretch::Bool = true,
    dz_bottom::Real = 500.0,
    bubble::Bool = false,
    deep_atmosphere::Bool = true,
    periodic_x::Bool = true,
    topography::AbstractTopography = NoTopography(),
    topography_damping_factor::Real = 5.0,
    mesh_warp_type::MeshWarpType = LinearWarp(),
    sleve_eta::Real = 0.7,
    sleve_s::Real = 10.0,
    topo_smoothing::Bool = false,
)
    n_quad_points = nh_poly + 1

    stretch =
        z_stretch ? Meshes.HyperbolicTangentStretching{FT}(dz_bottom) : Meshes.Uniform()

    hypsography_fun = hypsography_function_from_topography(
        FT, topography, topography_damping_factor, mesh_warp_type,
        sleve_eta, sleve_s, topo_smoothing,
    )

    z_mesh = DefaultZMesh(
        FT;
        z_min = 0,
        z_max,
        z_elem,
        stretch,
    )

    grid = SliceXZGrid(
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
        topography, topography_damping_factor, mesh_warp_type, 
        sleve_eta, sleve_s, topo_smoothing, context)

Create a hypsography function that handles topography integration.
"""
function hypsography_function_from_topography(
    FT::Type{<:AbstractFloat},
    topography::AbstractTopography,
    topography_damping_factor::Real,
    mesh_warp_type,
    sleve_eta::Real,
    sleve_s::Real,
    topo_smoothing::Bool,
)
    return function (h_grid, z_grid)
        topography isa NoTopography && return Hypsography.Flat()

        # Create horizontal space to work with topography
        h_space = if h_grid isa Grids.SpectralElementGrid1D
            Spaces.SpectralElementSpace1D(h_grid)
        elseif h_grid isa Grids.SpectralElementGrid2D
            Spaces.SpectralElementSpace2D(h_grid)
        else
            error("Unsupported horizontal grid type $(typeof(h_grid))")
        end

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
            mask(x::FT) where {FT} = x * FT(x > 0)
            z_surface = @. mask(z_surface)
            # diff_cfl = νΔt/Δx²
            diff_courant = 0.05 # Arbitrary example value.
            Δh_scale = Spaces.node_horizontal_length_scale(h_space)
            κ = FT(diff_courant * Δh_scale^2)
            maxiter = Int(round(log(topography_damping_factor) / diff_courant))
            Hypsography.diffuse_surface_elevation!(
                z_surface;
                κ,
                dt = FT(1),
                maxiter,
            )
            # Coefficient for horizontal diffusion may alternatively be
            # determined from the empirical parameters suggested by
            # E3SM  v1/v2 Topography documentation found here: 
            # https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/1456603764/V1+Topography+GLL+grids
            z_surface = @. mask(z_surface)
            if mesh_warp_type isa SLEVEWarp
                @info "SLEVE mesh warp"
                hypsography = Hypsography.SLEVEAdaption(
                    Geometry.ZPoint.(z_surface),
                    FT(sleve_eta),
                    FT(sleve_s),
                )
            elseif mesh_warp_type isa LinearWarp
                @info "Linear mesh warp"
                hypsography =
                    Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
            else
                error("Undefined mesh-warping option $(nameof(typeof(mesh_warp_type)))")
            end
        else
            # DCMIP200Topography, Hughes2023Topography, etc.
            if topo_smoothing
                Hypsography.diffuse_surface_elevation!(z_surface)
            end
            if mesh_warp_type isa SLEVEWarp
                @info "SLEVE mesh warp"
                hypsography = Hypsography.SLEVEAdaption(
                    Geometry.ZPoint.(z_surface),
                    FT(sleve_eta),
                    FT(sleve_s),
                )
            elseif mesh_warp_type isa LinearWarp
                @info "Linear mesh warp"
                hypsography =
                    Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
            end
        end
        return hypsography
    end
end
