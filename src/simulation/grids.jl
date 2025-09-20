"""
Grid constructors that replace the domain-based approach with direct ClimaCore.CommonGrids usage.

These constructors handle topography integration and provide the same interface as the previous
domain system but use ClimaCore grids directly.
"""

using ClimaCore: Geometry, Hypsography, Fields, Spaces, Meshes
using ClimaCore.CommonGrids: ExtrudedCubedSphereGrid, ColumnGrid, Box3DGrid, SliceXZGrid, DefaultZMesh
using ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput
import .AtmosArtifacts as AA
import ClimaComms

export SphereGrid, ColGrid, BoxGrid, PlaneGrid

abstract type MeshWarpType end

struct LinearWarp <: MeshWarpType end
struct SLEVEWarp <: MeshWarpType end

"""
    SphereGrid(FT, params, comms_ctx; kwargs...)

Create an ExtrudedCubedSphereGrid with topography support.

# Arguments
- `FT`: Floating point type
- `params`: ClimaAtmos parameters
- `comms_ctx`: Communications context

# Keyword Arguments
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
    FT,
    params,
    comms_ctx;
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
        topography, topography_damping_factor, mesh_warp_type,
        sleve_eta, sleve_s, topo_smoothing, comms_ctx,
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
        device = ClimaComms.device(comms_ctx),
        context = comms_ctx,
        stretch,
        hypsography_fun,
        global_geometry,
        enable_bubble = bubble,
    )

    return grid
end

"""
    ColGrid(FT, params, comms_ctx; kwargs...)

Create a ColumnGrid.

# Arguments
- `FT`: Floating point type
- `params`: ClimaAtmos parameters
- `comms_ctx`: Communications context

# Keyword Arguments
- `z_elem::Int`: Number of vertical elements
- `z_max::Real`: Maximum height
- `z_stretch::Bool`: Whether to use vertical stretching
- `dz_bottom::Real`: Bottom layer thickness for stretching
"""
function ColGrid(
    FT,
    params,
    comms_ctx;
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
        device = ClimaComms.device(comms_ctx),
        context = comms_ctx,
        stretch,
    )

    return grid
end

"""
    BoxGrid(FT, params, comms_ctx; kwargs...)

Create a Box3DGrid with topography support.

# Arguments
- `FT`: Floating point type
- `params`: ClimaAtmos parameters
- `comms_ctx`: Communications context

# Keyword Arguments
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
    FT,
    params,
    comms_ctx;
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
        topography, topography_damping_factor, mesh_warp_type,
        sleve_eta, sleve_s, topo_smoothing, comms_ctx,
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
        device = ClimaComms.device(comms_ctx),
        context = comms_ctx,
        stretch,
        hypsography_fun,
        global_geometry = Geometry.CartesianGlobalGeometry(),
        enable_bubble = bubble,
    )

    return grid
end

"""
    PlaneGrid(FT, params, comms_ctx; kwargs...)

Create a SliceXZGrid with topography support.

# Arguments
- `FT`: Floating point type
- `params`: ClimaAtmos parameters
- `comms_ctx`: Communications context

# Keyword Arguments
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
    FT,
    params,
    comms_ctx;
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
        topography, topography_damping_factor, mesh_warp_type,
        sleve_eta, sleve_s, topo_smoothing, comms_ctx,
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
        device = ClimaComms.device(comms_ctx),
        context = comms_ctx,
        stretch,
        hypsography_fun,
        global_geometry = Geometry.CartesianGlobalGeometry(),
    )

    return grid
end

"""
    hypsography_function_from_topography(
        topography, topography_damping_factor, mesh_warp_type, 
        sleve_eta, sleve_s, topo_smoothing, comms_ctx)

Create a hypsography function that handles topography integration.
"""
function hypsography_function_from_topography(
    topography::AbstractTopography,
    topography_damping_factor::Real,
    mesh_warp_type,
    sleve_eta::Real,
    sleve_s::Real,
    topo_smoothing::Bool,
    comms_ctx,
)
    # TODO: de-duplicate this with common_spaces.jl
    return function (h_grid, z_grid)
        FT = eltype(h_grid)
        # Create horizontal space to work with topography
        h_topology = Spaces.topology(Spaces.SpectralElementSpace2D(h_grid))
        h_space = Spaces.SpectralElementSpace2D(h_topology, h_grid.quadrature_style)

        if topography isa NoTopography
            z_surface = zeros(h_space)
        elseif topography isa EarthTopography
            z_surface = SpaceVaryingInput(
                AA.earth_orography_file_path(;
                    context = ClimaComms.context(h_space),
                ),
                "z",
                h_space,
            )
            @info "Remapping Earth orography from ETOPO2022 data onto horizontal space"
        else
            z_surface = SpaceVaryingInput(topography_function(topography), h_space)
            @info "Using $(nameof(typeof(topography))) orography"
        end

        if topography isa NoTopography
            hypsography = Hypsography.Flat()
        elseif topography isa EarthTopography
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
                @error "Undefined mesh-warping option"
            end
        else
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
