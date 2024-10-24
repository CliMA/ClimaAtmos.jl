using ClimaCore:
    Geometry, Domains, Meshes, Topologies, Spaces, Grids, Hypsography, Fields
using ClimaComms

function periodic_line_mesh(; x_max, x_elem)
    domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    return Meshes.IntervalMesh(domain; nelems = x_elem)
end

function periodic_rectangle_mesh(; x_max, y_max, x_elem, y_elem)
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(zero(y_max)),
        Geometry.YPoint(y_max);
        periodic = true,
    )
    domain = Domains.RectangleDomain(x_domain, y_domain)
    return Meshes.RectilinearMesh(domain, x_elem, y_elem)
end

# h_elem is the number of elements per side of every panel (6 panels in total)
function cubed_sphere_mesh(; radius, h_elem)
    domain = Domains.SphereDomain(radius)
    return Meshes.EquiangularCubedSphere(domain, h_elem)
end

function make_horizontal_space(
    mesh,
    quad,
    comms_ctx::ClimaComms.SingletonCommsContext,
    bubble,
)
    if mesh isa Meshes.AbstractMesh1D
        topology = Topologies.IntervalTopology(comms_ctx, mesh)
        space = Spaces.SpectralElementSpace1D(topology, quad)
    elseif mesh isa Meshes.AbstractMesh2D
        topology = Topologies.Topology2D(
            comms_ctx,
            mesh,
            Topologies.spacefillingcurve(mesh),
        )
        space = Spaces.SpectralElementSpace2D(
            topology,
            quad;
            enable_bubble = bubble,
        )
    end
    return space
end

function make_horizontal_space(mesh, quad, comms_ctx, bubble)
    if mesh isa Meshes.AbstractMesh1D
        error("Distributed mode does not work with 1D horizontal spaces.")
    elseif mesh isa Meshes.AbstractMesh2D
        topology = Topologies.DistributedTopology2D(
            comms_ctx,
            mesh,
            Topologies.spacefillingcurve(mesh),
        )
        space = Spaces.SpectralElementSpace2D(
            topology,
            quad;
            enable_bubble = bubble,
        )
    end
    return space
end

function diffuse_surface_elevation_biharmonic!(
    f::Fields.Field;
    κ::T = 5e8,
    maxiter::Int = 100,
    dt::T = 5e-2,
) where {T}
    if eltype(f) <: Real
        f_z = f
    elseif eltype(f) <: Geometry.ZPoint
        f_z = f.z
    end
    # Define required ops
    wdiv = Operators.WeakDivergence()
    grad = Operators.Gradient()
    # Create dss buffer
    ghost_buffer = (bf = Spaces.create_dss_buffer(f_z),)
    # Apply smoothing
    χf = @. wdiv(grad(f_z))
    Spaces.weighted_dss!(χf, ghost_buffer.bf)
    @. χf = wdiv(grad(χf))
    for iter in 1:maxiter
        # Euler steps
        if iter ≠ 1
            @. χf = wdiv(grad(f_z))
            Spaces.weighted_dss!(χf, ghost_buffer.bf)
            @. χf = wdiv(grad(χf))
        end
        Spaces.weighted_dss!(χf, ghost_buffer.bf)
        @. f_z -= κ * dt * χf
    end
    # Return mutated surface elevation profile
    return f
end

function make_hybrid_spaces(
    h_space,
    z_max,
    z_elem,
    z_stretch;
    surface_warp = nothing,
    topo_smoothing = false,
    deep = false,
    parsed_args = nothing,
)
    FT = eltype(z_max)
    # TODO: change this to make_hybrid_grid
    h_grid = Spaces.grid(h_space)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, z_stretch; nelems = z_elem)
    @info "z heights" z_mesh.faces
    device = ClimaComms.device(h_space)
    z_topology = Topologies.IntervalTopology(
        ClimaComms.SingletonCommsContext(device),
        z_mesh,
    )
    z_grid = Grids.FiniteDifferenceGrid(z_topology)
    if isnothing(surface_warp) && parsed_args["topography"] != "Earth"
        hypsography = Hypsography.Flat()
    elseif isnothing(surface_warp) && parsed_args["topography"] == "Earth"
        @info "SpaceVaryingInputs: Remapping orography onto spectral space"
        z_surface = SpaceVaryingInputs.SpaceVaryingInput(
            AA.earth_orography_file_path(;context=ClimaComms.context(h_space)), "z", h_space
        )
        parent(z_surface) .= ifelse.(parent(z_surface) .< FT(0), FT(0), parent(z_surface))
        Δh_scale = Spaces.node_horizontal_length_scale(h_space)
        diffuse_surface_elevation_biharmonic!(z_surface; κ=FT((Δh_scale)^4/1000), dt=FT(1), maxiter=128)
        parent(z_surface) .= ifelse.(parent(z_surface) .< FT(0), FT(0), parent(z_surface))
        if parsed_args["mesh_warp_type"] == "SLEVE"
            @info "SLEVE mesh warp"
            hypsography = Hypsography.SLEVEAdaption(
                Geometry.ZPoint.(z_surface),
                FT(parsed_args["sleve_eta"]),
                FT(parsed_args["sleve_s"]),
            )
        elseif parsed_args["mesh_warp_type"] == "Linear"
            @info "Linear mesh warp"
            hypsography =
                Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
        end
    else
        topo_smoothing = parsed_args["topo_smoothing"]
        z_surface = surface_warp(Fields.coordinate_field(h_space))
        if topo_smoothing
            Hypsography.diffuse_surface_elevation!(z_surface)
        end
        if parsed_args["mesh_warp_type"] == "SLEVE"
            @info "SLEVE mesh warp"
            hypsography = Hypsography.SLEVEAdaption(
                Geometry.ZPoint.(z_surface),
                FT(parsed_args["sleve_eta"]),
                FT(parsed_args["sleve_s"]),
            )
        elseif parsed_args["mesh_warp_type"] == "Linear"
            @info "Linear mesh warp"
            hypsography =
                Hypsography.LinearAdaption(Geometry.ZPoint.(z_surface))
        end
    end
    grid = Grids.ExtrudedFiniteDifferenceGrid(h_grid, z_grid, hypsography; deep)
    # TODO: return the grid
    center_space = Spaces.CenterExtrudedFiniteDifferenceSpace(grid)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(grid)
    return center_space, face_space
end
