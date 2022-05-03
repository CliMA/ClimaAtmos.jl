using ClimaCore: Geometry, Domains, Meshes, Topologies, Spaces

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

function make_horizontal_space(mesh, quad, ::Nothing)
    if mesh isa Meshes.AbstractMesh1D
        topology = Topologies.IntervalTopology(mesh)
        space = Spaces.SpectralElementSpace1D(topology, quad)
    elseif mesh isa Meshes.AbstractMesh2D
        topology = Topologies.Topology2D(mesh)
        space = Spaces.SpectralElementSpace2D(topology, quad)
    end
    return space
end

function make_horizontal_space(mesh, quad, comms_ctx)
    if mesh isa Meshes.AbstractMesh1D
        error("Distributed mode does not work with 1D horizontal spaces.")
    elseif mesh isa Meshes.AbstractMesh2D
        topology = Topologies.DistributedTopology2D(comms_ctx, mesh)
        space = Spaces.SpectralElementSpace2D(topology, quad)
    end
    return space
end

function make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(zero(z_max)),
        Geometry.ZPoint(z_max);
        boundary_tags = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain, z_stretch; nelems = z_elem)
    @info "z heights" z_mesh.faces
    z_topology = Topologies.IntervalTopology(z_mesh)
    z_space = Spaces.CenterFiniteDifferenceSpace(z_topology)
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)
    return center_space, face_space
end
