function make_function_space(domain::Column)
    column = ClimaCore.Domains.IntervalDomain(
        Geometry.ZPoint(domain.zlim[1])..Geometry.ZPoint(domain.zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(column; nelems = domain.nelements)
    center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

function make_function_space(domain::Plane)
    rectangle = ClimaCore.Domains.RectangleDomain(
        Geometry.XPoint(domain.xlim[1])..Geometry.XPoint(domain.xlim[2]),
        Geometry.YPoint(domain.ylim[1])..Geometry.YPoint(domain.ylim[2]);
        x1periodic = domain.periodic[1],
        x2periodic = domain.periodic[2],
        x1boundary = domain.periodic[1] ? nothing : (:left, :right),
        x2boundary = domain.periodic[2] ? nothing : (:left, :right),
    )
    mesh = Meshes.EquispacedRectangleMesh(rectangle, domain.nelements...)
    grid_topology = Topologies.GridTopology(mesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    return space
end

function make_function_space(domain::HybridPlane)
    vertdomain = ClimaCore.Domains.IntervalDomain(
        Geometry.ZPoint(domain.zlim[1])..Geometry.ZPoint(domain.zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = domain.nelements[2])
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = ClimaCore.Domains.IntervalDomain(
        Geometry.XPoint(domain.xlim[1])..Geometry.XPoint(domain.xlim[2]);
        periodic = domain.xperiodic,
        boundary_tags = domain.xperiodic ? nothing : (:left, :right),
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = domain.nelements[1])
    horztopology = Topologies.IntervalTopology(horzmesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    return hv_center_space, hv_face_space
end

function make_function_space(domain::Sphere)
    sphere = ClimaCore.Domains.SphereDomain(domain.radius)
    mesh =
        Meshes.Mesh2D(sphere, Meshes.EquiangularSphereWarp(), domain.nelements)
    grid_topology = Topologies.Grid2DTopology(mesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    return space
end
