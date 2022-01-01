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
