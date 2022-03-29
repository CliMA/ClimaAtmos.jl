function make_function_space(domain::Column)
    column = ClimaCore.Domains.IntervalDomain(
        Geometry.ZPoint(domain.zlim[1])..Geometry.ZPoint(domain.zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(
        column,
        domain.stretching;
        nelems = domain.nelements,
    )
    center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

function make_function_space(domain::HybridPlane)
    vertdomain = ClimaCore.Domains.IntervalDomain(
        Geometry.ZPoint(domain.zlim[1])..Geometry.ZPoint(domain.zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(
        vertdomain,
        domain.stretching;
        nelems = domain.nelements[2],
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    horzdomain = ClimaCore.Domains.IntervalDomain(
        Geometry.XPoint(domain.xlim[1])..Geometry.XPoint(domain.xlim[2]);
        periodic = domain.xperiodic,
        boundary_tags = domain.xperiodic ? nothing : (:left, :right),
    )
    horzmesh = Meshes.IntervalMesh(horzdomain, nelems = domain.nelements[1])
    horztopology = Topologies.IntervalTopology(horzmesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    if domain.topography isa WarpedSurface
        surface_function = domain.topography.surface_function
        z_surface =
            surface_function.(ClimaCore.Fields.coordinate_field(horzspace))
        hv_face_space = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_face_space,
            domain.topography.interior_warping,
            z_surface,
        )
        hv_center_space =
            Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    else
        hv_face_space = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_face_space,
        )
        hv_center_space =
            Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    end

    return hv_center_space, hv_face_space
end

function make_function_space(domain::HybridBox)
    vertdomain = ClimaCore.Domains.IntervalDomain(
        Geometry.ZPoint(domain.zlim[1])..Geometry.ZPoint(domain.zlim[2]);
        boundary_tags = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(
        vertdomain,
        domain.stretching;
        nelems = domain.nelements[3],
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    vert_face_space = Spaces.FaceFiniteDifferenceSpace(vertmesh)

    horzdomain = ClimaCore.Domains.RectangleDomain(
        Geometry.XPoint(domain.xlim[1])..Geometry.XPoint(domain.xlim[2]),
        Geometry.YPoint(domain.ylim[1])..Geometry.YPoint(domain.ylim[2]),
        x1periodic = true,
        x2periodic = true,
    )
    horzmesh = Meshes.RectilinearMesh(
        horzdomain,
        domain.nelements[1],
        domain.nelements[2],
    )
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    if domain.topography isa WarpedSurface
        surface_function = domain.topography.surface_function
        z_surface =
            surface_function.(ClimaCore.Fields.coordinate_field(horzspace))
        hv_face_space = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_face_space,
            domain.topography.interior_warping,
            z_surface,
        )
        hv_center_space =
            Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    else
        hv_face_space = ClimaCore.Spaces.ExtrudedFiniteDifferenceSpace(
            horzspace,
            vert_face_space,
        )
        hv_center_space =
            Spaces.CenterExtrudedFiniteDifferenceSpace(hv_face_space)
    end
    return hv_center_space, hv_face_space
end

function make_function_space(domain::SphericalShell{FT}) where {FT}
    vertdomain = ClimaCore.Domains.IntervalDomain(
        Geometry.ZPoint(FT(0)),
        Geometry.ZPoint(FT(domain.height));
        boundary_tags = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(
        vertdomain,
        domain.stretching;
        nelems = domain.nelements[2],
    )
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = ClimaCore.Domains.SphereDomain(domain.radius)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, domain.nelements[1])
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    return hv_center_space, hv_face_space
end
