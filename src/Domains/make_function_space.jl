"""
    make_function_space(domain::Column)
"""
function make_function_space(domain::Column{FT}) where {FT}
    column = ClimaCore.Domains.IntervalDomain(
        domain.zlim[1],
        domain.zlim[2];
        x3boundary = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(column; nelems = domain.nelements)
    center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

"""
    make_function_space(domain::Plane)
"""
function make_function_space(domain::Plane{FT}) where {FT}
    rectangle = ClimaCore.Domains.RectangleDomain(
        Interval(domain.xlim),
        Interval(domain.ylim),
        x1periodic = domain.periodic[1],
        x2periodic = domain.periodic[2],
    )
    mesh = Meshes.EquispacedRectangleMesh(
        rectangle,
        domain.nelements[1],
        domain.nelements[2],
    )
    grid_topology = Topologies.GridTopology(mesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial}()
    space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    return space
end
Interval(I::Tuple{Number, Number}) = Interval(I[1], I[2])

"""
    make_function_space(domain::HybridPlane)
"""
function make_function_space(domain::HybridPlane{FT}) where {FT}
    vertdomain = ClimaCore.Domains.IntervalDomain(
        FT(domain.zlim[1]),
        FT(domain.zlim[2]);
        x3boundary = (:bottom, :top),
    )

    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = domain.nelements[2])
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = ClimaCore.Domains.RectangleDomain(
        domain.xlim[1]..domain.xlim[2],
        -0..0,
        x1periodic = true,
        x2boundary = (:a, :b),
    )
    horzmesh =
        Meshes.EquispacedRectangleMesh(horzdomain, domain.nelements[1], 1)
    horztopology = Topologies.GridTopology(horzmesh)

    quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
    horzspace = Spaces.SpectralElementSpace1D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    return hv_center_space, hv_face_space
end
