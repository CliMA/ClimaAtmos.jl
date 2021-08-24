"""
    make_function_space(domain::Column)
"""
function make_function_space(domain::Column)
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
function make_function_space(domain::Plane)
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
    function_space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    return function_space
end

Interval(I::Tuple{Number, Number}) = Interval(I[1], I[2])
