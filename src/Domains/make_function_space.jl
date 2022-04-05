function make_function_space(domain::Column; distributed = false)

    if distributed
        error("Distributed 1D column domains are unsupported")
    end
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

    # Distributed topology not supported in Column configurations
    return (center_space, face_space, nothing, nothing)
end

function make_function_space(domain::HybridPlane; distributed = false)
    if distributed
        error("Distributed 2D plane domains are unsupported")
    end
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

    # Distributed topology unsupported in HybridPlane configurations
    return (hv_center_space, hv_face_space, nothing, nothing)
end

function make_function_space(domain::HybridBox; distributed = false)

    if distributed
        @info("Generating distributed 3D box topology.")
    end
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
    Nv = Meshes.nelements(vertmesh)
    Nf_center, Nf_face = 2, 1
    if distributed
        horztopology = Topologies.DistributedTopology2D(horzmesh, Context)
        comms_ctx_center =
            Spaces.setup_comms(Context, horztopology, quad, Nv, Nf_center)
        comms_ctx_face =
            Spaces.setup_comms(Context, horztopology, quad, Nv + 1, Nf_face)
    else
        horztopology = Topologies.Topology2D(horzmesh)
        quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
        comms_ctx_center = nothing
        comms_ctx_face = nothing
    end
    horzspace =
        Spaces.SpectralElementSpace2D(horztopology, quad, comms_ctx_center)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    return (hv_center_space, hv_face_space, comms_ctx_center, comms_ctx_face)
end

function make_function_space(
    domain::SphericalShell{FT};
    distributed = false,
) where {FT}

    if distributed
        @info("Generating distributed 3D sphere topology.")
    end

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
    if distributed
        horztopology = Topologies.DistributedTopology2D(horzmesh, Context)
        comms_ctx_center =
            Spaces.setup_comms(Context, horztopology, quad, Nv, Nf_center)
        comms_ctx_face =
            Spaces.setup_comms(Context, horztopology, quad, Nv + 1, Nf_face)
    else
        horztopology = Topologies.Topology2D(horzmesh)
        comms_ctx_center = nothing
        comms_ctx_face = nothing
    end

    quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    return (hv_center_space, hv_face_space, nothing, nothing)
end
