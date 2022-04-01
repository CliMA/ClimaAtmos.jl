usempi = get(ENV, "CLIMACORE_DISTRIBUTED", "")  == "MPI"
if usempi 
  using ClimaComms
  using ClimaCommsMPI
  const Context = ClimaCommsMPI.MPICommsContext
  const pid, nprocs = ClimaComms.init(Context)
  if pid = 1
    println("Parallel run with $nprocs processes.")
  end
  logger_stream = ClimaComms.iamroot(Context) ? stderr : devnull
  prev_logger = global_logger(ConsoleLogger(logger_stream, Logging.Info))
  atexit() do
    global_logger(prev_logger)
  end
else
  using Logging: global_logger
  using TerminalLoggers: TerminalLogger
  global_logger(TerminalLogger())
end

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

    # Distributed topology not supported in Column configurations
    return (center_space, face_space, nothing, nothing)
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
    if !usempi
        horztopology = Topologies.Topology2D(horzmesh)
        quad = Spaces.Quadratures.GLL{domain.npolynomial + 1}()
        comms_ctx_center =
          nothing
        comms_ctx_face =
          nothing
    else
        horztopology = Topologies.DistributedTopology2D(horzmesh, Context)
        comms_ctx_center =
          Spaces.setup_comms(Context, horztopology, quad, Nv, Nf_center)
        comms_ctx_face =
          Spaces.setup_comms(Context, horztopology, quad, Nv + 1, Nf_face)
    end
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad, comms_ctx_center)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)

    return (hv_center_space, hv_face_space, comms_ctx_center, comms_ctx_face)
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

    return (hv_center_space, hv_face_space, nothing, nothing)
end
