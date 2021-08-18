using ClimaCore.Spaces: SpectralElementSpace2D, CenterFiniteDifferenceSpace, FaceFiniteDifferenceSpace

"""
    create_function_space
"""
function create_function_space(::ClimaCoreBackend, ::AbstractDomain)
    error("Domain not supported by ClimaCoreBackend.")
end

function create_function_space(::ClimaCoreBackend, domain::Rectangle)
    rectangle = Domains.RectangleDomain(
        domain.xlim,
        domain.ylim,
        x1periodic = domain.periodic[1],
        x2periodic = domain.periodic[2],
    )
    mesh = Meshes.EquispacedRectangleMesh(
        rectangle, 
        domain.nelements[1], 
        domain.nelements[2]
    )
    grid_topology = Topologies.GridTopology(mesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial}()
    function_space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    return function_space
end

function create_function_space(::ClimaCoreBackend, domain::SingleColumn)
    column = Domains.IntervalDomain(
        domain.zlim.left, 
        domain.zlim.right; 
        x3boundary = (:bottom, :top)
    )
    mesh = Meshes.IntervalMesh(column; nelems = domain.nelements)
    center_space = Spaces.CenterFiniteDifferenceSpace(mesh)
    face_space = Spaces.FaceFiniteDifferenceSpace(center_space)

    return center_space, face_space
end

"""
    create_initial_conditions
"""
function create_initial_conditions(::ClimaCoreBackend, model::AbstractModel, spectral_space::SpectralElementSpace2D)
    UnPack.@unpack x1, x2 = Fields.coordinate_field(spectral_space)
    inital_state = model.initial_conditions.(x1, x2, Ref(model.parameters))

    return inital_state
end

function create_initial_conditions(::ClimaCoreBackend, model::AbstractModel, function_space::Tuple{CenterFiniteDifferenceSpace, FaceFiniteDifferenceSpace})
    center_space, face_space = function_space

    z_centers = Fields.coordinate_field(center_space)
    z_faces = Fields.coordinate_field(face_space)
    Yc = model.initial_conditions.centers.(z_centers, Ref(model.parameters))
    Yf = model.initial_conditions.faces.(z_faces, Ref(model.parameters))
    inital_state = ArrayPartition(Yc, Yf)

    return inital_state
end

"""
    create_ode_problem
"""
function create_ode_problem(backend::ClimaCoreBackend, model::AbstractModel, timestepper::AbstractTimestepper)
    function_space = create_function_space(backend, model.domain)
    rhs!           = create_rhs(backend, model, function_space)
    y_init         = create_initial_conditions(backend, model, function_space)

    return ODEProblem(
        rhs!, 
        y_init, 
        timestepper.tspan,
    )
end