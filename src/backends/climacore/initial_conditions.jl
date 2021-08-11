using ClimaCore.Spaces: SpectralElementSpace2D, CenterFiniteDifferenceSpace, FaceFiniteDifferenceSpace

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