"""
    make_initial_conditions(model::ShallowWaterModel{<:AbstractHorizontalDomain})
"""
function make_initial_conditions(model::ShallowWaterModel{<:AbstractHorizontalDomain})
    function_space = make_function_space(model.domain)

    @unpack x1, x2 = Fields.coordinate_field(function_space)
    state_init = model.initial_conditions.(x1, x2, Ref(model.parameters))

    return state_init
end

"""
    make_initial_conditions(model::SingleColumnModel{<:AbstractVerticalDomain})
"""
function make_initial_conditions(model::SingleColumnModel{<:AbstractVerticalDomain})
    center_space, face_space = make_function_space(model.domain)

    z_centers = Fields.coordinate_field(center_space)
    z_faces = Fields.coordinate_field(face_space)
    y_centers = model.initial_conditions.centers.(z_centers, Ref(model.parameters))
    y_faces = model.initial_conditions.faces.(z_faces, Ref(model.parameters))
    state_init = ArrayPartition(y_centers, y_faces)

    return state_init
end