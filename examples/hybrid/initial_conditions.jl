##
## Initial conditions
##

function init_state(
    center_initial_condition,
    face_initial_condition,
    center_space,
    face_space,
    params,
    atmos,
    perturb_initstate,
)
    (; energy_form, moisture_model, turbconv_model) = atmos
    ᶜlocal_geometry = Fields.local_geometry_field(center_space)
    ᶠlocal_geometry = Fields.local_geometry_field(face_space)
    c =
        center_initial_condition.(
            ᶜlocal_geometry,
            params,
            energy_form,
            moisture_model,
            turbconv_model,
            perturb_initstate,
        )
    f = face_initial_condition.(ᶠlocal_geometry, params, turbconv_model)
    Y = Fields.FieldVector(; c, f)
    return Y
end
