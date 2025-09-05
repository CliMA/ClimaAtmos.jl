import .Topography:
    NoTopography, CosineTopography, AgnesiTopography, ScharTopography

"""
    compute_steady_state_velocity(topography, params, z_top)

Compute the steady-state velocity for the given topography type.
"""
function compute_steady_state_velocity(::NoTopography, params, Y)
    top_level = Spaces.nlevels(axes(Y.c)) + Fields.half
    z_top = Fields.level(Fields.coordinate_field(Y.f).z, top_level)

    ᶜu = steady_state_velocity_no_warp.(params, Fields.coordinate_field(Y.c), z_top)
    ᶠu = steady_state_velocity_no_warp.(params, Fields.coordinate_field(Y.f), z_top)

    return (; ᶜu, ᶠu)
end

function compute_steady_state_velocity(topography::CosineTopography, params, Y)
    top_level = Spaces.nlevels(axes(Y.c)) + Fields.half
    z_top = Fields.level(Fields.coordinate_field(Y.f).z, top_level)

    if topography.dimension == 2
        ᶜu = steady_state_velocity_cosine_2d.(params, Fields.coordinate_field(Y.c), z_top)
        ᶠu = steady_state_velocity_cosine_2d.(params, Fields.coordinate_field(Y.f), z_top)
    elseif topography.dimension == 3
        ᶜu = steady_state_velocity_cosine_3d.(params, Fields.coordinate_field(Y.c), z_top)
        ᶠu = steady_state_velocity_cosine_3d.(params, Fields.coordinate_field(Y.f), z_top)
    else
        throw(
            ArgumentError(
                "CosineTopography dimension must be 2 or 3, got $(topography.dimension)",
            ),
        )
    end

    return (; ᶜu, ᶠu)
end

function compute_steady_state_velocity(::AgnesiTopography, params, Y)
    top_level = Spaces.nlevels(axes(Y.c)) + Fields.half
    z_top = Fields.level(Fields.coordinate_field(Y.f).z, top_level)

    ᶜu = steady_state_velocity_agnesi.(params, Fields.coordinate_field(Y.c), z_top)
    ᶠu = steady_state_velocity_agnesi.(params, Fields.coordinate_field(Y.f), z_top)

    return (; ᶜu, ᶠu)
end

function compute_steady_state_velocity(topography::ScharTopography, params, Y)
    top_level = Spaces.nlevels(axes(Y.c)) + Fields.half
    z_top = Fields.level(Fields.coordinate_field(Y.f).z, top_level)

    ᶜu = steady_state_velocity_schar.(params, Fields.coordinate_field(Y.c), z_top)
    ᶠu = steady_state_velocity_schar.(params, Fields.coordinate_field(Y.f), z_top)

    return (; ᶜu, ᶠu)
end
