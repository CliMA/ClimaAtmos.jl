import .Topography: AbstractTopography, NoTopography, CosineTopography, AgnesiTopography, ScharTopography
import .Topography: supports_steady_state, topography_name
import .InitialConditions as ICs

"""
    supports_steady_state_velocity(domain, initial_condition)

Check if the given domain and initial condition combination supports analytical steady-state velocity.
"""
supports_steady_state_velocity(::AtmosDomain, ::ICs.InitialCondition) = false

# Only ConstantBuoyancyFrequencyProfile with Linear mesh warping supports steady
# state
supports_steady_state_velocity(
    domain::Union{BoxDomain, PlaneDomain}, 
    ::ICs.ConstantBuoyancyFrequencyProfile
) = domain.mesh_warp_type == LinearWarp && supports_steady_state(domain.topography)

topography(domain::AtmosDomain) = domain.topography

"""
    get_steady_state_velocity(domain, initial_condition, params, Y; check_steady_state::Bool = false)

Compute analytical steady-state velocity if supported by the domain and initial condition.
Returns `nothing` if not supported or not requested.
"""
function get_steady_state_velocity(
    domain,
    initial_condition::ICs.InitialCondition,
    params,
    Y;
    check_steady_state::Bool = false
)
    # Early return if not requested
    check_steady_state || return nothing
    
    if !supports_steady_state_velocity(domain, initial_condition)
        throw(ArgumentError(
            "Steady-state velocity computation is not supported for the given " *
            "domain ($(nameof(typeof(domain)))) and initial condition ($(nameof(typeof(initial_condition)))). " *
            "Currently only supported for ConstantBuoyancyFrequencyProfile with Linear mesh warping."
        ))
    end
    
    topography = topography(domain)
    return compute_steady_state_velocity(topography, params, Y)
end

"""
    compute_steady_state_velocity(topography, params, Y)

Compute the steady-state velocity for the given topography type.
"""
function compute_steady_state_velocity(topography::NoTopography, params, Y)
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
        throw(ArgumentError("CosineTopography dimension must be 2 or 3, got $(topography.dimension)"))
    end
    
    return (; ᶜu, ᶠu)
end

function compute_steady_state_velocity(topography::AgnesiTopography, params, Y)
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
