"""
    Nonhydrostatic3DModel <: AbstractModel

A three-dimensional non-hydrostatic model, which is typically used for simulating
the Euler equations. Required fields are `domain`, `boundary_conditions`, and
`parameters`.
"""
Base.@kwdef struct Nonhydrostatic3DModel{D, B, T, M, VD, F, BC, P, FT, C, FC} <:
                   AbstractNonhydrostatic3DModel
    domain::D
    base::B = AdvectiveForm()
    thermodynamics::T = TotalEnergy()
    moisture::M = Dry()
    vertical_diffusion::VD = NoVerticalDiffusion()
    flux_corr::F = true
    hyperdiffusivity::FT
    boundary_conditions::BC
    parameters::P
    cache::C = CacheEmpty()
    source::FC = ()
end

function Models.components(model::Nonhydrostatic3DModel)
    (
        base = model.base,
        thermodynamics = model.thermodynamics,
        moisture = model.moisture,
        vertical_diffusion = model.vertical_diffusion,
    )
end

function Models.default_initial_conditions(
    model::Nonhydrostatic3DModel,
    space_center,
    space_face,
)
    # we need to provide default initial conditions for the model, because the ode solver
    # requires inital conditions when getting instantiated, but we also want to support the `set!` function
    # interface for initialization and re-initialization.
    local_geometry_center = Fields.local_geometry_field(space_center)
    local_geometry_face = Fields.local_geometry_field(space_face)

    # initialize everything to zeros of the correct types on the correct spaces
    FT = Spaces.undertype(space_center)
    zero_inits = map(Models.components(model)) do component
        variable_names = Models.variable_names(component)
        if !isnothing(variable_names) # e.g., a Dry() doesn't have moisture variable names
            variable_types = Models.variable_types(component, model, FT)
            variable_space_types = Models.variable_spaces(component, model)
            zero_inits =
                map(zip(variable_types, variable_space_types)) do (T, ST)
                    zero_instance = zero(T) # somehow need this, otherwise eltype inference error
                    if space_center isa ST
                        map(_ -> zero_instance, local_geometry_center)
                    elseif space_face isa ST
                        map(_ -> zero_instance, local_geometry_face)
                    else
                        error("$ST is neither a $space_center nor a $space_face.")
                    end
                end
            tmp = NamedTuple{variable_names}(zero_inits)
            Fields.FieldVector(; tmp...)
        end
    end

    # filter out the nothing subcomponents (e.g., a Dry() doesn't have moisture variables)
    zero_inits = NamedTuple(
        c => zero_inits[c]
        for c in keys(zero_inits) if !isnothing(zero_inits[c])
    )

    return Fields.FieldVector(; zero_inits...)
end

function Models.default_ode_cache(
    model::Nonhydrostatic3DModel,
    cache::CacheEmpty,
    space_center,
    space_face,
)
    return nothing
end

function Models.make_ode_function(model::Nonhydrostatic3DModel)
    FT = eltype(model.domain) # model works on different float types

    # shorthands for model components & model styles
    base_style = model.base
    thermo_style = model.thermodynamics
    moisture_style = model.moisture
    params = model.parameters
    hyperdiffusivity = model.hyperdiffusivity
    flux_correction = model.flux_corr
    vert_diffusion_style = model.vertical_diffusion
    source = model.source

    # this is the complete explicit right-hand side function
    # assembled here to be delivered to the time stepper.
    function rhs!(dY, Y, Ya, t)
        # auxiliary calculation is done here so we don't
        # redo it all the time and can cache the values
        Φ = calculate_gravitational_potential(Y, Ya, params, FT)
        p = calculate_pressure(
            Y,
            Ya,
            base_style,
            thermo_style,
            moisture_style,
            params,
            FT,
        )

        # base model equations
        # Ex.: ∂ₜρ = ..., ∂ₜρuh = ..., etc.
        rhs_base_model!(
            dY,
            Y,
            Ya,
            t,
            p,
            Φ,
            base_style,
            params,
            hyperdiffusivity,
            flux_correction,
            FT,
        )
        # Ex.: ∂ₜρθ = ...
        rhs_thermodynamics!(
            dY,
            Y,
            Ya,
            t,
            p,
            base_style,
            thermo_style,
            params,
            hyperdiffusivity,
            flux_correction,
            FT,
        )
        # Ex.: ∂ₜρq_tot = ...
        rhs_moisture!(
            dY,
            Y,
            Ya,
            t,
            p,
            base_style,
            moisture_style,
            params,
            hyperdiffusivity,
            FT,
        )

        # vertical diffusion
        rhs_vertical_diffusion!(
            dY,
            Y,
            Ya,
            t,
            p,
            base_style,
            thermo_style,
            moisture_style,
            vert_diffusion_style,
            params,
            FT,
        )

        # rhs_tracer!
        # rhs_edmf!

        
            for i_source in source
                rhs_source(i_source)
            end
            
        
    end

    return rhs!
end
