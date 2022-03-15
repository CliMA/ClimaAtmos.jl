"""
    Nonhydrostatic2DModel <: AbstractModel

A two-dimensional non-hydrostatic model, which is typically used for simulating
the Euler equations. Required fields are `domain`, `boundary_conditions`, and
`parameters`.
"""
Base.@kwdef struct Nonhydrostatic2DModel{D, B, T, M, VD, BC, P} <:
                   AbstractNonhydrostatic2DModel
    domain::D
    base::B = ConservativeForm()
    thermodynamics::T = PotentialTemperature()
    moisture::M = Dry()
    vertical_diffusion::VD = NoVerticalDiffusion()
    boundary_conditions::BC
    parameters::P
end

Models.components(model::Nonhydrostatic2DModel) = (
    base = model.base,
    thermodynamics = model.thermodynamics,
    moisture = model.moisture,
    vertical_diffusion = model.vertical_diffusion,
)

function Models.default_initial_conditions(model::Nonhydrostatic2DModel)
    # we need to provide default initial conditions for the model, because the ode solver
    # requires inital conditions when getting instantiated, but we also want to support the `set!` function
    # interface for initialization and re-initialization.
    space_center, space_face = Domains.make_function_space(model.domain)
    local_geometry_center = Fields.local_geometry_field(space_center)
    local_geometry_face = Fields.local_geometry_field(space_face)

    # initialize everything to zeros of the correct types on the correct spaces
    FT = Spaces.undertype(space_center)
    zero_inits = map(Models.components(model)) do component
        variable_names = Models.variable_names(component)
        if !isnothing(variable_names) # e.g., a Dry() doesn't have moisture variable names
            variable_types = Models.variable_types(component, model, FT) # variable types are differnent from model to model
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
            Fields.FieldVector(; NamedTuple{variable_names}(zero_inits)...)
        end
    end

    # filter out the nothing subcomponents (e.g., a Dry() doesn't have moisture variables)
    zero_inits = NamedTuple(
        c => zero_inits[c]
        for c in keys(zero_inits) if !isnothing(zero_inits[c])
    )

    return Fields.FieldVector(; zero_inits...)
end

function Models.make_ode_function(model::Nonhydrostatic2DModel)
    FT = eltype(model.domain) # model works on different float types

    # shorthands for model components & model styles
    thermo_style = model.thermodynamics
    moisture_style = model.moisture
    params = model.parameters
    v_diffusion_style = model.vertical_diffusion

    # this is the complete explicit right-hand side function
    # assembled here to be delivered to the time stepper.
    function rhs!(dY, Y, Ya, t)
        # auxiliary calculation is done here so we don't
        # redo it all the time and can cache the values
        p = calculate_pressure(Y, Ya, thermo_style, moisture_style, params, FT)

        # main model equations
        rhs_base_model!(dY, Y, Ya, t, p, params, FT) #E x.: ∂ₜρ = ..., ∂ₜρuh = ..., etc.
        rhs_thermodynamics!(dY, Y, Ya, t, p, thermo_style, params, FT) # Ex.: ∂ₜρθ = ...
        rhs_moisture!(dY, Y, Ya, t, p, moisture_style, params, FT) # Ex.: ∂ₜρq_tot = ...

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
            v_diffusion_style,
            params,
            FT,
        )

        # rhs_tracer!
        # rhs_edmf!
    end

    return rhs!
end

using ClimaCore: ClimaCore
IF2C = ClimaCore.Operators.InterpolateF2C()
function Models.get_velocities(Y, model::Nonhydrostatic2DModel)
    uh = Y.base.ρuh ./ Y.base.ρ
    w = @. IF2C(Y.base.ρw) / Y.base.ρ
    # Returns tuple with horizontal and vertical velocities for 2D model
    # Cell centered variables are returned (interpolated vertical velocity)
    return (uh, w)
end
