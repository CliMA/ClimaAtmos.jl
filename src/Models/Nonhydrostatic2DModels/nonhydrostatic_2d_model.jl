"""
    Nonhydrostatic2DModel <: AbstractModel

A two-dimensional non-hydrostatic model, which is typically used for simulating
the Euler equations. Required fields are `domain`, `boundary_conditions`, and
`parameters`.
"""
Base.@kwdef struct Nonhydrostatic2DModel{D, B, T, M, PR, VD, FT, BC, P, C} <:
                   AbstractNonhydrostatic2DModel
    domain::D
    base::B = ConservativeForm()
    thermodynamics::T = PotentialTemperature()
    moisture::M = Dry()
    precipitation::PR = NoPrecipitation()
    vertical_diffusion::VD = NoVerticalDiffusion()
    hyperdiffusivity::FT
    boundary_conditions::BC
    parameters::P
    cache::C = CacheBase()
end

Models.components(model::Nonhydrostatic2DModel) = (
    base = model.base,
    thermodynamics = model.thermodynamics,
    moisture = model.moisture,
    precipitation = model.precipitation,
    vertical_diffusion = model.vertical_diffusion,
)

function Models.default_initial_conditions(
    model::Nonhydrostatic2DModel,
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
    base_style = model.base
    thermo_style = model.thermodynamics
    moisture_style = model.moisture
    precip_style = model.precipitation
    params = model.parameters
    vert_diffusion_style = model.vertical_diffusion
    hyperdiffusivity = model.hyperdiffusivity

    # this is the complete explicit right-hand side function
    # assembled here to be delivered to the time stepper.
    function rhs!(dY, Y, Ya, t)

        precompute_cache!(
            dY,
            Y,
            Ya,
            thermo_style,
            moisture_style,
            precip_style,
            params,
            FT,
        )

        # main model equations
        rhs_base_model!(dY, Y, Ya, t, params, hyperdiffusivity, FT) #E x.: ∂ₜρ = ..., ∂ₜρuh = ..., etc.

        rhs_thermodynamics!(
            dY,
            Y,
            Ya,
            t,
            base_style,
            thermo_style,
            params,
            hyperdiffusivity,
            FT,
        )

        rhs_moisture!(
            dY,
            Y,
            Ya,
            t,
            base_style,
            moisture_style,
            precip_style,
            params,
            hyperdiffusivity,
            FT,
        ) # Ex.: ∂ₜρq_tot = ...

        rhs_precipitation!(
            dY,
            Y,
            Ya,
            t,
            base_style,
            moisture_style,
            precip_style,
            params,
            hyperdiffusivity,
            FT,
        ) # Ex.: ∂ₜρq_tot = ...

        # vertical diffusion
        rhs_vertical_diffusion!(
            dY,
            Y,
            Ya,
            t,
            base_style,
            thermo_style,
            moisture_style,
            vert_diffusion_style,
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
