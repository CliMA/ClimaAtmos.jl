"""
    Nonhydrostatic3DModel <: AbstractModel

A three-dimensional non-hydrostatic model, which is typically used for simulating
the Euler equations. Required fields are `domain`, `boundary_conditions`, and
`parameters`.
"""
Base.@kwdef struct Nonhydrostatic3DModel{
    D <: AbstractHybridDomain,
    BC,
    P,
    FT,
} <: AbstractModel
    domain::D
    base::AbstractBaseModelStyle = AdvectiveForm()
    thermodynamics::AbstractThermodynamicsStyle = TotalEnergy()
    moisture::AbstractMoistureStyle = Dry()
    flux_corr::Bool = true
    hyperdiffusivity::FT
    boundary_conditions::BC
    parameters::P
end

function Models.subcomponents(model::Nonhydrostatic3DModel)
    # we need a helper function here to extract the possible subcomponents
    # of a model. This can be used to generically work with models later.
    (
        base = model.base,
        thermodynamics = model.thermodynamics,
        moisture = model.moisture,
    )
end

function Models.state_variable_names(model::Nonhydrostatic3DModel)
    # we need to extract the active only subcomponents of the model because
    # later we don't want to carry around empty vectors of moisture in a dry model
    # so we need to extract information as to what variables are active.
    subcomponents = keys(Models.subcomponents(model))
    subcomponent_vars = (
        Models.state_variable_names(sc) for
        sc in values(Models.subcomponents(model))
    )

    # we need to collect the the non-nothing subcomponents, e.g., in case if Dry() moisture
    # we don't have any variables for moisture.
    active_subcomponents = (
        s for
        (s, vars) in zip(subcomponents, subcomponent_vars) if vars ≢ nothing
    )
    active_vars = (
        vars for
        (s, vars) in zip(subcomponents, subcomponent_vars) if vars ≢ nothing
    )

    return (; zip(active_subcomponents, active_vars)...)
end

function Models.default_initial_conditions(model::Nonhydrostatic3DModel)
    # we need to provide default initial conditions for the model, because the ode solver
    # requires inital conditions when getting instantiated, but we also want to support the `set!` function
    # interface for initialization and re-initialization.
    space_c, space_f = make_function_space(model.domain)
    local_geometry_c = Fields.local_geometry_field(space_c)
    local_geometry_f = Fields.local_geometry_field(space_f)

    # functions that make zeros for this model. These will be broadcasted
    # across the entire domain.    
    zero_val = zero(Spaces.undertype(space_c))
    zero_scalar(lg) = zero_val
    zero_12vector(lg) = Geometry.Covariant12Vector(zero_val, zero_val)
    zero_3vector(lg) = Geometry.Covariant3Vector(zero_val)

    # this sets up the default zero initial condition for the different model
    # components (e.g., base model component, thermodynamics component, etc.)
    varnames = state_variable_names(model)

    # base model components
    ρ = zero_scalar.(local_geometry_c)
    uh = zero_12vector.(local_geometry_c)
    w = zero_3vector.(local_geometry_f) # on faces
    base = (ρ = ρ, uh = uh, w = w)

    # all other components are assumed to be scalars
    # thermodynamics components
    thermodynamics_values =
        (zero_scalar.(local_geometry_c) for name in varnames.thermodynamics)
    thermodynamics = NamedTuple{varnames.thermodynamics}(thermodynamics_values)

    # moisture components
    if :moisture ∈ varnames
        moisture_values =
            (zero_scalar.(local_geometry_c) for name in varnames.moisture)
        moisture = NamedTuple{varnames.moisture}(moisture_values)
    end

    # construct fieldvector to return
    if :moisture ∈ varnames
        return Fields.FieldVector(
            base = base,
            thermodynamics = thermodynamics,
            moisture = moisture,
        )
    else
        return Fields.FieldVector(base = base, thermodynamics = thermodynamics)
    end
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
        rhs_moisture!(dY, Y, Ya, t, p, base_style, moisture_style, params, FT)
        # rhs_tracer!
        # rhs_edmf!
    end

    return rhs!
end
