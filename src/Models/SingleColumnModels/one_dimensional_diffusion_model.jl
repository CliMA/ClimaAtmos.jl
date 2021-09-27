mutable struct OneDimensionalDiffusionModel{D, T, B, C, F, H, O, TS}
    domain::D
    tracers::T
    boundary_conditions::B
    turbulence_closure::C # f(Y,∇Y,Ya,p,t)
    forcing::F
    hyperdiffusion::H
    operators::O
    timestepper::TS
end

tupleit(t) = try
    Tuple(t)
catch
    tuple(t)
end

function OneDimensionalDiffusionModel(;
    domain,
    tracers,
    boundary_conditions,
    hyperdiffusion = nothing,
    turbulence_closure = nothing,
    forcing = nothing,
    )

    tracers = tupleit(tracers)

    # regularize forcing?
    # regularize boundary_conditions?

    operators = NamedTuple(tracer => tracer_operators(domain, get_boundary_conditions(boundary_conditions, name)) for tracer in tracer_names)

    # Build DiffEq integrator (and call it timestepper)
    timestepper = nothing

    # Build references to data in the timestepper...

    return OneDimensionalDiffusionModel(domain, tracers, boundary_conditions, turbulence_closure, forcing, hyperdiffusion, turbulence_closure, timestepper)
end

function get_boundary_conditions(bcs, name)
    if name ∈ keys(bcs)
        return bcs[name]
    else
        return FieldBoundaryConditions(nothing, nothing)
    end
end

#####
##### Boundary conditions
#####

struct FieldBoundaryConditions{T, B}
    top::T
    bottom::B
end

function boundary_condition_setter(domain::Column, ::Nothing)
    FT = typeof(domain.zlim[1]) # TODO: define eltype(domain) so we can use that instead
    return Operators.SetGradient(zero(FT)) 
    # TODO Check ClimaCore underlying types (Cartesian / 
    # Covariant / Contravariant - domain dependent)
end

function tracer_operators(domain::Column, tracer_bcs)
    set_bottom_bc = boundary_condition_setter(domain, tracer_bcs.bottom)
    set_top_bc = boundary_condition_setter(domain, tracer_bcs.top)
    ∇ = Operators.GradientC2F(bottom = set_bottom_bc, top = set_top_bc)
    return ∇
end