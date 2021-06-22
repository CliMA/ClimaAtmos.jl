"""
    Declaration of state variables

    vars_state returns a NamedTuple of data types.
"""
function vars_state(::ThreeDimensionalCompressibleEulerWithBarotropicFluid, st::Auxiliary, FT)
    @vars begin
        x::FT
        y::FT
        z::FT
    end
end

function vars_state(::ThreeDimensionalCompressibleEulerWithBarotropicFluid, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρθ::FT
    end
end

"""
    Initialization of state variables

    init_state_xyz! sets up the initial fields within our state variables
    (e.g., prognostic, auxiliary, etc.), however it seems to not initialized
    the gradient flux variables by default.
"""
function init_state_prognostic!(
        balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
        state::Vars,
        aux::Vars,
        localgeo,
        t
    )
    x = aux.x
    y = aux.y
    z = aux.z

    parameters = balance_law.physics.parameters
    ic = balance_law.initial_conditions

    # TODO!: Set to 0 by default or assign IC
    if !isnothing(ic)
        state.ρ  = ic.ρ(parameters, x, y, z)
        state.ρu = ic.ρu(parameters, x, y, z)
        state.ρθ = ic.ρθ(parameters, x, y, z)
    end

    return nothing
end

function nodal_init_state_auxiliary!(
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
    state_auxiliary,
    tmp,
    geom,
)
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
end

"""
    LHS computations
"""
@inline function flux_first_order!(
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    lhs = balance_law.physics.lhs
    physics = balance_law.physics
    
    ntuple(Val(length(lhs))) do s
        Base.@_inline_meta
        calc_component!(flux, lhs[s], state, aux, physics)
    end
end

"""
    RHS computations
"""
function source!(
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid, 
    source, 
    state_prognostic, 
    state_auxiliary, 
    _...
)
    sources = balance_law.physics.sources
    physics = balance_law.physics

    ntuple(Val(length(sources))) do s
        Base.@_inline_meta
        calc_component!(source, sources[s], state_prognostic, state_auxiliary, physics)
    end
end

"""
    Utils
"""
function altitude(balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid, ::SphericalOrientation, geom)
    return norm(geom.coord) - balance_law.physics.parameters.a
end

function altitude(::ThreeDimensionalCompressibleEulerWithBarotropicFluid, ::FlatOrientation, geom)
    @inbounds geom.coord[3]
end