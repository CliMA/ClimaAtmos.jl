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
    parameters = balance_law.parameters
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
    Main model computations
"""
@inline function flux_first_order!(
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # base model equaations
    ρ  = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ
    eos = balance_law.equation_of_state
    parameters = balance_law.parameters
    
    u = ρu / ρ
    p = calc_pressure(eos, state, aux, parameters)

    flux.ρ  += ρu
    flux.ρu += ρu ⊗ u + p * I
    flux.ρθ += ρθ * u
    
    nothing
end

"""
    Source computations
"""
function source!(
    balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid, 
    source, 
    state_prognostic, 
    state_auxiliary, 
    _...
)
    sources = balance_law.sources

    ntuple(Val(length(sources))) do s
        Base.@_inline_meta
        calc_source!(source, balance_law, sources[s], state_prognostic, state_auxiliary)
    end
end