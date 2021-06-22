struct DryReferenceState{TP}
    temperature_profile::TP
end

"""
    Declaration of state variables

    vars_state returns a NamedTuple of data types.
"""
function vars_state(m::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, st::Auxiliary, FT)
    @vars begin
        x::FT
        y::FT
        z::FT
        Φ::FT
        ∇Φ::SVector{3, FT} # TODO: only needed for the linear balance_law
        ref_state::vars_state(m, m.physics.ref_state, st, FT)
    end
end

vars_state(::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, ::DryReferenceState, ::Auxiliary, FT) =
    @vars(T::FT, p::FT, ρ::FT, ρu::SVector{3, FT}, ρe::FT, ρq::FT)
vars_state(::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, ::NoReferenceState, ::Auxiliary, FT) = @vars()

function vars_state(::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end

"""
    Initialization of state variables

    init_state_xyz! sets up the initial fields within our state variables
    (e.g., prognostic, auxiliary, etc.), however it seems to not initialized
    the gradient flux variables by default.
"""
function init_state_prognostic!(
        balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
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
        state.ρe = ic.ρe(parameters, x, y, z)
    end

    return nothing
end

function nodal_init_state_auxiliary!(
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    state_auxiliary,
    tmp,
    geom,
)
    init_state_auxiliary!(balance_law, balance_law.physics.orientation, state_auxiliary, geom)
    init_state_auxiliary!(balance_law, balance_law.physics.ref_state, state_auxiliary, geom)
end

function init_state_auxiliary!(
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    ::SphericalOrientation,
    state_auxiliary,
    geom,
)
    g = balance_law.physics.parameters.g

    r = norm(geom.coord)
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = g * r
    state_auxiliary.∇Φ = g * geom.coord / r
end

function init_state_auxiliary!(
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    ::FlatOrientation,
    state_auxiliary,
    geom,
)
    g = balance_law.physics.parameters.g

    FT = eltype(state_auxiliary)
    
    r = geom.coord[3]
    state_auxiliary.x = geom.coord[1]
    state_auxiliary.y = geom.coord[2]
    state_auxiliary.z = geom.coord[3]
    state_auxiliary.Φ = g * r
    state_auxiliary.∇Φ = SVector{3, FT}(0, 0, g)
end

function init_state_auxiliary!(
    ::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    ::NoReferenceState,
    state_auxiliary,
    geom,
) end

function init_state_auxiliary!(
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
    ref_state::DryReferenceState,
    state_auxiliary,
    geom,
)
    orientation = balance_law.physics.orientation   
    R_d         = balance_law.physics.parameters.R_d
    γ           = balance_law.physics.parameters.γ
    Φ           = state_auxiliary.Φ

    FT = eltype(state_auxiliary)

    # Calculation of a dry reference state
    z = altitude(balance_law, orientation, geom)
    T, p = ref_state.temperature_profile(balance_law.physics.parameters, z)
    ρ  = p / R_d / T
    ρu = SVector{3, FT}(0, 0, 0)
    ρe = p / (γ - 1) + dot(ρu, ρu) / 2ρ + ρ * Φ

    state_auxiliary.ref_state.T  = T
    state_auxiliary.ref_state.p  = p
    state_auxiliary.ref_state.ρ  = ρ
    state_auxiliary.ref_state.ρu = ρu
    state_auxiliary.ref_state.ρe = ρe
end

"""
    LHS computations
"""
@inline function flux_first_order!(
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy,
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
    balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, 
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
function altitude(balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, ::SphericalOrientation, geom)
    return norm(geom.coord) - balance_law.physics.parameters.a
end

function altitude(::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, ::FlatOrientation, geom)
    @inbounds geom.coord[3]
end