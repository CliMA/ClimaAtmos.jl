"""
    TurbconvState{FT}

A collection of values that are required to initialize the `turbconv_model` of
an `AtmosModel`.
"""
abstract type TurbconvState{FT} end

"""
    PrecipState{FT}

A collection of values that are required to initialize the `precip_model` of
an `AtmosModel`.
"""
abstract type PrecipState{FT} end

"""
    LocalState(; params, geometry, thermo_state, velocity, turbconv_state, precip_state)

A generic representation of all the data required to initialize an `AtmosModel`
at some point in the domain. If `velocity` or `turbconv_state` are omitted, they
are set to 0.
"""
struct LocalState{
    FT,
    P <: CAP.ClimaAtmosParameters{FT},
    G <: Geometry.LocalGeometry{<:Any, <:Any, FT},
    TS <: TD.ThermodynamicState{FT},
    V <: Geometry.LocalVector{FT},
    TC <: TurbconvState{FT},
    PS <: PrecipState{FT},
    TP,
}
    params::P
    geometry::G
    thermo_state::TS
    velocity::V
    turbconv_state::TC
    precip_state::PS

    # commonly used values that can be inferred from the values above
    thermo_params::TP
    ρ::FT
end

function LocalState(;
    params,
    geometry,
    thermo_state,
    velocity = nothing,
    turbconv_state = nothing,
    precip_state = nothing,
)
    FT = eltype(params)
    return LocalState(
        params,
        geometry,
        thermo_state,
        isnothing(velocity) ? Geometry.UVVector(FT(0), FT(0)) : velocity,
        isnothing(turbconv_state) ? NoTurbconvState{FT}() : turbconv_state,
        isnothing(precip_state) ? NoPrecipState{FT}() : precip_state,
        CAP.thermodynamics_params(params),
        TD.air_density(CAP.thermodynamics_params(params), thermo_state),
    )
end

Base.eltype(::LocalState{FT}) where {FT} = FT

"""
    NoTurbconvState{FT}()

Indicates that no initial conditions are available for the `turbconv_model`. Any
values required by the `turbconv_model` are set to 0.
"""
struct NoTurbconvState{FT} <: TurbconvState{FT} end

@inline Base.getproperty(::NoTurbconvState{FT}, ::Symbol) where {FT} = FT(0)

"""
    EDMFState(; tke, draft_area)

Stores the values of `tke` and `draft_area` for the `turbconv_model`. If
`draft_area` is omitted, it is set to 0.
Currently, we input physical velocity w into EDMFState, and this will be wrong with topography
"""
struct EDMFState{FT} <: TurbconvState{FT}
    tke::FT
    draft_area::FT
    velocity::Geometry.WVector{FT}
end
EDMFState(; tke, draft_area = 0, velocity = Geometry.WVector(0)) =
    EDMFState{typeof(tke)}(tke, draft_area, velocity)

"""
    NoPrecipState{FT}()

Indicates that no initial conditions are available for the `precip_model`. Any
values required by the `precip_model` are set to 0.
"""
struct NoPrecipState{FT} <: PrecipState{FT} end

@inline Base.getproperty(::NoPrecipState{FT}, ::Symbol) where {FT} = FT(0)

"""
    PrecipState2M(; q_rai, q_sno)


Stores the values of `n_liq`, `n_rai`, `q_rai`, and `q_sno` for the `precip_model`.
This struct includes both mass and number densities and can be used for initializing 
precipitation states in both one-moment and two-moment microphysics schemes. 
In one-moment schemes, the number densities (`n_liq`, `n_rai`) are ignored.
If no values are provided, all fields are initialized to zero.
"""
struct PrecipStateMassNum{FT} <: PrecipState{FT}
    n_liq::FT
    n_rai::FT
    q_rai::FT
    q_sno::FT
end
PrecipStateMassNum(; n_liq = 0, n_rai = 0, q_rai = 0, q_sno = 0) =
    PrecipStateMassNum{typeof(q_rai)}(n_liq, n_rai, q_rai, q_sno)
