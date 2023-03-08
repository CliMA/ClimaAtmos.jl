"""
    TurbconvState{FT}

A collection of values that are required to initialize the `turbconv_model` of
an `AtmosModel`.
"""
abstract type TurbconvState{FT} end

"""
    LocalState(; params, geometry, thermo_state, velocity, turbconv_state)

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
    TP,
}
    params::P
    geometry::G
    thermo_state::TS
    velocity::V
    turbconv_state::TC

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
)
    FT = eltype(params)
    return LocalState(
        params,
        geometry,
        thermo_state,
        isnothing(velocity) ? Geometry.UVVector(FT(0), FT(0)) : velocity,
        isnothing(turbconv_state) ? NoTurbconvState{FT}() : turbconv_state,
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

@inline Base.getproperty(ts::NoTurbconvState{FT}, s::Symbol) where {FT} =
    s in (:tke, :Hvar, :QTVar, :HQTcov) ? FT(0) : getfield(ts, s)

"""
    EDMFState(; tke)

Stores the value of `tke` for a `turbconv_model::EDMFModel`. Any other values
required by the `turbconv_model` are set to 0.
"""
Base.@kwdef struct EDMFState{FT} <: TurbconvState{FT}
    tke::FT
end

@inline Base.getproperty(ts::EDMFState{FT}, s::Symbol) where {FT} =
    s in (:Hvar, :QTVar, :HQTcov) ? FT(0) : getfield(ts, s)

"""
    EDMFStateWithThermo2ndMoments(; tke, Hvar, QTvar, HQTcov)

Stores the values of `tke`, `Hvar`, `QTvar`, and `HQTcov` for a
`turbconv_model::EDMFModel` that uses a `PrognosticThermoCovariances` model.
"""
Base.@kwdef struct EDMFStateWithThermo2ndMoments{FT} <: TurbconvState{FT}
    tke::FT
    Hvar::FT
    QTvar::FT
    HQTcov::FT
end
