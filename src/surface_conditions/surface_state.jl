# Naming convention: surface_<value> is the value at the surface, whereas
# sfc_<value> is a field of values at the surface, and likewise for
# interior_<value> and int_<value>.

abstract type PrescribedFluxes{FT} end

abstract type SurfaceParameterization{FT} end

"""
    SurfaceState(; parametrization, T, p, q_vap, u, v, gustiness, beta)

A container for state variables at the ground level, for use with SurfaceFluxes.
"""
struct SurfaceState{
    FT,
    SFP <: SurfaceParameterization{FT},
    FTN1 <: Union{FT, Nothing},
    FTN2 <: Union{FT, Nothing},
    FTN3 <: Union{FT, Nothing},
    FTN4 <: Union{FT, Nothing},
    FTN5 <: Union{FT, Nothing},
    FTN6 <: Union{FT, Nothing},
    FTN7 <: Union{FT, Nothing},
}
    parameterization::SFP
    T::FTN1
    p::FTN2
    q_vap::FTN3
    u::FTN4
    v::FTN5
    gustiness::FTN6
    beta::FTN7
end

float_type(::Type{<:SurfaceParameterization{FT}}) where {FT} = FT

SurfaceState(;
    parameterization::SFP,
    T::FTN1 = nothing,
    p::FTN2 = nothing,
    q_vap::FTN3 = nothing,
    u::FTN4 = nothing,
    v::FTN5 = nothing,
    gustiness::FTN6 = nothing,
    beta::FTN7 = nothing,
) where {
    SFP <: SurfaceParameterization,
    FTN1,
    FTN2,
    FTN3,
    FTN4,
    FTN5,
    FTN6,
    FTN7,
} =
    SurfaceState{float_type(SFP), SFP, FTN1, FTN2, FTN3, FTN4, FTN5, FTN6, FTN7}(
        parameterization,
        T,
        p,
        q_vap,
        u,
        v,
        gustiness,
        beta,
    )

"""
    HeatFluxes(; shf, lhf)

Container for heat fluxes
 - shf: Sensible heat flux
 - lhf: Latent heat flux
"""
Base.@kwdef struct HeatFluxes{FT, FTN <: Union{FT, Nothing}} <:
                   PrescribedFluxes{FT}
    shf::FT
    lhf::FTN = nothing
end

"""
    θAndQFluxes(; θ_flux, q_flux)

Container for quantities used to calculate sensible and latent heat fluxes.
"""
Base.@kwdef struct θAndQFluxes{FT, FTN <: Union{FT, Nothing}} <:
                   PrescribedFluxes{FT}
    θ_flux::FT
    q_flux::FTN = nothing
end

"""
    ExchangeCoefficients(; Cd, Ch)
    ExchangeCoefficients(; C)

Exchange coefficients
 - Cd: Momentum Exchange Coefficient
 - Ch: Thermal Exchange Coefficient
"""
Base.@kwdef struct ExchangeCoefficients{FT} <: SurfaceParameterization{FT}
    Cd::FT
    Ch::FT
end
ExchangeCoefficients(C) = ExchangeCoefficients(Cd = C, Ch = C)

"""
    MoninObukhov(; z0, z0m, z0b, fluxes, ustar)

Container for storing values used to calculate surface conditions using
Monin-Obukhov Similarity Theory. See SurfaceFluxes docs for more information.

- z0: Roughness
- z0m: Roughness for momentum
- z0b: Roughness for scalars
- fluxes: Sensible and latent heat fluxes
- ustar: Friction velocity

Valid combinations:
 - roughness
 - roughness and fluxes
 - roughness and ustar
 - roughness and fluxes and ustar
Roughnesses can be specified by either z0 or both z0m and z0b.
"""
struct MoninObukhov{
    FT,
    PFN <: Union{PrescribedFluxes{FT}, Nothing},
    FTN <: Union{FT, Nothing},
} <: SurfaceParameterization{FT}
    z0m::FT
    z0b::FT
    fluxes::PFN
    ustar::FTN
end

function MoninObukhov(;
    z0 = nothing,
    z0m = nothing,
    z0b = nothing,
    fluxes = nothing,
    ustar = nothing,
)
    if !isnothing(z0)
        if isnothing(z0m) && isnothing(z0b)
            z0m = z0b = z0
        else
            error("Cannot specify z0 and z0m/z0b")
        end
    end
    m_o(z0m::Number, z0b::Number, fluxes, ::Nothing) =
        MoninObukhov(z0m, z0b, fluxes, nothing)
    m_o(z0m::Number, z0b::Number, ::Nothing, ustar::Number) =
        MoninObukhov(z0m, z0b, nothing, ustar)
    m_o(z0m::Number, z0b::Number, fluxes::PrescribedFluxes, ustar::Number) =
        MoninObukhov(z0m, z0b, fluxes, ustar)
    return m_o(z0m, z0b, fluxes, ustar)
end
