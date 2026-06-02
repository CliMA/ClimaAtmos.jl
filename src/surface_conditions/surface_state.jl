# Naming convention: surface_<value> is the value at the surface, whereas
# sfc_<value> is a field of values at the surface, and likewise for
# interior_<value> and int_<value>.

abstract type PrescribedFluxes{FT} end

"""
    SurfaceParameterization

Abstract supertype for surface flux closures. Concrete subtypes
([`MoninObukhov`](@ref), [`ExchangeCoefficients`](@ref)) determine how
`surface_state_to_conditions` turns the air–surface state difference into the
turbulent surface fluxes.
"""
abstract type SurfaceParameterization{FT} end

Base.broadcastable(p::SurfaceParameterization) = tuple(p)

float_type(::Type{<:SurfaceParameterization{FT}}) where {FT} = FT

"""
    SurfaceBoundaryOverrides(; p, q_vap, u, v, gustiness, beta)

Per-point overrides for surface boundary values used by `SurfaceFluxes`. Fields
default to `nothing`, in which case sensible defaults are used:

- `p`: surface pressure (default: hydrostatic from interior)
- `q_vap`: surface specific humidity (default: `q_vap_sat` at `T_sfc`)
- `u`, `v`: surface horizontal winds (default: 0)
- `gustiness`: turbulent gustiness (default: 1)
- `beta`: moisture availability (default: 1)

For the coupler use case, a `Fields.Field{<:SurfaceBoundaryOverrides}` may be
stored on the cache so that the coupler can override per-cell values.
"""
Base.@kwdef struct SurfaceBoundaryOverrides{PN, QN, UN, VN, GN, BN}
    p::PN = nothing
    q_vap::QN = nothing
    u::UN = nothing
    v::VN = nothing
    gustiness::GN = nothing
    beta::BN = nothing
end

"""
    HeatFluxes(; shf, lhf = nothing)

Prescribed surface turbulent energy fluxes, used as the `fluxes` field of a
[`MoninObukhov`](@ref) closure. Both use the sign convention that positive is
*upward* (directed from the surface into the atmosphere).

- `shf`: sensible heat flux (W/m²).
- `lhf`: latent heat flux (W/m²). Optional — `nothing` is treated as zero, and
  `lhf` must be left unset for a `DryModel` (specifying it with a dry model is an
  error).
"""
Base.@kwdef struct HeatFluxes{FT, FTN <: Union{FT, Nothing}} <: PrescribedFluxes{FT}
    shf::FT
    lhf::FTN = nothing
end

"""
    θAndQFluxes(; θ_flux, q_flux = nothing)

Prescribed surface *kinematic* fluxes of potential temperature and total
specific humidity, used as the `fluxes` field of a [`MoninObukhov`](@ref)
closure. They are converted per surface point into the sensible/latent heat
fluxes actually applied, via `shf = θ_flux * ρ_sfc * cp_m` and
`lhf = q_flux * ρ_sfc * Lᵥ`. Positive is *upward* (surface into atmosphere).

- `θ_flux`: potential-temperature flux (K·m/s).
- `q_flux`: total-specific-humidity flux (kg/kg·m/s). Optional — `nothing` is
  treated as zero, and `q_flux` must be left unset for a `DryModel`.
"""
Base.@kwdef struct θAndQFluxes{FT, FTN <: Union{FT, Nothing}} <: PrescribedFluxes{FT}
    θ_flux::FT
    q_flux::FTN = nothing
end

"""
    ExchangeCoefficients(; Cd, Ch)
    ExchangeCoefficients(C)

Bulk-aerodynamic surface flux closure with fixed, dimensionless exchange
coefficients — a [`SurfaceParameterization`](@ref) alternative to
[`MoninObukhov`](@ref) in which the turbulent fluxes scale linearly with the
near-surface wind speed and the air–surface differences (rather than being
derived from Monin–Obukhov stability).

- `Cd`: momentum (drag) exchange coefficient.
- `Ch`: thermal/scalar (heat and moisture) exchange coefficient.

The single-argument form `ExchangeCoefficients(C)` sets `Cd = Ch = C`.
"""
Base.@kwdef struct ExchangeCoefficients{FT} <: SurfaceParameterization{FT}
    Cd::FT
    Ch::FT
end
ExchangeCoefficients(C) = ExchangeCoefficients(Cd = C, Ch = C)

"""
    MoninObukhov(; z0, z0m, z0b, fluxes, shf, lhf, θ_flux, q_flux, ustar)

Container for storing values used to calculate surface conditions using
Monin-Obukhov Similarity Theory. See the
[SurfaceFluxes.jl MOST documentation](https://clima.github.io/SurfaceFluxes.jl/dev/SurfaceFluxes/#Monin-Obukhov-Similarity-Theory-(MOST))
for more information.

## Roughness (required)
- `z0`: Roughness (sets both `z0m` and `z0b`)
- `z0m`, `z0b`: Roughness for momentum and scalars (specify both, or use `z0`)

## Prescribed fluxes (optional) — specify via one of:
- `fluxes`: A [`HeatFluxes`](@ref)/[`θAndQFluxes`](@ref) struct, or a callable
  `(t, FT) -> HeatFluxes/θAndQFluxes` for time-varying fluxes (resolved once per
  surface update by `resolve_flux_scheme`, before the per-cell broadcast)
- `shf`, `lhf`: Sensible/latent heat fluxes (W/m²) — constructs `HeatFluxes`
- `θ_flux`, `q_flux`: θ and q fluxes (K·m/s, kg/kg·m/s) — constructs `θAndQFluxes`

## Other (optional)
- `ustar`: Friction velocity (m/s)

Valid combinations:
 - roughness
 - roughness and fluxes
 - roughness and ustar
 - roughness and fluxes and ustar
"""
struct MoninObukhov{
    FT,
    PFN <: Union{PrescribedFluxes, Nothing, Function},
    FTN <: Union{FT, Nothing},
} <: SurfaceParameterization{FT}
    z0m::FT
    z0b::FT
    fluxes::PFN
    ustar::FTN
end

function MoninObukhov(;
    z0 = nothing, z0m = nothing, z0b = nothing,
    fluxes = nothing, shf = nothing, lhf = nothing,
    θ_flux = nothing, q_flux = nothing, ustar = nothing,
)
    # Roughness handling
    if !isnothing(z0)
        if isnothing(z0m) && isnothing(z0b)
            z0m = z0b = z0
        else
            error("Cannot specify z0 and z0m/z0b")
        end
    end

    # Flux handling: flat kwargs --> PrescribedFluxes struct
    has_heat = !isnothing(shf) || !isnothing(lhf)
    has_theta = !isnothing(θ_flux) || !isnothing(q_flux)
    if !isnothing(fluxes) && (has_heat || has_theta)
        error(
            "Cannot specify `fluxes` alongside individual flux kwargs (shf, lhf, θ_flux, q_flux)",
        )
    end
    if has_heat && has_theta
        error(
            "Cannot specify both heat-flux kwargs (shf, lhf) and θ-flux kwargs (θ_flux, q_flux)",
        )
    end
    if has_heat
        fluxes = HeatFluxes(; shf, lhf)
    elseif has_theta
        fluxes = θAndQFluxes(; θ_flux, q_flux)
    end

    m_o(z0m::Number, z0b::Number, fluxes, ::Nothing) =
        MoninObukhov(z0m, z0b, fluxes, nothing)
    m_o(z0m::Number, z0b::Number, ::Nothing, ustar::Number) =
        MoninObukhov(z0m, z0b, nothing, ustar)
    # `fluxes` may be a `PrescribedFluxes` struct or a time-varying callable
    # `(t, FT) -> PrescribedFluxes` (resolved later by `resolve_flux_scheme`).
    m_o(
        z0m::Number,
        z0b::Number,
        fluxes::Union{PrescribedFluxes, Function},
        ustar::Number,
    ) =
        MoninObukhov(z0m, z0b, fluxes, ustar)
    return m_o(z0m, z0b, fluxes, ustar)
end
