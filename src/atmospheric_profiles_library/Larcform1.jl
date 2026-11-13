# Values are derived from Pithan 2016 and following assumptions:
# 1) atmospheric instantiated in hydrostatic balance
# 2) atmospheric is well described as an ideal gas
# 3) surface pressure is 1013hPa

"""
    Larcform1_constants(::Type{FT}, R_d, grav)

Larcform1 case constants [Pithan2016](@cite), for dry air gas constant `R_d`
(J/kg/K) and gravitational acceleration `grav` (m/s²).
"""
function Larcform1_constants(::Type{FT}, R_d, grav) where {FT}
    # Given constants — Pithan 2016, Table 1 and footnote a
    γ = FT(8e-3)               # lapse rate (K/m)
    T_0 = FT(273)              # near-surface air temperature (K)
    P_0 = FT(101300)           # surface pressure (Pa)
    P_tropopause = FT(30000)   # tropopause pressure, 300 hPa (Pa)
    q_top = FT(3e-6)           # specific humidity above tropopause (kg/kg)
    R_d = FT(R_d)
    grav = FT(grav)

    # Derived constants
    α = R_d * γ / grav  # exponent in hypsometric equation (Rγ/g)
    z_tropopause = (T_0 / γ) * (1 - (P_tropopause / P_0)^α)   # hypsometric equation for tropopause height (m)
    T_tropopause = T_0 - γ * z_tropopause                     # tropopause temperature (K)
    return (; γ, T_0, P_0, R_d, grav, P_tropopause, q_top, α, z_tropopause, T_tropopause)
end

"""
[Pithan2016](@cite)
"""
function Larcform1_T(::Type{FT}, R_d, grav) where {FT}
    LC = Larcform1_constants(FT, R_d, grav)
    return ZProfile(z -> FT(if z ≤ LC.z_tropopause
        LC.T_0 - LC.γ * z
    else
        LC.T_tropopause
    end))
end

"""
[Pithan2016](@cite)
"""
function Larcform1_p(::Type{FT}, R_d, grav) where {FT}
    LC = Larcform1_constants(FT, R_d, grav)
    return ZProfile(
        z -> FT(
            if z ≤ LC.z_tropopause
                LC.P_0 * (1 - LC.γ / LC.T_0 * z)^(1 / LC.α)
            else
                LC.P_tropopause *
                exp(-LC.grav / (LC.R_d * LC.T_tropopause) * (z - LC.z_tropopause))
            end,
        ),
    )
end

"""
[Pithan2016](@cite)
"""
function Larcform1_geostrophic_u(::Type{FT}, R_d, grav) where {FT}
    (; z_tropopause) = Larcform1_constants(FT, R_d, grav)
    return ZProfile(z -> z ≤ z_tropopause ? FT(5) : FT(0))
end

"""
[Pithan2016](@cite)
"""
Larcform1_geostrophic_v(::Type{FT}) where {FT} = ZProfile(Returns(FT(0)))

"""
[Pithan2016](@cite)
"""
function Larcform1_RH(::Type{FT}, R_d, grav) where {FT}
    (; P_0) = Larcform1_constants(FT, R_d, grav)
    rh_of_p = linear_interp(FT[0, 60000, P_0], FT[0.2, 0.2, 0.8])
    return ZProfile(rh_of_p ∘ Larcform1_p(FT, R_d, grav))
end
