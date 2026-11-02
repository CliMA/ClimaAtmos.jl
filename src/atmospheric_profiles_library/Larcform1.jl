# Values are derived from Pithan 2016 and following assumptions:
# 1) atmospheric instantiated in hydrostatic balance
# 2) atmospheric is well described as an ideal gas
# 3) surface pressure is 1013hPa

const LC = let
    # Given constants — Pithan 2016, Table 1 and footnote a
    γ = 8e-3      # lapse rate (K/m)
    T_0 = 273.0     # near-surface air temperature (K)
    P_0 = 101300.0  # surface pressure (Pa)
    R = 287.0     # gas constant for air (J/kg/K)
    g = 9.81      # gravitational acceleration (m/s²)
    P_tropopause = 30000.0   # tropopause pressure, 300 hPa (Pa)
    q_top = 3e-6      # specific humidity above tropopause (kg/kg)

    # Derived constants
    α = R * γ / g  # exponent in hypsometric equation (Rγ/g)
    z_tropopause = (T_0 / γ) * (1 - (P_tropopause / P_0)^α)   # hypsometric equation for tropopause height (m)
    T_tropopause = T_0 - γ * z_tropopause                     # tropopause temperature (K)
    (; γ, T_0, P_0, R, g, P_tropopause, q_top, α, z_tropopause, T_tropopause)
end
const Larcform1_constants = LC

"""
[Pithan2016](@cite)
"""
Larcform1_T(::Type{FT}) where {FT} = ZProfile(z -> FT(if z ≤ LC.z_tropopause
    LC.T_0 - LC.γ * z
else
    LC.T_tropopause
end))

"""
[Pithan2016](@cite)
"""
Larcform1_p(::Type{FT}) where {FT} = ZProfile(
    z -> FT(
        if z ≤ LC.z_tropopause
            LC.P_0 * (1 - LC.γ / LC.T_0 * z)^(1 / LC.α)
        else
            LC.P_tropopause * exp(-LC.g / (LC.R * LC.T_tropopause) * (z - LC.z_tropopause))
        end,
    ),
)

"""
[Pithan2016](@cite)
"""
Larcform1_geostrophic_u(::Type{FT}) where {FT} =
    ZProfile(z -> z ≤ LC.z_tropopause ? FT(5) : FT(0))

"""
[Pithan2016](@cite)
"""
Larcform1_geostrophic_v(::Type{FT}) where {FT} = ZProfile(Returns(FT(0)))

"""
[Pithan2016](@cite)
"""
function Larcform1_RH(::Type{FT}) where {FT}
    rh_of_p = linear_interp(FT[0, 60000, LC.P_0], FT[0.2, 0.2, 0.8])
    ZProfile(rh_of_p ∘ Larcform1_p(FT))
end
