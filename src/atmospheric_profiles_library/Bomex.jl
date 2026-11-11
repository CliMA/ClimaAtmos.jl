"""
[Siebesma2003](@cite)
"""
Bomex_θ_liq_ice(::Type{FT}) where {FT} = ZProfile(z -> if z <= 520.0
    FT(298.7)
elseif z > 520.0 && z <= 1480.0
    FT(298.7) + (z - 520) * (FT(302.4) - FT(298.7)) / (1480 - 520)
elseif z > 1480.0 && z <= 2000
    FT(302.4) + (z - 1480) * (FT(308.2) - FT(302.4)) / (2000 - 1480)
elseif z > 2000.0
    FT(308.2) + (z - 2000) * (FT(311.85) - FT(308.2)) / (3000 - 2000)
else
    FT(0)
end)
"""
[Siebesma2003](@cite)
"""
Bomex_q_tot(::Type{FT}) where {FT} = ZProfile(z -> if z <= 520
    (FT(17.0) + z * (FT(16.3) - 17) / 520) / 1000
elseif z > 520.0 && z <= 1480.0
    (FT(16.3) + (z - 520) * (FT(10.7) - FT(16.3)) / (1480 - 520)) / 1000
elseif z > 1480.0 && z <= 2000.0
    (FT(10.7) + (z - 1480) * (FT(4.2) - FT(10.7)) / (2000 - 1480)) / 1000
else
    (FT(4.2) + (z - 2000) * (3 - FT(4.2)) / (3000 - 2000)) / 1000
end)
"""
[Siebesma2003](@cite)
"""
Bomex_u(::Type{FT}) where {FT} = ZProfile(z -> if z <= 700.0
    FT(-8.75)
else
    FT(-8.75) + (z - 700) * (FT(-4.61) - FT(-8.75)) / (3000 - 700)
end)
"""
[Siebesma2003](@cite)
"""
Bomex_tke(::Type{FT}) where {FT} = ZProfile(z -> if (z <= 2500.0)
    FT(1) - z / 3000
else
    FT(0)
end)
"""
TMP TKE profile for testing
"""
function Bomex_tke_prescribed(::Type{FT}) where {FT}
    z_in = FT[25.0, 75.0, 125.0, 175.0, 225.0, 275.0, 325.0, 375.0, 425.0, 475.0, 525.0,
        575.0, 625.0, 675.0, 725.0, 775.0, 825.0, 875.0, 925.0, 975.0, 1025.0, 1075.0,
        1125.0, 1175.0, 1225.0, 1275.0, 1325.0, 1375.0, 1425.0, 1475.0, 1525.0, 1575.0,
        1625.0, 1675.0, 1725.0, 1775.0, 1825.0, 1875.0, 1925.0, 1975.0, 2025.0, 2075.0,
        2125.0, 2175.0, 2225.0, 2275.0, 2325.0, 2375.0, 2425.0, 2475.0, 2525.0, 2575.0,
        2625.0, 2675.0, 2725.0, 2775.0, 2825.0, 2875.0, 2925.0, 2975.0]
    tke_in =
        FT[0.3260, 0.4316, 0.4764, 0.4782, 0.4666, 0.4533, 0.4303, 0.3668, 0.2608, 0.1669,
            0.1264, 0.1129, 0.1127, 0.1183, 0.1258, 0.1322, 0.1367, 0.1403, 0.1434, 0.1456,
            0.1468, 0.1470, 0.1458, 0.1423, 0.1351, 0.1224, 0.1033, 0.0793, 0.0542, 0.0322,
            0.0162, 0.0068, 0.0024, 0.0007, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0]
    return ZProfile(linear_interp(z_in, tke_in))
end

# Geostrophic velocity profiles
"""
:(
"""
Bomex_geostrophic_u(::Type{FT}) where {FT} = ZProfile(z -> -10 + FT(1.8e-3) * z)
"""
:(
"""
Bomex_geostrophic_v(::Type{FT}) where {FT} = ZProfile(z -> FT(0))

"""
dTdt(Π, z)
"""
Bomex_dTdt(::Type{FT}) where {FT} = ΠZProfile(
    (Π, z) -> if z <= 1500.0
        FT(-2 / (3600 * 24)) * Π
    else
        FT(-2 / (3600 * 24) + (z - 1500) * (0 - -2 / (3600 * 24)) / (3000 - 1500)) * Π
    end,
)

# Large-scale drying
"""
:(
"""
Bomex_dqtdt(::Type{FT}) where {FT} = ZProfile(z -> if z <= 300.0
    FT(-1.2e-8)   #kg/(kg * s)
elseif z > 300.0 && z <= 500.0
    FT(-1.2e-8) + (z - 300) * (0 - FT(-1.2e-8)) / (500 - 300) #kg/(kg * s)
else
    FT(0)
end)

# Large scale subsidence
"""
:(
"""
Bomex_subsidence(::Type{FT}) where {FT} = ZProfile(z -> if z <= 1500.0
    FT(0) + z * (FT(-0.65) / 100 - 0) / (1500 - 0)
elseif z > 1500.0 && z <= 2100.0
    FT(-0.65) / 100 + (z - 1500) * (0 - FT(-0.65) / 100) / (2100 - 1500)
else
    FT(0)
end)
