"""
[Stevens2005](@cite)
"""
Dycoms_RF01_q_tot(::Type{FT}) where {FT} = ZProfile(z -> if z <= 840.0
    FT(9.0) / FT(1000.0)
else
    FT(1.5) / FT(1000.0)
end)

"""
[Stevens2005](@cite)
"""
Dycoms_RF01_θ_liq_ice(::Type{FT}) where {FT} = ZProfile(z -> if z <= 840.0
    FT(289)
else
    FT(297.5) + (z - FT(840))^FT(1.0 / 3.0)
end)

"""
[Stevens2005](@cite)
"""
Dycoms_RF01_u0(::Type{FT}) where {FT} = ZProfile(z -> FT(7))

"""
[Stevens2005](@cite)
"""
Dycoms_RF01_v0(::Type{FT}) where {FT} = ZProfile(z -> FT(-5.5))

"""
[Stevens2005](@cite)
"""
Dycoms_RF01_tke(::Type{FT}) where {FT} = ZProfile(z -> if z <= 800.0
    FT(1) - z / FT(1000)
else
    FT(0)
end)
"""
TMP TKE profile for testing
"""
function Dycoms_RF01_tke_prescribed(::Type{FT}) where {FT}
    z_in =
        FT[25.0, 75.0, 125.0, 175.0, 225.0, 275.0, 325.0, 375.0, 425.0, 475.0, 525.0, 575.0,
            625.0, 675.0, 725.0,
            775.0, 825.0, 875.0, 925.0, 975.0, 1025.0, 1075.0, 1125.0, 1175.0, 1225.0,
            1275.0, 1325.0, 1375.0, 1425.0, 1475.0]
    tke_in =
        FT[0.2726, 0.5479, 0.6597, 0.7079, 0.7285, 0.7343, 0.7319, 0.7252, 0.7166, 0.7064,
            0.6887, 0.6317,
            0.6362, 0.6266, 0.5832, 0.4633, 0.0504, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return ZProfile(linear_interp(z_in, tke_in))
end
