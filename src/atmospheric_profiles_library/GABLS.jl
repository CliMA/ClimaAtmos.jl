"""
:(
"""
GABLS_u(::Type{FT}) where {FT} = ZProfile(z -> FT(8.0))
"""
:(
"""
GABLS_v(::Type{FT}) where {FT} = ZProfile(z -> FT(0.0))

"""
:(
"""
GABLS_θ_liq_ice(::Type{FT}) where {FT} =
    ZProfile(z -> if z <= 100.0
        FT(265)
    else
        FT(265) + (z - 100) * FT(0.01)
    end)

"""
:(
"""
GABLS_q_tot(::Type{FT}) where {FT} = ZProfile(z -> FT(0))

"""
:(
"""
GABLS_tke(::Type{FT}) where {FT} =
    ZProfile(z -> if z <= 250.0
        FT(0.4) * (1 - z / 250) * (1 - z / 250) * (1 - z / 250)
    else
        FT(0)
    end)

"""
TMP TKE profile for testing
"""
function GABLS_tke_prescribed(::Type{FT}) where {FT}
    z_in = FT[25.0, 75.0, 125.0, 175.0, 225.0, 275.0, 325.0, 375.0]
    tke_in = FT[0.4662, 0.3873, 0.2777, 0.0277, 0.0003, 5.89e-8, 0.0, 0.0]
    return ZProfile(linear_interp(z_in, tke_in))
end

"""
:(
"""
GABLS_geostrophic_ug(::Type{FT}) where {FT} = ZProfile(z -> FT(8))
"""
:(
"""
GABLS_geostrophic_vg(::Type{FT}) where {FT} = ZProfile(z -> FT(0))
