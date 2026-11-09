"""
:(
"""
SP_u(::Type{FT}) where {FT} = ZProfile(z -> FT(1))

"""
:(
"""
SP_v(::Type{FT}) where {FT} = ZProfile(z -> FT(0))

"""
:(
"""
SP_θ_liq_ice(::Type{FT}) where {FT} =
    ZProfile(z -> if z <= 974.0
        FT(300)
    elseif z < 1074.0
        FT(300) + (z - 974) * FT(0.08)
    else
        FT(308) + (z - 1074) * FT(0.003)
    end)

"""
:(
"""
SP_q_tot(::Type{FT}) where {FT} = ZProfile(z -> FT(0))

"""
:(
"""
SP_geostrophic_u(::Type{FT}) where {FT} = ZProfile(z -> FT(1))

"""
:(
"""
SP_geostrophic_v(::Type{FT}) where {FT} = ZProfile(z -> FT(0))
