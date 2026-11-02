
# 825m: base of initial inversion
# 2045m: nominal domain top

"""
[ovchinnikov_intercomparison_2014](@cite)
"""
function ISDAC_q_tot(::Type{FT}) where {FT}
    ZProfile(z -> FT(if z < 400
        1.5 - 0.00075 * (z - 400)
    elseif 400 ≤ z < 825
        1.5
    elseif 825 ≤ z < 2045
        1.2
    else  #2045 ≤ z
        0.5 - 0.000075 * (z - 2045)
    end / 1000))  # kg/kg
end

"""
[ovchinnikov_intercomparison_2014](@cite)
"""
function ISDAC_θ_liq_ice(::Type{FT}) where {FT}
    ZProfile(z -> FT(if z < 400
        265 + 0.004 * (z - 400)
    elseif 400 ≤ z < 825
        265
    elseif 825 ≤ z < 2045
        266 + (z - 825)^0.3
    else  #2045 ≤ z
        271 + (z - 2000)^0.33
    end))  # K
end

"""
[ovchinnikov_intercomparison_2014](@cite)  # m/s
"""
ISDAC_u(::Type{FT}) where {FT} = ZProfile(z -> FT(-7))  # m/s

"""
[ovchinnikov_intercomparison_2014](@cite)  # m/s
"""
ISDAC_v(::Type{FT}) where {FT} = ZProfile(z -> FT(-2 + 0.003z))  # m/s

"""
[ovchinnikov_intercomparison_2014](@cite)
"""
ISDAC_subsidence(::Type{FT}) where {FT} =
    ZProfile(z -> FT(z < 825 ? -5e-6z : -0.4125e-2))  # m/s

"""
[ovchinnikov_intercomparison_2014](@cite)  # m²/s²
"""
ISDAC_tke(::Type{FT}) where {FT} = ZProfile(z -> FT(0.1))  # m²/s²

"""
[ovchinnikov_intercomparison_2014](@cite)
"""
function ISDAC_inv_τ_scalar(::Type{FT}) where {FT}
    z₁ = 1200  # m
    z₂ = 1500  # m
    hr = 3600  # s
    ZProfile(z -> FT(if z < z₁
        0
    elseif z₁ ≤ z ≤ z₂
        1 / hr * (1 - cos(π * (z - z₁) / (z₂ - z₁))) / 2
    else  #z > z₂
        1 / hr
    end))  # s⁻¹
end

"""
[ovchinnikov_intercomparison_2014](@cite)
"""
function ISDAC_inv_τ_wind(::Type{FT}) where {FT}
    zᵤᵥ = 825  # m
    hr = 3600  # s
    ZProfile(z -> FT(if z ≤ zᵤᵥ
        1 / 2hr * (1 - cos(π * z / zᵤᵥ)) / 2
    else  #z > zᵤᵥ
        1 / 2hr
    end))  # s⁻¹
end
