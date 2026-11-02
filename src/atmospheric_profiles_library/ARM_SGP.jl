"""
[Brown2002](@cite)
"""
function ARM_SGP_z(::Type{FT}) where {FT}
    z_in = FT[0.0, 50.0, 350.0, 650.0, 700.0, 1300.0, 2500.0, 5500.0] #LES z is in meters
    return z_in
end
"""
[Brown2002](@cite)
"""
function ARM_SGP_θ_liq_ice(::Type{FT}) where {FT}
    z_in = ARM_SGP_z(FT)
    θ_liq_ice_in = FT[299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2] # K
    return ZProfile(linear_interp(z_in, θ_liq_ice_in))
end
"""
[Brown2002](@cite)
"""
ARM_SGP_u(::Type{FT}) where {FT} = ZProfile(z -> FT(10))

"""
[Brown2002](@cite)
"""
function ARM_SGP_q_tot(::Type{FT}) where {FT}
    z_in = ARM_SGP_z(FT)
    r_in = FT[15.2, 15.17, 14.98, 14.8, 14.7, 13.5, 3.0, 3.0] ./ 1000 # qt should be in kg/kg
    q_tot_in = r_in ./ (1 .+ r_in)
    return ZProfile(linear_interp(z_in, q_tot_in))
end
"""
[Brown2002](@cite)
"""
function ARM_SGP_tke(::Type{FT}) where {FT}
    return ZProfile(z -> if z <= 2500.0
        FT(1) - z / 3000
    else
        FT(0)
    end)
end

"""
TMP TKE profile for testing
"""
function ARM_SGP_tke_prescribed(::Type{FT}) where {FT}
    z_in =
        FT[22.0, 66.0, 110.0, 154.0, 198.0, 242.0, 286.0, 330.0, 374.0, 418.0, 462.0, 506.0,
            550.0, 594.0, 638.0, 682.0, 726.0, 770.0, 814.0, 858.0, 902.0, 946.0, 990.0,
            1034.0,
            1078.0, 1122.0, 1166.0, 1210.0, 1254.0, 1298.0, 1342.0, 1386.0, 1430.0, 1474.0,
            1518.0, 1562.0, 1606.0, 1650.0, 1694.0, 1738.0, 1782.0, 1826.0, 1870.0, 1914.0,
            1958.0, 2002.0, 2046.0, 2090.0, 2134.0, 2178.0, 2222.0, 2266.0, 2310.0, 2354.0,
            2398.0, 2442.0, 2486.0, 2530.0, 2574.0, 2618.0, 2662.0, 2706.0, 2750.0, 2794.0,
            2838.0, 2882.0, 2926.0, 2970.0, 3014.0, 3058.0, 3102.0, 3146.0, 3190.0, 3234.0,
            3278.0, 3322.0, 3366.0, 3410.0, 3454.0, 3498.0, 3542.0, 3586.0, 3630.0, 3674.0,
            3718.0, 3762.0, 3806.0, 3850.0, 3894.0, 3938.0, 3982.0, 4026.0, 4070.0, 4114.0,
            4158.0, 4202.0, 4246.0, 4290.0, 4334.0, 4378.0]
    tke_in =
        FT[0.2940, 0.0431, 0.0015, 0.0032, 0.0115, 0.0177, 0.0221, 0.0251, 0.0269, 0.0277,
            0.0274, 0.0263, 0.0242, 0.0212, 0.0174, 0.0128, 0.0078, 0.0032, 0.0006, 0.0001,
            0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0]
    return ZProfile(linear_interp(z_in, tke_in))
end

"""
[Brown2002](@cite)
"""
function ARM_SGP_time(::Type{FT}) where {FT}
    t_in = FT[0.0, 3.0, 6.0, 9.0, 12.0, 14.5] .* 3600 #LES time is in sec
    return t_in
end

"""
[Brown2002](@cite)
"""
function ARM_SGP_dTdt(::Type{FT}) where {FT}
    t_in = ARM_SGP_time(FT)
    # Advective forcing for theta [K/h] converted to [K/sec]
    AT_in = FT[0.0, 0.0, 0.0, -0.08, -0.016, -0.016] ./ 3600
    # Radiative forcing for theta [K/h] converted to [K/sec]
    RT_in = FT[-0.125, 0.0, 0.0, 0.0, 0.0, -0.1] ./ 3600
    dTdt_A = linear_interp(t_in, AT_in)
    dTdt_R = linear_interp(t_in, RT_in)
    prof = (t, z) -> if z <= 1000.0
        FT(dTdt_A(t)+dTdt_R(t))
    elseif z > 1000.0 && z <= 2000.0
        FT((dTdt_A(t)+dTdt_R(t)) * (1 - (z - 1000) / 1000))
    else
        FT(0)
    end
    return TimeZProfile(prof)
end

"""
[Brown2002](@cite)
"""
function ARM_SGP_dqtdt(::Type{FT}) where {FT}
    t_in = ARM_SGP_time(FT)
    # Radiative forcing for qt converted to [kg/kg/sec]
    Rqt_in = FT[0.08, 0.02, 0.04, -0.1, -0.16, -0.3] ./ 1000 ./ 3600
    dqtdt = linear_interp(t_in, Rqt_in)
    prof = (Π, t, z) -> if z <= 1000.0
        FT(dqtdt(t) * Π)
    elseif z > 1000.0 && z <= 2000.0
        FT(dqtdt(t) * Π * (1 - (z - 1000) / 1000))
    else
        FT(0)
    end
    return ΠTimeZProfile(prof)
end

"""
[Brown2002](@cite)
"""
function ARM_SGP_shf(::Type{FT}) where {FT}
    t_Sur_in = FT[0.0, 4.0, 6.5, 7.5, 10.0, 12.5, 14.5] .* 3600 #LES time is in sec
    shf = FT[-30.0, 90.0, 140.0, 140.0, 100.0, -10, -10] # W/m^2
    profile = linear_interp(t_Sur_in, shf)
    return TimeProfile(profile)
end

"""
[Brown2002](@cite)
"""
function ARM_SGP_lhf(::Type{FT}) where {FT}
    t_Sur_in = FT[0.0, 4.0, 6.5, 7.5, 10.0, 12.5, 14.5] .* 3600 #LES time is in sec
    lhf = FT[5.0, 250.0, 450.0, 500.0, 420.0, 180.0, 0.0] # W/m^2
    profile = linear_interp(t_Sur_in, lhf)
    return TimeProfile(profile)
end
