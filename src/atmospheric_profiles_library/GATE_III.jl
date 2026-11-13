"""
[Khairoutdinov2009](@cite)
"""
function GATE_III_z(::Type{FT}) where {FT}
    # TODO: should the last element in here be 22, not 27, km?
    # TODO: unify with GATE_III_z_in
    z_in =
        FT[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
            7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
            14.0,
            14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 27.0] .* 1000 #z is in meters
    return z_in
end

"""
[Khairoutdinov2009](@cite)
"""
function GATE_III_q_tot(::Type{FT}) where {FT}
    z_in = GATE_III_z(FT)
    r_in =
        FT[16.5, 16.5, 13.5, 12.0, 10.0, 8.7, 7.1, 6.1, 5.2, 4.5, 3.6, 3.0, 2.3, 1.75, 1.3,
            0.9, 0.5, 0.25, 0.125, 0.065, 0.003, 0.0015, 0.0007, 0.0003, 0.0001, 0.0001,
            0.0001, 0.0001,
            0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
        ] ./ 1000 # mixing ratio should be in kg/kg
    q_tot = r_in ./ (1 .+ r_in) # convert mixing ratio to specific humidity
    return ZProfile(linear_interp(z_in, q_tot))
end


"""
[Khairoutdinov2009](@cite)
"""
function GATE_III_u(::Type{FT}) where {FT}
    z_in = GATE_III_z(FT)
    U_in = FT[-1, -1.75, -2.5, -3.6, -6.0, -8.75, -11.75, -13.0, -13.1, -12.1, -11.0, -8.5,
        -5.0, -2.6, 0.0,
        0.5, 0.4, 0.3, 0.0, -1.0, -2.5, -3.5, -4.5, -4.8, -5.0, -3.5, -2.0, -1.0, -1.0,
        -1.0,
        -1.5, -2.0, -2.5, -2.6, -2.7, -3.0, -3.0, -3.0] # [m/s]
    return ZProfile(linear_interp(z_in, U_in))
end

"""
[Khairoutdinov2009](@cite)
"""
function GATE_III_T(::Type{FT}) where {FT}
    # temperature is taken from a different input plot at different z levels
    z_in = GATE_III_z(FT)
    T_in = FT[
        299.184,
        294.836,
        294.261,
        288.773,
        276.698,
        265.004,
        253.930,
        243.662,
        227.674,
        214.266,
        207.757,
        201.973,
        198.278,
        197.414,
        198.110,
        198.110,
    ]
    z_T_in =
        FT[
            0.0,
            0.492,
            0.700,
            1.698,
            3.928,
            6.039,
            7.795,
            9.137,
            11.055,
            12.645,
            13.521,
            14.486,
            15.448,
            16.436,
            17.293,
            22.0,
        ] .* 1000 # for km
    return ZProfile(linear_interp(z_T_in, T_in))
end

"""
[Khairoutdinov2009](@cite)
"""
function GATE_III_tke(::Type{FT}) where {FT}
    return ZProfile(z -> if z <= 2500.0
        FT(1 - z / 3000)
    else
        FT(0)
    end)
end

#=

Unused data:

u_in = FT[-1, -1.75, -2.5, -3.6,  -6.0, -8.75, -11.75,
          -12.9, -13.1, -12.1, -11.0,  -8.5,  -5.0, -2.6,
          0.0, 0.5, 0.4, 0.3, 0.0, -1.0, -3.0, -3.5, -4.5,
          -4.6, -5.0, -3.5, -2.0, -1.0, -1.0, -1.0, -1.5,
          -2.0, -2.5, -2.6, -2.7, -3.0, -3.0]
=#
"""
[Khairoutdinov2009](@cite)
"""
function GATE_III_z_in(::Type{FT}) where {FT}
    #LES z is in meters
    z_in =
        FT[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
            6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
            13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0] .* 1000
    return z_in
end

"""
[Khairoutdinov2009](@cite)
"""
function GATE_III_dqtdt(::Type{FT}) where {FT}
    z_in = GATE_III_z_in(FT)
    # Advective mixing ratio forcing [g/kg/d] converted to [kg/kg/sec]
    r_tend_in =
        FT[0.0, 1.2, 2.0, 2.3, 2.2, 2.1, 1.9, 1.7, 1.5, 1.35, 1.22, 1.08, 0.95,
            0.82, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0025, 0.0012, 0.0006, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ./ (24 * 3600) ./ 1000
    # Specific humidity of the initial profile at the forcing heights
    q_in = GATE_III_q_tot(FT).(z_in)
    # dq/dt = (dr/dt) / (1 + r)² = (dr/dt) (1 - q)², from q = r / (1 + r)
    q_tend_in = r_tend_in .* (1 .- q_in) .^ 2
    return ZProfile(linear_interp(z_in, q_tend_in))
end

"""
[Khairoutdinov2009](@cite)
"""
function GATE_III_dTdt(::Type{FT}) where {FT}
    z_in = GATE_III_z_in(FT)
    # Radiative forcing for T [K/d] converted to [K/sec]
    RAD_in =
        FT[-2.9, -1.1, -0.8, -1.1, -1.25, -1.35, -1.4, -1.4, -1.44, -1.52, -1.6, -1.54,
            -1.49,
            -1.43, -1.36, -1.3, -1.25, -1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85,
            -0.8,
            -0.75, -0.7, -0.6, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ./ (24 * 3600)
    # Radiative T forcing [K/d] converted to [K/sec]
    Ttend_in =
        FT[0.0, -1.0, -2.2, -3.0, -3.5, -3.8, -4.0, -4.1, -4.2, -4.2, -4.1, -4.0, -3.85,
            -3.7, -3.5, -3.25, -3.0, -2.8, -2.5, -2.1, -1.7, -1.3, -1.0, -0.7, -0.5, -0.4,
            -0.3, -0.2, -0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ./ (24*3600)

    profile_T = linear_interp(z_in, Ttend_in)
    profile_R = linear_interp(z_in, RAD_in)
    return ZProfile(z -> profile_T(z) + profile_R(z))
end
