@inline function _mie_efficiencies(D_m, T, radar_cfg, phase)
    FT = typeof(D_m + T)
    wavelength = FT(0.299792458) / FT(radar_cfg.freq)
    x = FT(pi) * D_m / wavelength
    m = phase === :liquid ? _m_wat(radar_cfg.freq, T) : _m_ice(radar_cfg.freq, T)
    return _mie_int(x, m)
end

# Port of QuickBeam MieInt for Inp == 1.
@inline function _mie_int(x, m)
    FT = typeof(x + real(m))
    x > zero(FT) || return zero(FT), zero(FT)
    x <= FT(12000) || return zero(FT), zero(FT)
    (isfinite(real(m)) && isfinite(imag(m))) || return zero(FT), zero(FT)

    cm = Complex{FT}(real(m), imag(m))
    y = x * cm
    nstop = _mie_nstop(x)
    nmx = floor(Int, max(FT(nstop), abs(y)) + FT(15))
    nmx <= 30000 || return zero(FT), zero(FT)

    ir = inv(cm)
    psi0 = cos(x)
    psi1 = sin(x)
    chi0 = -sin(x)
    chi1 = cos(x)
    apsi1 = psi1
    xi1 = Complex{FT}(apsi1, chi1)

    qext_sum = zero(FT)
    sp_back = zero(Complex{FT})
    sm_back = zero(Complex{FT})
    pi0_back = zero(FT)
    pi1_back = one(FT)
    tnp1 = 1

    for n in 1:nstop
        dn = FT(n)
        tnp1 += 2
        tnm1 = tnp1 - 2
        a2 = FT(tnp1) / (dn * (dn + one(FT)))
        turbo = (dn + one(FT)) / dn
        rnx = dn / x
        psi = FT(tnm1) * psi1 / x - psi0
        apsi = psi
        chi = FT(tnm1) * chi1 / x - chi0
        xi = Complex{FT}(apsi, chi)
        # Recompute the logarithmic derivative as scalar state to avoid
        # allocating the Fortran recurrence array inside broadcasted kernels.
        log_derivative = _mie_log_derivative(n, nmx, y)

        a =
            ((log_derivative * ir + rnx) * apsi - apsi1) /
            ((log_derivative * ir + rnx) * xi - xi1)
        b =
            ((log_derivative * cm + rnx) * apsi - apsi1) /
            ((log_derivative * cm + rnx) * xi - xi1)

        qext_sum += FT(tnp1) * real(a + b)

        apb = a2 * (a + b)
        amb = a2 * (a - b)
        s_back = -pi1_back
        t_back = s_back - pi0_back
        taun_back = dn * t_back - pi0_back
        sp_back += apb * (pi1_back + taun_back)
        sm_back += amb * (pi1_back - taun_back)
        pi0_back = pi1_back
        pi1_back = s_back + t_back * turbo

        psi0 = psi1
        psi1 = psi
        apsi1 = psi1
        chi0 = chi1
        chi1 = chi
        xi1 = Complex{FT}(apsi1, chi1)
    end

    qext = FT(2) * qext_sum / x^2
    qbsca = FT(4) * abs2((sp_back + sm_back) / FT(2)) / x^2
    return qext, qbsca
end

@inline function _mie_nstop(x)
    FT = typeof(x)
    if x < FT(0.02)
        return 2
    elseif x <= FT(8)
        return floor(Int, x + FT(4) * cbrt(x) + FT(2))
    elseif x < FT(4200)
        return floor(Int, x + FT(4.05) * cbrt(x) + FT(2))
    else
        return floor(Int, x + FT(4) * cbrt(x) + FT(2))
    end
end

@inline function _mie_log_derivative(n, nmx, y)
    CT = typeof(y)
    d = zero(CT)
    for k in (nmx - 1):-1:n
        a1 = CT(k + 1) / y
        d = a1 - inv(a1 + d)
    end
    return d
end

@inline function _m_wat(freq, T)
    FT = typeof(freq + T)
    tc = T - FT(273.15)
    ld = FT(100) * FT(2.99792458e8) / (freq * FT(1e9))
    es =
        FT(78.54) *
        (
            one(FT) -
            (
                FT(4.579e-3) * (tc - FT(25)) +
                FT(1.19e-5) * (tc - FT(25))^2 -
                FT(2.8e-8) * (tc - FT(25))^3
            )
        )
    ei = FT(5.27137) + FT(0.021647) * tc - FT(0.00131198) * tc^2
    a = -(FT(16.8129) / (tc + FT(273))) + FT(0.0609265)
    ls = FT(0.00033836) * exp(FT(2513.98) / (tc + FT(273)))
    sg = FT(12.5664e8)
    tm1 = (ls / ld)^(one(FT) - a)
    cos1 = cos(FT(0.5) * a * FT(pi))
    sin1 = sin(FT(0.5) * a * FT(pi))
    denominator = one(FT) + FT(2) * tm1 * sin1 + tm1^2
    e_r = ei + (((es - ei) * (one(FT) + tm1 * sin1)) / denominator)
    e_i = (((es - ei) * tm1 * cos1) / denominator) + ((sg * ld) / FT(1.885e11))
    refractive_index = sqrt(Complex(e_r, e_i))
    return Complex(real(refractive_index), -imag(refractive_index))
end

@inline function _m_ice(freq, T)
    FT = typeof(freq + T)
    alam = FT(3e5) / freq
    (_ice_const(FT, ICE_WLMIN) <= alam <= _ice_const(FT, ICE_WLMAX)) ||
        return Complex(FT(NaN), FT(NaN))

    if alam < _ice_const(FT, ICE_CUTICE)
        i = _ice_upper_index(ICE_WL, alam)
        x1 = log(_ice_const(FT, ICE_WL[i - 1]))
        x2 = log(_ice_const(FT, ICE_WL[i]))
        x = log(alam)
        nr = _linear_interpolate(
            x,
            x1,
            x2,
            _ice_const(FT, ICE_TABRE[i - 1]),
            _ice_const(FT, ICE_TABRE[i]),
        )
        log_ni = _linear_interpolate(
            x,
            x1,
            x2,
            log(abs(_ice_const(FT, ICE_TABIM[i - 1]))),
            log(abs(_ice_const(FT, ICE_TABIM[i]))),
        )
        return Complex(nr, -exp(log_ni))
    else
        tk = min(max(T, _ice_const(FT, ICE_TEMREF[4])), _ice_const(FT, ICE_TEMREF[1]))
        lt1 = _ice_temperature_lower_index(tk)
        lt2 = lt1 - 1
        i = _ice_upper_index(ICE_WLT, alam)
        x1 = log(_ice_const(FT, ICE_WLT[i - 1]))
        x2 = log(_ice_const(FT, ICE_WLT[i]))
        x = log(alam)

        ylo = _linear_interpolate(
            x,
            x1,
            x2,
            _ice_const(FT, _ice_table_value(ICE_TABRET, i - 1, lt1)),
            _ice_const(FT, _ice_table_value(ICE_TABRET, i, lt1)),
        )
        yhi = _linear_interpolate(
            x,
            x1,
            x2,
            _ice_const(FT, _ice_table_value(ICE_TABRET, i - 1, lt2)),
            _ice_const(FT, _ice_table_value(ICE_TABRET, i, lt2)),
        )
        t1 = _ice_const(FT, ICE_TEMREF[lt1])
        t2 = _ice_const(FT, ICE_TEMREF[lt2])
        nr = _linear_interpolate(tk, t1, t2, ylo, yhi)

        ylo = _linear_interpolate(
            x,
            x1,
            x2,
            log(abs(_ice_const(FT, _ice_table_value(ICE_TABIMT, i - 1, lt1)))),
            log(abs(_ice_const(FT, _ice_table_value(ICE_TABIMT, i, lt1)))),
        )
        yhi = _linear_interpolate(
            x,
            x1,
            x2,
            log(abs(_ice_const(FT, _ice_table_value(ICE_TABIMT, i - 1, lt2)))),
            log(abs(_ice_const(FT, _ice_table_value(ICE_TABIMT, i, lt2)))),
        )
        ni = exp(_linear_interpolate(tk, t1, t2, ylo, yhi))
        return Complex(nr, -ni)
    end
end

@inline _linear_interpolate(x, x1, x2, y1, y2) =
    ((x - x1) * (y2 - y1) / (x2 - x1)) + y1

@inline function _ice_upper_index(table, x)
    FT = typeof(x)
    for i in 2:length(table)
        x <= _ice_const(FT, table[i]) && return i
    end
    return length(table)
end

@inline function _ice_temperature_lower_index(tk)
    FT = typeof(tk)
    for i in 2:length(ICE_TEMREF)
        tk >= _ice_const(FT, ICE_TEMREF[i]) && return i
    end
    return length(ICE_TEMREF)
end

@inline _ice_const(::Type{FT}, x) where {FT} = FT(Float32(x))

@inline _ice_table_value(table, i, j) = table[i + (j - 1) * 62]
