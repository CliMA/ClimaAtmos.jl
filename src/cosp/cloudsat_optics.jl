module COSPCloudSatOptics

export CloudSatRadarConfig,
    DEFAULT_CLOUDSAT_RADAR_CONFIG, quickbeam_optics!, cloudsat_optics!

const HYDRO_CLASSES = (:lcl, :icl, :rai, :sno)
const Q_KEYS = (; lcl = :q_lcl, icl = :q_icl, rai = :q_rai, sno = :q_sno)

struct CloudSatRadarConfig{FT}
    freq::FT
    k2::FT
    use_gas_abs::Bool
    min_mixing_ratio::FT
end

Base.broadcastable(cfg::CloudSatRadarConfig) = Ref(cfg)

CloudSatRadarConfig(; kwargs...) = CloudSatRadarConfig(Float64; kwargs...)
CloudSatRadarConfig(::Type{FT}; kwargs...) where {FT} =
    CloudSatRadarConfig{FT}(; kwargs...)

function CloudSatRadarConfig{FT}(;
    freq = 94,
    k2 = -1,
    use_gas_abs = true,
    min_mixing_ratio = 1e-15,
    hydrometeor_optics = :clima_1m_psd,
) where {FT}
    hydrometeor_optics === :clima_1m_psd ||
        throw(ArgumentError("only :clima_1m_psd hydrometeor optics is implemented"))
    return CloudSatRadarConfig{FT}(
        FT(freq),
        FT(k2),
        use_gas_abs != 0,
        FT(min_mixing_ratio),
    )
end

const DEFAULT_CLOUDSAT_RADAR_CONFIG = CloudSatRadarConfig(Float64)

struct Clima1MPSDParameters{FT}
    class::Symbol
    phase::Symbol
    r0::FT
    m0::FT
    me::FT
    n0::FT
end

Base.broadcastable(params::Clima1MPSDParameters) = Ref(params)

const O2_V0 = (
    49.4523790, 49.9622570, 50.4742380, 50.9877480, 51.5033500, 52.0214090,
    52.5423930, 53.0669060, 53.5957480, 54.1299999, 54.6711570, 55.2213650,
    55.7838000, 56.2647770, 56.3378700, 56.9681000, 57.6124810, 58.3238740,
    58.4465890, 59.1642040, 59.5909820, 60.3060570, 60.4347750, 61.1505580,
    61.8001520, 62.4112120, 62.4862530, 62.9979740, 63.5685150, 64.1277640,
    64.6789000, 65.2240670, 65.7647690, 66.3020880, 66.8368270, 67.3695950,
    67.9008620, 68.4310010, 68.9603060, 69.4890210, 70.0173420, 118.7503410,
    368.4983500, 424.7631200, 487.2493700, 715.3931500, 773.8387300,
    834.1453300,
)
const O2_A1 = (
    0.0000001, 0.0000003, 0.0000009, 0.0000025, 0.0000061, 0.0000141,
    0.0000310, 0.0000641, 0.0001247, 0.0002280, 0.0003918, 0.0006316,
    0.0009535, 0.0005489, 0.0013440, 0.0017630, 0.0000213, 0.0000239,
    0.0000146, 0.0000240, 0.0000211, 0.0000212, 0.0000246, 0.0000250,
    0.0000230, 0.0000193, 0.0000152, 0.0000150, 0.0000109, 0.0007335,
    0.0004635, 0.0002748, 0.0001530, 0.0000801, 0.0000395, 0.0000183,
    0.0000080, 0.0000033, 0.0000013, 0.0000005, 0.0000002, 0.0000094,
    0.0000679, 0.0006380, 0.0002350, 0.0000996, 0.0006710, 0.0001800,
)
const O2_A2 = (
    11.83, 10.72, 9.69, 8.89, 7.74, 6.84, 6.0, 5.22, 4.48, 3.81, 3.19,
    2.62, 2.115, 0.01, 1.655, 1.255, 0.91, 0.621, 0.079, 0.386, 0.207,
    0.207, 0.386, 0.621, 0.91, 1.255, 0.078, 1.66, 2.11, 2.62, 3.19,
    3.81, 4.48, 5.22, 6.0, 6.84, 7.74, 8.69, 9.69, 10.72, 11.83, 0.0,
    0.02, 0.011, 0.011, 0.089, 0.079, 0.079,
)
const O2_A3 = (
    0.0083, 0.0085, 0.0086, 0.0087, 0.0089, 0.0092, 0.0094, 0.0097,
    0.01, 0.0102, 0.0105, 0.01079, 0.0111, 0.01646, 0.01144, 0.01181,
    0.01221, 0.01266, 0.01449, 0.01319, 0.0136, 0.01382, 0.01297,
    0.01248, 0.01207, 0.01171, 0.01468, 0.01139, 0.01108, 0.01078,
    0.0105, 0.0102, 0.01, 0.0097, 0.0094, 0.0092, 0.0089, 0.0087,
    0.0086, 0.0085, 0.0084, 0.01592, 0.0192, 0.01916, 0.0192, 0.0181,
    0.0181, 0.0181,
)
const O2_A4 = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
)
const O2_A5 = (
    0.0056, 0.0056, 0.0056, 0.0055, 0.0056, 0.0055, 0.0057, 0.0053,
    0.0054, 0.0048, 0.0048, 0.00417, 0.00375, 0.00774, 0.00297, 0.00212,
    0.00094, -0.00055, 0.00597, -0.00244, 0.00344, -0.00413, 0.00132,
    -0.00036, -0.00159, -0.00266, -0.00477, -0.00334, -0.00417, -0.00448,
    -0.0051, -0.0051, -0.0057, -0.0055, -0.0059, -0.0056, -0.0058, -0.0057,
    -0.0056, -0.0056, -0.0056, -0.00044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)
const O2_A6 = (
    1.7, 1.7, 1.7, 1.7, 1.8, 1.8, 1.8, 1.9, 1.8, 2.0, 1.9, 2.1, 2.1,
    0.9, 2.3, 2.5, 3.7, -3.1, 0.8, 0.1, 0.5, 0.7, -1.0, 5.8, 2.9, 2.3,
    0.9, 2.2, 2.0, 2.0, 1.8, 1.9, 1.8, 1.8, 1.7, 1.8, 1.7, 1.7, 1.7,
    1.7, 1.7, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
)
const H2O_V1 = (
    22.23508, 67.81396, 119.99594, 183.310117, 321.225644, 325.152919,
    336.187, 380.197372, 390.134508, 437.346667, 439.150812, 443.018295,
    448.001075, 470.888974, 474.689127, 488.491133, 503.568532, 504.482692,
    556.936002, 620.700807, 658.0065, 752.033227, 841.073595, 859.865,
    899.407, 902.555, 906.205524, 916.171582, 970.315022, 987.926764,
)
const H2O_B1 = (
    0.109, 0.0011, 0.0007, 2.3, 0.0464, 1.54, 0.001, 11.9, 0.0044, 0.0637,
    0.921, 0.194, 10.6, 0.33, 1.28, 0.253, 0.0374, 0.0125, 510.0, 5.09,
    0.274, 250.0, 0.013, 0.133, 0.055, 0.038, 0.183, 8.56, 9.16, 138.0,
)
const H2O_B2 = (
    2.143, 8.73, 8.347, 0.653, 6.156, 1.515, 9.802, 1.018, 7.318, 5.015,
    3.561, 5.015, 1.37, 3.561, 2.342, 2.814, 6.693, 6.693, 0.114, 2.15,
    7.767, 0.336, 8.113, 7.989, 7.845, 8.36, 5.039, 1.369, 1.842, 0.178,
)
const H2O_B3 = (
    0.02784, 0.0276, 0.027, 0.02835, 0.0214, 0.027, 0.0265, 0.0276,
    0.019, 0.0137, 0.0164, 0.0144, 0.0238, 0.0182, 0.0198, 0.0249,
    0.0115, 0.0119, 0.03, 0.0223, 0.03, 0.0286, 0.0141, 0.0286, 0.0286,
    0.0264, 0.0234, 0.0253, 0.024, 0.0286,
)

"""
    cloudsat_optics!(z_vol, kr_vol, g_vol, q, thermo_state, rho_air, radar_cfg)

Fill first-stage CloudSat optical-volume quantities from existing subcolumn caches.

`z_vol` and `kr_vol` are one field per subcolumn, matching COSPv2
`quickbeam_optics` output shape. The four available 1M hydrometeor classes are
summed into those totals. 

Hydrometeor particle distributions use hard-coded ClimaMicrophysics 1M PSD
and mass-size assumptions, then an approximate Rayleigh-limit scattering
closure with a bounded finite-size correction computes `z_vol` and `kr_vol`.
Gas absorption uses the COSPv2 `gases` calculation.

Note: For cloud water in the 1M scheme, the prescribed number concentration is
1e8 m^-3. For cloud ice and rain, the PSD uses a prescribed Marshall-Palmer intercept
with slope diagnosed from mass conservation. For snow, the intercept is
empirical and the slope is also diagnosed from mass conservation.
"""
function cloudsat_optics!(
    z_vol_cloudsat::NTuple{N},
    kr_vol_cloudsat::NTuple{N},
    g_vol_cloudsat,
    q_subcol::NamedTuple,
    thermo_state,
    rho_air,
    radar_cfg::Union{Nothing, CloudSatRadarConfig} = nothing,
) where {N}
    return quickbeam_optics!(
        z_vol_cloudsat,
        kr_vol_cloudsat,
        g_vol_cloudsat,
        q_subcol,
        thermo_state,
        rho_air,
        radar_cfg,
    )
end

function quickbeam_optics!(
    z_vol_cloudsat::NTuple{N},
    kr_vol_cloudsat::NTuple{N},
    g_vol_cloudsat,
    q_subcol::NamedTuple,
    thermo_state,
    rho_air,
    radar_cfg::Union{Nothing, CloudSatRadarConfig} = nothing,
) where {N}
    _check_keys(q_subcol, values(Q_KEYS), "q_subcol")

    nsubcolumns = length(q_subcol.q_lcl)
    nsubcolumns > 0 ||
        throw(ArgumentError("q_subcol must contain at least one subcolumn"))
    N == nsubcolumns ||
        throw(DimensionMismatch("z_vol_cloudsat must contain $nsubcolumns subcolumns"))
    length(kr_vol_cloudsat) == nsubcolumns ||
        throw(DimensionMismatch("kr_vol_cloudsat must contain $nsubcolumns subcolumns"))

    reference = q_subcol.q_lcl[1]
    cfg = _radar_config(radar_cfg, reference)
    axes(g_vol_cloudsat) == axes(reference) ||
        throw(DimensionMismatch("g_vol_cloudsat must have matching axes"))
    axes(rho_air) == axes(reference) ||
        throw(DimensionMismatch("rho_air must have matching axes"))
    _check_field_tuple_axes(z_vol_cloudsat, reference, "z_vol_cloudsat")
    _check_field_tuple_axes(kr_vol_cloudsat, reference, "kr_vol_cloudsat")
    _check_subcolumn_axes(q_subcol, values(Q_KEYS), reference, nsubcolumns)

    pressure = _thermo_field(
        thermo_state,
        (:p, :pressure, Symbol("ᶜp")),
        "pressure",
    )
    temperature = _thermo_field(
        thermo_state,
        (:T, :temperature, Symbol("ᶜT")),
        "temperature",
    )
    specific_humidity = _thermo_field(
        thermo_state,
        (:qv, :q_vap, :specific_humidity, Symbol("ᶜqv"), Symbol("ᶜq_vap")),
        "water vapor specific humidity",
    )
    axes(pressure) == axes(reference) ||
        throw(DimensionMismatch("thermo_state pressure must have matching axes"))
    axes(temperature) == axes(reference) ||
        throw(DimensionMismatch("thermo_state temperature must have matching axes"))
    axes(specific_humidity) == axes(reference) ||
        throw(DimensionMismatch("thermo_state specific humidity must have matching axes"))

    @. g_vol_cloudsat =
        _gas_attenuation(pressure, temperature, specific_humidity, cfg)

    zero_value = zero(eltype(reference))
    for isubcolumn in 1:nsubcolumns
        @. z_vol_cloudsat[isubcolumn] = zero_value
        @. kr_vol_cloudsat[isubcolumn] = zero_value
    end

    for class in HYDRO_CLASSES
        q_fields = getproperty(q_subcol, getproperty(Q_KEYS, class))
        params = _clima_1m_psd_parameters(eltype(reference), Val(class))
        for isubcolumn in 1:nsubcolumns
            q_hydro = q_fields[isubcolumn]
            # TODO: fuse these two broadcasts once there is a backend-safe way
            # to return and accumulate both scalar optics values together.
            @. z_vol_cloudsat[isubcolumn] += _clima_hydrometeor_z_volume(
                q_hydro,
                rho_air,
                temperature,
                cfg,
                params,
            )
            @. kr_vol_cloudsat[isubcolumn] += _clima_hydrometeor_attenuation(
                q_hydro,
                rho_air,
                temperature,
                cfg,
                params,
            )
        end
    end

    return nothing
end

_radar_config(::Nothing, reference) = CloudSatRadarConfig(eltype(reference))
function _radar_config(cfg::CloudSatRadarConfig, reference)
    return CloudSatRadarConfig(
        eltype(reference);
        freq = cfg.freq,
        k2 = cfg.k2,
        use_gas_abs = cfg.use_gas_abs,
        min_mixing_ratio = cfg.min_mixing_ratio,
        hydrometeor_optics = :clima_1m_psd,
    )
end

@inline function _gas_attenuation(p, T, qv, radar_cfg)
    FT = typeof(p + T + qv)
    radar_cfg.use_gas_abs || return zero(FT)
    return _quickbeam_gases(_pressure_hpa(p), T, max(qv, zero(FT)), radar_cfg.freq)
end

@inline function _quickbeam_gases(pres_hpa, T, qv, freq)
    FT = typeof(pres_hpa + T + qv + freq)
    T_safe = max(T, FT(1))
    th = FT(300) / T_safe
    e = qv * pres_hpa / (qv + FT(0.622)) / FT(1000)
    p = pres_hpa / FT(1000) - e
    e_th = e * th
    one_th = one(FT) - th
    pth3 = p * th^3
    eth35 = e * th^FT(3.5)

    sumo = zero(FT)
    aux1 = FT(1.1) * e_th
    for i in eachindex(O2_V0)
        v0 = FT(O2_V0[i])
        a1 = FT(O2_A1[i])
        a2 = FT(O2_A2[i])
        a3 = FT(O2_A3[i])
        a4 = FT(O2_A4[i])
        a5 = FT(O2_A5[i])
        a6 = FT(O2_A6[i])
        aux2 = freq / v0
        aux3 = v0 - freq
        aux4 = v0 + freq
        gm = a3 * (p * th^(FT(0.8) - a4) + aux1)
        gm2 = gm * gm
        delt = a5 * p * th^a6
        x = aux3 * aux3 + gm2
        y = aux4 * aux4 + gm2
        # COSPv2 quickbeam_optics.F90 uses `aux4 / x` here, not `aux4 / y`.
        fpp_o2 =
            ((one(FT) / x + one(FT) / y) * (gm * aux2)) -
            (delt * aux2) * (aux3 / x - aux4 / x)
        s_o2 = a1 * pth3 * exp(a2 * one_th)
        sumo += fpp_o2 * s_o2
    end
    term1 = sumo

    gm0 = FT(5.6e-3) * (p + FT(1.1) * e) * th^FT(0.8)
    a0 = FT(3.07e-4)
    ap = FT(1.4) * (one(FT) - FT(1.2) * freq^FT(1.5) * FT(1e-5)) * FT(1e-10)
    term2 =
        (
            2 * a0 / (gm0 * (one(FT) + (freq / gm0)^2) * (one(FT) + (freq / FT(60))^2)) +
            ap * p * th^FT(2.5)
        ) * freq * p * th * th

    sumo = zero(FT)
    aux1 = FT(4.8) * e_th
    for i in eachindex(H2O_V1)
        v1 = FT(H2O_V1[i])
        b1 = FT(H2O_B1[i])
        b2 = FT(H2O_B2[i])
        b3 = FT(H2O_B3[i])
        aux2 = freq / v1
        aux3 = v1 - freq
        aux4 = v1 + freq
        gm = b3 * (p * th^FT(0.8) + aux1)
        gm2 = gm * gm
        x = aux3 * aux3 + gm2
        y = aux4 * aux4 + gm2
        fpp_h2o = (one(FT) / x + one(FT) / y) * (gm * aux2)
        s_h2o = b1 * eth35 * exp(b2 * one_th)
        sumo += fpp_h2o * s_h2o
    end
    term3 = sumo

    bf = FT(1.4e-6)
    be = FT(5.41e-5)
    term4 = (bf * p + be * e * th^3) * freq * e * th^FT(2.5)
    return FT(0.182) * freq * (term1 + term2 + term3 + term4)
end

# This prototype uses ClimaMicrophysics 1M PSD and mass-size assumptions to
# construct hydrometeor particle distributions, then uses QuickBeam-style
# scattering to compute CloudSat optical properties for the four active Clima 1M
# large-scale hydrometeor classes. This code intentionally avoids the original
# COSPv2 hydro_class_init/calc_Re/dsd and LUT/cache paths.
# TODO: replace hard-coded constants with values from ClimaParams once the
# prototype is validated.
@inline _rho_water(::Type{FT}) where {FT} = FT(1000)
@inline _rho_ice(::Type{FT}) where {FT} = FT(917)
@inline _n_lcl(::Type{FT}) where {FT} = FT(1e8)

@inline function _clima_1m_psd_parameters(::Type{FT}, ::Val{:lcl}) where {FT}
    r0 = FT(1e-5)
    return Clima1MPSDParameters(
        :lcl,
        :liquid,
        r0,
        FT(4) / FT(3) * FT(pi) * _rho_water(FT) * r0^3,
        FT(3),
        _n_lcl(FT),
    )
end

@inline function _clima_1m_psd_parameters(::Type{FT}, ::Val{:icl}) where {FT}
    r0 = FT(1e-5)
    return Clima1MPSDParameters(
        :icl,
        :ice,
        r0,
        FT(4) / FT(3) * FT(pi) * _rho_ice(FT) * r0^3,
        FT(3),
        FT(2e7),
    )
end

@inline function _clima_1m_psd_parameters(::Type{FT}, ::Val{:rai}) where {FT}
    r0 = FT(1e-3)
    return Clima1MPSDParameters(
        :rai,
        :liquid,
        r0,
        FT(4) / FT(3) * FT(pi) * _rho_water(FT) * r0^3,
        FT(3),
        FT(16e6),
    )
end

@inline function _clima_1m_psd_parameters(::Type{FT}, ::Val{:sno}) where {FT}
    r0 = FT(1e-3)
    return Clima1MPSDParameters(
        :sno,
        :ice,
        r0,
        FT(0.1) * r0^2,
        FT(2),
        zero(FT),
    )
end

@inline function _clima_hydrometeor_z_volume(q, rho_air, T, radar_cfg, params)
    return _clima_hydrometeor_optics(q, rho_air, T, radar_cfg, params)[1]
end

@inline function _clima_hydrometeor_attenuation(q, rho_air, T, radar_cfg, params)
    return _clima_hydrometeor_optics(q, rho_air, T, radar_cfg, params)[2]
end

@inline function _clima_hydrometeor_optics(q, rho_air, T, radar_cfg, params)
    FT = typeof(q + rho_air + T)
    q_pos = max(zero(FT), q)
    rho_pos = max(zero(FT), rho_air)
    (q_pos > FT(radar_cfg.min_mixing_ratio) && rho_pos > eps(FT)) ||
        return zero(FT), zero(FT)

    if params.class === :lcl
        return _clima_liquid_cloud_psd_optics(q_pos, rho_pos, T, radar_cfg)
    else
        return _clima_marshall_palmer_psd_optics(
            q_pos,
            rho_pos,
            T,
            radar_cfg,
            params,
        )
    end
end

@inline function _clima_liquid_cloud_radius(q, rho_air)
    FT = typeof(q + rho_air)
    lwc = max(zero(FT), rho_air * q)
    if lwc <= eps(FT)
        return zero(FT)
    end
    return cbrt(FT(3) * lwc / (FT(4) * FT(pi) * _rho_water(FT) * _n_lcl(FT)))
end

@inline function _clima_liquid_cloud_mass(q, rho_air)
    FT = typeof(q + rho_air)
    r = _clima_liquid_cloud_radius(q, rho_air)
    return _n_lcl(FT) * FT(4) / FT(3) * FT(pi) * _rho_water(FT) * r^3
end

@inline function _clima_liquid_cloud_psd_optics(q, rho_air, T, radar_cfg)
    FT = typeof(q + rho_air + T)
    r = _clima_liquid_cloud_radius(q, rho_air)
    r > zero(FT) || return zero(FT), zero(FT)
    # Cloud liquid uses a monodisperse approximation at the radius that exactly
    # matches total liquid mass for N_lcl = 1e8 m^-3.
    return _zeff_particle_integral(
        FT(2) * r,
        _n_lcl(FT),
        T,
        radar_cfg,
        :liquid,
    )
end

@inline function _snow_intercept(q, rho_air)
    FT = typeof(q + rho_air)
    rho0 = one(FT)
    nu = FT(0.63)
    n0_coeff = FT(4.36e9) * rho0^nu
    return n0_coeff * max(zero(FT), rho_air / rho0 * q)^nu
end

@inline function _psd_intercept(q, rho_air, params)
    params.class === :sno && return _snow_intercept(q, rho_air)
    return params.n0
end

@inline function _marshall_palmer_lambda(q, rho_air, params)
    FT = typeof(q + rho_air)
    q_pos = max(zero(FT), q)
    rho_pos = max(zero(FT), rho_air)
    n0 = _psd_intercept(q_pos, rho_pos, params)
    (q_pos > zero(FT) && rho_pos > zero(FT) && n0 > zero(FT)) ||
        return zero(FT)
    gamma_me1 = _gamma_integer(params.me + one(FT))
    return (
        gamma_me1 * params.m0 * n0 /
        (q_pos * rho_pos * params.r0^params.me)
    )^(one(FT) / (params.me + one(FT)))
end

@inline function _marshall_palmer_mass_integral(q, rho_air, params)
    FT = typeof(q + rho_air)
    n0 = _psd_intercept(q, rho_air, params)
    lambda = _marshall_palmer_lambda(q, rho_air, params)
    lambda > zero(FT) || return zero(FT)
    return params.m0 / params.r0^params.me *
           n0 *
           _gamma_integer(params.me + one(FT)) /
           lambda^(params.me + one(FT))
end

@inline function _clima_marshall_palmer_psd_optics(q, rho_air, T, radar_cfg, params)
    FT = typeof(q + rho_air + T)
    lambda = _marshall_palmer_lambda(q, rho_air, params)
    lambda > zero(FT) || return zero(FT), zero(FT)
    n0 = _psd_intercept(q, rho_air, params)
    z_vol = zero(FT)
    kr_vol = zero(FT)
    r_prev = FT(1e-7)
    log_step = exp(log(FT(5e-3) / r_prev) / FT(40))
    for _ in 1:40
        r_next = r_prev * log_step
        r_mid = sqrt(r_prev * r_next)
        dr = r_next - r_prev
        number_density = n0 * exp(-lambda * r_mid) * dr
        z_i, kr_i = _zeff_particle_integral(
            FT(2) * r_mid,
            number_density,
            T,
            radar_cfg,
            params.phase,
        )
        z_vol += z_i
        kr_vol += kr_i
        r_prev = r_next
    end
    return max(zero(FT), z_vol), max(zero(FT), kr_vol)
end

@inline function _zeff_particle_integral(D_m, number_m3, T, radar_cfg, phase)
    FT = typeof(D_m + number_m3 + T)
    D_m > zero(FT) && number_m3 > zero(FT) || return zero(FT), zero(FT)
    qe, backscatter_correction = _mie_efficiencies(D_m, T, radar_cfg, phase)
    z_vol =
        _rayleigh_reflectivity_factor(D_m, number_m3, T, radar_cfg, phase) *
        backscatter_correction
    area = FT(pi) * (D_m / FT(2))^2
    kr_vol = FT(4.343e3) * number_m3 * max(zero(FT), qe) * area
    return z_vol, kr_vol
end

@inline function _rayleigh_reflectivity_factor(D_m, number_m3, T, radar_cfg, phase)
    FT = typeof(D_m + number_m3 + T)
    D_mm = D_m * FT(1000)
    K2_particle = _dielectric_factor_particle(radar_cfg.freq, T, phase)
    K2_reference = _dielectric_factor(radar_cfg)
    # Effective reflectivity factor convention:
    # in the small-particle Rayleigh limit, z_vol = N * D_mm^6 after
    # dielectric normalization. Finite-size behavior is applied separately as a
    # dimensionless Mie/Rayleigh backscatter correction in `_zeff_particle_integral`.
    return number_m3 * D_mm^6 * K2_particle / K2_reference
end

# This is not a full Mie solver. It is a Rayleigh-limit approximation with a
# bounded finite-size backscatter correction, used until QuickBeam's MieInt
# scattering kernel is ported.
@inline function _mie_efficiencies(D_m, T, radar_cfg, phase)
    FT = typeof(D_m + T)
    wavelength = FT(299792458) / (FT(radar_cfg.freq) * FT(1e9))
    x = FT(pi) * D_m / wavelength
    m = phase === :liquid ? _m_wat(radar_cfg.freq, T) : _m_ice(radar_cfg.freq, T)
    K = (m^2 - one(m)) / (m^2 + 2)
    kabs = abs(imag(K))
    kback = abs2(K)
    qe_rayleigh = max(zero(FT), FT(4) * x * kabs + FT(8) / FT(3) * x^4 * kback)
    qbsca_rayleigh = FT(4) * x^4 * kback
    qe = min(FT(2), qe_rayleigh)
    backscatter_correction =
        qbsca_rayleigh > eps(FT) ? min(one(FT), FT(2) / qbsca_rayleigh) : one(FT)
    return qe, backscatter_correction
end

@inline function _dielectric_factor_particle(freq, T, phase)
    m = phase === :liquid ? _m_wat(freq, T) : _m_ice(freq, T)
    return abs2((m^2 - one(m)) / (m^2 + 2))
end

# Simplified liquid-water refractive index used by the prototype Rayleigh path.
# This is not a port of QuickBeam `optics_lib.F90::m_wat`; replace it with the
# full frequency- and temperature-dependent routine when `MieInt` is ported.
@inline function _m_wat(freq, T)
    FT = typeof(freq + T)
    temp_c = T - FT(273.15)
    nr = FT(8.7) - FT(0.004) * temp_c
    ni = FT(2.1) * (freq / FT(94)) * exp(-temp_c / FT(80))
    return Complex(nr, ni)
end

# Simplified ice refractive index used by the prototype Rayleigh path. This is
# not a port of QuickBeam `optics_lib.F90::m_ice`; replace it with the full
# routine when `MieInt` is ported.
@inline function _m_ice(freq, T)
    FT = typeof(freq + T)
    nr = FT(1.78)
    ni = FT(0.003) * (freq / FT(94)) * exp((T - FT(273.15)) / FT(50))
    return Complex(nr, ni)
end

@inline function _gamma_integer(x)
    FT = typeof(x)
    if abs(x - FT(3)) < sqrt(eps(FT))
        return FT(2)
    elseif abs(x - FT(4)) < sqrt(eps(FT))
        return FT(6)
    else
        return exp(_log_gamma_lanczos(x))
    end
end

@inline function _log_gamma_lanczos(x)
    FT = typeof(x)
    coeffs = (
        FT(676.5203681218851),
        FT(-1259.1392167224028),
        FT(771.32342877765313),
        FT(-176.61502916214059),
        FT(12.507343278686905),
        FT(-0.13857109526572012),
        FT(9.9843695780195716e-6),
        FT(1.5056327351493116e-7),
    )
    y = FT(0.99999999999980993)
    z = x - one(FT)
    for i in eachindex(coeffs)
        y += coeffs[i] / (z + FT(i))
    end
    t = z + FT(7.5)
    return FT(0.9189385332046727) + (z + FT(0.5)) * log(t) - t + log(y)
end

@inline _pressure_hpa(p_pa) = p_pa / typeof(p_pa)(100)

@inline function _dielectric_factor(radar_cfg)
    if radar_cfg.k2 >= zero(radar_cfg.k2)
        return radar_cfg.k2
    elseif abs(radar_cfg.freq - typeof(radar_cfg.freq)(94)) < typeof(radar_cfg.freq)(3)
        return typeof(radar_cfg.freq)(0.75)
    elseif abs(radar_cfg.freq - typeof(radar_cfg.freq)(35)) < typeof(radar_cfg.freq)(3)
        return typeof(radar_cfg.freq)(0.88)
    elseif abs(radar_cfg.freq - typeof(radar_cfg.freq)(13.8)) < typeof(radar_cfg.freq)(3)
        return typeof(radar_cfg.freq)(0.925)
    else
        return typeof(radar_cfg.freq)(0.933)
    end
end

function _thermo_field(thermo_state, names, description)
    for name in names
        hasproperty(thermo_state, name) && return getproperty(thermo_state, name)
    end
    throw(ArgumentError("thermo_state must provide $description as one of $names"))
end

function _check_keys(nt::NamedTuple, expected, name)
    keys(nt) == expected ||
        throw(ArgumentError("$name keys must be $(expected), got $(keys(nt))"))
end

function _check_field_tuple_axes(fields, reference, name)
    for field in fields
        axes(field) == axes(reference) ||
            throw(DimensionMismatch("$name fields must have matching axes"))
    end
end

function _check_subcolumn_axes(container, names, reference, nsubcolumns)
    for name in names
        fields = getproperty(container, name)
        length(fields) == nsubcolumns ||
            throw(DimensionMismatch("$name must contain $nsubcolumns subcolumns"))
        _check_field_tuple_axes(fields, reference, string(name))
    end
end

end
