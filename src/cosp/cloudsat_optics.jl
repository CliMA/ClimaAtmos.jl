module COSPCloudSatOptics

import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics.Parameters as CMP

export CloudSatRadarConfig,
    DEFAULT_CLOUDSAT_RADAR_CONFIG,
    cloudsat_gas_attenuation!,
    cloudsat_grid_mean_sizes!,
    cloudsat_optics_subcolumn!

const HYDRO_CLASSES = (:lcl, :icl, :rai, :sno)
const Q_KEYS = (; lcl = :q_lcl, icl = :q_icl, rai = :q_rai, sno = :q_sno)
const SIZE_KEYS = (;
    lcl = :r_lcl,
    icl = :lambda_inv_icl,
    rai = :lambda_inv_rai,
    sno = :lambda_inv_sno,
)

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

struct Clima1MPSDParameters{P}
    phase::Symbol
    hydrometeor::P
end

Base.broadcastable(params::Clima1MPSDParameters) = Ref(params)

include("cloudsat_const_tables.jl")
include("cloudsat_mie.jl")

"""
    cloudsat_gas_attenuation!(
        g_vol_cloudsat,
        temperature,
        pressure,
        specific_humidity,
        radar_cfg,
    )

Compute the CloudSat gas attenuation coefficient once for a model column.
"""
function cloudsat_gas_attenuation!(
    g_vol_cloudsat,
    temperature,
    pressure,
    specific_humidity,
    radar_cfg::Union{Nothing, CloudSatRadarConfig} = nothing,
)
    reference = g_vol_cloudsat
    cfg = _radar_config(radar_cfg, reference)
    for (field, name) in (
        (temperature, "temperature"),
        (pressure, "pressure"),
        (specific_humidity, "specific_humidity"),
    )
        axes(field) == axes(reference) ||
            throw(DimensionMismatch("$name must have matching axes"))
    end
    @. g_vol_cloudsat =
        _gas_attenuation(pressure, temperature, specific_humidity, cfg)
    return nothing
end
"""
    cloudsat_grid_mean_sizes!(
        grid_mean_sizes,
        grid_mean_hydrometeors,
        rho_air,
        microphysics_params,
    )

Diagnose one grid-mean particle-size field for each 1M hydrometeor before
stochastic subcolumn sampling. Cloud liquid uses a monodisperse radius based on
`microphysics_params.cloud.liquid.N_0`; cloud ice, rain, and snow use the
inverse Marshall-Palmer slope returned by `CM1.lambda_inverse`.
"""
function cloudsat_grid_mean_sizes!(
    grid_mean_sizes::NamedTuple,
    grid_mean_hydrometeors::NamedTuple,
    rho_air,
    microphysics_params::CMP.Microphysics1MParams,
)
    _check_keys(
        grid_mean_hydrometeors,
        values(Q_KEYS),
        "grid_mean_hydrometeors",
    )
    _check_keys(grid_mean_sizes, values(SIZE_KEYS), "grid_mean_sizes")
    for field in values(grid_mean_hydrometeors)
        axes(field) == axes(rho_air) ||
            throw(DimensionMismatch("grid-mean hydrometeors must have matching axes"))
    end
    for field in values(grid_mean_sizes)
        axes(field) == axes(rho_air) ||
            throw(DimensionMismatch("grid-mean sizes must have matching axes"))
    end

    liquid = _clima_1m_psd_parameters(microphysics_params, Val(:lcl))
    ice = _clima_1m_psd_parameters(microphysics_params, Val(:icl))
    rain = _clima_1m_psd_parameters(microphysics_params, Val(:rai))
    snow = _clima_1m_psd_parameters(microphysics_params, Val(:sno))
    @. grid_mean_sizes.r_lcl = _clima_liquid_cloud_radius(
        grid_mean_hydrometeors.q_lcl,
        rho_air,
        liquid,
    )
    @. grid_mean_sizes.lambda_inv_icl = _clima_grid_lambda_inverse(
        grid_mean_hydrometeors.q_icl,
        rho_air,
        ice,
    )
    @. grid_mean_sizes.lambda_inv_rai = _clima_grid_lambda_inverse(
        grid_mean_hydrometeors.q_rai,
        rho_air,
        rain,
    )
    @. grid_mean_sizes.lambda_inv_sno = _clima_grid_lambda_inverse(
        grid_mean_hydrometeors.q_sno,
        rho_air,
        snow,
    )
    return nothing
end

# Single-subcolumn hydrometeor optics entry point.
"""
    cloudsat_optics_subcolumn!(
        z_vol_cloudsat,
        kr_vol_cloudsat,
        hydrometeors,
        grid_mean_sizes,
        temperature,
        rho_air,
        microphysics_params,
        radar_cfg,
    )

Compute hydrometeor optical quantities for one streamed subcolumn using particle
sizes diagnosed from the grid-mean state. The output fields are working storage
and are overwritten on every call.
"""
function cloudsat_optics_subcolumn!(
    z_vol_cloudsat,
    kr_vol_cloudsat,
    hydrometeors::NamedTuple,
    grid_mean_sizes::NamedTuple,
    temperature,
    rho_air,
    microphysics_params::CMP.Microphysics1MParams,
    radar_cfg::Union{Nothing, CloudSatRadarConfig} = nothing,
)
    _check_keys(hydrometeors, values(Q_KEYS), "hydrometeors")
    _check_keys(grid_mean_sizes, values(SIZE_KEYS), "grid_mean_sizes")
    reference = z_vol_cloudsat
    cfg = _radar_config(radar_cfg, reference)
    axes(kr_vol_cloudsat) == axes(reference) ||
        throw(DimensionMismatch("kr_vol_cloudsat must have matching axes"))
    axes(temperature) == axes(reference) ||
        throw(DimensionMismatch("temperature must have matching axes"))
    axes(rho_air) == axes(reference) ||
        throw(DimensionMismatch("rho_air must have matching axes"))
    for key in values(Q_KEYS)
        axes(getproperty(hydrometeors, key)) == axes(reference) ||
            throw(DimensionMismatch("hydrometeor fields must have matching axes"))
    end
    for key in values(SIZE_KEYS)
        axes(getproperty(grid_mean_sizes, key)) == axes(reference) ||
            throw(DimensionMismatch("grid-mean size fields must have matching axes"))
    end

    zero_value = zero(eltype(reference))
    @. z_vol_cloudsat = zero_value
    @. kr_vol_cloudsat = zero_value

    for class in HYDRO_CLASSES
        q_hydro = getproperty(hydrometeors, getproperty(Q_KEYS, class))
        grid_size = getproperty(grid_mean_sizes, getproperty(SIZE_KEYS, class))
        params = _clima_1m_psd_parameters(microphysics_params, Val(class))
        # TODO: fuse these two broadcasts once there is a backend-safe way to
        # return and accumulate both scalar optics values together.
        @. z_vol_cloudsat += _clima_hydrometeor_z_volume(
            q_hydro,
            grid_size,
            rho_air,
            temperature,
            cfg,
            params,
        )
        @. kr_vol_cloudsat += _clima_hydrometeor_attenuation(
            q_hydro,
            grid_size,
            rho_air,
            temperature,
            cfg,
            params,
        )
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
# Pure-ice material density used only to convert particle mass to the diameter
# of a volume-equivalent compact ice sphere for Mie scattering.
@inline _rho_solid_ice(::Type{FT}) where {FT} = FT(917)

@inline _clima_1m_psd_parameters(params, ::Val{:lcl}) =
    Clima1MPSDParameters(:liquid, params.cloud.liquid)

@inline _clima_1m_psd_parameters(params, ::Val{:icl}) =
    Clima1MPSDParameters(:ice, params.cloud.ice)

@inline _clima_1m_psd_parameters(params, ::Val{:rai}) =
    Clima1MPSDParameters(:liquid, params.precip.rain)

@inline _clima_1m_psd_parameters(params, ::Val{:sno}) =
    Clima1MPSDParameters(:ice, params.precip.snow)

@inline function _clima_hydrometeor_z_volume(
    q,
    grid_size,
    rho_air,
    T,
    radar_cfg,
    params,
)
    return _clima_hydrometeor_optics(
        q,
        grid_size,
        rho_air,
        T,
        radar_cfg,
        params,
    )[1]
end

@inline function _clima_hydrometeor_attenuation(
    q,
    grid_size,
    rho_air,
    T,
    radar_cfg,
    params,
)
    return _clima_hydrometeor_optics(
        q,
        grid_size,
        rho_air,
        T,
        radar_cfg,
        params,
    )[2]
end

@inline function _clima_hydrometeor_optics(
    q,
    grid_radius,
    rho_air,
    T,
    radar_cfg,
    params::Clima1MPSDParameters{<:CMP.CloudLiquid},
)
    FT = typeof(q + rho_air + T)
    q_pos = max(zero(FT), q)
    rho_pos = max(zero(FT), rho_air)
    (q_pos > FT(radar_cfg.min_mixing_ratio) && rho_pos > eps(FT)) ||
        return zero(FT), zero(FT)
    return _clima_liquid_cloud_psd_optics(
        q_pos,
        grid_radius,
        rho_pos,
        T,
        radar_cfg,
        params,
    )
end

@inline function _clima_hydrometeor_optics(
    q,
    grid_lambda_inv,
    rho_air,
    T,
    radar_cfg,
    params,
)
    FT = typeof(q + rho_air + T)
    q_pos = max(zero(FT), q)
    rho_pos = max(zero(FT), rho_air)
    (q_pos > FT(radar_cfg.min_mixing_ratio) && rho_pos > eps(FT)) ||
        return zero(FT), zero(FT)
    return _clima_marshall_palmer_psd_optics(
        q_pos,
        grid_lambda_inv,
        rho_pos,
        T,
        radar_cfg,
        params,
    )
end

@inline function _clima_liquid_cloud_radius(q, rho_air, liquid)
    FT = typeof(q + rho_air)
    q_pos = max(zero(FT), q)
    rho_pos = max(zero(FT), rho_air)
    lwc = rho_pos * q_pos
    if lwc <= zero(FT)
        return zero(FT)
    end
    liquid_params = liquid.hydrometeor
    return cbrt(
        FT(3) * lwc /
        (FT(4) * FT(pi) * liquid_params.ρw * liquid_params.N_0),
    )
end

@inline function _clima_grid_lambda_inverse(q, rho_air, params)
    FT = typeof(q + rho_air)
    q_pos = max(zero(FT), q)
    rho_pos = max(zero(FT), rho_air)
    hydro = params.hydrometeor
    return CM1.lambda_inverse(hydro.pdf, hydro.mass, q_pos, rho_pos)
end

@inline function _clima_liquid_cloud_number(q, grid_radius, rho_air, liquid)
    FT = typeof(q + grid_radius + rho_air)
    q_pos = max(zero(FT), q)
    rho_pos = max(zero(FT), rho_air)
    r = max(zero(FT), grid_radius)
    particle_mass =
        FT(4) / FT(3) * FT(pi) * liquid.ρw * r^3
    return particle_mass > zero(FT) ?
           rho_pos * q_pos / particle_mass :
           zero(FT)
end

@inline function _clima_liquid_cloud_psd_optics(
    q,
    grid_radius,
    rho_air,
    T,
    radar_cfg,
    params,
)
    FT = typeof(q + rho_air + T)
    r = max(zero(FT), grid_radius)
    r > zero(FT) || return zero(FT), zero(FT)
    number_m3 =
        _clima_liquid_cloud_number(q, r, rho_air, params.hydrometeor)
    return _zeff_particle_integral(
        FT(2) * r,
        number_m3,
        T,
        radar_cfg,
        :liquid,
    )
end

@inline function _particle_mass(r, params)
    (; r0, m0, me, Δm, χm) = params.hydrometeor.mass
    return χm * m0 * (r / r0)^(me + Δm)
end

@inline function _discrete_psd_intercept(q, grid_lambda_inv, rho_air, params)
    FT = typeof(q + grid_lambda_inv + rho_air)
    q_pos = max(zero(FT), q)
    rho_pos = max(zero(FT), rho_air)
    lambda_inv = max(zero(FT), grid_lambda_inv)
    lambda_inv > zero(FT) || return zero(FT)
    mass_integral = zero(FT)
    r_prev = FT(1e-7)
    log_step = exp(log(FT(5e-3) / r_prev) / FT(40))
    for _ in 1:40
        r_next = r_prev * log_step
        r_mid = sqrt(r_prev * r_next)
        dr = r_next - r_prev
        spectral_weight = exp(-r_mid / lambda_inv) * dr
        mass_integral += _particle_mass(r_mid, params) * spectral_weight
        r_prev = r_next
    end
    # `spectral_weight` has units of m, so this amplitude has units of m^-4.
    # The resulting bin number `n0 * spectral_weight` is in m^-3 and its
    # discrete mass sum is exactly `rho_air * q`.
    return mass_integral > zero(FT) ?
           rho_pos * q_pos / mass_integral :
           zero(FT)
end

@inline function _clima_marshall_palmer_psd_optics(
    q,
    grid_lambda_inv,
    rho_air,
    T,
    radar_cfg,
    params,
)
    FT = typeof(q + rho_air + T)
    lambda_inv = max(zero(FT), grid_lambda_inv)
    n0 = _discrete_psd_intercept(q, lambda_inv, rho_air, params)
    n0 > zero(FT) || return zero(FT), zero(FT)
    z_vol = zero(FT)
    kr_vol = zero(FT)
    r_prev = FT(1e-7)
    log_step = exp(log(FT(5e-3) / r_prev) / FT(40))
    for _ in 1:40
        r_next = r_prev * log_step
        r_mid = sqrt(r_prev * r_next)
        dr = r_next - r_prev
        bin_number = n0 * exp(-r_mid / lambda_inv) * dr
        D_m = _scattering_diameter(r_mid, params)
        z_i, kr_i = _zeff_particle_integral(
            D_m,
            bin_number,
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

@inline function _scattering_diameter(r, params)
    params.phase === :liquid && return 2 * r
    FT = typeof(r)
    mass = _particle_mass(r, params)
    return cbrt(FT(6) * mass / (FT(pi) * _rho_solid_ice(FT)))
end

@inline function _zeff_particle_integral(D_m, number_m3, T, radar_cfg, phase)
    FT = typeof(D_m + number_m3 + T)
    D_m > zero(FT) && number_m3 > zero(FT) || return zero(FT), zero(FT)
    qext, qbsca = _mie_efficiencies(D_m, T, radar_cfg, phase)
    wavelength = FT(0.299792458) / FT(radar_cfg.freq)
    k2 = _dielectric_factor(radar_cfg)

    # QuickBeam `zeff` convention for one particle-size bin:
    # eta_mie = 0.25 * pi * qbsca * N * D^2, then converted to mm^6 m^-3.
    eta_mie = FT(0.25) * FT(pi) * qbsca * number_m3 * D_m^2
    z_vol = wavelength^4 / FT(pi)^5 / k2 * eta_mie * FT(1e18)

    # kr = 0.25 * pi * qext * N * D^2 * (1000 * 10 / log(10)), in dB km^-1.
    k_sum = qext * number_m3 * D_m^2
    kr_vol = FT(0.25) * FT(pi) * k_sum * (FT(1000) * FT(10) / log(FT(10)))
    return max(zero(FT), z_vol), max(zero(FT), kr_vol)
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

function _check_keys(nt::NamedTuple, expected, name)
    keys(nt) == expected ||
        throw(ArgumentError("$name keys must be $(expected), got $(keys(nt))"))
end

end
