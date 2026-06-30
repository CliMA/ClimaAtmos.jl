module COSP1MReffNpDiagnostics

export set_1M_reff_np_subcolumns!

const HYDROMETEOR_KEYS = (:q_lcl, :q_icl, :q_rai, :q_sno)
const REFF_KEYS = (:Reff_lcl, :Reff_icl, :Reff_rai, :Reff_sno)
const NP_KEYS = (:Np_lcl, :Np_icl, :Np_rai, :Np_sno)

#####
##### CloudMicrophysics 1M diagnostic constants
#####
#####

struct ReffNp1MParameters{FT}
    n_lcl::FT
    k_lcl::FT
    rho_w::FT
    r0_rain::FT
    me_rain::FT
    m0_rain::FT
    n0_rain::FT
    gamma_me_plus_one_rain::FT
    r0_ice::FT
    me_ice::FT
    m0_ice::FT
    n0_ice::FT
    gamma_me_plus_one_ice::FT
    r0_snow::FT
    me_snow::FT
    m0_snow::FT
    mu_snow::FT
    nu_snow::FT
    gamma_me_plus_one_snow::FT
end

Base.broadcastable(params::ReffNp1MParameters) = Ref(params)

const N_LCL = 1e8
const K_LCL = 0.8
const RHO_W = 1000.0

const R0_RAIN = 1e-3
const ME_RAIN = 3.0
const N0_RAIN = 16e6
const M0_RAIN = 4 / 3 * pi * RHO_W * R0_RAIN^3
const GAMMA_ME_PLUS_ONE_RAIN = 6.0

const R0_ICE = 1e-5
const ME_ICE = 3.0
const N0_ICE = 2e7
const RHO_I = 916.7
const M0_ICE = 4 / 3 * pi * RHO_I * R0_ICE^3
const GAMMA_ME_PLUS_ONE_ICE = 6.0

const R0_SNOW = 1e-3
const ME_SNOW = 2.0
const M0_SNOW = 0.1 * R0_SNOW^2
const MU_SNOW = 4.36e9
const NU_SNOW = 0.63
const GAMMA_ME_PLUS_ONE_SNOW = 2.0

const DEFAULT_REFF_NP_1M_PARAMETERS = ReffNp1MParameters(
    N_LCL,
    K_LCL,
    RHO_W,
    R0_RAIN,
    ME_RAIN,
    M0_RAIN,
    N0_RAIN,
    GAMMA_ME_PLUS_ONE_RAIN,
    R0_ICE,
    ME_ICE,
    M0_ICE,
    N0_ICE,
    GAMMA_ME_PLUS_ONE_ICE,
    R0_SNOW,
    ME_SNOW,
    M0_SNOW,
    MU_SNOW,
    NU_SNOW,
    GAMMA_ME_PLUS_ONE_SNOW,
)


"""
    set_1M_reff_np_subcolumns!(
        subcolumn_reff,
        subcolumn_Np,
        subcolumn_hydrometeors,
        rho,
    )

Diagnose effective radius and particle number concentration for sliced 1M
subcolumn hydrometeor specific contents.
"""
function set_1M_reff_np_subcolumns!(
    subcolumn_reff::NamedTuple,
    subcolumn_Np::NamedTuple,
    subcolumn_hydrometeors::NamedTuple,
    rho,
)
    return set_1M_reff_np_subcolumns!(
        subcolumn_reff,
        subcolumn_Np,
        subcolumn_hydrometeors,
        rho,
        DEFAULT_REFF_NP_1M_PARAMETERS,
    )
end

function set_1M_reff_np_subcolumns!(
    subcolumn_reff::NamedTuple,
    subcolumn_Np::NamedTuple,
    subcolumn_hydrometeors::NamedTuple,
    rho,
    params::ReffNp1MParameters,
)
    _check_keys(subcolumn_hydrometeors, HYDROMETEOR_KEYS, "subcolumn_hydrometeors")
    _check_keys(subcolumn_reff, REFF_KEYS, "subcolumn_reff")
    _check_keys(subcolumn_Np, NP_KEYS, "subcolumn_Np")

    nsubcolumns = length(subcolumn_hydrometeors.q_lcl)
    nsubcolumns > 0 ||
        throw(ArgumentError("subcolumn_hydrometeors must contain subcolumns"))
    reference = subcolumn_hydrometeors.q_lcl[1]
    axes(rho) == axes(reference) ||
        throw(DimensionMismatch("rho must have matching axes"))

    _check_subcolumn_axes(subcolumn_hydrometeors, HYDROMETEOR_KEYS, reference, nsubcolumns)
    _check_subcolumn_axes(subcolumn_reff, REFF_KEYS, reference, nsubcolumns)
    _check_subcolumn_axes(subcolumn_Np, NP_KEYS, reference, nsubcolumns)

    for isubcolumn in 1:nsubcolumns
        q_lcl = subcolumn_hydrometeors.q_lcl[isubcolumn]
        q_icl = subcolumn_hydrometeors.q_icl[isubcolumn]
        q_rai = subcolumn_hydrometeors.q_rai[isubcolumn]
        q_sno = subcolumn_hydrometeors.q_sno[isubcolumn]

        @. subcolumn_reff.Reff_lcl[isubcolumn] =
            _liquid_reff(q_lcl, rho, params)
        @. subcolumn_Np.Np_lcl[isubcolumn] =
            _liquid_number(q_lcl, rho, params)

        @. subcolumn_reff.Reff_icl[isubcolumn] =
            _ice_reff(q_icl, rho, params)
        @. subcolumn_Np.Np_icl[isubcolumn] =
            _ice_number(q_icl, rho, params)

        @. subcolumn_reff.Reff_rai[isubcolumn] =
            _rain_reff(q_rai, rho, params)
        @. subcolumn_Np.Np_rai[isubcolumn] =
            _rain_number(q_rai, rho, params)

        @. subcolumn_reff.Reff_sno[isubcolumn] =
            _snow_reff(q_sno, rho, params)
        @. subcolumn_Np.Np_sno[isubcolumn] =
            _snow_number(q_sno, rho, params)
    end

    return nothing
end

@inline _present(q, rho) = q > eps(q) && rho > eps(rho)

@inline _liquid_reff(q_lcl, rho) =
    _liquid_reff(q_lcl, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
@inline function _liquid_reff(q_lcl, rho, params)
    FT = typeof(q_lcl + rho)
    if _present(q_lcl, rho)
        r_vol =
            cbrt(
                FT(3) * rho * q_lcl /
                (FT(4) * FT(pi) * FT(params.rho_w) * FT(params.n_lcl)),
            )
        return r_vol / cbrt(FT(params.k_lcl))
    else
        return zero(FT)
    end
end

@inline _liquid_number(q_lcl, rho) =
    _liquid_number(q_lcl, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
@inline function _liquid_number(q_lcl, rho, params)
    FT = typeof(q_lcl + rho)
    return _present(q_lcl, rho) ? FT(params.n_lcl) : zero(FT)
end

@inline _rain_lambda_inverse(q, rho) = _lambda_inverse(
    q,
    rho,
    DEFAULT_REFF_NP_1M_PARAMETERS.r0_rain,
    DEFAULT_REFF_NP_1M_PARAMETERS.me_rain,
    DEFAULT_REFF_NP_1M_PARAMETERS.m0_rain,
    DEFAULT_REFF_NP_1M_PARAMETERS.n0_rain,
    DEFAULT_REFF_NP_1M_PARAMETERS.gamma_me_plus_one_rain,
)

@inline _rain_lambda_inverse(q, rho, params) = _lambda_inverse(
    q,
    rho,
    params.r0_rain,
    params.me_rain,
    params.m0_rain,
    params.n0_rain,
    params.gamma_me_plus_one_rain,
)

@inline _ice_lambda_inverse(q, rho) = _lambda_inverse(
    q,
    rho,
    DEFAULT_REFF_NP_1M_PARAMETERS.r0_ice,
    DEFAULT_REFF_NP_1M_PARAMETERS.me_ice,
    DEFAULT_REFF_NP_1M_PARAMETERS.m0_ice,
    DEFAULT_REFF_NP_1M_PARAMETERS.n0_ice,
    DEFAULT_REFF_NP_1M_PARAMETERS.gamma_me_plus_one_ice,
)

@inline _ice_lambda_inverse(q, rho, params) = _lambda_inverse(
    q,
    rho,
    params.r0_ice,
    params.me_ice,
    params.m0_ice,
    params.n0_ice,
    params.gamma_me_plus_one_ice,
)

@inline function _snow_lambda_inverse(q, rho)
    return _snow_lambda_inverse(q, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
end
@inline function _snow_lambda_inverse(q, rho, params)
    FT = typeof(q + rho)
    n0 = _snow_n0(q, rho, params)
    return _lambda_inverse(
        q,
        rho,
        params.r0_snow,
        params.me_snow,
        params.m0_snow,
        n0,
        params.gamma_me_plus_one_snow,
    )::FT
end

@inline function _lambda_inverse(q, rho, r0, me, m0, n0, gamma_me_plus_one)
    FT = typeof(q + rho)
    if _present(q, rho) && n0 > eps(FT)
        me_ft = FT(me)
        return (
            rho * q * FT(r0)^me_ft /
            (FT(m0) * FT(n0) * FT(gamma_me_plus_one))
        )^(one(FT) / (me_ft + one(FT)))
    else
        return zero(FT)
    end
end

@inline function _snow_n0(q_sno, rho)
    return _snow_n0(q_sno, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
end
@inline function _snow_n0(q_sno, rho, params)
    FT = typeof(q_sno + rho)
    return _present(q_sno, rho) ?
           FT(params.mu_snow) * (rho * q_sno)^FT(params.nu_snow) : zero(FT)
end

@inline _marshall_palmer_reff(lambda_inv) = typeof(lambda_inv)(3) * lambda_inv
@inline _marshall_palmer_number(n0, lambda_inv) = n0 * lambda_inv

@inline function _rain_reff(q_rai, rho)
    return _rain_reff(q_rai, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
end
@inline function _rain_reff(q_rai, rho, params)
    return _marshall_palmer_reff(_rain_lambda_inverse(q_rai, rho, params))
end

@inline function _rain_number(q_rai, rho)
    return _rain_number(q_rai, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
end
@inline function _rain_number(q_rai, rho, params)
    lambda_inv = _rain_lambda_inverse(q_rai, rho, params)
    return _marshall_palmer_number(typeof(q_rai + rho)(params.n0_rain), lambda_inv)
end

@inline function _ice_reff(q_icl, rho)
    return _ice_reff(q_icl, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
end
@inline function _ice_reff(q_icl, rho, params)
    return _marshall_palmer_reff(_ice_lambda_inverse(q_icl, rho, params))
end

@inline function _ice_number(q_icl, rho)
    return _ice_number(q_icl, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
end
@inline function _ice_number(q_icl, rho, params)
    lambda_inv = _ice_lambda_inverse(q_icl, rho, params)
    return _marshall_palmer_number(typeof(q_icl + rho)(params.n0_ice), lambda_inv)
end

@inline function _snow_reff(q_sno, rho)
    return _snow_reff(q_sno, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
end
@inline function _snow_reff(q_sno, rho, params)
    return _marshall_palmer_reff(_snow_lambda_inverse(q_sno, rho, params))
end

@inline function _snow_number(q_sno, rho)
    return _snow_number(q_sno, rho, DEFAULT_REFF_NP_1M_PARAMETERS)
end
@inline function _snow_number(q_sno, rho, params)
    n0 = _snow_n0(q_sno, rho, params)
    lambda_inv = _snow_lambda_inverse(q_sno, rho, params)
    return _marshall_palmer_number(n0, lambda_inv)
end

function _check_keys(nt::NamedTuple, expected, name)
    keys(nt) == expected ||
        throw(ArgumentError("$name keys must be $(expected), got $(keys(nt))"))
end

function _check_subcolumn_axes(container, names, reference, nsubcolumns)
    for name in names
        fields = getproperty(container, name)
        length(fields) == nsubcolumns ||
            throw(DimensionMismatch("$name must contain $nsubcolumns subcolumns"))
        for field in fields
            axes(field) == axes(reference) ||
                throw(DimensionMismatch("$name fields must have matching axes"))
        end
    end
end

end
