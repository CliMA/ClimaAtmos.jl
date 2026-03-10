# Scalar helpers for the Jacobian diagonal computation.
# Use S = (S / q)_precomputed * q as an approximation for microphysics sinks,
# and S = S_precomputed for sources.
@inline function _jac_coeff(Sq, q)
    FT = typeof(Sq)
    ε = q_min(FT)
    return ifelse(Sq >= zero(FT), zero(FT), Sq / max(q, ε))
end

@inline function _jac_coeffs_1m(mp_tendency, q_liq, q_ice, q_rai, q_sno)
    ∂tendency_∂q_lcl = _jac_coeff(mp_tendency.dq_lcl_dt, q_liq)
    ∂tendency_∂q_icl = _jac_coeff(mp_tendency.dq_icl_dt, q_ice)
    ∂tendency_∂q_rai = _jac_coeff(mp_tendency.dq_rai_dt, q_rai)
    ∂tendency_∂q_sno = _jac_coeff(mp_tendency.dq_sno_dt, q_sno)
    return (; ∂tendency_∂q_lcl, ∂tendency_∂q_icl, ∂tendency_∂q_rai, ∂tendency_∂q_sno)
end
