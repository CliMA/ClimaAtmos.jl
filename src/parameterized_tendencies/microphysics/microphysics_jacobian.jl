"""
    microphysics_tendency_model(mp_tendency, mp_derivative, q)

Model for applying a microphysics tendency.

Currently:
- positive tendencies are applied directly,
- negative tendencies are represented as a linear sink `mp_derivative * q`.

This interface allows replacing the current simple model with more
sophisticated treatments in the future. Mass conservation across tracers 
is not enforced here and remains a TODO.
"""
@inline function microphysics_tendency_model(mp_tendency, mp_derivative, q)
    # Use linear sink only when tendency is negative and q > 0
    ifelse(mp_tendency < 0 && q > 0, mp_derivative * q, mp_tendency)
end
@inline microphysics_tendency_model(mp_tendency, mp_derivative, ρq, ρ) =
    microphysics_tendency_model(mp_tendency, mp_derivative, ρq / ρ)

# Scalar helpers for the Jacobian diagonal computation.
@inline function _jac_coeff(S, q)
    FT = typeof(S)
    ε = q_min(FT)
    return ifelse(S < 0 && q > 0, S / max(q, ε), zero(FT))
end

@inline function _jac_coeff_from_ratio(Sq, ρq, ρ)
    q = ρq / ρ
    return _jac_coeff(Sq, q)
end

@inline function _jac_coeffs_1m(mp_tendency, q_liq, q_ice, q_rai, q_sno)
    ∂tendency_∂q_lcl = _jac_coeff(mp_tendency.dq_lcl_dt, q_liq)
    ∂tendency_∂q_icl = _jac_coeff(mp_tendency.dq_icl_dt, q_ice)
    ∂tendency_∂q_rai = _jac_coeff(mp_tendency.dq_rai_dt, q_rai)
    ∂tendency_∂q_sno = _jac_coeff(mp_tendency.dq_sno_dt, q_sno)
    return (; ∂tendency_∂q_lcl, ∂tendency_∂q_icl, ∂tendency_∂q_rai, ∂tendency_∂q_sno)
end
