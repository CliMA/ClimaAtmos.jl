"""
    microphysics_tendency_model(mp_tendency, mp_derivative, q, dt)

Model for applying a microphysics tendency to a tracer-like variable `q`.

Currently:
- if `mp_tendency ≥ 0`, the tendency is applied directly;
- if `mp_tendency < 0`, the tendency is represented as a linear sink
  `mp_derivative * q`.

Notes:
- `mp_derivative` is assumed to represent the local linearized sink coefficient
  associated with `mp_tendency`;
- mass conservation across tracers is not enforced here and remains a TODO.
"""
@inline function microphysics_tendency_model(mp_tendency, mp_derivative, q, dt)
    ifelse(mp_tendency < 0, mp_derivative * q, mp_tendency)
end
@inline microphysics_tendency_model(mp_tendency, mp_derivative, ρq, ρ, dt) =
    microphysics_tendency_model(mp_tendency, mp_derivative, ρq / ρ, dt)

# Approximate diagonal sink coefficient used in the microphysics Jacobian.
# For negative tendencies, use S / q as a local linearized sink rate, limited
# to a minimum time scale of 1 second to avoid excessively large Jacobian entries.
# Positive tendencies do not contribute to the diagonal sink term.
@inline function _jac_coeff(S, q, q_min)
    FT = typeof(S)
    return ifelse(S < 0 && q > 0, max(-one(FT), S / max(q, q_min)), zero(FT))
end

@inline function _jac_coeff_from_ratio(Sq, ρq, ρ, q_min)
    q = ρq / ρ
    return _jac_coeff(Sq, q, q_min)
end

@inline function _jac_coeffs_1m(mp_tendency, q_liq, q_ice, q_rai, q_sno, q_min)
    ∂tendency_∂q_lcl = _jac_coeff(mp_tendency.dq_lcl_dt, q_liq, q_min)
    ∂tendency_∂q_icl = _jac_coeff(mp_tendency.dq_icl_dt, q_ice, q_min)
    ∂tendency_∂q_rai = _jac_coeff(mp_tendency.dq_rai_dt, q_rai, q_min)
    ∂tendency_∂q_sno = _jac_coeff(mp_tendency.dq_sno_dt, q_sno, q_min)
    return (; ∂tendency_∂q_lcl, ∂tendency_∂q_icl, ∂tendency_∂q_rai, ∂tendency_∂q_sno)
end
