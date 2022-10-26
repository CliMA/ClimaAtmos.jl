#### Non-dimensional Entrainment-Detrainment functions
function max_area_limiter(εδ_model, max_area, a_up)
    γ_lim = εδ_params(εδ_model).γ_lim
    β_lim = εδ_params(εδ_model).β_lim
    logistic_term = (2 - 1 / (1 + exp(-γ_lim * (max_area - a_up))))
    return (logistic_term)^β_lim - 1
end

"""
    non_dimensional_function(εδ_model::MDEntr, εδ_model_vars)

Returns the nondimensional entrainment and detrainment
functions following Cohen et al. (JAMES, 2020), given:
 - `εδ_model`       :: MDEntr - Moisture deficit entrainment closure
 - `εδ_model_vars`  :: structure containing variables
"""
function non_dimensional_function(εδ_model::MDEntr, εδ_model_vars)
    FT = eltype(εδ_model_vars.q_cond_up)
    Δw = get_Δw(εδ_model, εδ_model_vars.w_up, εδ_model_vars.w_en)
    c_ε = εδ_params(εδ_model).c_ε
    μ_0 = εδ_params(εδ_model).μ_0
    β = εδ_params(εδ_model).β
    χ = εδ_params(εδ_model).χ
    c_δ =
        if !TD.has_condensate(εδ_model_vars.q_cond_up + εδ_model_vars.q_cond_en)
            FT(0)
        else
            εδ_model.params.c_δ
        end

    Δb = εδ_model_vars.b_up - εδ_model_vars.b_en
    μ_ij =
        (χ - εδ_model_vars.a_up / (εδ_model_vars.a_up + εδ_model_vars.a_en)) *
        Δb / Δw
    exp_arg = μ_ij / μ_0
    D_ε = 1 / (1 + exp(-exp_arg))
    D_δ = 1 / (1 + exp(exp_arg))

    M_δ = (max((εδ_model_vars.RH_up)^β - (εδ_model_vars.RH_en)^β, 0))^(1 / β)
    M_ε = (max((εδ_model_vars.RH_en)^β - (εδ_model_vars.RH_up)^β, 0))^(1 / β)

    nondim_ε = (c_ε * D_ε + c_δ * M_ε)
    nondim_δ = (c_ε * D_δ + c_δ * M_δ)
    return nondim_ε, nondim_δ
end
