# ============================================================================
# experiment_lognormal_cf.jl
#
# Description:
# This script is an offline analysis tool used to evaluate and tune the 
# mathematical parametrization for the Sub-Grid Scale (SGS) cloud fraction
# when assuming a Lognormal distribution for total water specific humidity (q_tot).
#
# Context:
# In atmospheric models, resolving exact cloud boundaries is impossible at
# macroscopic grid scales. Instead, the "Cloud Fraction" (CF) is calculated
# by integrating the right-tail of an assumed probability density function (PDF)
# of moisture that exceeds the saturation threshold. 
#
# For a Gaussian PDF, the integration yields an Error Function, which is often
# approximated as: CF_gaussian ≈ tanh(c * Q) where Q is the normalized condensate 
# and c ≈ π/√6.
#
# This script determines the best equivalent analytical approximation for a 
# *Lognormal* PDF, which has a positive skew (a sharp left side and fat right tail)
# that becomes more pronounced as the Coefficient of Variation (Cv) increases.
#
# Purpose:
# 1. Calculates the mathematically exact cloud fraction by applying Black-Scholes 
#    style root-finding to the Lognormal incomplete first moment.
# 2. Fits multiple analytical proxy functions (Exponential, Tanh, Rational, Arctan)
#    to finding the one that best minimizes RMSE against the exact numerical integration.
# 3. Specifically evaluates how the optimal coefficient `c` in `tanh(c * Q)`
#    should scale varying skewness domains (Cv).
# ============================================================================

using Pkg
Pkg.activate(mktempdir())
Pkg.add(["Roots", "Optim", "SpecialFunctions", "Plots", "StatsBase"])

using Optim
using SpecialFunctions
using Plots
using StatsBase
using Printf
using LinearAlgebra

# Standard Normal CDF used as the basis for Black-Scholes style integrals
Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))

"""
    exact_lognormal_stats(Cv, q_sat_over_μ)

Calculates the mathematically exact Cloud Fraction and normalized condensate 
for a Lognormal distribution given a Coefficient of Variation (Cv) and a 
saturation ratio (q_sat / μ_q).

Based on Black-Scholes style analytical integrals for Lognormal variables.
"""
function exact_lognormal_stats(Cv, q_sat_over_μ)
    σ_ln = sqrt(log(1 + Cv^2))
    d1 = (log(1.0 / q_sat_over_μ) + σ_ln^2 / 2) / σ_ln
    d2 = d1 - σ_ln

    # Cloud Fraction = Probability(q > q_sat)
    cf_true = Φ(d2)

    # Grid-mean Condensate = Expected value of max(q - q_sat, 0)
    q_cond_over_μ = Φ(d1) - q_sat_over_μ * Φ(d2)

    # Q_hat is the condensate normalized by the standard deviation
    Q_hat = q_cond_over_μ / Cv
    return cf_true, Q_hat
end

"""
    generate_dataset()

Creates a synthetic evaluation dataset spanning a realistic parameter space
of atmospheric variances (Cv from 0.1 to 10) and saturation ratios.
"""
function generate_dataset()
    Cv_vals = exp10.(range(-1, 1, length = 50))
    q_sat_ratios = exp10.(range(-2, 2, length = 200))

    data_Cv = Float64[]
    data_Q = Float64[]
    data_CF = Float64[]

    # Scan parameter space and filter out trivial physical extremes (all-cloud or clear-sky)
    for Cv in Cv_vals
        for r in q_sat_ratios
            cf, Q = exact_lognormal_stats(Cv, r)
            if cf > 1e-4 && cf < 0.9999 && Q > 1e-4
                push!(data_Cv, Cv)
                push!(data_Q, Q)
                push!(data_CF, cf)
            end
        end
    end
    return data_Cv, data_Q, data_CF
end

"""
    experiment()

Main routine. Runs nonlinear optimization combinations (L-BFGS) to fit various 
S-curve functional forms against the exact Lognormal dataset generated above.
"""
function experiment()
    Cvs, Qs, CFs = generate_dataset()

    # Define candidate proxy functions to map normalized condensate (Q) to Cloud Fraction
    # 1. Exponential: Asymptotes quickly
    f_exp(c, Q) = 1 - exp(-c * Q)

    # 2. Tanh: Symmetric s-curve (matches default Gaussian formulation)
    f_tanh(c, Q) = tanh(c * Q)

    # 3. Rational: Smooth algebraic drop-off
    f_rat(c, Q) = 1 - 1 / (1 + c * Q)

    # 4. Atan: Heavy-tailed trigonometric proxy
    f_atan(c, Q) = 2 / π * atan(c * Q)

    # General L-BFGS optimizer to minimize the L2 loss for a given candidate function
    function fit_model(f)
        loss(c) = sum(abs2.(f.(c[1], Qs) .- CFs))
        res = optimize(loss, [1.0], LBFGS())
        c_opt = Optim.minimizer(res)[1]
        rmse = sqrt(Optim.minimum(res) / length(Qs))
        return c_opt, rmse
    end

    # Step 1: Evaluate Global Fits 
    # Assumes a single constant parameter across all atmospheric variance states
    c_exp, err_exp = fit_model(f_exp)
    c_tanh, err_tanh = fit_model(f_tanh)
    c_rat, err_rat = fit_model(f_rat)
    c_atan, err_atan = fit_model(f_atan)

    println("--- Global Fits (Constant Parameter) ---")
    @printf("Exponential   CF = 1 - exp(-%.3f * Q), RMSE: %.4f\n", c_exp, err_exp)
    @printf("Tanh          CF = tanh(%.3f * Q), RMSE: %.4f\n", c_tanh, err_tanh)
    @printf("Rational      CF = 1 - 1/(1 + %.3f * Q), RMSE: %.4f\n", c_rat, err_rat)
    @printf("Arctan        CF = 2/π * atan(%.3f * Q), RMSE: %.4f\n", c_atan, err_atan)

    # Generate scatter plot comparing global fits against visually separated Cv samples
    Q_plot = collect(range(0, 3, length = 100))
    p = plot(
        title = "Model Comparison (Global Fits)",
        xlabel = "Normalized Condensate Q_hat",
        ylabel = "Cloud Fraction",
    )

    for Cv_sample in [0.1, 0.5, 2.0]
        pts_Q = Float64[]
        pts_CF = Float64[]
        for r in exp10.(range(-2, 2, length = 100))
            cf, Q = exact_lognormal_stats(Cv_sample, r)
            if cf > 1e-3 && cf < 0.999
                push!(pts_Q, Q)
                push!(pts_CF, cf)
            end
        end
        scatter!(
            p,
            pts_Q,
            pts_CF,
            label = "True (Cv=$(Cv_sample))",
            alpha = 0.5,
            markersize = 3,
            markerstrokewidth = 0,
        )
    end

    plot!(
        p,
        Q_plot,
        f_exp.(c_exp, Q_plot),
        label = "1 - exp(-$(round(c_exp, digits=2)) Q)",
        linewidth = 2,
    )
    plot!(
        p,
        Q_plot,
        f_tanh.(c_tanh, Q_plot),
        label = "tanh($(round(c_tanh, digits=2)) Q)",
        linewidth = 2,
    )

    savefig(p, "cf_models.png")

    # Step 2: Evaluate Covariate Fits
    # Tests how the ideal parameter for the Tanh model needs to scale as Cv (skewness) increases
    println("\n--- Tanh Parameter Dependence on Cv ---")
    Cv_evals = exp10.(range(-1, 1, length = 10))
    for Cv in Cv_evals
        subset_idx = findall(x -> isapprox(x, Cv, rtol = 1e-2), Cvs)
        sub_Qs = Qs[subset_idx]
        sub_CFs = CFs[subset_idx]
        if length(sub_Qs) > 5
            loss_sub(c) = sum(abs2.(f_tanh.(c[1], sub_Qs) .- sub_CFs))
            res = optimize(loss_sub, [1.0], LBFGS())
            c_opt = Optim.minimizer(res)[1]
            rmse = sqrt(Optim.minimum(res) / length(sub_Qs))
            @printf("Cv = %5.2f | tanh(c*Q) c_opt: %5.2f | RMSE: %.4f\n", Cv, c_opt, rmse)
        end
    end
end

experiment()
