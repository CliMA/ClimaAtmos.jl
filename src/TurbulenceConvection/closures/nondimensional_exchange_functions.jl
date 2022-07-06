#### Non-dimensional Entrainment-Detrainment functions
function max_area_limiter(εδ_model, max_area, a_up)
    FT = eltype(a_up)
    γ_lim = εδ_params(εδ_model).γ_lim
    β_lim = εδ_params(εδ_model).β_lim
    logistic_term = (2 - 1 / (1 + exp(-γ_lim * (max_area - a_up))))
    return (logistic_term)^β_lim - 1
end

function non_dimensional_groups(εδ_model, εδ_model_vars)
    FT = eltype(εδ_model_vars.tke_en)
    Δw = get_Δw(εδ_model, εδ_model_vars.w_up, εδ_model_vars.w_en)
    Δb = εδ_model_vars.b_up - εδ_model_vars.b_en
    Π_norm = εδ_params(εδ_model).Π_norm
    Π₁ = (εδ_model_vars.zc_i * Δb) / (Δw^2 + εδ_model_vars.wstar^2) / Π_norm[1]
    Π₂ =
        (εδ_model_vars.tke_gm - εδ_model_vars.a_en * εδ_model_vars.tke_en) / (εδ_model_vars.tke_gm + eps(FT)) /
        Π_norm[2]
    Π₃ = √(εδ_model_vars.a_up) / Π_norm[3]
    Π₄ = (εδ_model_vars.RH_up - εδ_model_vars.RH_en) / Π_norm[4]
    Π₅ = εδ_model_vars.zc_i / εδ_model_vars.H_up / Π_norm[5]
    Π₆ = εδ_model_vars.zc_i / εδ_model_vars.ref_H / Π_norm[6]
    Π_groups = (Π₁, Π₂, Π₃, Π₄, Π₅, Π₆)

    return map(i -> Π_groups[i], εδ_model_vars.entr_Π_subset)
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
    c_δ = if !TD.has_condensate(εδ_model_vars.q_cond_up + εδ_model_vars.q_cond_en)
        FT(0)
    else
        εδ_model.params.c_δ
    end

    Δb = εδ_model_vars.b_up - εδ_model_vars.b_en
    μ_ij = (χ - εδ_model_vars.a_up / (εδ_model_vars.a_up + εδ_model_vars.a_en)) * Δb / Δw
    exp_arg = μ_ij / μ_0
    D_ε = 1 / (1 + exp(-exp_arg))
    D_δ = 1 / (1 + exp(exp_arg))

    M_δ = (max((εδ_model_vars.RH_up)^β - (εδ_model_vars.RH_en)^β, 0))^(1 / β)
    M_ε = (max((εδ_model_vars.RH_en)^β - (εδ_model_vars.RH_up)^β, 0))^(1 / β)

    nondim_ε = (c_ε * D_ε + c_δ * M_ε)
    nondim_δ = (c_ε * D_δ + c_δ * M_δ)
    return nondim_ε, nondim_δ
end

"""
    non_dimensional_function!(nondim_ε, nondim_δ, Π_groups, εδ_model::FNOEntr)

Uses a non local (Fourier) neural network to predict the fields of
    non-dimensional components of dynamical entrainment/detrainment.
 - `nondim_ε`   :: output - non dimensional entr from FNO, as column fields
 - `nondim_δ`   :: output - non dimensional detr from FNO, as column fields
 - `Π_groups`   :: input - non dimensional groups, as column fields
 - `::FNOEntr ` a non-local entrainment-detrainment model type
"""
function non_dimensional_function!(
    nondim_ε::AbstractArray{FT}, # output
    nondim_δ::AbstractArray{FT}, # output
    Π_groups::AbstractArray{FT}, # input
    εδ_model::FNOEntr,
) where {FT <: Real}

    width = εδ_model.w_fno
    modes = εδ_model.nm_fno
    c_fno = εδ_model.c_fno
    n_params = length(c_fno)
    n_input_vars = size(Π_groups)[2]
    M = size(Π_groups)[1]
    z = LinRange(0, 1, M)
    Π_groups = hcat(Π_groups, z)
    Π = Π_groups'
    Π = reshape(Π, (size(Π)..., 1)) # (C, X, B)
    even = Bool(mod(M, 2)) ? false : true

    expected_n_params = (
        (n_input_vars + 1) * width +
        width +
        modes * width * width * 2 +
        width * width +
        width +
        width * width +
        width +
        2 * width +
        2
    )# 3rd dense layer

    if expected_n_params != n_params
        error("Incorrect number of parameters ($n_params) for requested FNO architecture ($expected_n_params)!")
    end

    trafo = OF.FourierTransform(modes = (modes,), even = even)
    model = OF.Chain(
        Flux.Dense(n_input_vars + 1, width, Flux.tanh),
        OF.SpectralKernelOperator(trafo, width => width, Flux.tanh),
        Flux.Dense(width, width, Flux.tanh),
        Flux.Dense(width, 2),
    )

    index = 1
    for p in Flux.params(model)
        len_p = length(p)
        if eltype(p) <: Real
            p[:] .= c_fno[index:(index + len_p - 1)]
            index += len_p
        elseif eltype(p) <: Complex
            p[:] .= c_fno[index:(index + len_p - 1)] + c_fno[(index + len_p):(index + len_p * 2 - 1)] * im
            index += len_p * 2
        else
            error("Bad eltype in Flux params")
        end
    end

    output = model(Π)
    nondim_ε .= Flux.relu.(output[1, :] .+ 0.5)
    nondim_δ .= Flux.relu.(output[2, :])

    return nothing
end

"""
    Count number of parameters in fully-connected NN model given Array specifying architecture following
        the pattern: [#inputs, #neurons in L1, #neurons in L2, ...., #outputs]. Equal to the number of weights + biases.
"""
num_params_from_arc(nn_arc::AbstractArray{Int}) = num_weights_from_arc(nn_arc) + num_biases_from_arc(nn_arc)

"""
    Count number of weights in fully-connected NN architecture.
"""
num_weights_from_arc(nn_arc::AbstractArray{Int}) = sum(i -> nn_arc[i] * nn_arc[i + 1], 1:(length(nn_arc) - 1))

"""
    Count number of biases in fully-connected NN architecture.
"""
num_biases_from_arc(nn_arc::AbstractArray{Int}) = sum(i -> nn_arc[i + 1], 1:(length(nn_arc) - 1))


"""
    construct_fully_connected_nn(
        arc::AbstractArray{Int},
        params::AbstractArray{FT};
        biases_bool::bool = false,
        activation_function::Flux.Function = Flux.sigmoid,
        output_layer_activation_function::Flux.Function = Flux.relu,)

    Given network architecture and parameter vectors, construct NN model and unpack weights (and biases if `biases_bool` is true).
    - `arc` :: vector specifying network architecture
    - `params` :: parameter vector containing weights (and biases if `biases_bool` is true)
    - `biases_bool` :: bool specifying whether `params` includes biases.
    - `activation_function` :: activation function for hidden layers
    - `output_layer_activation_function` :: activation function for output layer
"""
function construct_fully_connected_nn(
    arc::AbstractArray{Int},
    params::AbstractArray{FT};
    biases_bool::Bool = false,
    activation_function::Flux.Function = Flux.sigmoid,
    output_layer_activation_function::Flux.Function = Flux.relu,
) where {FT <: Real}

    # check consistency of architecture and parameters
    if biases_bool
        n_params_nn = num_params_from_arc(arc)
        n_params_vect = length(params)
    else
        n_params_nn = num_weights_from_arc(arc)
        n_params_vect = length(params)
    end
    if n_params_nn != n_params_vect
        error("Incorrect number of parameters ($n_params_vect) for requested NN architecture ($n_params_nn)!")
    end

    layers = []
    parameters_i = 1
    # unpack parameters in parameter vector into network
    for layer_i in 1:(length(arc) - 1)
        if layer_i == length(arc) - 1
            activation_function = output_layer_activation_function
        end
        layer_num_weights = arc[layer_i] * arc[layer_i + 1]

        nn_biases = if biases_bool
            params[(parameters_i + layer_num_weights):(parameters_i + layer_num_weights + arc[layer_i + 1] - 1)]
        else
            biases_bool
        end

        layer = Flux.Dense(
            reshape(params[parameters_i:(parameters_i + layer_num_weights - 1)], arc[layer_i + 1], arc[layer_i]),
            nn_biases,
            activation_function,
        )
        parameters_i += layer_num_weights

        if biases_bool
            parameters_i += arc[layer_i + 1]
        end
        push!(layers, layer)
    end

    return Flux.Chain(layers...)
end

"""
    non_dimensional_function!(nondim_ε, nondim_δ, Π_groups, εδ_model::NNEntrNonlocal)

Uses a fully connected neural network to predict the non-dimensional components of dynamical entrainment/detrainment.
    non-dimensional components of dynamical entrainment/detrainment.
 - `nondim_ε`   :: output - non dimensional entr from FNO, as column fields
 - `nondim_δ`   :: output - non dimensional detr from FNO, as column fields
 - `Π_groups`   :: input - non dimensional groups, as column fields
 - `::NNEntrNonlocal ` a non-local entrainment-detrainment model type
"""
function non_dimensional_function!(
    nondim_ε::AbstractArray{FT}, # output
    nondim_δ::AbstractArray{FT}, # output
    Π_groups::AbstractArray{FT}, # input
    εδ_model::NNEntrNonlocal,
) where {FT <: Real}
    # neural network architecture
    nn_arc = εδ_model.nn_arc
    c_nn_params = εδ_model.c_nn_params
    nn_model = construct_fully_connected_nn(nn_arc, c_nn_params; biases_bool = εδ_model.biases_bool)
    output = nn_model(Π_groups')
    nondim_ε .= output[1, :]
    nondim_δ .= output[2, :]

    return nothing
end

"""
    non_dimensional_function(εδ_model::NNEntr, εδ_model_vars)

Uses a fully connected neural network to predict the non-dimensional components of dynamical entrainment/detrainment.
 - `εδ_model`       :: NNEntr - Neural network entrainment closure
 - `εδ_model_vars`  :: structure containing variables
"""
function non_dimensional_function(εδ_model::NNEntr, εδ_model_vars)
    nn_arc = εδ_model.nn_arc
    c_nn_params = εδ_model.c_nn_params

    nondim_groups = collect(non_dimensional_groups(εδ_model, εδ_model_vars))
    # neural network architecture
    nn_model = construct_fully_connected_nn(nn_arc, c_nn_params; biases_bool = εδ_model.biases_bool)

    nondim_ε, nondim_δ = nn_model(nondim_groups)
    return nondim_ε, nondim_δ
end

"""
    non_dimensional_function(εδ_model::LinearEntr, εδ_model_vars)

Uses a simple linear model to predict the non-dimensional components of dynamical entrainment/detrainment.
 - `εδ_model`       :: LinearEntr - linear entrainment closure
 - `εδ_model_vars`  :: structure containing variables
"""
function non_dimensional_function(εδ_model::LinearEntr, εδ_model_vars)
    c_linear = εδ_model.c_linear

    nondim_groups = collect(non_dimensional_groups(εδ_model, εδ_model_vars))
    # Linear closure
    lin_arc = (length(nondim_groups), 1)  # (#inputs, #outputs)
    lin_model_ε = Flux.Dense(reshape(c_linear[1:6], lin_arc[2], lin_arc[1]), [c_linear[7]], Flux.relu)
    lin_model_δ = Flux.Dense(reshape(c_linear[8:13], lin_arc[2], lin_arc[1]), [c_linear[14]], Flux.relu)

    nondim_ε = lin_model_ε(nondim_groups)[1]
    nondim_δ = lin_model_δ(nondim_groups)[1]
    return nondim_ε, nondim_δ
end


"""
    non_dimensional_function(εδ_model::RFEntr, εδ_model_vars)

Uses a Random Feature model to predict the non-dimensional components of dynamical entrainment/detrainment.
 - `εδ_model`       :: RFEntr - basic RF entrainment closure
 - `εδ_model_vars`  :: structure containing variables
"""
function non_dimensional_function(εδ_model::RFEntr{d, m}, εδ_model_vars) where {d, m}
    # d inputs, p=2 outputs, m random features
    nondim_groups = non_dimensional_groups(εδ_model, εδ_model_vars)

    # Learnable and fixed parameters
    c_rf_fix = εδ_model.c_rf_fix # 2 x m x (1 + d), fix
    c_rf_opt = εδ_model.c_rf_opt # 2 x (m + 1 + d), learn

    # Random Features
    scale_x_entr = (c_rf_opt[1, (m + 2):(m + d + 1)] .^ 2) .* nondim_groups
    scale_x_detr = (c_rf_opt[2, (m + 2):(m + d + 1)] .^ 2) .* nondim_groups
    f_entr = c_rf_opt[1, m + 1]^2 * sqrt(2) * cos.(c_rf_fix[1, :, 2:(d + 1)] * scale_x_entr + c_rf_fix[1, :, 1])
    f_detr = c_rf_opt[2, m + 1]^2 * sqrt(2) * cos.(c_rf_fix[2, :, 2:(d + 1)] * scale_x_detr + c_rf_fix[2, :, 1])

    # Square output for nonnegativity for prediction
    nondim_ε = sum(c_rf_opt[1, 1:m] .* f_entr) / sqrt(m)
    nondim_δ = sum(c_rf_opt[2, 1:m] .* f_detr) / sqrt(m)
    return nondim_ε^2, nondim_δ^2
end

"""
    non_dimensional_function(εδ_model::LogNormalScalingProcess, εδ_model_vars)

Uses a LogNormal random variable to scale a deterministic process
to predict the non-dimensional components of dynamical entrainment/detrainment.

Arguments:
 - `εδ_model_vars`  :: structure containing variables
 - `εδ_model`       :: LogNormalScalingProcess - Stochastic lognormal scaling
"""
function non_dimensional_function(εδ_model::LogNormalScalingProcess, εδ_model_vars)
    FT = eltype(εδ_model_vars.q_cond_up)
    # model parameters
    mean_model = εδ_model.mean_model
    c_gen_stoch = εδ_model.c_gen_stoch
    ε_σ² = c_gen_stoch[1]
    δ_σ² = c_gen_stoch[2]

    # Mean model closure
    ε_mean_nondim, δ_mean_nondim = non_dimensional_function(εδ_model, εδ_model_vars, mean_model)

    # lognormal scaling
    nondim_ε = ε_mean_nondim * lognormal_sampler(FT(1), ε_σ²)
    nondim_δ = δ_mean_nondim * lognormal_sampler(FT(1), δ_σ²)

    return nondim_ε, nondim_δ
end

function lognormal_sampler(m::FT, var::FT)::FT where {FT}
    μ = log(m^2 / √(m^2 + var))
    σ = √(log(1 + var / m^2))
    return rand(Distributions.LogNormal(μ, σ))
end

"""
    non_dimensional_function(εδ_model::NoisyRelaxationProcess, εδ_model_vars)

Uses a noisy relaxation process to predict the non-dimensional components
of dynamical entrainment/detrainment. A deterministic closure is used as the
equilibrium mean function for the relaxation process.

Arguments:
 - `εδ_model_vars`  :: structure containing variables
 - `εδ_model`       :: NoisyRelaxationProcess - A noisy relaxation process closure
"""
function non_dimensional_function(εδ_model::NoisyRelaxationProcess, εδ_model_vars)
    # model parameters
    mean_model = εδ_model.mean_model
    c_gen_stoch = εδ_model.c_gen_stoch
    ε_σ² = c_gen_stoch[1]
    δ_σ² = c_gen_stoch[2]
    ε_λ = c_gen_stoch[3]
    δ_λ = c_gen_stoch[4]

    # Mean model closure
    ε_mean_nondim, δ_mean_nondim = non_dimensional_function(mean_model, εδ_model_vars)

    # noisy relaxation process
    ε_u0 = εδ_model_vars.ε_nondim
    δ_u0 = εδ_model_vars.δ_nondim
    Δt = εδ_model_vars.Δt
    nondim_ε = noisy_relaxation_process(ε_mean_nondim, ε_λ, ε_σ², ε_u0, Δt)
    nondim_δ = noisy_relaxation_process(δ_mean_nondim, δ_λ, δ_σ², δ_u0, Δt)

    return nondim_ε, nondim_δ
end

"""
    Solve a noisy relaxation process numerically

In this formulation, the noise amplitude is scaled by the speed of
reversion λ and the long-term mean μ, in addition to the variance σ² as is usual,

    `du = λ(μ - u)⋅dt + √(2λμσ²)⋅dW`

To ensure non-negativity, the solution is passed through a relu filter.
"""
function noisy_relaxation_process(μ::FT, λ::FT, σ²::FT, u0::FT, Δt::FT)::FT where {FT}
    f(u, p, t) = λ * (μ - u)        # mean-reverting process
    g(u, p, t) = √(2λ * μ * σ²)     # noise fluctuation
    tspan = (FT(0), Δt)
    prob = SDE.SDEProblem(f, g, u0, tspan; save_start = false, saveat = last(tspan))
    sol = SDE.solve(prob, SDE.SOSRI())
    return Flux.relu(sol.u[end])
end
