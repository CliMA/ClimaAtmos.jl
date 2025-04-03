using Flux


"""
    Count number of parameters in fully-connected NN model given Array specifying architecture following
        the pattern: [#inputs, #neurons in L1, #neurons in L2, ...., #outputs]. Equal to the number of weights + biases.
"""
num_params_from_arc(nn_arc::AbstractArray{Int}) =
    num_weights_from_arc(nn_arc) + num_biases_from_arc(nn_arc)

"""
    Count number of weights in fully-connected NN architecture.
"""
num_weights_from_arc(nn_arc::AbstractArray{Int}) =
    sum(i -> nn_arc[i] * nn_arc[i + 1], 1:(length(nn_arc) - 1))

"""
    Count number of biases in fully-connected NN architecture.
"""
num_biases_from_arc(nn_arc::AbstractArray{Int}) =
    sum(i -> nn_arc[i + 1], 1:(length(nn_arc) - 1))


"""
    construct_fully_connected_nn(
        arc::AbstractArray{Int},
        params::AbstractArray{FT};
        biases_bool::bool = false,
        activation_function::Flux.Function = Flux.relu,
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
    activation_function::Flux.Function = Flux.relu,
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
        error(
            "Incorrect number of parameters ($n_params_vect) for requested NN architecture ($n_params_nn)!",
        )
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
            reshape(
                params[parameters_i:(parameters_i + layer_num_weights - 1)],
                arc[layer_i + 1],
                arc[layer_i],
            ),
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
serialize_ml_model(
    ml_model::Flux.Chain,
    )

    Given Flux NN model, serialize model weights into a vector of parameters.
    - `ml_model` - A Flux model instance.
"""

function serialize_ml_model(ml_model::Flux.Chain)
    parameters = []
    param_type = eltype.(Flux.params(ml_model))[1]
    for layer in ml_model
        for param in Flux.params(layer)
            param_flattened = reshape(param, length(param))
            push!(parameters, param_flattened...)
        end
    end
    return convert(Array{param_type}, parameters)
end


"""
    construct_fully_connected_nn_default(
        arc::AbstractArray{Int};
        biases_bool::Bool = false,
        activation_function::Flux.Function = Flux.relu,
        output_layer_activation_function::Flux.Function = Flux.relu,
    )

Given network architecture, construct NN model with default `Flux.jl` weight initialization (glorot_uniform).
- `arc` :: vector specifying network architecture
- `biases_bool` :: bool specifying whether to include biases.
- `activation_function` :: activation function for hidden layers
- `output_layer_activation_function` :: activation function for output layer
"""

function construct_fully_connected_nn_default(
    arc::AbstractArray{Int};
    biases_bool::Bool = false,
    activation_function::Flux.Function = Flux.relu,
    output_layer_activation_function::Flux.Function = Flux.relu,
)

    layers = []
    for layer_i in 1:(length(arc) - 1)
        if layer_i == length(arc) - 1
            activation_function = output_layer_activation_function
        end

        layer = Flux.Dense(
            arc[layer_i] => arc[layer_i + 1],
            activation_function;
            bias = biases_bool,
        )
        push!(layers, layer)
    end

    return Flux.Chain(layers...)
end



"""
    serialize_std_model(
        ml_model::Flux.Chain;
        std_weight::Real,
        std_bias::Real
    ) -> Vector{Float64}

Given a Flux NN model, serialize the standard deviations for weights and biases into a vector.
- `ml_model` :: A Flux.Chain model instance.
- `std_weight` :: Standard deviation for weights.
- `std_bias` :: Standard deviation for biases.

Returns a vector of stds where weights have `std_weight` and biases have `std_bias`,
aligned with the parameter order from `serialize_ml_model`.
"""
function serialize_std_model(
    ml_model::Flux.Chain;
    std_weight::Real,
    std_bias::Real,
)
    stds = Float64[]  # Initialize an empty Float64 vector

    for layer in ml_model.layers
        # Ensure the layer is a Dense layer
        if isa(layer, Flux.Dense)
            # Serialize weights
            weights = layer.weight
            n_weights = length(weights)
            push!(stds, fill(Float64(std_weight), n_weights)...)

            # Serialize biases if they exist
            if layer.bias !== nothing
                biases = layer.bias
                n_biases = length(biases)
                push!(stds, fill(Float64(std_bias), n_biases)...)
            end
        else
            error(
                "serialize_std_model currently supports only Flux.Dense layers. Found layer of type $(typeof(layer)).",
            )
        end
    end

    return stds
end
