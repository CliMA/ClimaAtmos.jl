#= 

    Here we will train a transformer based error corrector.

    This should not be included as part of the package, but the training should be done 
    separately and the trained model can be loaded into the package.

    The training script should be able to handle different training setups, different rollouts, 
    physics constrained et.c

=#

using ClimaAtmos: ClimaAtmos as CA, ErrorCorrection as CAEC
using ClimaAtmos.ErrorCorrection: TransformerErrorCorrectors
using ClimaAtmos.ErrorCorrection.TransformerErrorCorrectors: TransformerErrorCorrector

using Lux: Lux # They say its' better than Flux


"""
    Here we'd like to be able to train a transformer based error corrector.
    
    We'd like to enable various training setups, different rollouts, physics constrained et.c

"""
function train!(::TransformerErrorCorrector,)
    error("Not implemented yet")
end

# Need to figure out how to enforce hard contraints on the transformer, like conservation of energy and moisture.
# You could add mismatch to the loss or you could enforce hard scalings that prevent violation
# Y is organized as ?
"""
This gets you a transformer, including if you need any custom constraint layers at the end
"""
function get_transformer() 
    error("Not implemented yet")

    return transformer
end

# No constraints or anything
function basic_transformer()
    error("Not implemented yet")

    return transformer
end