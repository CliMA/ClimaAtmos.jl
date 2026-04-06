#=

    Here we will define and train transformer based error correctors.

    Transformers are nice because they can learn long range dependencies in the vertical, e.g. response to the surface.
    Additionally they permit variable length inputs, which is useful for handling different vertical resolutions.

    We will use the 

=#

module TransformerErrorCorrectors

using ClimaAtmos: ClimaAtmos as CA, ErrorCorrection as CAEC



struct TransformerErrorCorrector{TT} <: AbstractErrorCorrector
    transformer::TT
end
(::TransformerErrorCorrector)(Y_TEC, Y, p, t, col, cache, FT, params, t_step::Real, dt::Real, _...) = TEC.transformer(Y_TEC, Y, p, t, col, cache, FT, params, t_step, dt) # tjhis should really be in place mutating, and write into Y_TEC

function ErrorCorrection.apply_error_correction!(TEC::TransformerErrorCorrector, Y, Y_TEC, p, t, col, cache, FT, params, t_step::Real, dt::Real, _...)
    # The transformer should be able to write directly into its array
    TEC(Y_TEC, Y, p, t, col, cache, FT, params, t_step, dt)
    @. Y += Y_TEC
end

function load!(TEC::TransformerErrorCorrector, path::String)
    # Load the transformer from the path    
    error("Not implemented yet")
end


end
