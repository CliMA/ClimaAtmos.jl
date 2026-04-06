module ErrorCorrection

    abstract type AbstractErrorCorrector end

    function apply_error_correction!(::AbstractErrorCorrector, Y, p, t, col, cache, FT, params, t_step::Real, dt::Real, _...) end

    include("transformers.jl") # Transformer error correctors really should be an extension since need's stuff


    abstract type AbstractConstraint end
    abstract type AbstractConservationConstraint <: AbstractConstraint end
    abstract type AbstractNonNegativityConstraint <: AbstractConstraint end
    
end