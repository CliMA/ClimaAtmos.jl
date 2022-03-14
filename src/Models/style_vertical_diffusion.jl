"""
Supertype for all vertical diffusion styles.
"""
abstract type AbstractVerticalDiffusionStyle <: AbstractModelStyle end

struct NoVerticalDiffusion <: AbstractVerticalDiffusionStyle end
Base.@kwdef struct ConstantViscosity{FT} <: AbstractVerticalDiffusionStyle
    ν::FT = 0
end

Models.variable_names(::AbstractVerticalDiffusionStyle) = nothing

Models.variable_types(::AbstractVerticalDiffusionStyle, ::AbstractModel, FT) =
    Nothing

Models.variable_spaces(::AbstractVerticalDiffusionStyle, ::AbstractModel, FT) =
    Nothing
