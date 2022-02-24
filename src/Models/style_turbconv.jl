"""
Supertype for all model subgrid-scale styles.
"""
abstract type AbstractSubgridscaleStyle <: AbstractModelStyle end

Base.@kwdef struct ConstantViscosity{FT} <: AbstractSubgridscaleStyle
    ν::FT = 0
end

Models.variable_names(::Union{Nothing, ConstantViscosity}) = nothing

Models.variable_types(
    ::Union{Nothing, ConstantViscosity},
    ::AbstractModel,
    FT,
) = Nothing

Models.variable_spaces(
    ::Union{Nothing, ConstantViscosity},
    ::AbstractModel,
    FT,
) = Nothing
