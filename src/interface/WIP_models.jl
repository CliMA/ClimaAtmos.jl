abstract type AbstractModel end

"""
    BarotropicFluidModel <: AbstractModel
"""
Base.@kwdef struct BarotropicFluidModel{A,B,C,D} <: AbstractModel
    domain::A
    boundary_conditions::B
    initial_conditions::C
    parameters::D
end