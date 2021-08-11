abstract type AbstractModel end

"""
    BarotropicFluidModel <: AbstractModel
"""
Base.@kwdef struct BarotropicFluidModel{A,B,C,D,E} <: AbstractModel
    domain::A
    physics::B = ModelPhysics()
    boundary_conditions::C
    initial_conditions::D
    parameters::E
end

"""
    HydrostaticModel <: AbstractModel
"""
Base.@kwdef struct HydrostaticModel{A,B,C,D} <: AbstractModel
    domain::A
    boundary_conditions::B
    initial_conditions::C
    parameters::D
end
