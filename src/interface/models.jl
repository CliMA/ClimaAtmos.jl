abstract type AbstractModel end
abstract type AbstractEquationOfState end
abstract type AbstractThermodynamicVariable end
abstract type AbstractCompressibility end
abstract type AbstractEquationSet end

struct TotalEnergy <: AbstractThermodynamicVariable
# Examples:
# struct InternalEnergy <: AbstractThermodynamicVariable
# struct PotentialTemperature <: AbstractThermodynamicVariable

struct DryIdealGas <: AbstractEquationOfState
struct MoistIdealGas <: AbstractEquationOfState

struct Compressible <: AbstractCompressibility
# Examples:
# struct Anelastic <: AbstractCompressibility
# struct Hydrostatic <: AbstractCompressibility

"""
    ThreeDimensionalNavierStokes <: AbstractEquationSet
"""
Base.@kwdef struct ThreeDimensionalNavierStokes{𝒜,ℬ,𝒞} <: AbstractEquationSet
    thermodynamic_variable::𝒜
    equation_of_state::ℬ
    compressibility::𝒞
end

"""
    ModelSetup <: AbstractFluidModel
"""
Base.@kwdef struct ModelSetup{𝒜,ℬ,𝒞,𝒟} <: AbstractModel
    equations::𝒜 # 3D navier stokes, 2D navier stokes
    physics::ℬ # sources, parameterizations, diffusion
    boundary_conditions::𝒞 # no flux / free slip
    initial_conditions::𝒟 # initialize with zero, one, etc.
end

# TODO!: Default atmospheric configuration
# function IdealizedDryAtmosModelSetup(initial_conditions)
#     equations = ThreeDimensionalNavierStokes(
#         thermodynamic_variable = TotalEnergy(),
#         equation_of_state = DryIdealGas(),
#         compressibility = Compressible(),
#     )
#     physics = (
#         gravity = Gravity(),
#         coriolis = Coriolis(),    
#     )
#     boundary_conditions = (
#         top = FreeSlip(), 
#         bottom = FreeSlip()
#     )
#
#     return ModelSetup(
#         equations = equations, 
#         physics = physics,
#         boundary_conditions = boundary_conditions,
#         initial_conditions = initial_conditions,
#     )
# end