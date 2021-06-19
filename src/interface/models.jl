abstract type AbstractThermodynamicVariable end
abstract type AbstractEquationOfState end
abstract type AbstractCompressibility end
abstract type AbstractEquationSet end
abstract type AbstractModel end

struct TotalEnergy <: AbstractThermodynamicVariable end
# examples:
# struct InternalEnergy <: AbstractThermodynamicVariable
# struct PotentialTemperature <: AbstractThermodynamicVariable

struct DryIdealGas <: AbstractEquationOfState end
struct MoistIdealGas <: AbstractEquationOfState end

struct Compressible <: AbstractCompressibility end
# examples:
# struct Anelastic <: AbstractCompressibility
# struct Hydrostatic <: AbstractCompressibility

"""
    ThreeDimensionalEuler <: AbstractEquationSet
"""
Base.@kwdef struct ThreeDimensionalEuler{𝒜,ℬ,𝒞,𝒟} <: AbstractEquationSet
    thermodynamic_variable::𝒜
    equation_of_state::ℬ
    pressure_convention::𝒞
    physics::𝒟
end

"""
    ModelSetup <: AbstractModel
"""
Base.@kwdef struct ModelSetup{𝒜,ℬ,𝒞,𝒟} <: AbstractModel
    equations::𝒜 # 3D navier stokes, 2D navier stokes
    boundary_conditions::ℬ # no flux / free slip
    initial_conditions::𝒞 # initialize with zero, one, etc.
    parameters::𝒟
end

# TODO!: Default atmospheric configuration
# function IdealizedDryAtmosModelSetup(initial_conditions)
#     equations = ThreeDimensionalEuler(
#         thermodynamic_variable = TotalEnergy(),
#         equation_of_state = DryIdealGas(),
#         compressibility = Compressible(),
#     ),
#     physics = (
#         gravity = Gravity(),
#         coriolis = Coriolis(),
#     ),
#     boundary_conditions = (
#         ρ  = (top = NoFlux(), bottom = NoFlux(),),
#         ρu = (top = FreeSlip(), bottom = FreeSlip(),),
#         ρe = (top = NoFlux(), bottom = NoFlux(),),
#     ),
#     initial_conditions = initial_conditions,
#     parameters = parameters,
# )
# end