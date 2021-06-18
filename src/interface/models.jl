abstract type AbstractThermodynamicVariable end
abstract type AbstractEquationOfState end
abstract type AbstractCompressibility end
abstract type AbstractEquationSet end
abstract type AbstractModel end

struct TotalEnergy <: AbstractThermodynamicVariable
# examples:
# struct InternalEnergy <: AbstractThermodynamicVariable
# struct PotentialTemperature <: AbstractThermodynamicVariable

struct DryIdealGas <: AbstractEquationOfState
struct MoistIdealGas <: AbstractEquationOfState

struct Compressible <: AbstractCompressibility
# examples:
# struct Anelastic <: AbstractCompressibility
# struct Hydrostatic <: AbstractCompressibility

"""
    ThreeDimensionalEuler <: AbstractEquationSet
"""
Base.@kwdef struct ThreeDimensionalEuler{𝒜,ℬ,𝒞} <: AbstractEquationSet
    thermodynamic_variable::𝒜
    equation_of_state::ℬ
    compressibility::𝒞
end

"""
    ModelSetup <: AbstractModel
"""
Base.@kwdef struct ModelSetup{𝒜,ℬ,𝒞,𝒟,ℰ} <: AbstractModel
    equations::𝒜 # 3D navier stokes, 2D navier stokes
    physics::ℬ # sources, parameterizations, diffusion
    boundary_conditions::𝒞 # no flux / free slip
    initial_conditions::𝒟 # initialize with zero, one, etc.
    parameters::ℰ
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