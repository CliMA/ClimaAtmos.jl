abstract type AbstractEquationSet end
abstract type AbstractModel end

"""
    ThreeDimensionalEuler <: AbstractEquationSet
"""
Base.@kwdef struct ThreeDimensionalEuler{𝒜,ℬ,𝒞,𝒟,ℰ} <: AbstractEquationSet
    thermodynamic_variable::𝒜
    equation_of_state::ℬ
    pressure_convention::𝒞
    sources::𝒟
    ref_state::ℰ
end

"""
    ModelSetup <: AbstractModel
"""
Base.@kwdef struct ModelSetup{𝒜,ℬ,𝒞,𝒟,ℰ} <: AbstractModel
    equation_set::𝒜 # 3D navier stokes, 2D navier stokes
    domain::ℰ # discretized_domain
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
#         sources = (
#             Gravity(),
#             Coriolis(),
#             Radiation(),
#         )
#         ref_state = DryReferenceState(),
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