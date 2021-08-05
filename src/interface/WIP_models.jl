abstract type AbstractModel end

"""
    ModelSetup <: AbstractModel
"""
Base.@kwdef struct ModelSetup{A,B,C,D,E} <: AbstractModel
    equation_set::A
    domain::B
    boundary_conditions::C 
    initial_conditions::D
    parameters::E
end

# TODO!: Default atmospheric configuration
# function IdealizedDryAtmosModelSetup(initial_conditions, parameters)
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