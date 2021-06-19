abstract type AbstractPhysics end
abstract type AbstractEquationOfState end

# coriolis force
struct DeepShellCoriolis <: AbstractPhysics end

# gravity
struct DeepGravity <: AbstractPhysics end
struct ShallowGravity <: AbstractPhysics end

# thermodynamics
struct BarotropicFluid <: AbstractEquationOfState end
struct DryIdealGas <: AbstractEquationOfState end
struct MoistIdealGas <: AbstractEquationOfState end