abstract type AbstractEquationOfState end
abstract type AbstractThermodynamicVariable end
abstract type AbstractCompressibility end
abstract type AbstractPhysics end

# thermodynamics
struct Density <: AbstractThermodynamicVariable end
struct TotalEnergy <: AbstractThermodynamicVariable end

struct BarotropicFluid <: AbstractEquationOfState end
struct DryIdealGas <: AbstractEquationOfState end
struct MoistIdealGas <: AbstractEquationOfState end

# compressibility
struct Compressible <: AbstractCompressibility end

# coriolis force
struct DeepShellCoriolis <: AbstractPhysics end
struct ThinShellCoriolis <: AbstractPhysics end
struct BetaPlaneCoriolis <: AbstractPhysics end

# gravity
struct DeepGravity <: AbstractPhysics end
struct ShallowGravity <: AbstractPhysics end
