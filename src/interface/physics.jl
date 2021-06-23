abstract type AbstractEquationOfState end
abstract type AbstractThermodynamicVariable end
abstract type AbstractCompressibility end
abstract type AbstractSource end

abstract type AbstractMicrophysics <: AbstractSource end

# thermodynamics
struct Density <: AbstractThermodynamicVariable end
struct TotalEnergy <: AbstractThermodynamicVariable end

struct BarotropicFluid <: AbstractEquationOfState end
struct DryIdealGas <: AbstractEquationOfState end
struct MoistIdealGas <: AbstractEquationOfState end

# compressibility
struct Compressible <: AbstractCompressibility end

# coriolis force
struct DeepShellCoriolis <: AbstractSource end

# gravity
struct Gravity <: AbstractSource end

# microphysics

struct ZeroMomentMicrophysics <: AbstractMicrophysics end