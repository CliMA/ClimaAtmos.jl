abstract type AbstractBoundaryCondition end

struct DirichletBC <: AbstractBoundaryCondition end
struct NeumannBC <: AbstractBoundaryCondition end
struct FluxBC <: AbstractBoundaryCondition end

# impenetrable freeslip
struct DefaultBC <: AbstractBoundaryCondition end