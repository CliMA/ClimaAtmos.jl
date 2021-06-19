abstract type AbstractBoundaryCondition end

struct NoFlux <: AbstractBoundaryCondition end
struct FreeSlip <: AbstractBoundaryCondition end