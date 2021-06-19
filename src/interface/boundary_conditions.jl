abstract type AbstractBoundaryCondition end

struct NoFlux <: AbstractBoundaryCondition end
struct DefaultBC <: AbstractBoundaryCondition end

Base.@kwdef struct BulkFormulaTemperature{𝒯,𝒰,𝒱} <: AbstractBoundaryCondition 
  drag_coef_temperature::𝒯
  drag_coef_moisture::𝒰
  surface_temperature::𝒱
end