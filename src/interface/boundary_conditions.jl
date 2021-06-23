abstract type AbstractBoundaryCondition end

struct DefaultBC <: AbstractBoundaryCondition end

Base.@kwdef struct BulkFormulaTemperature{𝒜,ℬ,𝒞} <: AbstractBoundaryCondition 
  drag_coef_temperature::𝒜
  drag_coef_moisture::ℬ
  surface_temperature::𝒞
end