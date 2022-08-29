"""
    get_boundary_flux(model, ::NoFlux, var::Fields.Field, Y, Ya)

    Vertical flux assignment for no-flux conditions (e.g., insulating wall)
"""
function get_boundary_flux(model, ::NoFlux, var::Fields.Field, Y, Ya)
    FT = eltype(Y)
    flux = Geometry.WVector(FT(0))
end

"""
    get_boundary_flux(model, ::NoVectorFlux, var::Fields.Field, Y, Ya)

    Vertical flux assignment for an AxisTensor with CovariantAxis{(1,2)} in no-flux conditions (e.g., free-slip wall)
"""
function get_boundary_flux(model, ::NoVectorFlux, var::Fields.Field, Y, Ya)
    FT = eltype(Y)
    flux =
        Geometry.Covariant3Vector(FT(0)) ⊗
        Geometry.Covariant12Vector(FT(0), FT(0))
end
