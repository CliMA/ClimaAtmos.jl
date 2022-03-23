"""
    get_boundary_flux(model, ::NoFlux, var::Fields.Field, Y, Ya)

    Vertical flux assignment for no-flux conditions (e.g., insulating wall)
"""
function get_boundary_flux(model, ::NoFlux, var::Fields.Field, Y, Ya)
    FT = eltype(Y)
    flux = Geometry.Contravariant3Vector(FT(0))
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

"""
    get_boundary_flux(model, bc::DragLaw, uv, Y, Ya)

    Vertical momentum fluxes for a bulk-formulation drag law, given the momentum exchange coefficient
"""
function get_boundary_flux(model, bc::DragLaw, uv, Y, Ya)
    FT = eltype(Y)
    coefficient =
        eltype(bc.coefficient) == FT ? bc.coefficient : bc.coefficient(Y, Ya)

    uv_1 = Fields.level(uv, 1)
    u_wind = norm(uv_1)

    flux =
        Geometry.Contravariant3Vector(coefficient * u_wind) ⊗
        Geometry.Covariant12Vector(FT(0), FT(0)) #uv_1 # change when ClimaCore enables Fields for specifying boundary conditions (#325)
end

"""
    get_boundary_flux(
        model,
        bc::BulkFormula,
        ρc::Fields.Field,
        Y,
        Ya,
    )

    Vertical fluxes for arbitrary variables with the bulk aerodynamic turbulent formula, 
    given variable exchange coefficients. (e.g. for energy, or tracers)
"""

function get_boundary_flux(model, bc::BulkFormula, ρc::Fields.Field, Y, Ya)
    FT = eltype(Y.base)
    c_sfc = bc.surface_field

    ρ_1 = Fields.level(Y.base.ρ, 1)
    ρc_1 = Fields.level(ρc, 1)
    uh_1 = Fields.level(Y.base.uh, 1)
    u_wind = norm(uh_1)

    bulk_flux = bc.coefficients .* ρ_1 .* u_wind .* (ρc_1 ./ ρ_1 .- c_sfc)

    flux = Geometry.Contravariant3Vector.(parent(bulk_flux)[1]) # change when ClimaCore enables Fields for specifying boundary conditions (#325)
end
