"""
    get_boundary_flux(::NoFluxCondition, var::Fields.Field, Y, Ya)

    Vertical flux assignment for no-flux conditions (i.e. free-slip wall)
"""
function get_boundary_flux(::NoFluxCondition, var::Fields.Field, Y, Ya) # what about var is UVVector/UVVector?
    FT = eltype(Y)
    flux = Geometry.WVector(FT(0))
end

"""
    get_boundary_flux(bc::DragLawCondition, uv, Y, Ya)

    Vertical momentum fluxes for a bulk-formulation drag law, given momentum exchange coefficients
"""
function get_boundary_flux(bc::DragLawCondition, uv, Y, Ya)
    FT = eltype(Y)
    coefficients =
        eltype(bc.coefficients) == FT ? bc.coefficients : bc.coefficients(Y, Ya)

    uv_1 = first_interior(uv)
    u_wind = norm(uv_1)

    flux = Geometry.WVector(coefficients.Cd * u_wind) ⊗ uv_1
end

"""
    get_boundary_flux(
        bc::BulkFormulaCondition,
        ρc::Fields.Field,
        Y,
        Ya,
    )

    Vertical fluxes for arbitrary variables with a bulk-formulation drag law, 
    given variable exchange coefficients. (e.g. for energy, or tracers)
"""
function get_boundary_flux(
    bc::BulkFormulaCondition,
    ρc::Fields.Field,
    Y,
    Ya,
)
    FT = eltype(Y.base)
    coefficients =
        eltype(bc.coefficients) == FT ? bc.coefficients : bc.coefficients(Y, Ya)
    c_sfc = bc.surface_field

    ρ_1 = first_interior(Y.base.ρ)
    ρc_1 = first_interior(ρc)
    uv_1 = first_interior(Y.base.uv)
    u_wind = norm(uv_1)

    flux =
        Geometry.WVector(coefficients.Ch * u_wind * ρ_1 * (ρc_1 / ρ_1 - c_sfc))
end

# Obtain the first interior datapoint of variable `v`
first_interior(v) = Operators.getidx(v, Operators.Interior(), 1)
