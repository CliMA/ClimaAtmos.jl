# General
"""
    get_boundary_flux(model, ::NoFluxCondition, _...)

Construct zero boundary fluxes for a scalar or 2-element vector(x,y)
"""
function get_boundary_flux(model, ::NoFluxCondition, var::Fields.Field, Ym, Ya)
    FT = eltype(Ym)
    flux = Geometry.WVector(FT(0))
end

# Momentum-specific
"""
    get_boundary_flux(model, ::DragLawCondition, _...)

Construct non-zero boundary fluxes using the bulk formula and constant or Mohnin-Obukhov-based drag coefficient, Cd
"""
function get_boundary_flux(model, bc::DragLawCondition, uv, Ym, Ya)
    FT = eltype(Ym)
    coefficients = eltype(bc.coefficients) == FT ? bc.coefficients :
        bc.coefficients(Ym, Ya)

    uv_1 = first_interior(Ym.uv)
    u_wind = LinearAlgebra.norm(uv_1)

    flux = Geometry.WVector(coefficients.Cd * u_wind) ⊗ uv_1
end


# Potential temperature density-specific
"""
    get_boundary_flux(model, ::BulkFormulaCondition, _...)

    Construct non-zero boundary fluxes using the bulk formula and constant or Mohnin-Obukhov-based heat transfer coefficient, Ch
"""
function get_boundary_flux(
    model,
    bc::BulkFormulaCondition,
    ρθ::Fields.Field,
    Ym,
    Ya,
)
    FT = eltype(Ym)
    coefficients = eltype(bc.coefficients) == FT ? bc.coefficients :
        bc.coefficients(Ym, Ya)
    θ_sfc = bc.θ_sfc

    ρ_1 = first_interior(Ym.ρ)
    ρθ_1 = first_interior(Ym.ρθ)
    uv_1 = first_interior(Ym.uv)
    u_wind = LinearAlgebra.norm(uv_1)

    flux =
        Geometry.WVector(coefficients.Ch * u_wind * ρ_1 * (ρθ_1 / ρ_1 - θ_sfc))
end

# helpers
"""
    first_interior(f)
- obtains the first interior datapoint of variable `v`
"""
first_interior(v) = Operators.getidx(v, Operators.Interior(), 1)
