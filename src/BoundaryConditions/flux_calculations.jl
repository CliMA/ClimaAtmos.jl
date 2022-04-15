"""
    get_boundary_flux(model, ::NoFlux, var::Fields.Field, Y, Ya)

    Vertical flux assignment for no-flux conditions (i.e., insulating wall)
"""
function get_boundary_flux(model, ::NoFlux, var::Fields.Field, Y, Ya)
    FT = eltype(Y)
    flux = Geometry.Contravariant3Vector(FT(0))
end

"""
    get_boundary_flux(model, ::NoVectorFlux, var::Fields.Field, Y, Ya)

    Vertical flux assignment for an AxisTensor with CovariantAxis{(1,2)} in no-flux conditions (i.e., insulating wall)
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

function get_boundary_flux(model, bc::BulkFormulaDryTotalEnergy, ρc::Fields.Field, Y, Ya)
    FT = eltype(Y.base)

    sfc = bc.surface_fields

    # get relevant states of the first atmos internal point and the surface 
    p = calculate_pressure(
        Y,
        Ya,
        model.base,
        model.thermodynamics,
        model.moisture,
        model.parameters,
        FT,
    )

    qt_sfc = FT(0)
    cv_d = TD.cv_m(model.parameters, FT)

    # internal energies
    e_int_in = Fields.level(TD.air_temperature_from_ideal_gas_law.(model.parameters, p, Y.base.ρ) .* cv_d, 1)
    e_int_sfc = sfc.surface_temperature .* cv_d

    # horizontal velocity
    uh = Geometry.UVWVector.(Y.base.uh)
    u_in = Fields.level(uh.components.data.:1, 1)
    v_in = Fields.level(uh.components.data.:2, 1)
    u_sfc = sfc.surface_horizontal_velocity[1]
    v_sfc = sfc.surface_horizontal_velocity[2]

    # height
    z_in = similar(e_int_in)
    parent(z_in) .= parent(Fields.coordinate_field(axes(e_int_in)).z)[1]
    z_sfc = sfc.surface_height

    # density
    ρ_in = Fields.level(Y.base.ρ,1)
    ρ_sfc = Fields.level(Y.base.ρ,1) #TODO: use the new density weighted extrapolate operator

    # broadcast SurfaceFluxes.jl functions
    bulk_flux = map_fluxes.(parent(e_int_in), e_int_sfc, parent(u_in), u_sfc, parent(v_in), v_sfc, parent(z_in), z_sfc, parent(ρ_in), parent(ρ_sfc), qt_sfc; model = model, bc = bc)

    flux = Geometry.Contravariant3Vector.(parent(bulk_flux)[1]) # TODO: change when ClimaCore enables Fields for specifying boundary conditions (#325)
end

function map_fluxes(e_int_in, e_int_sfc, u_in, u_sfc,  v_in, v_sfc, z_in, z_sfc, ρ_in, ρ_sfc, qt_sfc; model = nothing, bc = nothing)

    FT = eltype(e_int_in)

    # recombine into a horizontal velocity vector
    uh_in = [u_in, v_in]
    uh_sfc = [u_sfc, v_sfc]

    # thermodynamic states
    ts_in = TD.PhaseEquil_ρeq(model.parameters, ρ_in, e_int_in, qt_sfc) 
    ts_sfc = TD.PhaseEquil_ρeq(model.parameters, ρ_sfc, e_int_sfc, qt_sfc)  

    # collect states
    state_sfc = SF.SurfaceValues(z_sfc, uh_sfc, ts_sfc) 
    state_in = SF.InteriorValues(z_in, uh_in, ts_in)

    # collect all surface conditions
    kwargs = (; state_in, state_sfc)
    sc = SF.Coefficients{FT}.(; kwargs..., Cd = bc.coefficients.Cd, Ch = bc.coefficients.Ch, z0m = FT(0), z0b = FT(0))

    uf = UF.Businger()
    bulk_flux = SF.sensible_heat_flux(model.parameters, sc.Ch, sc, uf) 
end

function calculate_pressure(Y, Ya, b, t, m, p, F)
    ClimaAtmos.Models.Nonhydrostatic3DModels.calculate_pressure(Y, Ya, b, t, m, p, F)
end

function calculate_gravitational_potential(Y, Ya, params, FT)
    ClimaAtmos.Models.Nonhydrostatic3DModels.calculate_gravitational_potential(Y, Ya, params, FT)
end

