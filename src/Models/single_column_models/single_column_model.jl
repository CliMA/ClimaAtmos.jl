"""
    SingleColumnModel <: AbstractModel
"""
Base.@kwdef struct SingleColumnModel{DT<:AbstractVerticalDomain,BCT,ICT,PT} <: AbstractModel
    domain::DT
    boundary_conditions::BCT
    initial_conditions::ICT
    parameters::PT
end

"""
    make_ode_function(model::SingleColumnModel)
"""
function make_ode_function(model::SingleColumnModel)
    function rhs!(dY, Y, _, t)
        # unpack parameters
        @unpack Cd, f, ν, ug, vg, C_p, MSLP, R_d, R_m, C_v, grav = model.parameters

        # unpack prognostic state
        (Yc, Yf) = Y.x
        (dYc, dYf) = dY.x
        @unpack ρ, u, v, ρθ = Yc
        @unpack w = Yf

        # unpack prognostic state tendency
        dρ, du, dv, dw, dρθ = dYc.ρ, dYc.u, dYc.v, dYf.w, dYc.ρθ

        # unpack boundary conditions
        bc_ρ  = model.boundary_conditions.ρ
        bc_u  = model.boundary_conditions.u
        bc_v  = model.boundary_conditions.v
        bc_w  = model.boundary_conditions.w
        bc_ρθ = model.boundary_conditions.ρθ

        # density (centers)
        flux_bottom = get_boundary_flux(model, bc_ρ.bottom, ρ, Y, model.parameters)
        flux_top    = get_boundary_flux(model, bc_ρ.top, ρ, Y, model.parameters)
        ∇c = Operators.GradientC2F()
        ∇f = Operators.GradientF2C(bottom = flux_bottom, top = flux_top)

        If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
        @. dρ = ∇f( -w * If(ρ) ) # Eq. 4.11

        # potential temperature (centers)
        flux_bottom = get_boundary_flux(model, bc_ρθ.bottom, ρθ, Y, model.parameters)
        flux_top    = get_boundary_flux(model, bc_ρθ.top, ρθ, Y, model.parameters)
        ∇c = Operators.GradientC2F()
        ∇f = Operators.GradientF2C(bottom = flux_bottom, top = flux_top) # Eq. 4.20, 4.21

        @. dρθ = ∇f( -w * If(ρθ) + ν * ∇c(ρθ/ρ) ) # Eq. 4.12

        # u velocity (centers)
        flux_bottom = get_boundary_flux(model, bc_u.bottom, u, Y, model.parameters)
        # TODO! move into custom boundary condition
        center_space = axes(Yc)
        dz_top = center_space.face_local_geometry.WJ[end]
        u_top = parent(u)[end]
        flux_top = ν * (u_top - ug)/ dz_top / 2
        ∇c = Operators.GradientC2F() # Eq. 4.18
        ∇f = Operators.GradientF2C(bottom = flux_bottom, top = Operators.SetValue(flux_top)) # Eq. 4.16

        A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
        @. du = ∇f(ν * ∇c(u)) + f * (v - vg) - A(w, u) # Eq. 4.8

        # v velocity (centers)
        flux_bottom = get_boundary_flux(model, bc_v.bottom, v, Y, model.parameters)
        # TODO! move into custom boundary condition
        center_space = axes(Yc)
        dz_top = center_space.face_local_geometry.WJ[end]
        v_top = parent(v)[end]
        flux_top = ν * (v_top - vg) / dz_top / 2
        ∇c = Operators.GradientC2F() # Eq. 4.18
        ∇f = Operators.GradientF2C(bottom = flux_bottom, top = Operators.SetValue(flux_top)) # Eq. 4.16
        
        A = Operators.AdvectionC2C(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
        @. dv = ∇f(ν * ∇c(v)) - f * (u - ug) - A(w, v) # Eq. 4.9

        # w velocity (faces)
        flux_bottom = get_boundary_flux(model, bc_w.bottom, w, Y, model.parameters)
        flux_top    = get_boundary_flux(model, bc_w.top, w, Y, model.parameters)
        ∇c = Operators.GradientC2F()
        ∇f = Operators.GradientF2C(bottom = flux_bottom, top = flux_top)

        If = Operators.InterpolateC2F(bottom = Operators.Extrapolate(), top = Operators.Extrapolate())
        B = Operators.SetBoundaryOperator(bottom = Operators.SetValue(0.0), top = Operators.SetValue(0.0))
        Π(ρθ) = C_p .* (R_d .* ρθ ./ MSLP).^(R_m ./ C_v)
        @. dw = B( -(If(ρθ / ρ) * ∇c(Π(ρθ))) - grav + ∇c(ν * ∇f(w)) - w * If(∇f(w))) # Eq. 4.10

        return dY
    end

    return rhs!
end

"""
    get_boundary_flux(::SingleColumnModel, ::NoFluxCondition, _...)
"""
@inline function get_boundary_flux(::SingleColumnModel, ::NoFluxCondition, _...)
    return Operators.SetValue(0.0)
end

"""
    get_boundary_flux(::SingleColumnModel, bc::CustomFluxCondition, args...)
"""
@inline function get_boundary_flux(::SingleColumnModel, bc::CustomFluxCondition, args...)
    return Operators.SetValue(bc.compute_flux(args...))
end

"""
    get_boundary_flux(model::SingleColumnModel, bc::DragLawCondition, field, Y, parameters)
"""
@inline function get_boundary_flux(model::SingleColumnModel, bc::DragLawCondition, field, Y, parameters)
    @unpack Cd = model.parameters
    Yc = Y.x[1]
    @unpack u, v = Yc

    u_bottom, v_bottom = parent(u)[1], parent(v)[1]
    wind_speed = sqrt(u_bottom^2 + v_bottom^2)
    field_bottom = parent(field)[1]

    return Operators.SetValue(Cd * wind_speed * field_bottom)
end