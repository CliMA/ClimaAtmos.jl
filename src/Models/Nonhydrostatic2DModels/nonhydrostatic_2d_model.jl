"""
    Nonhydrostatic2DModel <: AbstractModel

A two-dimensional non-hydrostatic model, which is typically used for simulating
the Euler equations. Required fields are `domain`, `boundary_conditions`, and
`parameters`.
"""
Base.@kwdef struct Nonhydrostatic2DModel{D <: AbstractHybridDomain, BC, P} <:
                   AbstractModel
    domain::D
    boundary_conditions::BC
    parameters::P
    name::Symbol = :nhm
    varnames::Tuple = (:ρ, :ρuh, :ρw, :ρθ) # ρuh is the horizontal momentum
end

function Models.default_initial_conditions(model::Nonhydrostatic2DModel)
    space_c, space_f = make_function_space(model.domain)
    local_geometry_c = Fields.local_geometry_field(space_c)
    local_geometry_f = Fields.local_geometry_field(space_f)

    # functions that make zeros for this model
    zero_val = zero(Spaces.undertype(space_c))
    zero_scalar(lg) = zero_val # .
    zero_1vector(lg) = Geometry.UVector(zero_val) # ---->
    zero_3vector(lg) = Geometry.WVector(zero_val) # (-_-') . ┓( ´∀` )┏ 

    ρ = zero_scalar.(local_geometry_c)
    ρuh = zero_1vector.(local_geometry_c)
    ρw = zero_3vector.(local_geometry_f) # faces
    ρθ = zero_scalar.(local_geometry_c)

    return Fields.FieldVector(
        nhm = Fields.FieldVector(ρ = ρ, ρuh = ρuh, ρw = ρw, ρθ = ρθ),
    )
end

function Models.make_ode_function(model::Nonhydrostatic2DModel)
    FT = eltype(model.parameters)
    @unpack p_0, R_d, cp_d, cv_d, g = model.parameters
    γ = cp_d / cv_d

    # TODO!: Replace with Thermodynamics.jl
    pressure(ρθ::FT) where {FT} =
        ρθ >= FT(0) ? p_0 * (R_d * ρθ / p_0)^γ : FT(NaN)

    # unity tensor for pressure term calculation 
    # in horizontal spectral divergence
    I = Ref(Geometry.Axis2Tensor(
        (Geometry.UAxis(), Geometry.UAxis()),
        @SMatrix [FT(1)]
    ),)

    # operators
    # spectral horizontal operators
    hdiv = Operators.Divergence()

    # vertical FD operators with BC's
    # interpolators
    scalar_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    vector_interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.SetValue(Geometry.UVector(FT(0))),
        top = Operators.SetValue(Geometry.UVector(FT(0))),
    )
    tensor_interp_f2c = Operators.InterpolateF2C()

    # gradients
    scalar_grad_c2f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )

    # divergences
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.WVector(FT(0))),
        top = Operators.SetValue(Geometry.WVector(FT(0))),
    )
    tensor_vdiv_c2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.WVector(FT(0))),
        top = Operators.SetDivergence(Geometry.WVector(FT(0))),
    )
    tensor_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.WVector(FT(0)) ⊗ Geometry.UVector(FT(0)),
        ),
        top = Operators.SetValue(
            Geometry.WVector(FT(0)) ⊗ Geometry.UVector(FT(0)),
        ),
    )

    function rhs!(dY, Y, Ya, t)
        dYm = dY.nhm
        dρ = dYm.ρ
        dρuh = dYm.ρuh
        dρw = dYm.ρw
        dρθ = dYm.ρθ
        Ym = Y.nhm
        ρ = Ym.ρ
        ρuh = Ym.ρuh
        ρw = Ym.ρw
        ρθ = Ym.ρθ

        # advection, pressure gradients, sources
        # density
        @. dρ = -hdiv(ρuh)
        @. dρ -= vector_vdiv_f2c(ρw)

        # potential temperature density
        @. dρθ = -hdiv(ρuh * ρθ / ρ)
        @. dρθ -= vector_vdiv_f2c(ρw * scalar_interp_c2f(ρθ / ρ))

        # horizontal momentum
        @. dρuh = -hdiv(ρuh ⊗ ρuh / ρ + pressure(ρθ) * I)
        @. dρuh -= tensor_vdiv_f2c(ρw ⊗ vector_interp_c2f(ρuh / ρ))

        # vertical momentum
        # TODO: How ugly is this?! (-_-)' . ￣□￣｜｜  (ಥ﹏ಥ)
        uh_f = @. vector_interp_c2f(ρuh / ρ)
        @. dρw = -hdiv(uh_f ⊗ ρw)
        @. dρw += B(
            Geometry.transform(
                Geometry.WAxis(),
                -(scalar_grad_c2f(pressure(ρθ))) +
                scalar_interp_c2f(ρ) * Geometry.Covariant3Vector(-g), # TODO!: Not generally a Covariant3Vector
            ) - tensor_vdiv_c2f(tensor_interp_f2c(
                ρw ⊗ ρw / scalar_interp_c2f(ρ),
            )),
        )

        # TODO! diffusion goes here!

        # discrete stiffness summation for spectral operations
        Spaces.weighted_dss!(dρ)
        Spaces.weighted_dss!(dρθ)
        Spaces.weighted_dss!(dρuh)
        Spaces.weighted_dss!(dρw)

        return dY
    end
end
