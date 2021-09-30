"""
    struct Nonhydrostatic2DModel <: AbstractModel

Construct a two-dimensional non-hydrostatic model on `domain`
with `parameters` and `boundary_conditions` that contain field boundary conditions.
Typically, this is used for simulating the Euler equations.
"""
Base.@kwdef struct Nonhydrostatic2DModel{FT, BCT, PT} <: AbstractModel
    domain::AbstractHybridDomain{FT}
    boundary_conditions::BCT
    parameters::PT
    name::Symbol = :nhm
    varnames::Tuple = (:ρ, :ρuh, :ρw, :ρθ) # ρuh is the horizontal momentum
end

function Models.default_initial_conditions(
    model::Nonhydrostatic2DModel{FT},
) where {FT}
    space_c, space_f = make_function_space(model.domain)
    local_geometry_c = Fields.local_geometry_field(space_c)
    local_geometry_f = Fields.local_geometry_field(space_f)

    # functions that make zeros for this model
    zero_scalar(lg) = zero(FT) # .
    zero_1vector(lg) = Geometry.Cartesian1Vector(zero(FT)) # ---->
    zero_3vector(lg) = Geometry.Cartesian3Vector(zero(FT)) # (-_-') . ┓( ´∀` )┏ 

    ρ = zero_scalar.(local_geometry_c)
    ρuh = zero_1vector.(local_geometry_c)
    ρw = zero_3vector.(local_geometry_f) # faces
    ρθ = zero_scalar.(local_geometry_c)

    return Fields.FieldVector(
        nhm = Fields.FieldVector(ρ = ρ, ρuh = ρuh, ρw = ρw, ρθ = ρθ),
    )
end

function Models.make_ode_function(model::Nonhydrostatic2DModel{FT}) where {FT}
    @unpack p_0, R_d, cp_d, cv_d, g = model.parameters
    γ = cp_d / cv_d

    # TODO!: Replace with Thermodynamics.jl
    pressure(ρθ) = ρθ >= 0 ? p_0 * (R_d * ρθ / p_0)^γ : NaN

    # unity tensor for pressure term calculation 
    # in horizontal spectral divergence
    I = Ref(Geometry.Axis2Tensor(
        (Geometry.Cartesian1Axis(), Geometry.Cartesian1Axis()),
        @SMatrix [1.0]
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
        bottom = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian1Vector(0.0)),
    )
    tensor_interp_f2c = Operators.InterpolateF2C()

    # gradients
    scalar_grad_c2f = Operators.GradientC2F()
    B = Operators.SetBoundaryOperator(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )

    # divergences
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetValue(Geometry.Cartesian3Vector(0.0)),
    )
    tensor_vdiv_c2f = Operators.DivergenceC2F(
        bottom = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
        top = Operators.SetDivergence(Geometry.Cartesian3Vector(0.0)),
    )
    tensor_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(
            Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
        ),
        top = Operators.SetValue(
            Geometry.Cartesian3Vector(0.0) ⊗ Geometry.Cartesian1Vector(0.0),
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
                Geometry.Cartesian3Axis(),
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

function get_velocities(u, model::Nonhydrostatic2DModel)
    u_1 = getproperty(u.nhm,:ρuh) ./ getproperty(u.nhm, :ρ)
    u_2 = getproperty(u.nhm,:ρw) ./ getproperty(u.nhm, :ρ)
    # Interpolation required to get matched spaces between vertical momentum and density
    return (u_1, u_2)
end
