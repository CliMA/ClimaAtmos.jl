const C123 = Geometry.Covariant123Vector
const compose = Operators.ComposeStencils()
const apply = Operators.ApplyStencil()

# Allow one() to be called on vectors.
Base.one(::T) where {T <: Geometry.AxisTensor} = one(T)
Base.one(::Type{T}) where {T′, A, S, T <: Geometry.AxisTensor{T′, 1, A, S}} =
    T(axes(T), S(one(T′)))

function jacobian!(W, Y, p, dtγ, t, FT)
    interp_c2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    interp_f2c = Operators.InterpolateF2C()
    vector_vdiv_f2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Covariant3Vector(FT(0))),
        top = Operators.SetValue(Geometry.Covariant3Vector(FT(0))),
    )
    scalar_vgrad_c2f = Operators.GradientC2F(
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
        top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    )

    interp_f2c_stencil = Operators.Operator2Stencil(interp_f2c)
    interp_c2f_stencil = Operators.Operator2Stencil(interp_c2f)
    vector_vdiv_f2c_stencil = Operators.Operator2Stencil(vector_vdiv_f2c)
    scalar_vgrad_c2f_stencil = Operators.Operator2Stencil(scalar_vgrad_c2f)

    (; params, flags, dtγ_ref, ∂dρ∂M, ∂dE∂M, ∂dM∂E, ∂dM∂ρ, ∂dM∂M) = W
    ρ = Y.base.ρ
    uh = Y.base.uh
    w = Y.base.w

    p_0::FT = CLIMAParameters.Planet.MSLP(params)
    R_d::FT = CLIMAParameters.Planet.R_d(params)
    T_tri::FT = CLIMAParameters.Planet.T_triple(params)
    grav::FT = CLIMAParameters.Planet.grav(params)
    cp_d::FT = CLIMAParameters.Planet.cp_d(params)
    cv_d::FT = CLIMAParameters.Planet.cv_d(params)
    γ = cp_d / cv_d

    dtγ_ref[] = dtγ

    # If we let w_data = w.components.data.:1 and w_unit = one.(w), then
    # w == w_data .* w_unit. The Jacobian blocks involve w_data, not w.
    w_data = w.components.data.:1

    Φ = Fields.coordinate_field(ρ).z .* grav

    # interp_f2c(w) =
    #     interp_f2c(w)_data * interp_f2c(w)_unit =
    #     interp_f2c(w_data) * interp_f2c(w)_unit
    # norm_sqr(interp_f2c(w)) =
    #     norm_sqr(interp_f2c(w_data) * interp_f2c(w)_unit) =
    #     interp_f2c(w_data)^2 * norm_sqr(interp_f2c(w)_unit)
    # K =
    #     norm_sqr(C123(uh) + C123(interp_f2c(w))) / 2 =
    #     norm_sqr(uh) / 2 + norm_sqr(interp_f2c(w)) / 2 =
    #     norm_sqr(uh) / 2 + interp_f2c(w_data)^2 * norm_sqr(interp_f2c(w)_unit) / 2
    # ∂(K)/∂(w_data) =
    #     ∂(K)/∂(interp_f2c(w_data)) * ∂(interp_f2c(w_data))/∂(w_data) =
    #     interp_f2c(w_data) * norm_sqr(interp_f2c(w)_unit) * interp_f2c_stencil(1)
    ∂K∂w_data = @. (
        interp_f2c(w_data) *
        norm_sqr(one(interp_f2c(w))) *
        interp_f2c_stencil(one(w_data))
    )

    # dρ = -vector_vdiv_f2c(interp_c2f(ρ) * w)
    # ∂(dρ)/∂(w_data) = -vector_vdiv_f2c_stencil(interp_c2f(ρ) * w_unit)
    @. ∂dρ∂M = -vector_vdiv_f2c_stencil(interp_c2f(ρ) * one(w))

    if :ρθ in propertynames(Y.thermodynamics)
        ρθ = Y.thermodynamics.ρθ
        p = @. p_0 * (ρθ * R_d / p_0)^γ

        if flags.∂dE∂M_mode != :exact
            error("∂dE∂M_mode must be :exact when using ρθ")
        end

        # ρθₜ = -vector_vdiv_f2c(interp_c2f(ρθ) * w)
        # ∂(ρθₜ)/∂(w_data) = -vector_vdiv_f2c_stencil(interp_c2f(ρθ) * w_unit)
        @. ∂dE∂M = -vector_vdiv_f2c_stencil(interp_c2f(ρθ) * one(w))
    elseif :ρe_tot in propertynames(Y.thermodynamics)
        ρe_tot = Y.thermodynamics.ρe_tot
        K = @. norm_sqr(C123(uh) + C123(interp_f2c(w))) / 2
        p = @. ρ * R_d * ((ρe_tot / ρ - K - Φ) / cv_d + T_tri)

        if flags.∂dE∂M_mode == :exact
            # ρeₜ = -vector_vdiv_f2c(interp_c2f(ρe_tot + p) * w)
            # ∂(ρeₜ)/∂(w_data) =
            #     -vector_vdiv_f2c_stencil(interp_c2f(ρe_tot + p) * w_unit) -
            #     vector_vdiv_f2c_stencil(w) * ∂(interp_c2f(ρe_tot + p))/∂(w_data)
            # ∂(interp_c2f(ρe_tot + p))/∂(w_data) =
            #     ∂(interp_c2f(ρe_tot + p))/∂(p) * ∂(p)/∂(w_data)
            # ∂(interp_c2f(ρe_tot + p))/∂(p) = interp_c2f_stencil(1)
            # ∂(p)/∂(w_data) = ∂(p)/∂(K) * ∂(K)/∂(w_data)
            # ∂(p)/∂(K) = -ρ * R_d / cv_d
            @. ∂dE∂M =
                -vector_vdiv_f2c_stencil(interp_c2f(ρe_tot + p) * one(w)) -
                compose(
                    vector_vdiv_f2c_stencil(w),
                    compose(
                        interp_c2f_stencil(one(p)),
                        -ρ * R_d / cv_d * ∂K∂w_data,
                    ),
                )
        elseif flags.∂dE∂M_mode == :no_∂p∂K
            # same as above, but we approximate ∂(p)/∂(K) = 0, so that ∂dE∂M
            # has 3 diagonals instead of 5
            @. ∂dE∂M = -vector_vdiv_f2c_stencil(interp_c2f(ρe_tot + p) * one(w))
        else
            error("∂dE∂M_mode must be :exact or :no_∂p∂K when using ρe_tot")
        end
    elseif :ρe_int in propertynames(Y.thermodynamics)
        ρe_int = Y.thermodynamics.ρe_int
        p = @. R_d * (ρe_int / cv_d + ρ * T_tri)

        if flags.∂dE∂M_mode != :exact
            error("∂dE∂M_mode must be :exact when using ρe_int")
        end

        # ρe_intₜ =
        #     -vector_vdiv_f2c(interp_c2f(ρe_int + p) * w) +
        #     interp_f2c(dot(scalar_vgrad_c2f(p), Geometry.Contravariant3Vector(w)))
        # ∂(ρe_intₜ)/∂(w_data) =
        #     -vector_vdiv_f2c_stencil(interp_c2f(ρe_int + p) * w_unit) + interp_f2c_stencil(
        #         dot(scalar_vgrad_c2f(p), Geometry.Contravariant3Vector(w_unit)),
        #     )
        @. ∂dE∂M =
            -vector_vdiv_f2c_stencil(interp_c2f(ρe_int + p) * one(w)) +
            interp_f2c_stencil(dot(
                scalar_vgrad_c2f(p),
                Geometry.Contravariant3Vector(one(w)),
            ),)
    end

    # To convert ∂(wₜ)/∂(E) to ∂(w_data)ₜ/∂(E) and ∂(wₜ)/∂(w_data) to
    # ∂(w_data)ₜ/∂(w_data), we must extract the third component of each
    # vector-valued stencil coefficient.
    to_scalar_coefs(vector_coefs) =
        map(vector_coef -> vector_coef.u₃, vector_coefs)

    # TODO: If we end up using :gradΦ_shenanigans, optimize it to
    # `cached_stencil / interp_c2f(ρ)`.
    if flags.∂dM∂ρ_mode != :exact && flags.∂dM∂ρ_mode != :gradΦ_shenanigans
        error("∂dM∂ρ_mode must be :exact or :gradΦ_shenanigans")
    end
    if :ρθ in propertynames(Y.thermodynamics)
        # wₜ = -scalar_vgrad_c2f(p) / interp_c2f(ρ) - scalar_vgrad_c2f(K + Φ)
        # ∂(wₜ)/∂(ρθ) = ∂(wₜ)/∂(scalar_vgrad_c2f(p)) * ∂(scalar_vgrad_c2f(p))/∂(ρθ)
        # ∂(wₜ)/∂(scalar_vgrad_c2f(p)) = -1 / interp_c2f(ρ)
        # ∂(scalar_vgrad_c2f(p))/∂(ρθ) =
        #     scalar_vgrad_c2f_stencil(γ * R_d * (ρθ * R_d / p_0)^(γ - 1))
        @. ∂dM∂E = to_scalar_coefs(
            -1 / interp_c2f(ρ) *
            scalar_vgrad_c2f_stencil(γ * R_d * (ρθ * R_d / p_0)^(γ - 1)),
        )

        if flags.∂dM∂ρ_mode == :exact
            # wₜ = -scalar_vgrad_c2f(p) / interp_c2f(ρ) - scalar_vgrad_c2f(K + Φ)
            # ∂(wₜ)/∂(ρ) = ∂(wₜ)/∂(interp_c2f(ρ)) * ∂(interp_c2f(ρ))/∂(ρ)
            # ∂(wₜ)/∂(interp_c2f(ρ)) = scalar_vgrad_c2f(p) / interp_c2f(ρ)^2
            # ∂(interp_c2f(ρ))/∂(ρ) = interp_c2f_stencil(1)
            @. ∂dM∂ρ = to_scalar_coefs(
                scalar_vgrad_c2f(p) / interp_c2f(ρ)^2 *
                interp_c2f_stencil(one(ρ)),
            )
        elseif flags.∂dM∂ρ_mode == :gradΦ_shenanigans
            # wₜ = (
            #     -scalar_vgrad_c2f(p) / interp_c2f(ρ′) -
            #     scalar_vgrad_c2f(Φ) / interp_c2f(ρ′) * interp_c2f(ρ)
            # ), where ρ′ = ρ but we approximate ∂(ρ′)/∂(ρ) = 0
            @. ∂dM∂ρ = to_scalar_coefs(
                -scalar_vgrad_c2f(Φ) / interp_c2f(ρ) *
                interp_c2f_stencil(one(ρ)),
            )
        end
    elseif :ρe_tot in propertynames(Y.thermodynamics)
        # wₜ = -scalar_vgrad_c2f(p) / interp_c2f(ρ) - scalar_vgrad_c2f(K + Φ)
        # ∂(wₜ)/∂(ρe_tot) = ∂(wₜ)/∂(scalar_vgrad_c2f(p)) * ∂(scalar_vgrad_c2f(p))/∂(ρe_tot)
        # ∂(wₜ)/∂(scalar_vgrad_c2f(p)) = -1 / interp_c2f(ρ)
        # ∂(scalar_vgrad_c2f(p))/∂(ρe_tot) = scalar_vgrad_c2f_stencil(R_d / cv_d)
        @. ∂dM∂E = to_scalar_coefs(
            -1 / interp_c2f(ρ) *
            scalar_vgrad_c2f_stencil(R_d / cv_d * one(ρe_tot)),
        )

        if flags.∂dM∂ρ_mode == :exact
            # wₜ = -scalar_vgrad_c2f(p) / interp_c2f(ρ) - scalar_vgrad_c2f(K + Φ)
            # ∂(wₜ)/∂(ρ) =
            #     ∂(wₜ)/∂(scalar_vgrad_c2f(p)) * ∂(scalar_vgrad_c2f(p))/∂(ρ) +
            #     ∂(wₜ)/∂(interp_c2f(ρ)) * ∂(interp_c2f(ρ))/∂(ρ)
            # ∂(wₜ)/∂(scalar_vgrad_c2f(p)) = -1 / interp_c2f(ρ)
            # ∂(scalar_vgrad_c2f(p))/∂(ρ) =
            #     scalar_vgrad_c2f_stencil(R_d * (-(K + Φ) / cv_d + T_tri))
            # ∂(wₜ)/∂(interp_c2f(ρ)) = scalar_vgrad_c2f(p) / interp_c2f(ρ)^2
            # ∂(interp_c2f(ρ))/∂(ρ) = interp_c2f_stencil(1)
            @. ∂dM∂ρ = to_scalar_coefs(
                -1 / interp_c2f(ρ) *
                scalar_vgrad_c2f_stencil(R_d * (-(K + Φ) / cv_d + T_tri)) +
                scalar_vgrad_c2f(p) / interp_c2f(ρ)^2 *
                interp_c2f_stencil(one(ρ)),
            )
        elseif flags.∂dM∂ρ_mode == :gradΦ_shenanigans
            # wₜ = (
            #     -scalar_vgrad_c2f(p′) / interp_c2f(ρ′) -
            #     scalar_vgrad_c2f(Φ) / interp_c2f(ρ′) * interp_c2f(ρ)
            # ), where ρ′ = ρ but we approximate ∂ρ′/∂ρ = 0, and where
            # p′ = p but with K = 0
            @. ∂dM∂ρ = to_scalar_coefs(
                -1 / interp_c2f(ρ) *
                scalar_vgrad_c2f_stencil(R_d * (-Φ / cv_d + T_tri)) -
                scalar_vgrad_c2f(Φ) / interp_c2f(ρ) *
                interp_c2f_stencil(one(ρ)),
            )
        end
    elseif :ρe_int in propertynames(Y.thermodynamics)
        # wₜ = -scalar_vgrad_c2f(p) / interp_c2f(ρ) - scalar_vgrad_c2f(K + Φ)
        # ∂(wₜ)/∂(ρe_int) = ∂(wₜ)/∂(scalar_vgrad_c2f(p)) * ∂(scalar_vgrad_c2f(p))/∂(ρe_int)
        # ∂(wₜ)/∂(scalar_vgrad_c2f(p)) = -1 / interp_c2f(ρ)
        # ∂(scalar_vgrad_c2f(p))/∂(ρe_int) = scalar_vgrad_c2f_stencil(R_d / cv_d)
        @. ∂dM∂E = to_scalar_coefs(
            -1 / interp_c2f(ρ) *
            scalar_vgrad_c2f_stencil(R_d / cv_d * one(ρe_int)),
        )

        if flags.∂dM∂ρ_mode == :exact
            # wₜ = -scalar_vgrad_c2f(p) / interp_c2f(ρ) - scalar_vgrad_c2f(K + Φ)
            # ∂(wₜ)/∂(ρ) =
            #     ∂(wₜ)/∂(scalar_vgrad_c2f(p)) * ∂(scalar_vgrad_c2f(p))/∂(ρ) +
            #     ∂(wₜ)/∂(interp_c2f(ρ)) * ∂(interp_c2f(ρ))/∂(ρ)
            # ∂(wₜ)/∂(scalar_vgrad_c2f(p)) = -1 / interp_c2f(ρ)
            # ∂(scalar_vgrad_c2f(p))/∂(ρ) = scalar_vgrad_c2f_stencil(R_d * T_tri)
            # ∂(wₜ)/∂(interp_c2f(ρ)) = scalar_vgrad_c2f(p) / interp_c2f(ρ)^2
            # ∂(interp_c2f(ρ))/∂(ρ) = interp_c2f_stencil(1)
            @. ∂dM∂ρ = to_scalar_coefs(
                -1 / interp_c2f(ρ) *
                scalar_vgrad_c2f_stencil(R_d * T_tri * one(ρe_int)) +
                scalar_vgrad_c2f(p) / interp_c2f(ρ)^2 *
                interp_c2f_stencil(one(ρ)),
            )
        elseif flags.∂dM∂ρ_mode == :gradΦ_shenanigans
            # wₜ = (
            #     -scalar_vgrad_c2f(p) / interp_c2f(ρ′) -
            #     scalar_vgrad_c2f(Φ) / interp_c2f(ρ′) * interp_c2f(ρ)
            # ), where p′ = p but we approximate ∂ρ′/∂ρ = 0
            @. ∂dM∂ρ = to_scalar_coefs(
                -1 / interp_c2f(ρ) *
                scalar_vgrad_c2f_stencil(R_d * T_tri * one(ρe_int)) -
                scalar_vgrad_c2f(Φ) / interp_c2f(ρ) *
                interp_c2f_stencil(one(ρ)),
            )
        end
    end

    # wₜ = -scalar_vgrad_c2f(p) / interp_c2f(ρ) - scalar_vgrad_c2f(K + Φ)
    # ∂(wₜ)/∂(w_data) =
    #     ∂(wₜ)/∂(scalar_vgrad_c2f(p)) * ∂(scalar_vgrad_c2f(p))/∂(w_dataₜ) +
    #     ∂(wₜ)/∂(scalar_vgrad_c2f(K + Φ)) * ∂(scalar_vgrad_c2f(K + Φ))/∂(w_dataₜ) =
    #     (
    #         ∂(wₜ)/∂(scalar_vgrad_c2f(p)) * ∂(scalar_vgrad_c2f(p))/∂(K) +
    #         ∂(wₜ)/∂(scalar_vgrad_c2f(K + Φ)) * ∂(scalar_vgrad_c2f(K + Φ))/∂(K)
    #     ) * ∂(K)/∂(w_dataₜ)
    # ∂(wₜ)/∂(scalar_vgrad_c2f(p)) = -1 / interp_c2f(ρ)
    # ∂(scalar_vgrad_c2f(p))/∂(K) =
    #     E_name == :ρe_tot ? scalar_vgrad_c2f_stencil(-ρ * R_d / cv_d) : 0
    # ∂(wₜ)/∂(scalar_vgrad_c2f(K + Φ)) = -1
    # ∂(scalar_vgrad_c2f(K + Φ))/∂(K) = scalar_vgrad_c2f_stencil(1)
    # ∂(K)/∂(w_data) =
    #     interp_f2c(w_data) * norm_sqr(interp_f2c(w)_unit) * interp_f2c_stencil(1)
    if :ρθ in propertynames(Y.thermodynamics) ||
       :ρe_int in propertynames(Y.thermodynamics)
        @. ∂dM∂M = to_scalar_coefs(compose(
            -1 * scalar_vgrad_c2f_stencil(one(K)),
            ∂K∂w_data,
        ),)
    elseif :ρe_tot in propertynames(Y.thermodynamics)
        @. ∂dM∂M = to_scalar_coefs(compose(
            -1 / interp_c2f(ρ) * scalar_vgrad_c2f_stencil(-ρ * R_d / cv_d) +
            -1 * scalar_vgrad_c2f_stencil(one(K)),
            ∂K∂w_data,
        ),)
    end
end
