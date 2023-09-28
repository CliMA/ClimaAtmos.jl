function CTS.step_u!(
    integrator,
    cache::CTS.IMEXSSPRKCache,
    f::AtmosODEFunction,
    name,
)
    (; u, p, t, dt, alg) = integrator
    (; T_lim!, T_exp!, T_imp!, lim!, dss!) = f
    (; post_explicit!, post_implicit!) = f
    (; tableau, newtons_method) = alg
    (; a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, U_lim, U_exp, T_lim, T_exp, T_imp, temp, β, γ, newtons_method_cache) =
        cache
    s = length(b_imp)

    if !isnothing(newtons_method)
        (; update_j) = newtons_method
        jacobian = newtons_method_cache.j
        if (!isnothing(jacobian)) &&
           CTS.needs_update!(update_j, CTS.NewTimeStep(t))
            if γ isa Nothing
                sdirk_error(name)
            else
                T_imp!.Wfact(jacobian, u, p, dt * γ, t)
            end
        end
    end

    @. U = u

    for i in 1:s
        t_exp = t + dt * c_exp[i]
        t_imp = t + dt * c_imp[i]

        if i == 1
            @. U_exp = u
        elseif !iszero(β[i - 1])
            @. U_lim = U_exp + dt * T_lim
            lim!(U_lim, p, t_exp, U_exp)
            @. U_exp = U_lim
            @. U_exp += dt * T_exp
            @. U_exp = (1 - β[i - 1]) * u + β[i - 1] * U_exp
        end

        dss!(U_exp, p, t_exp)
        # i ≠ 1 && post_explicit!(U_exp, p, t_exp) # TODO: is this needed?

        @. U = U_exp
        for j in 1:(i - 1)
            iszero(a_imp[i, j]) && continue
            @. U += dt * a_imp[i, j] * T_imp[j]
        end
        i ≠ 1 && post_explicit!(U, p, t_exp) # TODO: is this the correct placement? is t_exp correct here?

        if !iszero(a_imp[i, i]) # Implicit solve
            @assert !isnothing(newtons_method)
            @. temp = U
            # TODO: can/should we remove these closures?
            implicit_equation_residual! =
                (residual, Ui) -> begin
                    T_imp!(residual, Ui, p, t_imp)
                    @. residual = temp + dt * a_imp[i, i] * residual - Ui
                end
            implicit_equation_jacobian! =
                (jacobian, Ui) ->
                    T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
            call_post_implicit! = Ui -> begin
                post_implicit!(Ui, p, t_imp)
            end
            CTS.solve_newton!(
                newtons_method,
                newtons_method_cache,
                U,
                implicit_equation_residual!,
                implicit_equation_jacobian!,
                call_post_implicit!,
            )
        end

        # We do not need to DSS U again because the implicit solve should
        # give the same results for redundant columns (as long as the implicit
        # tendency only acts in the vertical direction).

        if !all(iszero, a_imp[:, i]) || !iszero(b_imp[i])
            if iszero(a_imp[i, i])
                # If its coefficient is 0, T_imp[i] is effectively being
                # treated explicitly.
                T_imp!(T_imp[i], U, p, t_imp)
            else
                # If T_imp[i] is being treated implicitly, ensure that it
                # exactly satisfies the implicit equation.
                @. T_imp[i] = (U - temp) / (dt * a_imp[i, i])
            end
        end

        if !iszero(β[i])
            T_lim!(T_lim, U, p, t_exp)
            T_exp!(T_exp, U, p, t_exp)
        end
    end

    t_final = t + dt

    if !iszero(β[s])
        @. U_lim = U_exp + dt * T_lim
        lim!(U_lim, p, t_final, U_exp)
        @. U_exp = U_lim
        @. U_exp += dt * T_exp
        @. u = (1 - β[s]) * u + β[s] * U_exp
    end

    for j in 1:s
        iszero(b_imp[j]) && continue
        @. u += dt * b_imp[j] * T_imp[j]
    end

    dss!(u, p, t_final)
    post_explicit!(u, p, t_final)

    return u
end
