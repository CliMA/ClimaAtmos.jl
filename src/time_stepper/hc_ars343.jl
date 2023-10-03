function CTS.step_u!(
    integrator,
    cache::CTS.IMEXARKCache,
    f::AtmosODEFunction,
    ::CTS.ARS343,
)
    (; u, p, t, dt, sol, alg) = integrator
    (; f) = sol.prob
    (; T_imp!, T_lim!, T_exp!, lim!, dss!) = f
    (; post_explicit!, post_implicit!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache) = cache

    if !isnothing(newtons_method_cache)
        jacobian = newtons_method_cache.j
        if (!isnothing(jacobian)) &&
           CTS.needs_update!(newtons_method.update_j, CTS.NewTimeStep(t))
            post_implicit!(u, p, t)
            T_imp!.Wfact(jacobian, u, p, dt * γ, t)
        end
    end

    s = 4

    i::Int = 1
    t_exp = t
    @. U = u
    lim!(U, p, t_exp, u)
    dss!(U, p, t_exp)
    post_explicit!(U, p, t_exp)
    T_lim!(T_lim[i], U, p, t_exp)
    T_exp!(T_exp[i], U, p, t_exp)

    i = 2
    t_exp = t + dt * c_exp[i]
    @. U = u
    @. U += dt * a_exp[i, 1] * T_lim[1]
    lim!(U, p, t_exp, u)
    @. U += dt * a_exp[i, 1] * T_exp[1]
    dss!(U, p, t_exp)

    @. temp = U # used in closures

    let i = i
        t_imp = t + dt * c_imp[i]
        post_explicit!(U, p, t_imp)
        implicit_equation_residual! =
            (residual, Ui) -> begin
                T_imp!(residual, Ui, p, t_imp)
                @. residual = temp + dt * a_imp[i, i] * residual - Ui
            end
        implicit_equation_jacobian! =
            (jacobian, Ui) -> begin
                T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
            end
        call_post_implicit! =
            Ui -> begin
                @. T_imp[i] = (Ui - temp) / (dt * a_imp[i, i])
                post_implicit!(Ui, p, t_imp)
            end
        _solve_newton!(
            newtons_method,
            newtons_method_cache,
            U,
            implicit_equation_residual!,
            call_post_implicit!,
            implicit_equation_jacobian!,
        )
    end

    T_lim!(T_lim[i], U, p, t_exp)
    T_exp!(T_exp[i], U, p, t_exp)

    i = 3
    t_exp = t + dt * c_exp[i]
    @. U = u
    @. U += dt * a_exp[i, 1] * T_lim[1]
    @. U += dt * a_exp[i, 2] * T_lim[2]
    lim!(U, p, t_exp, u)
    @. U += dt * a_exp[i, 1] * T_exp[1]
    @. U += dt * a_exp[i, 2] * T_exp[2]
    @. U += dt * a_imp[i, 2] * T_imp[2]
    dss!(U, p, t_exp)

    @. temp = U # used in closures

    let i = i
        t_imp = t + dt * c_imp[i]
        post_explicit!(U, p, t_imp)
        implicit_equation_residual! =
            (residual, Ui) -> begin
                T_imp!(residual, Ui, p, t_imp)
                @. residual = temp + dt * a_imp[i, i] * residual - Ui
            end
        implicit_equation_jacobian! =
            (jacobian, Ui) -> begin
                T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
            end
        call_post_implicit! =
            Ui -> begin
                @. T_imp[i] = (Ui - temp) / (dt * a_imp[i, i])
                post_implicit!(Ui, p, t_imp)
            end
        _solve_newton!(
            newtons_method,
            newtons_method_cache,
            U,
            implicit_equation_residual!,
            call_post_implicit!,
            implicit_equation_jacobian!,
        )
    end

    T_lim!(T_lim[i], U, p, t_exp)
    T_exp!(T_exp[i], U, p, t_exp)
    i = 4
    t_exp = t + dt
    @. U = u
    @. U += dt * a_exp[i, 1] * T_lim[1]
    @. U += dt * a_exp[i, 2] * T_lim[2]
    @. U += dt * a_exp[i, 3] * T_lim[3]
    lim!(U, p, t_exp, u)
    @. U += dt * a_exp[i, 1] * T_exp[1]
    @. U += dt * a_exp[i, 2] * T_exp[2]
    @. U += dt * a_exp[i, 3] * T_exp[3]
    @. U += dt * a_imp[i, 2] * T_imp[2]
    @. U += dt * a_imp[i, 3] * T_imp[3]
    dss!(U, p, t_exp)

    @. temp = U # used in closures

    let i = i
        t_imp = t + dt * c_imp[i]
        post_explicit!(U, p, t_imp)
        implicit_equation_residual! =
            (residual, Ui) -> begin
                T_imp!(residual, Ui, p, t_imp)
                @. residual = temp + dt * a_imp[i, i] * residual - Ui
            end
        implicit_equation_jacobian! =
            (jacobian, Ui) -> begin
                T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
            end
        call_post_implicit! =
            Ui -> begin
                @. T_imp[i] = (Ui - temp) / (dt * a_imp[i, i])
                post_implicit!(Ui, p, t_imp)
            end
        _solve_newton!(
            newtons_method,
            newtons_method_cache,
            U,
            implicit_equation_residual!,
            call_post_implicit!,
            implicit_equation_jacobian!,
        )
    end

    T_lim!(T_lim[i], U, p, t_exp)
    T_exp!(T_exp[i], U, p, t_exp)

    # final
    i = -1

    t_final = t + dt
    @. temp = u
    @. temp += dt * b_exp[2] * T_lim[2]
    @. temp += dt * b_exp[3] * T_lim[3]
    @. temp += dt * b_exp[4] * T_lim[4]
    lim!(temp, p, t_final, u)
    @. u = temp
    @. u += dt * b_exp[2] * T_exp[2]
    @. u += dt * b_exp[3] * T_exp[3]
    @. u += dt * b_exp[4] * T_exp[4]
    @. u += dt * b_imp[2] * T_imp[2]
    @. u += dt * b_imp[3] * T_imp[3]
    @. u += dt * b_imp[4] * T_imp[4]
    dss!(u, p, t_final)
    # post_explicit!(u, p, t_final)
    return u
end
