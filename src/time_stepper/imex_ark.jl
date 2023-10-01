function CTS.step_u!(
    integrator,
    cache::CTS.IMEXARKCache,
    f::AtmosODEFunction,
    name,
)
    (; u, p, t, dt, alg) = integrator
    (; T_lim!, T_exp!, T_imp!, lim!, dss!) = f
    (; post_explicit!, post_implicit!) = f
    (; tableau, newtons_method) = alg
    (; a_exp, b_exp, a_imp, b_imp, c_exp, c_imp) = tableau
    (; U, T_lim, T_exp, T_imp, temp, γ, newtons_method_cache) = cache
    s = length(b_exp)

    if !isnothing(newtons_method)
        (; update_j) = newtons_method
        jacobian = newtons_method_cache.j
        if (!isnothing(jacobian)) &&
           CTS.needs_update!(update_j, CTS.NewTimeStep(t))
            if γ isa Nothing
                sdirk_error(name)
            else
                post_implicit!(u, p, t)
                T_imp!.Wfact(jacobian, u, p, dt * γ, t)
            end
        end
    end

    for i in 1:s
        NVTX.@range "stage" payload = i begin
            t_exp = t + dt * c_exp[i]
            t_imp = t + dt * c_imp[i]

            @. U = u

            for j in 1:(i - 1)
                iszero(a_exp[i, j]) && continue
                @. U += dt * a_exp[i, j] * T_lim[j]
            end
            lim!(U, p, t_exp, u)

            for j in 1:(i - 1)
                iszero(a_exp[i, j]) && continue
                @. U += dt * a_exp[i, j] * T_exp[j]
            end

            for j in 1:(i - 1)
                iszero(a_imp[i, j]) && continue
                @. U += dt * a_imp[i, j] * T_imp[j]
            end

            dss!(U, p, t_exp)

            if iszero(a_imp[i, i])
                post_explicit!(U, p, t_exp)
            else # Implicit solve
                @assert !isnothing(newtons_method)
                @. temp = U
                post_explicit!(U, p, t_exp)
                # TODO: can/should we remove these closures?
                implicit_equation_residual! =
                    (residual, Ui) -> begin
                        post_implicit!(Ui, p, t_imp)
                        T_imp!(residual, Ui, p, t_imp)
                        @. residual =
                            temp + dt * a_imp[i, i] * residual - Ui
                    end
                implicit_equation_jacobian! =
                    (jacobian, Ui) -> begin
                        post_implicit!(Ui, p, t_imp)
                        T_imp!.Wfact(jacobian, Ui, p, dt * a_imp[i, i], t_imp)
                    end

                call_post_implicit! =
                    Ui -> begin
                        if (!all(iszero, a_imp[:, i]) || !iszero(b_imp[i])) && !iszero(a_imp[i, i])
                            # If T_imp[i] is being treated implicitly, ensure that it
                            # exactly satisfies the implicit equation.
                            @. T_imp[i] = (Ui - temp) / (dt * a_imp[i, i])
                        end
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

            if (!all(iszero, a_imp[:, i]) || !iszero(b_imp[i])) &&
               iszero(a_imp[i, i])
                # If its coefficient is 0, T_imp[i] is effectively being
                # treated explicitly.
                T_imp!(T_imp[i], U, p, t_imp)
            end

            if !all(iszero, a_exp[:, i]) || !iszero(b_exp[i])
                T_lim!(T_lim[i], U, p, t_exp)
                T_exp!(T_exp[i], U, p, t_exp)
            end
        end
    end

    t_final = t + dt

    @. temp = u
    for j in 1:s
        iszero(b_exp[j]) && continue
        @. temp += dt * b_exp[j] * T_lim[j]
    end
    lim!(temp, p, t_final, u)
    @. u = temp

    for j in 1:s
        iszero(b_exp[j]) && continue
        @. u += dt * b_exp[j] * T_exp[j]
    end

    for j in 1:s
        iszero(b_imp[j]) && continue
        @. u += dt * b_imp[j] * T_imp[j]
    end

    dss!(u, p, t_final)

    return u
end
