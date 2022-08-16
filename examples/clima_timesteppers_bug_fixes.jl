import ClimaTimeSteppers: step_u!

# Don't wrap a ForwardEulerODEFunction in an ODEFunction.
OrdinaryDiffEq.ODEFunction{iip}(f::ForwardEulerODEFunction) where {iip} = f
OrdinaryDiffEq.ODEFunction(f::ForwardEulerODEFunction) = f

# Fix the ARSAlgorithm
function step_u!(int, cache::ClimaTimeSteppers.ARSCache{Nstages}) where {Nstages}

    f = int.prob.f
    f1! = f.f1
    f2! = f.f2
    Wfact! = f1!.Wfact

    u = int.u
    p = int.p
    t = int.t
    dt = int.dt

    tab = cache.tableau
    W = cache.W
    U = cache.U
    Uhat = cache.Uhat
    residual = cache.idu
    linsolve! = cache.linsolve!

    # Update W
    # Wfact!(W, u, p, dt*tab.γ, t)


    # implicit eqn:
    #   ux = u + dt * f(ux, p, t)
    # Newton iteration:
    #   ux <- ux + (I - dt J) \ (u + dt f(ux, p, t) - ux)
    # initial iteration
    #   ux <- u + dt (I - dt J) \ f(u, p, t)

    function implicit_step!(ux, u, p, t, dt)
        FT = eltype(u)
        abstol = FT(1e7) * eps(FT) # empirical threshold
        ux .= u
        converged = false
        max_iters = 100
        prev_residual_norm = FT(Inf)
        for iter in 0:max_iters
            iter == 0 && Wfact!(W, ux, p, dt, t)
            f1!(residual, ux, p, t)
            @. residual = ux - dt * residual - u
            residual_norm = norm(residual)
            if residual_norm < abstol || residual_norm > prev_residual_norm
                # The residual oscillates around a minimum after the limit of
                # numerical precision is reached.
                converged = true
                break
            end
            prev_residual_norm = residual_norm
            iter == max_iters && error("Failed to converge in $max_iters iters")
            # The following two lines are technically part of the next Newton iteration
            linsolve!(residual, W, residual)
            @. ux += residual
        end
    end

    #### stage 1
    # explicit
    Uhat[1] .= u # utilde[i],  Q0[1] == 1
    f2!(Uhat[1], u, p, t+dt*tab.chat[1], dt*tab.ahat[2,1])

    # implicit
    implicit_step!(U[1], Uhat[1], p, t+dt*tab.c[1], dt*tab.a[1,1])
    if Nstages == 1
        u .= tab.Q0[2] .* u .+
            tab.Qhat[2,1] .* Uhat[1] .+ tab.Q[2,1] .* U[1] # utilde[2]
        f2!(u, U[1], p, t+dt*tab.chat[2], dt*tab.ahat[3,2])
        return
    end

    #### stage 2
    Uhat[2] .= tab.Q0[2] .* u .+
            tab.Qhat[2,1] .* Uhat[1] .+ tab.Q[2,1] .* U[1] # utilde[2]
    f2!(Uhat[2], U[1], p, t+dt*tab.chat[2], dt*tab.ahat[3,2])

    implicit_step!(U[2], Uhat[2], p, t+dt*tab.c[2], dt*tab.a[2,2])

    if Nstages == 2
        u .= tab.Q0[3] .* u .+
            tab.Qhat[3,1] .* Uhat[1] .+ tab.Q[3,1] .* U[1] .+
            tab.Qhat[3,2] .* Uhat[2] .+ tab.Q[3,2] .* U[2] # utilde[3]
        f2!(u, U[2], p, t+dt*tab.chat[3], dt*tab.ahat[4,3])
        return
    end

    #### stage 3
    Uhat[3] .= tab.Q0[3] .* u .+
            tab.Qhat[3,1] .* Uhat[1] .+ tab.Q[3,1] .* U[1] .+
            tab.Qhat[3,2] .* Uhat[2] .+ tab.Q[3,2] .* U[2] # utilde[3]
    f2!(Uhat[3], U[2], p, t+dt*tab.chat[3], dt*tab.ahat[4,3])
    # @show Uhat[3] t+dt*tab.chat[3]

    implicit_step!(U[3], Uhat[3], p, t+dt*tab.c[3], dt*tab.a[3,3])
    # @show U[3] t+dt*tab.c[3]

    ### final update
    u .= tab.Q0[4] .* u .+
    tab.Qhat[4,1] .* Uhat[1] .+ tab.Q[4,1] .* U[1] .+
    tab.Qhat[4,2] .* Uhat[2] .+ tab.Q[4,2] .* U[2] .+
    tab.Qhat[4,3] .* Uhat[3] .+ tab.Q[4,3] .* U[3]

    # @show u
    f2!(u, U[3], p, t+dt*tab.chat[4], dt*tab.ahat[5,4])
    # @show u t+dt*tab.chat[4]
    return
end
