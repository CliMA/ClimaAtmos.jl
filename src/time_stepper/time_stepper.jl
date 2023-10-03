import NVTX
import ClimaTimeSteppers as CTS
import LinearAlgebra as LA
import Krylov

Base.@kwdef struct AtmosODEFunction{TL, TE, TI, L, D, PE, PI} <:
                   CTS.AbstractClimaODEFunction
    T_lim!::TL = (uₜ, u, p, t) -> nothing
    T_exp!::TE = (uₜ, u, p, t) -> nothing
    T_imp!::TI = (uₜ, u, p, t) -> nothing
    lim!::L = (u, p, t, u_ref) -> nothing
    dss!::D = (u, p, t) -> nothing
    post_explicit!::PE = (u, p, t) -> nothing
    post_implicit!::PI = (u, p, t) -> nothing
end

include("imex_ark.jl")
include("imex_ssprk.jl")
include("hc_ars343.jl")
using LinearOperators: LinearOperator

# overload with 1 more arg
function CTS.jvp!(
    alg::CTS.ForwardDiffJVP,
    cache,
    jΔx,
    Δx,
    x,
    f!,
    f,
    post_implicit!,
)
    (; default_step, step_adjustment) = alg
    (; x2, f2) = cache
    FT = eltype(x)
    ε = FT(step_adjustment) * default_step(Δx, x)
    @. x2 = x + ε * Δx
    post_implicit!(x2) # f! is not called with x, so we need to call post_implicit!
    f!(f2, x2)
    @. jΔx = (f2 - f) / ε
end

# overload with 1 more arg
function CTS.solve_krylov!(
    alg::CTS.KrylovMethod,
    cache,
    Δx,
    x,
    f!,
    f,
    n,
    post_implicit!,
    j = nothing,
)
    (; jacobian_free_jvp, forcing_term, solve_kwargs) = alg
    (; disable_preconditioner, debugger) = alg
    type = CTS.solver_type(alg)
    (; jacobian_free_jvp_cache, forcing_term_cache, solver, debugger_cache) =
        cache
    jΔx!(jΔx, Δx) =
        isnothing(jacobian_free_jvp) ? mul!(jΔx, j, Δx) :
        CTS.jvp!(
            jacobian_free_jvp,
            jacobian_free_jvp_cache,
            jΔx,
            Δx,
            x,
            f!,
            f,
            post_implicit!,
        )
    opj = LinearOperator(eltype(x), length(x), length(x), false, false, jΔx!)
    M =
        disable_preconditioner || isnothing(j) || isnothing(jacobian_free_jvp) ?
        I : j
    CTS.print_debug!(debugger, debugger_cache, opj, M)
    ldiv = true
    atol = zero(eltype(Δx))
    rtol = CTS.get_rtol!(forcing_term, forcing_term_cache, f, n)
    verbose = Int(CTS.is_verbose(alg.verbose))
    Krylov.solve!(solver, opj, f; M, ldiv, atol, rtol, verbose, solve_kwargs...)
    iter = solver.stats.niter
    if !solver.stats.solved
        str1 = isnothing(j) ? () : ("the Jacobian",)
        str2 =
            isnothing(jacobian_free_jvp) ? () : ("the Jacobian-vector product",)
        str = join((str1..., str2...), " and/or ")
        if solver.stats.inconsistent
            @warn "$type detected that the Jacobian is singular on iteration \
                   $iter; if possible, try improving the approximation of $str"
        else
            @warn "$type did not converge within $iter iterations; if \
                   possible, try improving the approximation of $str, or try \
                   increasing the forcing term"
        end
    elseif iter == 0 &&
           solver.stats.status != "x = 0 is a zero-residual solution"
        @warn "$type set Δx to 0 without running any iterations; if possible, \
               try decreasing the forcing term"
    end
    Δx .= Krylov.solution(solver)
end


function _solve_newton!(
    alg::CTS.NewtonsMethod,
    cache,
    x,
    f!,
    post_implicit!,
    j! = nothing,
)
    (; max_iters, update_j, krylov_method, convergence_checker, verbose) = alg
    (; krylov_method_cache, convergence_checker_cache) = cache
    (; Δx, f, j) = cache
    if (!isnothing(j)) && CTS.needs_update!(update_j, CTS.NewNewtonSolve())
        j!(j, x)
    end
    for n in 1:max_iters
        # Compute Δx[n].
        if (!isnothing(j)) &&
           CTS.needs_update!(update_j, CTS.NewNewtonIteration())
            j!(j, x)
        end
        f!(f, x)
        if isnothing(krylov_method)
            if j isa DenseMatrix
                LA.ldiv!(Δx, lu(j), f) # Highly inefficient! Only used for testing.
            else
                LA.ldiv!(Δx, j, f)
            end
        else
            CTS.solve_krylov!(
                krylov_method,
                krylov_method_cache,
                Δx,
                x,
                f!,
                f,
                n,
                post_implicit!,
                j,
            )
        end
        CTS.is_verbose(verbose) &&
            @info "Newton iteration $n: ‖x‖ = $(LA.norm(x)), ‖Δx‖ = $(LA.norm(Δx))"

        x .-= Δx
        isnothing(post_implicit!) || post_implicit!(x)
        # Update x[n] with Δx[n - 1], and exit the loop if Δx[n] is not needed.
        # Check for convergence if necessary.
        if CTS.is_converged!(
            convergence_checker,
            convergence_checker_cache,
            x,
            Δx,
            n,
        )
            break
        end
        if CTS.is_verbose(verbose) && n == max_iters
            @warn "Newton's method did not converge within $n iterations: ‖x‖ = $(LA.norm(x)), ‖Δx‖ = $(LA.norm(Δx))"
        end
    end
end
