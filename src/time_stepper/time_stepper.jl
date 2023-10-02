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

function _solve_newton!(
    alg::CTS.NewtonsMethod,
    cache,
    x,
    f!,
    j! = nothing,
    post_implicit! = nothing,
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
