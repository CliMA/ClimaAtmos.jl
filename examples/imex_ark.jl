#=
An s-stage (DIRK) IMEX ARK method for solving ∂u/∂t = f_exp(u, t) + f_imp(u, t)
is given by
    u_next :=
        u + Δt * ∑_{χ∈(exp,imp)} ∑_{i=1}^s b_χ[i] * f_χ(U[i], t + Δt * c_χ[i]),
    where
    U[i] :=
        u + Δt * ∑_{j=1}^{i-1} a_exp[i,j] * f_exp(U[j], t + Δt * c_exp[j]) +
        Δt * ∑_{j=1}^i a_imp[i,j] * f_imp(U[j], t + Δt * c_imp[j])
    ∀ i ∈ 1:s
Here, u_next denotes the value of u(t) at t_next = t + Δt.
The values a_χ[i,j] are called the "internal coefficients", the b_χ[i] are
called the "weights", and the c_χ[i] are called the "abcissae" (or "nodes").
The abscissae are often defined as c_χ[i] := ∑_{j=1}^s a_χ[i,j] for the explicit
and implicit methods to be "internally consistent", with c_exp[i] = c_imp[i] for
the overall IMEX method to be "internally consistent", but this is not required.
If the weights are defined as b_χ[j] := a_χ[s,j], then u_next = U[s]; i.e., the
method is FSAL (first same as last).

To simplify our notation, let
    a_χ[s+1,j] := b_χ[j] ∀ j ∈ 1:s,
    F_χ[j] := f_χ(U[j], t + Δt * c_χ[j]) ∀ j ∈ 1:s, and
    Δu_χ[i,j] := Δt * a_χ[i,j] * F_χ[j] ∀ i ∈ 1:s+1, j ∈ 1:s
This allows us to rewrite our earlier definitions as
    u_next = u + ∑_{χ∈(exp,imp)} ∑_{i=1}^s Δu_χ[s+1,i], where
    U[i] = u + ∑_{j=1}^{i-1} Δu_exp[i,j] + ∑_{j=1}^i Δu_imp[i,j] ∀ i ∈ 1:s

We will now rewrite the algorithm so that we can express each value of F_χ in
terms of the first increment Δu_χ that it is used to generate.
First, ∀ j ∈ 1:s, let
    I_χ[j] := min(i ∈ 1:s+1 ∣ a_χ[i,j] != 0)
Note that I_χ[j] is undefined if the j-th column of a_χ only contains zeros.
Also, note that I_imp[j] >= j and I_exp[j] > j ∀ j ∈ 1:s.
In addition, ∀ i ∈ 1:s+1, let
    new_Js_χ[i] := [j ∈ 1:s ∣ I_χ[j] == i],
    old_Js_χ[i] := [j ∈ 1:s ∣ I_χ[j] < i && a_χ[i,j] != 0], and
    N_χ[i] := length(new_Js_χ[i])
We can then define, ∀ i ∈ 1:s+1,
    ũ[i] := u + ∑_{χ∈(exp,imp)} ∑_{j ∈ old_Js_χ[i]} Δu_χ[i,j] and
    Û_χ[i,k] := Û_χ[i,k-1] + Δu_χ[i,new_Js_χ[i][k]] ∀ k ∈ 1:N_χ[i], where
    Û_exp[i,0] := ũ[i] and Û_imp[i,0] := Û_exp[i,N_exp[i]]
We then find that
    u_next = Û_imp[s+1,N_imp[s+1]] and U[i] = Û_imp[i,N_imp[i]] ∀ i ∈ 1:s
Let
    all_Js_χ := [j ∈ 1:s | isdefined(I_χ[j])]
Next, ∀ j ∈ all_Js_χ, let
    K_χ[j] := k ∈ N_χ[I_χ[j]] | new_Js_χ[I_χ[j]][k] == j
We then have that, ∀ j ∈ all_Js_χ,
    Û_χ[I_χ[j],K_χ[j]] = Û_χ[I_χ[j],K_χ[j]-1] + Δu_χ[I_χ[j],j]
Since a_χ[I_χ[j],j] != 0, this means that, ∀ j ∈ all_Js_χ,
    F_χ[j] = (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1]) / (Δt * a_χ[I_χ[j],j])

Now, suppose that we want to modify this algorithm so that we can apply a
filter/limiter during the addition of the increments Δu_χ[i,new_Js_χ[i][k]].
Specifically, instead of specifying f_χ(u, t), we want to specify
g_χ(û, u, t, Δt) and redefine, ∀ i ∈ 1:s+1 and ∀ k ∈ 1:N_χ[i],
    Û_χ[i,k] :=
        g_χ(
            Û_χ[i,k-1],
            U[new_Js_χ[i][k]],
            t + Δt * c_χ[new_Js_χ[i][k]],
            Δt * a_χ[i,new_Js_χ[i][k]]
        )
Note that specifying g_χ(û, u, t, Δt) := û + Δt * f_χ(u, t) is equivalent to not
using any filters/limiters.
We can use our earlier expression to redefine F_χ[j] as, ∀ j ∈ all_Js_χ,
    F_χ[j] := (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1]) / (Δt * a_χ[I_χ[j],j])
We then have that, ∀ i ∈ 1:s+1 and ∀ j ∈ all_Js_χ,
    Δu_χ[i,j] = ā_χ[i,j] * ΔÛ_χ[j], where
    ā_χ[i,j] := a_χ[i,j]/a_χ[I_χ[j],j] and
    ΔÛ_χ[j] := Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1]
We can then use these values of Δu_χ[i,j] to determine each value of ũ[i].

Now, ∀ i ∈ 1:s+1, let
    Js_to_save_χ[i] := [j ∈ new_Js_χ[i] | max(i′ ∈ 1:s+1 ∣ a_χ[i′,j] != 0) > i]
Note that we only need to compute F_χ[j] (or, rather, ΔÛ_χ[j]) if there is some
i ∈ 1:s+1 for which j ∈ Js_to_save_χ[i], since only then is there some value of
Δu_χ[i,j] that is computed based on F_χ[j].

This procedure of computing the values of F_χ (or, rather, the values of ΔÛ_χ)
from the values of Û_χ and using them to compute ũ[i] is rather inefficient, and
it would be better to directly use the values of Û_χ to compute ũ[i].
From the previous section, we know that, ∀ i ∈ 1:s+1,
    ũ[i] =
        u +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_Js_χ[i]} ā_χ[i,j] * (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1])
Now, ∀ i ∈ 1:s+1, let
    old_Js1_χ[i] := [j ∈ old_Js_χ[i] | K_χ[j] == 1] and
    old_Js2_χ[i] := [j ∈ old_Js_χ[i] | K_χ[j] > 1]
Since Û_exp[i,0] = ũ[i] and Û_imp[i,0] = Û_exp[i,N_exp[i]], we then have that
    ũ[i] =
        u +
        ∑_{j ∈ old_Js1_exp[i]} ā_exp[i,j] * (Û_exp[I_exp[j],1] - ũ[I_exp[j]]) +
        ∑_{j ∈ old_Js1_imp[i]} ā_imp[i,j] * (Û_imp[I_imp[j],1] - Û_exp[I_imp[j],N_exp[I_imp[j]]]) +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_Js2_χ[i]} ā_χ[i,j] * (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1])
Next, ∀ i ∈ 1:s+1, let
    old_Js11_imp[i] := [j ∈ old_Js1_imp[i] | N_exp[I_imp[j]] == 0] and
    old_Js12_imp[i] := [j ∈ old_Js1_imp[i] | N_exp[I_imp[j]] > 0]
Since Û_exp[i,0] = ũ[i], this means that
    ũ[i] =
        u +
        ∑_{j ∈ old_Js1_exp[i]} ā_exp[i,j] * (Û_exp[I_exp[j],1] - ũ[I_exp[j]]) +
        ∑_{j ∈ old_Js11_imp[i]} ā_imp[i,j] * (Û_imp[I_imp[j],1] - ũ[I_imp[j]]) +
        ∑_{j ∈ old_Js12_imp[i]} ā_imp[i,j] * (Û_imp[I_imp[j],1] - Û_exp[I_imp[j],N_exp[I_imp[j]]]) +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_Js2_χ[i]} ā_χ[i,j] * (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1])
We will now show that, ∀ i ∈ 1:s+1, there are some Q₀ and Q_χ such that
    ũ[i] = Q₀[i] * u + ∑_{χ∈(exp,imp)} ∑_{j=1}^{i-1} ∑_{k=1}^{N_χ[j]} Q_χ[i, j, k] * Û_χ[j, k]
First, we check the base case: ũ[1] = u, so that
    ũ[1] = Q₀[1] * u, where Q₀[1] = 1
Next, we apply the inductive step...
Is this too messy to do in the general case?

Don't forget about the possible memory optimizations!
=#

using LinearAlgebra: norm
using StaticArrays: @SArray

struct IMEXARKAlgorithm{a_exp, a_imp, b_exp, b_imp, c_exp, c_imp, N} <:
        ClimaTimeSteppers.DistributedODEAlgorithm
    nlsolve::N
end
IMEXARKAlgorithm{a_exp, a_imp, b_exp, b_imp, c_exp, c_imp}(;
    nlsolve::N,
) where {a_exp, a_imp, b_exp, b_imp, c_exp, c_imp, N} =
    IMEXARKAlgorithm{a_exp, a_imp, b_exp, b_imp, c_exp, c_imp, N}(nlsolve)

function ARS121_tableau()
    a_exp = @SArray [0 0; 1 0]
    b_exp = @SArray [0, 1]
    c_exp = vec(sum(a_exp; dims = 2))

    a_imp = @SArray [0 0; 0 1]
    b_imp = @SArray [0, 1]
    c_imp = vec(sum(a_imp; dims = 2))

    return (a_exp, a_imp, b_exp, b_imp, c_exp, c_imp)
end
const ARS121new = IMEXARKAlgorithm{ARS121_tableau()...}

function ARS232_tableau()
    γ = 1 - sqrt(2) / 2
    δ = -2/3 * sqrt(2)

    a_exp = @SArray [
      0 0     0;
      γ 0     0;
      δ (1-δ) 0;
    ]
    b_exp = @SArray [0, 1 - γ, γ]
    c_exp = vec(sum(a_exp; dims = 2))

    a_imp = @SArray [
        0 0     0;
        0 γ     0;
        0 (1-γ) γ;
    ]
    b_imp = @SArray [0, 1 - γ, γ]
    c_imp = vec(sum(a_imp; dims = 2))

    return (a_exp, a_imp, b_exp, b_imp, c_exp, c_imp)
end
const ARS232new = IMEXARKAlgorithm{ARS232_tableau()...}

function ARS343_tableau()
    γ = 0.4358665215084590
    a42 = 0.5529291480359398
    a43 = 0.5529291480359398

    b1 = -3/2 * γ^2 + 4 * γ - 1/4
    b2 =  3/2 * γ^2 - 5 * γ + 5/4
    a31 = (1 - 9/2 * γ + 3/2 * γ^2) * a42 +
        (11/4 - 21/2 * γ + 15/4 * γ^2) * a43 - 7/2 + 13 * γ - 9/2 * γ^2
    a32 = (-1 + 9/2 * γ - 3/2 * γ^2) * a42 +
        (-11/4 + 21/2 * γ - 15/4 * γ^2) * a43 + 4 - 25/2 * γ + 9/2 * γ^2
    a41 = 1 - a42 - a43

    a_exp = @SArray [
      0   0   0   0;
      γ   0   0   0;
      a31 a32 0   0;
      a41 a42 a43 0;
    ]
    b_exp = @SArray [0, b1, b2, γ]
    c_exp = vec(sum(a_exp; dims = 2))

    a_imp = @SArray [
        0 0       0  0;
        0 γ       0  0;
        0 (1-γ)/2 γ  0;
        0 b1      b2 γ;
    ]
    b_imp = @SArray [0, b1, b2, γ]
    c_imp = vec(sum(a_imp; dims = 2))

    return (a_exp, a_imp, b_exp, b_imp, c_exp, c_imp)
end
const ARS343new = IMEXARKAlgorithm{ARS343_tableau()...}

function IMKG354a_tableau()
    a_exp = @SArray [
      0   0   0   0   0   0;
      1/5 0   0   0   0   0;
      0   1/5 0   0   0   0;
      0   0   2/3 0   0   0;
      1/3 0   0   1/3 0   0;
      1/4 0   0   0   3/4 0;
    ]
    b_exp = vec(a_exp[end, :])
    c_exp = vec(sum(a_exp; dims = 2))

    a_imp = @SArray [
      0   0   0     0    0   0;
      0   0   0     0    0   0;
      0   0   2/5   0    0   0;
      0   0   11/30 2/5  0   0;
      1/3 0   0     -2/3 1   0;
      1/4 0   0     0    3/4 0;
    ]
    b_imp = vec(a_imp[end, :])
    c_imp = vec(sum(a_imp; dims = 2))

    return (a_exp, a_imp, b_exp, b_imp, c_exp, c_imp)
end
const IMKG354a = IMEXARKAlgorithm{IMKG354a_tableau()...}

struct IMEXARKCache{is, js, a, c, I, new_Js, old_Js, Js_to_save, V, N} <:
        ClimaTimeSteppers.DistributedODEAlgorithm
    vars::V
    nlsolve!::N
end

function ClimaTimeSteppers.cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::IMEXARKAlgorithm{a_exp, a_imp, b_exp, b_imp, c_exp, c_imp};
    kwargs...
) where {a_exp, a_imp, b_exp, b_imp, c_exp, c_imp}
    @assert ndims(a_exp) == ndims(a_imp) == 2
    @assert ndims(b_exp) == ndims(b_imp) == ndims(c_exp) == ndims(c_imp) == 1
    @assert size(a_exp, 1) == size(a_exp, 2) == size(a_imp, 1) == size(a_imp, 2)
    @assert size(b_exp, 1) == size(b_imp, 1) == size(c_exp, 1) == size(c_imp, 1)
    s = size(a_exp, 1)
    @assert all(i -> all(j -> a_exp[i, j] == 0, i:s), 1:s)
    @assert all(i -> all(j -> a_imp[i, j] == 0, (i + 1):s), 1:s)
    is_fsal = b_exp == a_exp[s, :] && b_imp == a_imp[s, :]

    χs = (1, 2)
    FT = eltype(prob.u0)
    a = (FT.(vcat(a_exp, b_exp')), FT.(vcat(a_imp, b_imp')))
    c = (FT.(c_exp), FT.(c_imp))

    is = Tuple(1:(is_fsal ? s : s + 1))
    js = Tuple(1:s)
    I = map(χ -> map(j -> findfirst(i -> a[χ][i, j] != 0, is), js), χs)
    all_Js = map(χ -> filter(j -> !isnothing(I[χ][j]), js), χs)
    new_Js = map(χ -> map(i -> filter(j -> I[χ][j] == i, js), is), χs)
    old_Js_func(χ, i) = filter(j -> I[χ][j] < i && a[χ][i, j] != 0, all_Js[χ])
    old_Js = map(χ -> map(i -> old_Js_func(χ, i), is), χs)
    Js_to_save_func(χ, i) =
        filter(j -> findlast(i′ -> a[χ][i′, j] != 0, is) > i, new_Js[χ][i])
    Js_to_save = map(χ -> map(i -> Js_to_save_func(χ, i), is), χs)

    u = prob.u0
    Uis = map(i -> Symbol(:U, i) => similar(u), is)
    ΔÛχjs = map(χ -> map(j -> Symbol(:ΔÛ, χ, :_, j) => similar(u), tuplejoin(Js_to_save[χ]...)), χs)
    vars = NamedTuple((:Û_copy => similar(u), Uis..., tuplejoin(ΔÛχjs...)...))
    nlsolve! = nlcache(alg.nlsolve, u, prob.f.f1.jac_prototype)
    return IMEXARKCache{is, js, a, c, I, new_Js, old_Js, Js_to_save, typeof(vars), typeof(nlsolve!)}(
        vars,
        nlsolve!,
    )
end

@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

# Workaround for not being allowed to use closures in a generated function
struct ImplicitStepErrorFunction{F, U, P, T}
    ode_f!::F
    û::U
    p::P
    t::T
    Δt::T
end
struct ImplicitStepErrorJacobian{W, P, T}
    Wfact!::W
    p::P
    t::T
    Δt::T
end

(f!::ImplicitStepErrorFunction)(f, u) = f!(f, u, f!.ode_f!)
function (f!::ImplicitStepErrorFunction)(f, u, ode_f!::ForwardEulerODEFunction)
    (; û, p, t, Δt) = f!
    f .= û
    ode_f!(f, u, p, t, Δt)
    f .-= u
end
function (f!::ImplicitStepErrorFunction)(f, u, ode_f!::ODEFunction)
    (; û, p, t, Δt) = f!
    ode_f!(f, u, p, t)
    f .= û .+ Δt .* f .- u
end

((; Wfact!, p, t, Δt)::ImplicitStepErrorJacobian)(j, u) = Wfact!(j, u, p, Δt, t)

function step_u_expr(
    ::Type{<:IMEXARKCache{is, js, a, c, I, new_Js, old_Js, Js_to_save}},
) where {is, js, a, c, I, new_Js, old_Js, Js_to_save}
    function Δu_expr(χ, i, j)
        ΔÛχj = :(vars.$(Symbol(:ΔÛ, χ, :_, j)))
        return :(broadcasted(*, $(a[χ][i, j] / a[χ][I[χ][j], j]), $ΔÛχj))
    end

    expr = :(
        (; broadcasted, materialize!) = Base;
        (; u, p, t, dt, prob) = integrator;
        (; f) = prob;
        (; f1, f2) = f;
        (; nlsolve!, vars) = cache
    )

    χs = (1, 2)
    f = (:f2, :f1) # the fs are stored in the opposite order in a SplitFunction
    for i in is
        Ui = i == is[end] ? :u : :(vars.$(Symbol(:U, i)))
        Δu_exprs =
            tuplejoin(map(χ -> map(j -> Δu_expr(χ, i, j), old_Js[χ][i]), χs)...)
        ũi_expr =
            length(Δu_exprs) == 0 ? :u : :(broadcasted(+, u, $(Δu_exprs...)))
        expr = :($expr; materialize!($Ui, $ũi_expr))
        for χ in χs
            fχ! = f[χ]
            for j in new_Js[χ][i]
                Uj = :(vars.$(Symbol(:U, j)))
                Ûχik_expr = :(
                    t′ = t + dt * $(c[χ][j]);
                    Δt′ = dt * $(a[χ][i, j])
                )
                if j == i
                    Ûχik_expr = :(
                        $Ûχik_expr;
                        nlsolve!(
                            $Uj,
                            ImplicitStepErrorFunction($fχ!, $Ui, p, t′, Δt′),
                            ImplicitStepErrorJacobian($fχ!.Wfact, p, t′, Δt′),
                        )
                    )
                else
                    Ûχik_expr = :($Ûχik_expr; $fχ!($Ui, $Uj, p, t′, Δt′))
                end
                if j in Js_to_save[χ][i]
                    ΔÛχj = :(vars.$(Symbol(:ΔÛ, χ, :_, j)))
                    Ûχik_expr = :(
                        vars.Û_copy .= $Ui;
                        $Ûχik_expr;
                        $ΔÛχj .= $Ui .- vars.Û_copy
                    )
                end
                expr = :($expr; $Ûχik_expr)
            end
        end
    end
    return :($expr; return u)
end

@generated function ClimaTimeSteppers.step_u!(integrator, cache::IMEXARKCache)
    return step_u_expr(cache)
end

abstract type ConvergenceCondition end
function (::ConvergenceCondition)(residuals) end

struct AbsoluteTolerance{FT} <: ConvergenceCondition
    r::FT # minimum value of residual; should be ≥ 0
end
struct RelativeTolerance{FT} <: ConvergenceCondition
    r::FT # minimum value of residual divided by initial residual; should be ≥ 0
end
struct InsufficientConvergenceRate{FT} <: ConvergenceCondition
    r::FT # minimum value of inverse of linear convergence rate; should be ≥ 1
end
struct FirstConditionSatisfied{
    C <: NTuple{N, <:ConvergenceCondition} where {N}
} <: ConvergenceCondition
    conds::C
end
struct AllConditionsSatisfied{
    C <: NTuple{N, <:ConvergenceCondition} where {N}
} <: ConvergenceCondition
    conds::C
end

FirstConditionSatisfied(conds...) = FirstConditionSatisfied(conds)
AllConditionsSatisfied(conds...) = AllConditionsSatisfied(conds)

((; r)::AbsoluteTolerance)(residuals) = residuals[end] <= r
((; r)::RelativeTolerance)(residuals) = residuals[end] <= r * residuals[1]
((; r)::InsufficientConvergenceRate)(residuals) =
    length(residuals) > 1 && residuals[end - 1] <= r * residuals[end]
((; conds)::FirstConditionSatisfied)(residuals) = any(c -> c(residuals), conds)
((; conds)::AllConditionsSatisfied)(residuals) = all(c -> c(residuals), conds)

Base.@kwdef struct NewtonAlgorithm{L, C <: ConvergenceCondition}
    linsolve::L
    has_converged::C = nothing
    max_iters::Int = 1
    update_jac::Bool = false # this could also be the Jacobian update frequency
end
struct NewtonCache{A, L, X, J, R}
    alg::A
    linsolve!::L
    j_div_f::X
    f::X
    j::J
    residuals::R
end

nlcache(alg::NewtonAlgorithm, x_prototype, j_prototype) =
    NewtonCache(
        alg,
        alg.linsolve(Val{:init}, j_prototype, x_prototype),
        similar(x_prototype),
        similar(x_prototype),
        similar(j_prototype),
        isnothing(alg.has_converged) ? nothing :
            Array{eltype(x_prototype)}(undef, alg.max_iters + 1),
    )

#=
We want to solve the equation f(x) = 0, given j(x) = ∂f/∂x.
Let x[n] denote the value of x on the n-th Newton iteration.
A first-order Taylor series expansion tells us that
    f(x[n+1]) ≈ f(x[n]) + j(x[n]) * (x[n+1] - x[n])
Setting the right-hand side equal to 0 and solving for x[n+1] for gives us
    x[n+1] = x[n] - j_div_f[n], where j_div_f[n] = j(x[n])\f(x[n])
If j changes very slowly, we can approximate j(x[n]) ≈ j(x[0]).
We can either iterate from n = 0 to the maximum number of iterations, or we can
stop based on some convergence condition.
=#
function (nlsolve!::NewtonCache)(x, f!, j!)
    (; alg, linsolve!, j_div_f, f, j, residuals) = nlsolve!
    (; has_converged, max_iters, update_jac) = alg
    for iter in 0:max_iters
        if iter > 0
            linsolve!(j_div_f, j, f)
            x .-= j_div_f
        end
        if isnothing(has_converged)
            iter == max_iters && break
            f!(f, x)
        else
            f!(f, x)
            residuals[iter + 1] = norm(f)
            has_converged(view(residuals, 1:(iter + 1))) && break
            iter == max_iters && error(
                "Newton's method did not converge within $max_iters iterations \
                (residual change: $(residuals[1]) → $(residuals[end]))"
            )
        end
        (update_jac || iter == 0) && j!(j, x)
    end
end
