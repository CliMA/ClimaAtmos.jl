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
    first_i_χ[j] := min(i ∈ 1:s+1 ∣ a_χ[i,j] != 0)
Note that first_i_χ[j] is undefined if the j-th column of a_χ only contains zeros.
Also, note that first_i_imp[j] >= j and first_i_exp[j] > j ∀ j ∈ 1:s.
In addition, ∀ i ∈ 1:s+1, let
    new_js_χ[i] := [j ∈ 1:s ∣ first_i_χ[j] == i],
    old_js_χ[i] := [j ∈ 1:s ∣ first_i_χ[j] < i && a_χ[i,j] != 0], and
    N_χ[i] := length(new_js_χ[i])
We can then define, ∀ i ∈ 1:s+1,
    ũ[i] := u + ∑_{χ∈(exp,imp)} ∑_{j ∈ old_js_χ[i]} Δu_χ[i,j] and
    Û_χ[i,k] := Û_χ[i,k-1] + Δu_χ[i,new_js_χ[i][k]] ∀ k ∈ 1:N_χ[i], where
    Û_exp[i,0] := ũ[i] and Û_imp[i,0] := Û_exp[i,N_exp[i]]
We then find that
    u_next = Û_imp[s+1,N_imp[s+1]] and U[i] = Û_imp[i,N_imp[i]] ∀ i ∈ 1:s
Let
    all_js_χ := [j ∈ 1:s | isdefined(first_i_χ[j])]
Next, ∀ j ∈ all_js_χ, let
    K_χ[j] := k ∈ N_χ[first_i_χ[j]] | new_js_χ[first_i_χ[j]][k] == j
We then have that, ∀ j ∈ all_js_χ,
    Û_χ[first_i_χ[j],K_χ[j]] = Û_χ[first_i_χ[j],K_χ[j]-1] + Δu_χ[first_i_χ[j],j]
Since a_χ[first_i_χ[j],j] != 0, this means that, ∀ j ∈ all_js_χ,
    F_χ[j] = (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1]) / (Δt * a_χ[first_i_χ[j],j])

Now, suppose that we want to modify this algorithm so that we can apply a
filter/limiter during the addition of the increments Δu_χ[i,new_js_χ[i][k]].
Specifically, instead of specifying f_χ(u, t), we want to specify
g_χ(û, u, t, Δt) and redefine, ∀ i ∈ 1:s+1 and ∀ k ∈ 1:N_χ[i],
    Û_χ[i,k] :=
        g_χ(
            Û_χ[i,k-1],
            U[new_js_χ[i][k]],
            t + Δt * c_χ[new_js_χ[i][k]],
            Δt * a_χ[i,new_js_χ[i][k]]
        )
Note that specifying g_χ(û, u, t, Δt) := û + Δt * f_χ(u, t) is equivalent to not
using any filters/limiters.
We can use our earlier expression to redefine F_χ[j] as, ∀ j ∈ all_js_χ,
    F_χ[j] := (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1]) / (Δt * a_χ[first_i_χ[j],j])
We then have that, ∀ i ∈ 1:s+1 and ∀ j ∈ all_js_χ,
    Δu_χ[i,j] = ā_χ[i,j] * ΔÛ_χ[j], where
    ā_χ[i,j] := a_χ[i,j]/a_χ[first_i_χ[j],j] and
    ΔÛ_χ[j] := Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1]
We can then use these values of Δu_χ[i,j] to determine each value of ũ[i].

Now, ∀ i ∈ 1:s+1, let
    Js_to_save_χ[i] := [j ∈ new_js_χ[i] | max(i′ ∈ 1:s+1 ∣ a_χ[i′,j] != 0) > i]
Note that we only need to compute F_χ[j] (or, rather, ΔÛ_χ[j]) if there is some
i ∈ 1:s+1 for which j ∈ Js_to_save_χ[i], since only then is there some value of
Δu_χ[i,j] that is computed based on F_χ[j].

This procedure of computing the values of F_χ (or, rather, the values of ΔÛ_χ)
from the values of Û_χ and using them to compute ũ[i] is rather inefficient, and
it would be better to directly use the values of Û_χ to compute ũ[i].
From the previous section, we know that, ∀ i ∈ 1:s+1,
    ũ[i] =
        u +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_js_χ[i]} ā_χ[i,j] * (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1])
Now, ∀ i ∈ 1:s+1, let
    old_js1_χ[i] := [j ∈ old_js_χ[i] | K_χ[j] == 1] and
    old_js2_χ[i] := [j ∈ old_js_χ[i] | K_χ[j] > 1]
Since Û_exp[i,0] = ũ[i] and Û_imp[i,0] = Û_exp[i,N_exp[i]], we then have that
    ũ[i] =
        u +
        ∑_{j ∈ old_js1_exp[i]} ā_exp[i,j] * (Û_exp[first_i_exp[j],1] - ũ[first_i_exp[j]]) +
        ∑_{j ∈ old_js1_imp[i]} ā_imp[i,j] * (Û_imp[first_i_imp[j],1] - Û_exp[first_i_imp[j],N_exp[first_i_imp[j]]]) +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_js2_χ[i]} ā_χ[i,j] * (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1])
Next, ∀ i ∈ 1:s+1, let
    old_js11_imp[i] := [j ∈ old_js1_imp[i] | N_exp[first_i_imp[j]] == 0] and
    old_js12_imp[i] := [j ∈ old_js1_imp[i] | N_exp[first_i_imp[j]] > 0]
Since Û_exp[i,0] = ũ[i], this means that
    ũ[i] =
        u +
        ∑_{j ∈ old_js1_exp[i]} ā_exp[i,j] * (Û_exp[first_i_exp[j],1] - ũ[first_i_exp[j]]) +
        ∑_{j ∈ old_js11_imp[i]} ā_imp[i,j] * (Û_imp[first_i_imp[j],1] - ũ[first_i_imp[j]]) +
        ∑_{j ∈ old_js12_imp[i]} ā_imp[i,j] * (Û_imp[first_i_imp[j],1] - Û_exp[first_i_imp[j],N_exp[first_i_imp[j]]]) +
        ∑_{χ∈(exp,imp)} ∑_{j ∈ old_js2_χ[i]} ā_χ[i,j] * (Û_χ[first_i_χ[j],K_χ[j]] - Û_χ[first_i_χ[j],K_χ[j]-1])
We will now show that, ∀ i ∈ 1:s+1, there are some Q₀ and Q_χ such that
    ũ[i] = Q₀[i] * u + ∑_{χ∈(exp,imp)} ∑_{j=1}^{i-1} ∑_{k=1}^{N_χ[j]} Q_χ[i, j, k] * Û_χ[j, k]
First, we check the base case: ũ[1] = u, so that
    ũ[1] = Q₀[1] * u, where Q₀[1] = 1
Next, we apply the inductive step...
Is this too messy to do in the general case?

Don't forget about the possible memory optimizations!
=#

using LinearAlgebra: norm
using StaticArrays: SMatrix, SVector, @SArray

struct IMEXARKAlgorithm{as, cs, N} <: ClimaTimeSteppers.DistributedODEAlgorithm
    newtons_method::N
end
IMEXARKAlgorithm{as, cs}(; newtons_method::N) where {as, cs, N} =
    IMEXARKAlgorithm{as, cs, N}(newtons_method)

# Default values for b_χ assume FSAL, and default values for c_χ assume internal
# consistency.
function make_IMEXARKAlgorithm(;
    a_exp::SMatrix{s, s},
    b_exp::SVector{s} = vec(a_exp[end, :]),
    c_exp::SVector{s} = vec(sum(a_exp; dims = 2)),
    a_imp::SMatrix{s, s},
    b_imp::SVector{s} = vec(a_imp[end, :]),
    c_imp::SVector{s} = vec(sum(a_imp; dims = 2)),
) where {s}
    @assert all(i -> all(j -> a_exp[i, j] == 0, i:s), 1:s)
    @assert all(i -> all(j -> a_imp[i, j] == 0, (i + 1):s), 1:s)
    if a_exp[end, :] == b_exp && a_imp[end, :] == b_imp
        as = (a_exp, a_imp)
    else
        as = (vcat(a_exp, b_exp'), vcat(a_imp, b_imp'))
    end
    cs = (c_exp, c_imp)
    return IMEXARKAlgorithm{as, cs}
end

const ARS121new = make_IMEXARKAlgorithm(;
    a_exp = @SArray([0 0; 1 0]),
    b_exp = @SArray([0, 1]),
    a_imp = @SArray([0 0; 0 1]),
    b_imp = @SArray([0, 1]),
)
const ARS232new = let
    γ = 1 - sqrt(2) / 2
    δ = -2/3 * sqrt(2)

    make_IMEXARKAlgorithm(;
        a_exp = @SArray([
            0 0     0;
            γ 0     0;
            δ (1-δ) 0;
        ]),
        b_exp = @SArray([0, 1 - γ, γ]),
        a_imp = @SArray([
            0 0     0;
            0 γ     0;
            0 (1-γ) γ;
        ]),
        b_imp = @SArray([0, 1 - γ, γ]),
    )
end
const ARS343new = let
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

    make_IMEXARKAlgorithm(;
        a_exp = @SArray([
            0   0   0   0;
            γ   0   0   0;
            a31 a32 0   0;
            a41 a42 a43 0;
        ]),
        b_exp = @SArray([0, b1, b2, γ]),
        a_imp = @SArray([
            0 0       0  0;
            0 γ       0  0;
            0 (1-γ)/2 γ  0;
            0 b1      b2 γ;
        ]),
        b_imp = @SArray([0, b1, b2, γ]),
    )
end
const IMKG354a = make_IMEXARKAlgorithm(;
    a_exp = @SArray([
        0      0      0      0      0      0;
        1//5   0      0      0      0      0;
        0      1//5   0      0      0      0;
        0      0      2//3   0      0      0;
        1//3   0      0      1//3   0      0;
        1//4   0      0      0      3//4   0;
    ]),
    a_imp = @SArray([
        0      0      0      0      0      0;
        0      0      0      0      0      0;
        0      0      2//5   0      0      0;
        0      0      11//30 2//5   0      0;
        1//3   0      0      -2//3  1      0;
        1//4   0      0      0      3//4   0;
    ]),
)

# General helper functions
is_increment(f_type) = f_type <: ClimaTimeSteppers.ForwardEulerODEFunction
i_range(a) = 1:size(a, 1)
j_range(a) = 1:size(a, 2)

# Helper functions for increments
first_i(j, a) = findfirst(i -> a[i, j] != 0, i_range(a))
new_js(i, a) = filter(j -> first_i(j, a) == i, j_range(a))
js_to_save(i, a) = filter(
    j -> findlast(i′ -> a[i′, j] != 0, i_range(a)) > i,
    new_js(i, a),
)

# Helper functions for tendencies
has_implicit_step(i, a) = i <= size(a, 2) && a[i, i] != 0
save_tendency(i, a) = 
    !isnothing(findlast(i′ -> a[i′, i] != 0, (i + 1):size(a, 1)))

# Helper functions for increments and tendencies
old_js(i, a, f_type) = filter(
    j -> (
            is_increment(f_type) ?
            !isnothing(first_i(j, a)) && first_i(j, a) < i : j < i
        ) && a[i, j] != 0,
    j_range(a),
)

# TODO: Use the allocate_cache pattern for ODE algorithms as well
struct IMEXARKCache{as, cs, C, N}
    _cache::C
    newtons_method::N
end

function ClimaTimeSteppers.cache(
    prob::DiffEqBase.AbstractODEProblem,
    alg::IMEXARKAlgorithm{as, cs};
    kwargs...
) where {as, cs}
    f_cache(χ, a, f_type) = is_increment(f_type) ?
        map(
            j -> Symbol(:ΔÛ, χ, :_, j) => similar(u),
            Iterators.flatten(map(i -> js_to_save(i, a), i_range(a))),
        ) :
        map(
            i -> Symbol(:f, χ, :_, i) => similar(u),
            filter(i -> save_tendency(i, a), i_range(a)),
        )
    u = prob.u0
    _cache = NamedTuple((
        :U_temp => similar(u),
        map(i -> Symbol(:U, i) => similar(u), i_range(as[1]))..., # or as[2]
        f_cache(:exp, as[1], typeof(prob.f.f2))...,
        f_cache(:imp, as[2], typeof(prob.f.f1))...,
    ))
    newtons_method =
        allocate_cache(alg.newtons_method, u, prob.f.f1.jac_prototype)
    return IMEXARKCache{as, cs, typeof(_cache), typeof(newtons_method)}(
        _cache,
        newtons_method,
    )
end

# Workarounds for not being allowed to use closures in a generated function
struct ImplicitError{F, U, P, T}
    ode_f!::F
    û::U
    p::P
    t::T
    Δt::T
end
struct ImplicitErrorSaveTendency{F, U, P, T}
    ode_f!::F
    ode_f::U
    û::U
    p::P
    t::T
    Δt::T
end
struct ImplicitErrorJacobian{W, P, T}
    Wfact!::W
    p::P
    t::T
    Δt::T
end

(ie::ImplicitError)(f, u) = ie(f, u, ie.ode_f!)
function ((; û, p, t, Δt)::ImplicitError)(f, u, ode_f!::ClimaTimeSteppers.ForwardEulerODEFunction)
    f .= û
    ode_f!(f, u, p, t, Δt)
    f .-= u
end
function ((; û, p, t, Δt)::ImplicitError)(f, u, ode_f!::ODEFunction)
    ode_f!(f, u, p, t)
    f .= û .+ Δt .* f .- u
end
function ((; ode_f!, ode_f, û, p, t, Δt)::ImplicitErrorSaveTendency)(f, u)
    ode_f!(ode_f, u, p, t)
    f .= û .+ Δt .* ode_f .- u
end
((; Wfact!, p, t, Δt)::ImplicitErrorJacobian)(j, u) = Wfact!(j, u, p, Δt, t)

function step_u_expr(
    ::Type{<:IMEXARKCache{as, cs}},
    ::Type{f_exp_type},
    ::Type{f_imp_type},
    ::Type{FT},
) where {as, cs, f_exp_type, f_imp_type, FT}
    function Δu_expr(i, j, χ, a, f_type)
        if is_increment(f_type)
            ΔÛj = :(_cache.$(Symbol(:ΔÛ, χ, :_, j)))
            return :(broadcasted(*, $(FT(a[i, j] / a[first_i(j, a), j])), $ΔÛj))
        else
            fj = :(_cache.$(Symbol(:f, χ, :_, j)))
            return :(broadcasted(*, $(FT(a[i, j])), $fj))
        end
    end
    Δu_exprs(i, χ, a, f_type) =
        map(j -> Δu_expr(i, j, χ, a, f_type), old_js(i, a, f_type))

    χs = (:exp, :imp)
    fs = (:f2, :f1)
    f_types = (f_exp_type, f_imp_type)

    expr = :(
        (; broadcasted, materialize!) = Base;
        (; u, p, t, dt, prob) = integrator;
        (; f) = prob;
        (; f1, f2) = f;
        (; _cache, newtons_method) = cache
    )

    is = i_range(as[1]) # or as[2]
    for i in is
        Ui = i == is[end] ? :u : :(_cache.$(Symbol(:U, i)))
        all_Δu_exprs = (
            Δu_exprs(i, χs[1], as[1], f_types[1])...,
            Δu_exprs(i, χs[2], as[2], f_types[2])...,
        )
        ũi_expr = length(all_Δu_exprs) == 0 ? :u :
            :(broadcasted(+, u, $(all_Δu_exprs...)))
        expr = :($(expr.args...); materialize!($Ui, $ũi_expr))
        for (χ, a, c, f, f_type) in zip(χs, as, cs, fs, f_types)
            if is_increment(f_type)
                for j in new_js(i, a)
                    Ûik_expr = :(
                        t′ = t + dt * $(FT(c[j]));
                        Δt′ = dt * $(FT(a[i, j]))
                    )
                    if j == i
                        Ûik_expr = :(
                            $(Ûik_expr.args...);
                            _cache.U_temp .= $Ui;
                            newtons_method(
                                _cache.U_temp,
                                ImplicitError($f, $Ui, p, t′, Δt′),
                                ImplicitErrorJacobian($f.Wfact, p, t′, Δt′),
                            );
                            $Ui .= _cache.U_temp
                        )
                    else
                        Uj = :(_cache.$(Symbol(:U, j))) # this is why we store Uj
                        Ûik_expr = :(
                            $(Ûik_expr.args...);
                            $f($Ui, $Uj, p, t′, Δt′)
                        )
                    end
                    if j in js_to_save(i, a)
                        ΔÛj = :(_cache.$(Symbol(:ΔÛ, χ, :_, j)))
                        Ûik_expr = :(
                            $ΔÛj .= $Ui;
                            $(Ûik_expr.args...);
                            $ΔÛj .= $Ui .- $ΔÛj
                        )
                    end
                    expr = :($(expr.args...); $(Ûik_expr.args...))
                end
            else
                fi = :(_cache.$(Symbol(:f, χ, :_, i)))
                if has_implicit_step(i, a)
                    implicit_error_expr = save_tendency(i, a) ?
                        :(ImplicitErrorSaveTendency($f, $fi, $Ui, p, t′, Δt′)) :
                        :(ImplicitError($f, $Ui, p, t′, Δt′))
                    expr = :(
                        $(expr.args...);
                        t′ = t + dt * $(FT(c[i]));
                        Δt′ = dt * $(FT(a[i, i]));
                        _cache.U_temp .= $Ui;
                        newtons_method(
                            _cache.U_temp,
                            $implicit_error_expr,
                            ImplicitErrorJacobian($f.Wfact, p, t′, Δt′),
                        );
                        $Ui .= _cache.U_temp
                    )
                elseif save_tendency(i, a)
                    expr = :(
                        $(expr.args...);
                        t′ = t + dt * $(FT(c[i]));
                        $f(fi, $Ui, p, t′);
                    )
                end
            end
        end
    end
    return :($(expr.args...); @info(t); return u)
end

ClimaTimeSteppers.step_u!(integrator, cache::IMEXARKCache) =
    imex_ark_step_u!(
        integrator,
        cache,
        typeof(integrator.prob.f.f2),
        typeof(integrator.prob.f.f1),
        eltype(integrator.prob.u0),
    )
@generated imex_ark_step_u!(
    integrator,
    cache,
    ::Type{f_exp_type},
    ::Type{f_imp_type},
    ::Type{FT},
) where {f_exp_type, f_imp_type, FT} =
    step_u_expr(cache, f_exp_type, f_imp_type, FT)

abstract type ConvergenceCondition end
cache_type(::ConvergenceCondition, ::Type{FT}) where {FT} = Nothing
needs_cache_update(::ConvergenceCondition, iter) = false
# Every subtype of ConvergenceCondition must define
#     has_converged(::ConvergenceCondition, cache, val, err, iter)
# If needs_cache_update is ever true for that subtype, then it must also define
#     updated_cache(::ConvergenceCondition, cache, val, err, iter)
# The values of val and err passed to a ConvergenceCondition will always be ≥ 0.
# Although cache_type can call promote_type to prevent type instability errors,
# this should be avoided to ensure that users write type-stable code.

struct MaximumError{FT} <: ConvergenceCondition
    max_err::FT # should be ≥ 0 to allow has_converged to return true
end
has_converged((; max_err)::MaximumError, cache, val, err, iter) = err <= max_err

struct MaximumRelativeError{FT} <: ConvergenceCondition
    max_rel_err::FT # should be ≥ 0 to allow has_converged to return true
end
has_converged((; max_rel_err)::MaximumRelativeError, cache, val, err, iter) =
    err <= max_rel_err * val

struct MinimumErrorReduction{FT} <: ConvergenceCondition
    min_reduction::FT # should be ≥ 1 to prevent divergence
end
cache_type(::MinimumErrorReduction, ::Type{FT}) where {FT} =
    NamedTuple{(:max_err,), Tuple{FT}}
has_converged(::MinimumErrorReduction, cache, val, err, iter) =
    iter >= 1 && err <= cache.max_err
needs_cache_update(::MinimumErrorReduction, iter) = iter == 0
updated_cache((; min_reduction)::MinimumErrorReduction, cache, val, err, iter) =
    (; max_err = err / min_reduction)

struct MinimumRateOfConvergence{FT, FT2} <: ConvergenceCondition
    rate::FT # should be ≥ 0 to allow has_converged to return true
    order::FT2 # should be ≥ 1 to prevent divergence
end # if order == 1, then rate should be ≤ 1 to prevent divergence
cache_type(::MinimumRateOfConvergence, ::Type{FT}) where {FT} =
    NamedTuple{(:min_err,), Tuple{FT}}
has_converged(::MinimumRateOfConvergence, cache, val, err, iter) =
    iter >= 1 && err >= cache.min_err
needs_cache_update(::MinimumRateOfConvergence, iter) = true
updated_cache((; rate, order)::MinimumRateOfConvergence, cache, val, err, iter) =
    (; min_err = rate * err^order)

struct MultipleConditions{
    CC <: Union{typeof(any), typeof(all)},
    C <: NTuple{N, <:ConvergenceCondition} where {N},
} <: ConvergenceCondition
    condition_combiner::CC
    conditions::C
end
MultipleConditions(
    condition_combiner::Union{typeof(any), typeof(all)} = all,
    conditions::ConvergenceCondition...,
) = MultipleConditions(condition_combiner, conditions)
cache_type((; conditions)::MultipleConditions, ::Type{FT}) where {FT} =
    Tuple{map(condition -> cache_type(condition, FT), conditions)...}
has_converged(mc::MultipleConditions, cache, val, err, iter) =
    mc.condition_combiner(
        i -> has_converged(mc.conditions[i], val, err, cache[i], iter),
        1:length(mc.conditions),
    )
needs_cache_update((; conditions)::MultipleConditions, iter) =
    any(condition -> needs_cache_update(condition, iter), conditions)
updated_cache((; conditions)::MultipleConditions, cache, val, err, iter) =
    map(
        i -> needs_cache_update(conditions[i], iter) ?
            updated_cache(conditions[i], val, err, cache[i], iter) : cache[i],
        1:length(conditions),
    )

# Although we can't pass && or || to a ConvergenceChecker, we can ensure that
# the ConvergenceChecker uses short-circuit evaluation internally.
Base.@kwdef struct ConvergenceChecker{
    GNC <: Union{Nothing, ConvergenceCondition},
    CWC <: Union{Nothing, ConvergenceCondition},
    CC <: Union{typeof(&), typeof(|)},
    N,
    C,
}
    norm_condition::GNC = nothing
    component_condition::CWC = nothing
    condition_combiner::CC = &
    norm::N = LinearAlgebra.norm
    _cache::C = nothing
end
function allocate_cache(convergence_checker::ConvergenceChecker, val_prototype)
    (; norm_condition, component_condition) = convergence_checker
    if isnothing(norm_condition) && isnothing(component_condition)
        error("ConvergenceChecker must have at least one ConvergenceCondition")
    end
    FT = eltype(val_prototype)
    norm_cache = isnothing(norm_condition) ? nothing :
        Ref{cache_type(norm_condition, FT)}()
    component_cache = isnothing(component_condition) ? nothing :
        similar(val_prototype, cache_type(component_condition, FT))
    component_bools = isnothing(component_condition) ? nothing :
        similar(val_prototype, Bool)
    return ConvergenceChecker(
        norm_condition,
        component_condition,
        convergence_checker.condition_combiner,
        convergence_checker.norm,
        (; norm_cache, component_cache, component_bools),
    )
end
function (convergence_checker::ConvergenceChecker)(val, err, iter)
    (; norm_condition, component_condition, condition_combiner, norm, _cache) =
        convergence_checker
    (; norm_cache, component_cache, component_bools) = _cache

    # TODO: The caches don't need to be updated if the sequence has converged.

    function norm_func()
        norm_val = norm(val)
        norm_err = norm(err)
        norm_converged = has_converged(
            norm_condition,
            norm_cache[],
            norm_val,
            norm_err,
            iter,
        )
        if needs_cache_update(norm_condition, iter)
            norm_cache[] = updated_cache(
                norm_condition,
                norm_cache[],
                norm_val,
                norm_err,
                iter,
            )
        end
        return norm_converged
    end

    function component_func()
        # Caching abs.(val) and abs.(err) is probably not worth the overhead.
        @. component_bools = has_converged(
            component_condition,
            component_cache,
            abs(val),
            abs(err),
            iter,
        )
        component_converged = all(component_bools)
        if needs_cache_update(component_condition, iter)
            @. component_cache = updated_cache(
                component_condition,
                component_cache,
                abs(val),
                abs(err),
                iter,
            )
        end
        return component_converged
    end

    both_funcs(::ConvergenceCondition, ::Nothing, _) = norm_func()
    both_funcs(::Nothing, ::ConvergenceCondition, _) = component_func()
    both_funcs(::ConvergenceCondition, ::ConvergenceCondition, ::typeof(&)) =
        norm_func() && component_func()
    both_funcs(::ConvergenceCondition, ::ConvergenceCondition, ::typeof(|)) =
        norm_func() || component_func()

    return both_funcs(norm_condition, component_condition, condition_combiner)
end

Base.@kwdef struct NewtonsMethod{L, CC <: Union{Nothing, ConvergenceChecker}, C}
    linsolve::L
    convergence_checker::CC = nothing
    max_iters::Int = 1
    update_j::Bool = false # this could also be the Jacobian update frequency
    _cache::C = nothing
end

allocate_cache(newtons_method::NewtonsMethod, x_prototype, j_prototype) =
    NewtonsMethod(
        newtons_method.linsolve(Val{:init}, j_prototype, x_prototype), # TODO
        isnothing(newtons_method.convergence_checker) ? nothing :
            allocate_cache(newtons_method.convergence_checker, x_prototype),
        newtons_method.max_iters,
        newtons_method.update_j,
        (;
            Δx = similar(x_prototype),
            f = similar(x_prototype),
            j = similar(j_prototype),
        ),
    )

#=
We want to solve the equation f(x) = 0, given j(x) := ∂f/∂x.
Let x[n] denote the value of x on the n-th Newton iteration, and suppose that
x[n] is converging to some root x′.
A first-order Taylor series expansion tells us that
    f(x′) ≈ f(x[n]) + j(x[n]) * (x′ - x[n])
Since f(x′) = 0, we have that the error on the n-th iteration is roughly
    x[n] - x′ ≈ Δx[n], where Δx[n] := j(x[n]) \ f(x[n])
We can then set x[n+1] to be the value of x′ given by this approximation:
    x[n+1] := x[n] - Δx[n]
If j changes very slowly, we can approximate j(x[n]) ≈ j(x[0]); i.e, we can use
the "chord method".
If a convergence checker is provided, we can use it to determine whether to stop
iterating on iteration n based on the value x[n] and its error Δx[n]; otherwise,
we iterate from n = 0 to the maximum number of iterations.
=#
function (newtons_method::NewtonsMethod)(x, f!, j!)
    (; linsolve, convergence_checker, max_iters, update_j, _cache) =
        newtons_method
    (; Δx, f, j) = _cache
    for iter in 0:max_iters
        iter > 0 && (x .-= Δx)
        isnothing(convergence_checker) && iter == max_iters && break
        (update_j || iter == 0) && j!(j, x)
        f!(f, x)
        linsolve(Δx, j, f)
        if !isnothing(convergence_checker)
            convergence_checker(x, Δx, iter) && break
            iter == max_iters &&
                @warn "Newton's method didn't converge in $max_iters iterations"
        end
    end
end

# TODO: Instead of just passing j! to Newton's method, wrap it in various
# approximations, like ChordMethod, BroydensMethod, BadBroydensMethod, etc.
# Also, allow the Jacobian to be computed once per timestep, rather than once
# per stage.
