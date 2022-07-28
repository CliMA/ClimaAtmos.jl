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
    a_χ[i,j] := a_χ[i,j] ∀ i ∈ 1:s, j ∈ 1:s,
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
    all_new_Js_χ := [j ∈ 1:s | isdefined(I_χ[j])]
Next, ∀ j ∈ all_new_Js_χ, let
    K_χ[j] := k ∈ N_χ[I_χ[j]] | new_Js_χ[I_χ[j]][k] == j
We then have that, ∀ j ∈ all_new_Js_χ,
    Û_χ[I_χ[j],K_χ[j]] = Û_χ[I_χ[j],K_χ[j]-1] + Δu_χ[I_χ[j],j]
Since a_χ[I_χ[j],j] != 0, this means that, ∀ j ∈ all_new_Js_χ,
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
We can use our earlier expression to redefine F_χ[j] as, ∀ j ∈ all_new_Js_χ,
    F_χ[j] := (Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1]) / (Δt * a_χ[I_χ[j],j])
We then have that, ∀ i ∈ 1:s+1 and ∀ j ∈ all_new_Js_χ,
    Δu_χ[i,j] = ā_χ[i,j] * ΔÛ_χ[j], where
    ā_χ[i,j] := a_χ[i,j]/a_χ[I_χ[j],j] and
    ΔÛ_χ[j] := Û_χ[I_χ[j],K_χ[j]] - Û_χ[I_χ[j],K_χ[j]-1]
We can then use these values of Δu_χ[i,j] to determine each value of ũ[i].

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
    ũ[i] = Q₀[i] * u + ∑_{χ∈(exp,imp)} ∑_{l=1}^{i-1} ∑_{k=1}^{N_χ[l]} Q_χ[i, l, k] * Û_χ[l, k]
First, we check the base case: ũ[1] = u, so that
    ũ[1] = Q₀[1] * u, where Q₀[1] = 1
Next, we apply the inductive step...
Is this too messy to do in the general case?

Don't forget the register optimization!
=#

function step_u_expr(
    alg::IMEXARKAlgorithm{a_exp, a_imp, b_exp, b_imp, c_exp, c_imp},
) where {a_exp, a_imp, b_exp, b_imp, c_exp, c_imp}
    @assert ndims(a_exp) == ndims(a_imp) == 2 &&
    @assert ndims(b_exp) == ndims(b_imp) == ndims(c_exp) == ndims(c_imp) == 1
    @assert size(a_exp, 1) == size(a_exp, 2) == size(a_imp, 1) == size(a_imp, 2)
    @assert size(b_exp, 1) == size(b_imp, 1) == size(c_exp, 1) == size(c_imp, 1)
    s = size(a_exp, 1)
    @assert all(i -> all(j -> a_exp[i, j] == 0, i:s), 1:s)
    @assert all(i -> all(j -> a_imp[i, j] == 0, (i + 1):s), 1:s)
    is_fsal = b_exp == a_exp[s, :] && b_imp == a_imp[s, :]

    exp = 1
    imp = 2
    χs = (exp, imp)
    a = [vcat(a_exp, b_exp'), vcat(a_imp, b_imp')]
    c = [c_exp, c_imp]

    I = map(χ -> map(j -> findfirst(i -> a[χ][i, j] != 0, 1:(s + 1)), 1:s), χs)
    new_Js = map(χ -> map(i -> findall(isequal(i), I[χ]), 1:(s + 1)), χs)
    old_Js_func(χ, i) = findall(j -> I[χ][j] < i && a[χ][i, j] != 0, 1:s)
    old_Js = map(χ -> map(i -> old_Js_func(χ, i), 1:(s + 1)), χs)
    N = map(χ -> map(length, new_Js[χ]), χs)
    K_func(χ, j) =
        isnothing(I[χ][j]) ? nothing : findfirst(isequal(j), new_Js[χ][I[χ][j]])
    K = map(χ -> map(j -> K_func(χ, j), 1:s), χs)

    function Δus_func(χ, i, j)
        ā = a[χ][i, j] / a[χ][I[χ][j], j]
        ΔÛ = Symbol(:ΔÛ, χ, :_, j)
        return :(broadcasted(*, $ā, $ΔÛ))
    end
    expr = quote
        (; broadcasted, materialize!) = Base
        (; u, p, t) = integrator
        (; U) = cache
    end
    for i in 1:(is_fsal ? s : s + 1)
        Δus = vcat(map(χ -> map(j -> Δus_func(χ, i, j), old_Js[χ][i]), χs)...)
        ũ = length(Δus) == 0 ? :u : :(broadcasted(+, u, $(Δus...)))
        expr = :($expr; materialize!(U, $ũ))
        for k in 1:N[exp][i]

        end
    end
end

Base.@kwdef struct IMEXARKAlgorithm{a_exp, a_imp, c_exp, c_imp, N} <:
        ClimaTimeSteppers.DistributedODEAlgorithm
    nlsolve::N
end
