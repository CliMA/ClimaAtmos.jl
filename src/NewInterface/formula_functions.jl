using ClimaCore: Operators
using LinearAlgebra: norm_sqr

struct DefaultFluidFunction{V <: Var} <: AbstractFormulaFunction
    var::V
end

cache_reqs(::DefaultFluidFunction{Var{(:f, :ρ)}}, vars) = ()
function (::DefaultFluidFunction{Var{(:f, :ρ)}})(vars, Y, cache, consts, t)
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    ) # TODO: Should we do something more clever than simple extrapolation here?
    @lazydots @. If(Y.c.ρ)
end

# TODO: Should this be NaN when ρθ is negative, or should it throw an error?
cache_reqs(::DefaultFluidFunction{Var{(:c, :P)}}, vars) = ()
function (::DefaultFluidFunction{Var{(:c, :P)}})(vars, Y, cache, consts, t)
    @unpack γ = consts
    T = typeof(γ)
    @lazydots if Var(:c, :ρθ) ∈ vars
        @unpack R_d, p_0 = consts
        factor = p_0 * (R_d / p_0)^γ
        ρθ = @. ifelse(Y.c.ρθ < $T(0), $T(NaN), Y.c.ρθ)
        @. factor * ρθ^γ
    else
        Ic = Operators.InterpolateF2C()
        factor = γ - T(1)
        ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
        if Var(:f, :ρw) ∈ vars
            ρu_sqr = @. Ic(norm_sqr(Y.f.ρw))
            @. factor * (Y.c.ρe_tot - ρ * consts.c.Φ - ρu_sqr / ($T(2) * ρ))
        else
            u_sqr = @. Ic(norm_sqr(Y.f.w))
            @. factor * (Y.c.ρe_tot - ρ * (consts.c.Φ + u_sqr / $T(2)))
        end
    end
end

cache_reqs(::DefaultFluidFunction{Var{(:c, :ρe_tot)}}, vars) = ()
function (::DefaultFluidFunction{Var{(:c, :ρe_tot)}})(vars, Y, cache, consts, t)
    @lazydots if Var(:c, :ρθ) ∈ vars
        Ic = Operators.InterpolateF2C()
        @unpack γ, R_d, p_0 = consts
        T = typeof(γ)
        factor = p_0 * (R_d / p_0)^γ / (γ - T(1))
        ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
        ρθ = @. ifelse(Y.c.ρθ < $T(0), $T(NaN), Y.c.ρθ)
        if Var(:f, :ρw) ∈ vars
            ρu_sqr = @. Ic(norm_sqr(Y.f.ρw))
            @. factor * ρθ^γ + ρ * consts.c.Φ + ρu_sqr / ($T(2) * ρ)
        else
            u_sqr = @. Ic(norm_sqr(Y.f.w))
            @. factor * ρθ^γ + ρ * (consts.c.Φ + u_sqr / $T(2))
        end
    else
        Y.c.ρe_tot
    end
end
