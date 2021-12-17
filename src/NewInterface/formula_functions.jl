struct DefaultFluidFunction{V <: Var} <: AbstractFormulaFunction
    var::V
end

cache_reqs(::DefaultFluidFunction{Var{(:f, :ρ)}}, vars) = ()
function (::DefaultFluidFunction{Var{(:f, :ρ)}})(vars, Y, cache, consts, t)
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    @lazydots @. If(Y.c.ρ)
end

cache_reqs(::DefaultFluidFunction{Var{(:c, :P)}}, vars) = ()
function (::DefaultFluidFunction{Var{(:c, :P)}})(vars, Y, cache, consts, t)
    @lazydots if Var(:c, :ρθ) ∈ vars
        @unpack MSLP, R_d, γ = consts
        factor = MSLP * (R_d / MSLP)^γ
        @. factor * Y.c.ρθ^γ
    else
        Ic = Operators.InterpolateF2C()
        T = typeof(consts.γ)
        factor = consts.γ - T(1)
        ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
        if Var(:f, :ρw) ∈ vars
            @. factor *
               (Y.c.ρe_tot - ρ * consts.c.Φ - Ic(Y.f.ρw^2) / ($T(2) * ρ))
        else
            @. factor * (Y.c.ρe_tot - ρ * (consts.c.Φ + Ic(Y.f.w^2) / $T(2)))
        end
    end
end
