using ClimaCore: Geometry, Operators
using ClimaCore.Geometry: ⊗

struct VerticalAdvection{V <: Var, M} <: AbstractTendencyTerm{M}
    var::V
    mode::M
end
VerticalAdvection(var; mode = Implicit(DefaultFluidJacobian())) =
    VerticalAdvection(var, mode)
# TODO: If the default mode depends on var, make `tendency_mode` a function.

cache_reqs(::VerticalAdvection{Var{(:c, :ρ)}}, vars) =
    Var(:f, :ρw) ∉ vars ? (Var(:f, :ρ),) : ()
function (::VerticalAdvection{Var{(:c, :ρ)}})(vars, Y, cache, consts, t)
    ∇◦ᵥc = Operators.DivergenceF2C()
    @lazydots if Var(:f, :ρw) ∈ vars
        @. -∇◦ᵥc(Y.f.ρw)
    else
        @. -∇◦ᵥc(Y.f.w * cache.f.ρ)
    end
end

cache_reqs(::VerticalAdvection{Var{(:c, :ρθ)}}, vars) = ()
function (::VerticalAdvection{Var{(:c, :ρθ)}})(vars, Y, cache, consts, t)
    ∇◦ᵥc = Operators.DivergenceF2C()
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    ) # TODO: Should we do something more clever than simple extrapolation here?
    @lazydots if Var(:f, :ρw) ∈ vars
        ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
        @. -∇◦ᵥc(Y.f.ρw * If(Y.c.ρθ / ρ))
    else
        @. -∇◦ᵥc(Y.f.w * If(Y.c.ρθ))
    end
end

cache_reqs(::VerticalAdvection{Var{(:c, :ρe_tot)}}, vars) = (Var(:c, :P),)
function (::VerticalAdvection{Var{(:c, :ρe_tot)}})(vars, Y, cache, consts, t)
    ∇◦ᵥc = Operators.DivergenceF2C()
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    ) # TODO: Should we do something more clever than simple extrapolation here?
    @lazydots if Var(:f, :ρw) ∈ vars
        ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
        @. -∇◦ᵥc(Y.f.ρw * If((Y.c.ρe_tot + cache.c.P) / ρ))
    else
        @. -∇◦ᵥc(Y.f.w * If(Y.c.ρe_tot + cache.c.P))
    end
end

cache_reqs(::VerticalAdvection{Var{(:f, :ρw)}}, vars) = ()
function (::VerticalAdvection{Var{(:f, :ρw)}})(vars, Y, cache, consts, t)
    ∇◦ᵥf = Operators.DivergenceC2F()
    Ic = Operators.InterpolateF2C()
    ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
    @lazydots @. -∇◦ᵥf(Ic(Y.f.ρw ⊗ Y.f.ρw) / ρ)
end

# TODO: Replace this monkey patch with the curl operator and check that the
# results are identical.
using ClimaCore: RecursiveApply
RecursiveApply.rmul(x::AbstractArray, y::AbstractArray) = x * y
cache_reqs(::VerticalAdvection{Var{(:f, :w)}}, vars) = ()
function (::VerticalAdvection{Var{(:f, :w)}})(vars, Y, cache, consts, t)
    ∇ᵥf = Operators.GradientC2F()
    Ic = Operators.InterpolateF2C()
    @lazydots @. adjoint(∇ᵥf(Ic(Y.f.w))) *
                 Geometry.transform(Geometry.Contravariant3Axis(), Y.f.w)
end

###############################################################################

struct PressureGradient{V <: Var, M} <: AbstractTendencyTerm{M}
    var::V
    mode::M
end
PressureGradient(var; mode = Implicit(DefaultFluidJacobian())) =
    PressureGradient(var, mode)

cache_reqs(::PressureGradient{Var{(:f, :ρw)}}, vars) = (Var(:c, :P),)
function (::PressureGradient{Var{(:f, :ρw)}})(vars, Y, cache, consts, t)
    ∇ᵥf = Operators.GradientC2F()
    @lazydots @. -Geometry.transform(Geometry.WAxis(), ∇ᵥf(cache.c.P))
end

cache_reqs(::PressureGradient{Var{(:f, :w)}}, vars) = (Var(:c, :P), Var(:f, :ρ))
function (::PressureGradient{Var{(:f, :w)}})(vars, Y, cache, consts, t)
    ∇ᵥf = Operators.GradientC2F()
    @lazydots @. -Geometry.transform(Geometry.WAxis(), ∇ᵥf(cache.c.P)) /
                 cache.f.ρ
end

###############################################################################

struct Gravity{V <: Var, M} <: AbstractTendencyTerm{M}
    var::V
    mode::M
end
Gravity(var; mode = Implicit(DefaultFluidJacobian())) = Gravity(var, mode)

cache_reqs(::Gravity{Var{(:f, :ρw)}}, vars) = (Var(:f, :ρ),)
function (::Gravity{Var{(:f, :ρw)}})(vars, Y, cache, consts, t)
    @lazydots @. cache.f.ρ * consts.f.∇Φ
end

cache_reqs(::Gravity{Var{(:f, :w)}}, vars) = ()
(::Gravity{Var{(:f, :w)}})(vars, Y, cache, consts, t) = consts.f.∇Φ
