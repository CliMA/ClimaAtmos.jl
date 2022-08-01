using ClimaCore: Geometry, Operators
using ClimaCore.Geometry: ⊗

const Iᶠ = Operators.InterpolateC2F()
const Iᶜ = Operators.InterpolateF2C()
const ∇ᵥᶠ = Operators.GradientC2F()
const ∇ᵥᶠ◦ = Operators.DivergenceC2F()
const ∇ᵥᶜ◦ = Operators.DivergenceF2C()

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
    @lazydots if Var(:f, :ρw) ∈ vars
        @. -∇ᵥᶜ◦(Y.f.ρw)
    else
        @. -∇ᵥᶜ◦(Y.f.w * cache.f.ρ)
    end
end

cache_reqs(::VerticalAdvection{Var{(:c, :ρθ)}}, vars) = ()
function (::VerticalAdvection{Var{(:c, :ρθ)}})(vars, Y, cache, consts, t)
    @lazydots if Var(:f, :ρw) ∈ vars
        ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
        @. -∇ᵥᶜ◦(Y.f.ρw * Iᶠ(Y.c.ρθ / ρ))
    else
        @. -∇ᵥᶜ◦(Y.f.w * Iᶠ(Y.c.ρθ))
    end
end

cache_reqs(::VerticalAdvection{Var{(:c, :ρe_tot)}}, vars) = (Var(:c, :P),)
function (::VerticalAdvection{Var{(:c, :ρe_tot)}})(vars, Y, cache, consts, t)
    @lazydots if Var(:f, :ρw) ∈ vars
        ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
        @. -∇ᵥᶜ◦(Y.f.ρw * Iᶠ((Y.c.ρe_tot + cache.c.P) / ρ))
    else
        @. -∇ᵥᶜ◦(Y.f.w * Iᶠ(Y.c.ρe_tot + cache.c.P))
    end
end

cache_reqs(::VerticalAdvection{Var{(:f, :ρw)}}, vars) = ()
function (::VerticalAdvection{Var{(:f, :ρw)}})(vars, Y, cache, consts, t)
    ρ = Var(:c, :ρ) ∈ vars ? Y.c.ρ : consts.c.ρ
    @lazydots @. -∇ᵥᶠ◦(Iᶜ(Y.f.ρw ⊗ Y.f.ρw) / ρ)
end

# TODO: Replace this monkey patch with the curl operator and check that the
# results are identical.
using ClimaCore: RecursiveApply
RecursiveApply.rmul(x::AbstractArray, y::AbstractArray) = x * y
cache_reqs(::VerticalAdvection{Var{(:f, :w)}}, vars) = ()
function (::VerticalAdvection{Var{(:f, :w)}})(vars, Y, cache, consts, t)
    @lazydots @. adjoint(∇ᵥᶠ(Iᶜ(Y.f.w))) * Geometry.Contravariant3Vector(Y.f.w)
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
    @lazydots @. -Geometry.WVector(∇ᵥᶠ(cache.c.P))
end

cache_reqs(::PressureGradient{Var{(:f, :w)}}, vars) = (Var(:c, :P), Var(:f, :ρ))
function (::PressureGradient{Var{(:f, :w)}})(vars, Y, cache, consts, t)
    @lazydots @. -Geometry.WVector(∇ᵥᶠ(cache.c.P)) / cache.f.ρ
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
