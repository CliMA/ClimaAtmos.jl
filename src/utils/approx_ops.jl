#####
##### Approximate `Float32` division, reciprocal, square root and power
#####
#
# A copy of `fast_div`, `fast_pow` (here `approx_pow`, since ClimaAtmos has its own
# `fast_pow`), `fast_sqrt` and `@approx_ops` of
# CloudMicrophysics.jl (`src/Utilities.jl`), extended by the reciprocal. On a GPU
# the correctly rounded `Float32` division is evaluated with a check that calls a
# slow subroutine whenever an operand or the result is zero, subnormal or
# infinite, which is the common case in the SGS node construction (infinite
# thresholds, zero numerators). The approximate operations have no such path.

"""
    fast_div(a, b)

`a / b`, which for `Float32` carries the LLVM fast-math flag `afn`
("approximate functions").

On a CPU the flag changes nothing: the division stays the correctly rounded IEEE
division. On a GPU it selects the approximate division of the hardware (PTX
`div.approx.f32`, error up to 2 ulp for a divisor of magnitude between `2^-126`
and `2^126`; a finite divisor above `2^126`, about `8.5e37`, gives `0`, or `NaN`
for an infinite numerator). None of the value-changing fast-math assumptions (no
`NaN`, no `Inf`, reassociation) is made, so infinite thresholds and `ifelse`
guards keep their meaning. Every other number type uses `/`.
"""
@inline fast_div(a::Float32, b::Float32) = Base.llvmcall(
    "%3 = fdiv afn float %0, %1\nret float %3",
    Float32, Tuple{Float32, Float32}, a, b,
)
@inline fast_div(a::Real, b::Real) = _fast_div(promote(a, b)...)
@inline fast_div(a, b) = a / b
@inline _fast_div(a::Float32, b::Float32) = fast_div(a, b)
@inline _fast_div(a, b) = a / b

"""
    fast_inv(x)

`inv(x)`, for `Float32` as `fast_div(1, x)`. Every other number type uses `inv`.
"""
@inline fast_inv(x::Float32) = fast_div(one(x), x)
@inline fast_inv(x) = inv(x)

"""
    approx_pow(x, y)

`x^y` for a base `x ≥ 0`, evaluated for two `Float32` arguments with
`Base.FastMath.pow_fast`: on a GPU `exp2(y * log2(x))` in single precision, in
place of the `Float32` `^` of CUDA, which is evaluated in double precision.
Every other combination of types uses `^`.
"""
@inline approx_pow(x::Float32, y::Float32) = Base.FastMath.pow_fast(x, y)
@inline approx_pow(x, y) = x^y

"""
    fast_sqrt(x)

`sqrt(x)` for `x ≥ 0`. For `Float32` it is the LLVM square root with the
fast-math flag `afn`, without the domain check of `Base.sqrt`: a negative
argument gives `NaN` instead of an error. On a CPU the result is the correctly
rounded square root; on a GPU it is the approximate one of the hardware (PTX
`sqrt.approx.f32`). Every other number type uses `sqrt`.
"""
@inline fast_sqrt(x::Float32) = Base.llvmcall(
    (
        """
        declare float @llvm.sqrt.f32(float)
        define float @entry(float %0) alwaysinline {
            %2 = call afn float @llvm.sqrt.f32(float %0)
            ret float %2
        }
        """,
        "entry",
    ), Float32, Tuple{Float32}, x)
@inline fast_sqrt(x) = sqrt(x)

"""
    @approx_ops expr

Replace, in `expr`, every `a / b` by [`fast_div`](@ref)`(a, b)`, every `inv(x)`
by [`fast_inv`](@ref)`(x)`, every `sqrt(x)` by [`fast_sqrt`](@ref)`(x)` and every
`x^y` whose exponent is not an integer literal by [`approx_pow`](@ref)`(x, y)`.

The macro is applied to the definitions of functions evaluated at every grid
point on a GPU. It changes `Float32` evaluations only, and on a CPU only through
`approx_pow` and the missing domain check of `fast_sqrt`. Unlike `@fastmath` it
assumes nothing about `NaN` or `Inf` and does not reorder operations. Functions
called from `expr` are not rewritten; each needs the macro itself.
"""
macro approx_ops(expr)
    return esc(_approx_ops(expr))
end
_approx_ops(x) = x
function _approx_ops(ex::Expr)
    args = map(_approx_ops, ex.args)
    if ex.head === :call && length(args) == 3 && args[1] === :/
        return Expr(:call, fast_div, args[2], args[3])
    elseif ex.head === :call && length(args) == 3 && args[1] === :^ &&
           !(args[3] isa Integer)
        return Expr(:call, approx_pow, args[2], args[3])
    elseif ex.head === :call && length(args) == 2 && args[1] === :sqrt
        return Expr(:call, fast_sqrt, args[2])
    elseif ex.head === :call && length(args) == 2 && args[1] === :inv
        return Expr(:call, fast_inv, args[2])
    end
    return Expr(ex.head, args...)
end
