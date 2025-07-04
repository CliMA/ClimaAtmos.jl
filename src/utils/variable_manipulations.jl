import ClimaCore.MatrixFields: @name

"""
    specific(ρχ, ρ)
    specific(ρaχ, ρa, ρχ, ρ, turbconv_model)

Calculates the specific quantity `χ` (per unit mass) from a density-weighted
quantity. This function uses multiple dispatch to select the appropriate
calculation method based on the number of arguments.

**Grid-Scale Method (2 arguments)**

    specific(ρχ, ρ)

Performs a direct division of the density-weighted quantity `ρχ` by the density
`ρ`. This method is used for grid-mean quantities where the density `ρ` is
well-defined and non-zero.

**SGS Regularized Method (5 arguments)**

    specific(ρaχ, ρa, ρχ, ρ, turbconv_model)

Calculates the specific quantity `χ` for a subgrid-scale (SGS) component by
dividing the density-area-weighted quantity `ρaχ` by the density-area product
`ρa`.

This method includes regularization to handle cases where the SGS area fraction
(and thus `ρa`) is zero or vanishingly small. It performs a linear
interpolation between the SGS specific quantity (`ρaχ / ρa`) and the grid-mean
specific quantity (`ρχ / ρ`). The interpolation weight is computed by
`sgs_weight_function` to ensure a smooth and numerically stable transition,
preventing division by zero. Using this regularized version instead of directly
computing `ρaχ / ρa` breaks the assumption of domain decomposition (sum of SGS
domains equals GS) when the approximated area fraction `a` is small.

Arguments:
- `ρχ`: The grid-mean density-weighted quantity (e.g., `ρe_tot`, `ρq_tot`).
- `ρ`: The grid-mean density.
- `ρaχ`: The density-area-weighted SGS quantity (e.g., `sgs.ρa * sgs.h_tot`).
- `ρa`: The density-area product of the SGS component.
- `ρχ_fallback`: The grid-mean density-weighted quantity used for the fallback value.
- `ρ_fallback`: The grid-mean density used for the fallback value.
- `turbconv_model`: The turbulence convection model, containing parameters for regularization (e.g., `a_half`).
"""
@inline specific(ρχ, ρ) = ρχ / ρ

@inline function specific(ρaχ, ρa, ρχ, ρ, turbconv_model)
    # TODO: Replace turbconv_model struct by parameters, and include a_half in
    # parameters, not in config
    weight = sgs_weight_function(ρa / ρ, turbconv_model.a_half)
    # If ρa is exactly zero, the weight function will be zero, causing the first
    # term to be NaN (0 * ... / 0). The ifelse handles this case explicitly.
    return ρa == 0 ? ρχ / ρ : weight * ρaχ / ρa + (1 - weight) * ρχ / ρ
end

# Internal method that checks if its input is @name(ρχ) for some variable χ.
@generated is_ρ_weighted_name(
    ::MatrixFields.FieldName{name_chain},
) where {name_chain} =
    length(name_chain) == 1 && startswith(string(name_chain[1]), "ρ")

# Internal method that converts @name(ρχ) to @name(χ) for some variable χ.
@generated function specific_tracer_name(
    ::MatrixFields.FieldName{ρχ_name_chain},
) where {ρχ_name_chain}
    χ_symbol = Symbol(string(ρχ_name_chain[1])[(ncodeunits("ρ") + 1):end])
    return :(@name($χ_symbol))
end

"""
    gs_tracer_names(Y)

`Tuple` of `@name`s for the grid-scale tracers in the center field `Y.c`
(excluding `ρ`, `ρe_tot`, velocities, and SGS fields).
"""
gs_tracer_names(Y) =
    unrolled_filter(MatrixFields.top_level_names(Y.c)) do name
        is_ρ_weighted_name(name) && !(name in (@name(ρ), @name(ρe_tot)))
    end

"""
    specific_gs_tracer_names(Y)

`Tuple` of the specific tracer names `@name(χ)` that correspond to the
density-weighted tracer names `@name(ρχ)` in `gs_tracer_names(Y)`.
"""
specific_gs_tracer_names(Y) =
    unrolled_map(specific_tracer_name, gs_tracer_names(Y))

sedimenting_specific_sgs_tracer_names() =
  (@name(q_liq), @name(q_ice), @name(q_rai), @name(q_sno))

"""
    ᶜempty(Y)

Lazy center `Field` of empty `NamedTuple`s.
"""
ᶜempty(Y) = lazy.(Returns((;)).(Y.c))

"""
    ᶜgs_tracers(Y)

Lazy center `Field` of `NamedTuple`s that contain the values of all grid-scale
tracers given by `gs_tracer_names(Y)`.
"""
function ᶜgs_tracers(Y)
    isempty(gs_tracer_names(Y)) && return ᶜempty(Y)
    ρχ_symbols = unrolled_map(MatrixFields.extract_first, gs_tracer_names(Y))
    ρχ_fields = unrolled_map(gs_tracer_names(Y)) do ρχ_name
        MatrixFields.get_field(Y.c, ρχ_name)
    end
    return @. lazy(NamedTuple{ρχ_symbols}(tuple(ρχ_fields...)))
end

"""
    ᶜspecific_gs_tracers(Y)

Lazy center `Field` of `NamedTuple`s that contain the values of all specific
grid-scale tracers given by `specific_gs_tracer_names(Y)`.
"""
function ᶜspecific_gs_tracers(Y)
    isempty(gs_tracer_names(Y)) && return ᶜempty(Y)
    χ_symbols =
        unrolled_map(MatrixFields.extract_first, specific_gs_tracer_names(Y))
    χ_fields = unrolled_map(gs_tracer_names(Y)) do ρχ_name
        ρχ_field = MatrixFields.get_field(Y.c, ρχ_name)
        @. lazy(specific(ρχ_field, Y.c.ρ))
    end
    return @. lazy(NamedTuple{χ_symbols}(tuple(χ_fields...)))
end

"""
    foreach_gs_tracer(f, Y_or_similar_values...)

Applies a function `f` to each grid-scale tracer in the state `Y` or any similar
value like the tendency `Yₜ`. This is used to implement performant loops over
all tracers given by `gs_tracer_names(Y)`.

Although the first input value needs to be similar to `Y`, the remaining values
can also be center `Field`s similar to `Y.c`, and they can use specific tracers
given by `specific_gs_tracer_names(Y)` instead of density-weighted tracers.

Arguments:

- `f`: The function applied to each grid-scale tracer, which must have the
  signature `f(ρχ_or_χ_fields..., ρχ_name)`, where `ρχ_or_χ_fields` are
  grid-scale tracer subfields (either density-weighted or specific) and
  `ρχ_name` is the `MatrixFields.FieldName` of the tracer.
- `Y_or_similar_values`: The state `Y` or similar values like the tendency `Yₜ`.

# Examples

```julia
foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
    ᶜρχₜ .+= tendency_of_ρχ(ᶜρχ)
    if ρχ_name == @name(ρq_tot)
        ᶜρχₜ .+= additional_tendency_of_ρq_tot(ᶜρχ)
    end
end
```

```julia
foreach_gs_tracer(Yₜ, Base.materialize(ᶜspecific_gs_tracers(Y))) do ᶜρχₜ, ᶜχ, ρχ_name
    ᶜρχₜ .+= Y.c.ρ .* tendency_of_χ(ᶜχ)
    if ρχ_name == @name(ρq_tot)
        ᶜρχₜ .+= Y.c.ρ .* additional_tendency_of_q_tot(ᶜχ)
    end
end
```
"""
foreach_gs_tracer(f::F, Y_or_similar_values...) where {F} =
    unrolled_foreach(gs_tracer_names(Y_or_similar_values[1])) do ρχ_name
        ρχ_or_χ_fields = unrolled_map(Y_or_similar_values) do value
            field = value isa Fields.FieldVector ? value.c : value
            ρχ_or_χ_name =
                MatrixFields.has_field(field, ρχ_name) ? ρχ_name :
                specific_tracer_name(ρχ_name)
            MatrixFields.get_field(field, ρχ_or_χ_name)
        end
        f(ρχ_or_χ_fields..., ρχ_name)
    end
foreach_sgs_tracer(f::F, Y_or_similar_values...) where {F} =
    unrolled_foreach(sedimenting_specific_sgs_tracer_names()) do χ_name
        χ_fields = unrolled_map(Y_or_similar_values) do value
            field = value isa Fields.FieldVector ? value.c.sgsʲs.:1 : value
            MatrixFields.get_field(field, χ_name)
        end
        f(χ_fields..., χ_name)
    end

"""
    sgs_weight_function(a, a_half)

Computes the weight of the SGS variables in the linear interpolation used in
`divide_by_ρa`. This is a continuously differentiable and monotonically
increasing function of `a` that is equal to 0 when `a ≤ 0`, is equal to 1 when
`a ≥ 1`, is equal to `1 / 2` when `a = a_half`, grows very rapidly near
`a = a_half`, and grows very slowly at all other values of `a`.  If `a_half` is
sufficiently small, this function is essentially equal to 1 for all `a` more
than a few times larger than `a_half` (up to floating-point precision).

We will now provide a description of how this function was constructed. We need
the function to be equal to 0 when `a ≤ 0` and equal to 1 when `a ≥ 1`. Since
the function must also be continuously differentiable, its derivative at these
values of `a` has to be 0. To obtain a function with these properties, we use a
piecewise definition:
    - For all `a < 0`, the function is equal to 0.
    - For all `a > 1`, the function is equal to 1.
    - For all `0 ≤ a ≤ 1`, the function is a sigmoid that connects the point
      `(0, 0)` to the point `(1, 1)`, with a derivative of 0 at these points.
Most well-known sigmoid functions connect the "points" `(-Inf, 0)` and
`(Inf, 1)`, not `(0, 0)` and `(1, 1)`. To obtain the desired sigmoid curve, we
begin with two simple sigmoid functions that go from `(-Inf, 0)` to `(Inf, 1)`
at different rates. In this case, we use two `tanh` functions, scaled and
translated so that they lie between 0 and 1:
    - `fast_sigmoid(a) = (1 + tanh(a)) / 2` and
    - `slow_sigmoid(a) = (1 + tanh(a / 2)) / 2`.
Note that the second sigmoid is commonly called the "logistic" function. We then
take the inverse of the sigmoid that grows more slowly, and we make that the
input of the sigmoid that grows more quickly:
    - `sigmoid(a) = fast_sigmoid(slow_sigmoid⁻¹(a)) =
       (1 + tanh(2 * atanh(2 * a - 1))) / 2`.
The resulting function goes from `(0, 0)` to `(1, 1)`, and, since the outer
sigmoid grows more quickly, it has the same asymptotic behavior as the outer
sigmoid, which means that its derivative at the boundary points is 0. If we had
instead put the sigmoid that grows more slowly on the outside, the asymptotic
behavior would come from the inverted inner sigmoid, which means that the
derivative at the boundary points would be `Inf`.

The sigmoid function we have constructed reaches `1 / 2` when `a = 1 / 2`. More
generally, we need the weight function to reach `1 / 2` when `a` is some small
value `a_half`. To achieve this, we replace the input to the sigmoid function
with a smooth, monotonically increasing function that goes through `(0, 0)`,
`(a_half, 1 / 2)`, and `(1, 1)`. The simplest option is the power function
    - `power(a) = a^(-1 / log2(a_half))`.
However, this function does not work well because, when `a_half < 1 / 2`, its
derivative at `a = 0` is `Inf`, and its derivative at `a = 1` is some positive
number. Making this power function the input to the sigmoid function causes the
derivative of the sigmoid to become `Inf` at `a = 0` when `a_half < 1 / 4`, and
it causes the sigmoid to grow too slowly from `1 / 2` to 1, only reaching 1 when
`a` is significantly larger than `a_half`. In order to fix this, we transform
the power function by replacing `a` with `1 - a`, `a_half` with `1 - a_half`,
and `power(a)` with `1 - power(a)`, which gives us
    - `power(a) = 1 - (1 - a)^(-1 / log2(1 - a_half))`.
This transformed function works better because, when `a_half < 1 / 2`, its
derivative at `a = 0` is some positive number, and its derivative at `a = 1` is
0. When we make this the input to the sigmoid function, the result has a
continuous derivative and is essentially equal to 1 for all `a` more than a few
times larger than `a_half`. So, for all `0 ≤ a ≤ 1`, we define the weight
function as
    - `weight(a) = sigmoid(power(a)) =
       (1 + tanh(2 * atanh(1 - 2 * (1 - a)^(-1 / log2(1 - a_half))))) / 2`.
"""
sgs_weight_function(a, a_half) =
    if a <= 0 # autodiff generates NaNs when a is 0
        zero(a)
    elseif a > min(1, 42 * a_half) # autodiff generates NaNs when a is large
        one(a)
    else
        (1 + tanh(2 * atanh(1 - 2 * (1 - a)^(-1 / log2(1 - a_half))))) / 2
    end

"""
    divide_by_ρa(ρaχ, ρa, ρχ, ρ, turbconv_model)

Computes `ρaχ / ρa`, regularizing the result to avoid issues when `a` is small.
This is done by performing a linear interpolation from `ρaχ / ρa` to `ρχ / ρ`,
using `sgs_weight_function(ρa / ρ, turbconv_model.a_half)` as the weight of
`ρaχ / ρa` in the interpolation. Note that `ρa / ρ` is the "anelastic
approximation" of `a`; we cannot directly use `a` to compute the weight because
this function needs to be called before `a` has been computed. Also, note that
using this function instead of directly computing `ρaχ / ρa` breaks the
assumption of domain decomposition when the approximated `a` is small.
"""
function divide_by_ρa(ρaχ, ρa, ρχ, ρ, turbconv_model)
    weight = sgs_weight_function(ρa / ρ, turbconv_model.a_half)
    # If ρa = 0, we know that ρa / ρ = 0, which means that weight = 0. However,
    # 0 * ρaχ / 0 = NaN, regardless of what ρaχ is, so the linear interpolation
    # will always return NaN when ρa = 0. To avoid this problem, we need to add
    # a special case for ρa = 0.
    return ρa == 0 ? ρχ / ρ : weight * ρaχ / ρa + (1 - weight) * ρχ / ρ
end

"""
    ρa⁺(gs)

Computes the total mass-flux subdomain area-weighted density, assuming that the
mass-flux subdomain states are stored in `gs.sgsʲs`.
"""
ρa⁺(gs) = mapreduce_with_init(sgsʲ -> sgsʲ.ρa, +, gs.sgsʲs)

"""
    ρah_tot⁺(sgsʲs)

Computes the total mass-flux subdomain area-weighted ρh_tot, assuming that the
mass-flux subdomain states are stored in `sgsʲs`.
"""
ρah_tot⁺(sgsʲs) = mapreduce_with_init(sgsʲ -> sgsʲ.ρa * sgsʲ.h_tot, +, sgsʲs)

"""
    ρamse⁺(sgsʲs)

Computes the total mass-flux subdomain area-weighted ρmse, assuming that the
mass-flux subdomain states are stored in `sgsʲs`.
"""
ρamse⁺(sgsʲs) = mapreduce_with_init(sgsʲ -> sgsʲ.ρa * sgsʲ.mse, +, sgsʲs)

"""
    ρaq_tot⁺(sgsʲs)

Computes the total mass-flux subdomain area-weighted ρq_tot, assuming that the
mass-flux subdomain states are stored in `sgsʲs`.
"""
ρaq_tot⁺(sgsʲs) = mapreduce_with_init(sgsʲ -> sgsʲ.ρa * sgsʲ.q_tot, +, sgsʲs)

"""
    ρaq_liq⁺(sgsʲs)

Computes the liquid water mass-flux subdomain area-weighted ρq_liq, assuming that the
mass-flux subdomain states are stored in `sgsʲs`.
"""
ρaq_liq⁺(sgsʲs) = mapreduce_with_init(sgsʲ -> sgsʲ.ρa * sgsʲ.q_liq, +, sgsʲs)

"""
    ρaq_ice⁺(sgsʲs)

Computes the ice water  mass-flux subdomain area-weighted ρq_ice, assuming that the
mass-flux subdomain states are stored in `sgsʲs`.
"""
ρaq_ice⁺(sgsʲs) = mapreduce_with_init(sgsʲ -> sgsʲ.ρa * sgsʲ.q_ice, +, sgsʲs)

"""
    ρaq_rai⁺(sgsʲs)

Computes the rain mass-flux subdomain area-weighted ρq_rai, assuming that the
mass-flux subdomain states are stored in `sgsʲs`.
"""
ρaq_rai⁺(sgsʲs) = mapreduce_with_init(sgsʲ -> sgsʲ.ρa * sgsʲ.q_rai, +, sgsʲs)

"""
    ρaq_sno⁺(sgsʲs)

Computes the snow mass-flux subdomain area-weighted ρq_sno, assuming that the
mass-flux subdomain states are stored in `sgsʲs`.
"""
ρaq_sno⁺(sgsʲs) = mapreduce_with_init(sgsʲ -> sgsʲ.ρa * sgsʲ.q_sno, +, sgsʲs)

"""
    ρa⁰(gs)

Computes the environment area-weighted density, assuming that the mass-flux
subdomain states are stored in `gs.sgsʲs`.
"""
ρa⁰(gs) = gs.ρ - mapreduce_with_init(sgsʲ -> sgsʲ.ρa, +, gs.sgsʲs)

"""
    u₃⁺(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model)

Computes the average mass-flux subdomain vertical velocity `u₃⁺` by dividing the
total momentum `ρaw⁺` by the total area-weighted density `ρa⁺`, both of which
are computed from the tuples of subdomain densities and velocities `ρaʲs` and
`u₃ʲs`. The division is computed using `divide_by_ρa` to avoid issues when `a⁺`
is small.
"""
u₃⁺(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model) = divide_by_ρa(
    unrolled_dotproduct(ρaʲs, u₃ʲs),
    reduce(+, ρaʲs),
    ρ * u₃,
    ρ,
    turbconv_model,
)

"""
    u₃⁰(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model)

Computes the environment vertical velocity `u₃⁰` by dividing the environment
momentum `ρaw⁰` by the environment area-weighted density `ρa⁰`, both of which
are computed from the domain decomposition of the grid-scale quantities `ρw` and
`ρ` into the mass-flux subdomain quantities `ρawʲs` and `ρaʲs` and the
environment quantities. The division is computed using `divide_by_ρa` to avoid
issues when `a⁰` is small.
"""
u₃⁰(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model) = divide_by_ρa(
    ρ * u₃ - unrolled_dotproduct(ρaʲs, u₃ʲs),
    ρ - reduce(+, ρaʲs),
    ρ * u₃,
    ρ,
    turbconv_model,
)

import ClimaCore.RecursiveApply: ⊞, ⊠, rzero, rpromote_type
function mapreduce_with_init(f, op, iter...)
    r₀ = rzero(rpromote_type(typeof(f(map(first, iter)...))))
    mapreduce(f, op, iter...; init = r₀)
end

# Inference fails for certain mapreduce calls inside cuda
# kernels, so let's define a recursive unrolled dot product:
promote_type_mul(n::Number, x::Geometry.AxisTensor) = typeof(x)
promote_type_mul(x::Geometry.AxisTensor, n::Number) = typeof(x)
@inline function unrolled_dotproduct(a::Tuple, b::Tuple)
    r = rzero(promote_type_mul(first(a), first(b)))
    unrolled_dotproduct(r, a, b)
end
@inline unrolled_dotproduct(s, ::Tuple{}, ::Tuple{}) = s
@inline unrolled_dotproduct(s, a::Tuple, b::Tuple) =
    s ⊞ (first(a) ⊠ first(b)) ⊞
    unrolled_dotproduct(s, Base.tail(a), Base.tail(b))
@inline unrolled_dotproduct(s, a::Tuple{<:Any}, b::Tuple{<:Any}) =
    s ⊞ (first(a) ⊠ first(b))
