import ClimaCore.MatrixFields: @name
import ClimaCore.RecursiveApply: ⊞, ⊠, rzero, rpromote_type

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
            field = value isa Fields.Field ? value : value.c
            ρχ_or_χ_name =
                MatrixFields.has_field(field, ρχ_name) ? ρχ_name :
                specific_tracer_name(ρχ_name)
            MatrixFields.get_field(field, ρχ_or_χ_name)
        end
        f(ρχ_or_χ_fields..., ρχ_name)
    end

"""
    specific(ρχ, ρ)
    specific(ρaχ, ρa, ρχ, ρ, turbconv_model)

Calculates the specific quantity `χ` (per unit mass) from a density-weighted
quantity. This function uses multiple dispatch to select the appropriate
calculation method based on the number of arguments.

**Grid-Scale Method (2 arguments)**

    specific(ρχ, ρ)

Performs a direct division of the density-weighted quantity `ρχ` by the density `ρ`.
This method is used for grid-mean quantities where the density `ρ` is well-defined
and non-zero.

**SGS Regularized Method (5 arguments)**

    specific(ρaχ, ρa, ρχ, ρ, turbconv_model)

Calculates the specific quantity `χ` for a subgrid-scale (SGS) component by
dividing the density-area-weighted quantity `ρaχ` by the density-area
product `ρa`.

This method includes regularization to handle cases where the SGS area fraction
(and thus `ρa`) is zero or vanishingly small. It performs a linear interpolation
between the SGS specific quantity (`ρaχ / ρa`) and the grid-mean specific
quantity (`ρχ / ρ`). The interpolation weight is computed by `sgs_weight_function`
to ensure a smooth and numerically stable transition, preventing division by zero.
Using this regularized version instead of directly computing `ρaχ / ρa` breaks the
assumption of domain decomposition (sum of SGS domains equals GS) when the approximated 
area fraction `a` is small.

Arguments:
- `ρaχ`: The density-area-weighted SGS quantity (e.g., `sgs.ρa * sgs.h_tot`).
- `ρa`: The density-area product of the SGS component.
- `ρχ`: The fallback grid-mean density-weighted quantity (e.g., `ρe_tot`, `ρq_tot`).
- `ρ`: The fallback grid-mean density.
- `turbconv_model`: The turbulence convection model, containing parameters for regularization (e.g., `a_half`).
"""
function specific(ρχ, ρ)
    return ρχ / ρ 
end

function specific(ρaχ, ρa, ρχ, ρ, turbconv_model)
    # TODO: Replace turbconv_model struct by parameters, and include a_half in 
    # parameters, not in config
    weight = sgs_weight_function(ρa / ρ, turbconv_model.a_half)
    # If ρa is exactly zero, the weight function will be zero, causing the first
    # term to be NaN (0 * ... / 0). The ifelse handles this case explicitly.
    return ρa == 0 ? ρχ / ρ : weight * ρaχ / ρa + (1 - weight) * ρχ / ρ
end

 """
    sgs_weight_function(a, a_half)

Computes a smooth, monotonic weight function `w(a)` that ranges from 0 to 1.

This function is used as the interpolation weight in the regularized `specific`
function. It ensures a numerically stable and smooth transition between a subgrid-scale 
(SGS) quantity and its grid-mean counterpart, especially when the SGS area fraction `a` 
is small.

**Key Properties:**
- `w(a) = 0` for `a ≤ 0`.
- `w(a) = 1` for `a ≥ 1`.
- `w(a_half) = 0.5`.
- The function is continuously differentiable, with derivatives equal to zero at
  `a = 0` and `a = 1`, which ensures smooth blending.
- The functions grows very rapidly near `a = a_half`, and grows very slowly at all other 
  values of `a`.
- For small `a_half`, the weight rapidly approaches 1 for values of `a` that are
  a few times larger than `a_half`.

**Construction Method:**
The function is piecewise. For `a` between 0 and 1, it is a custom sigmoid curve
constructed in two main steps to satisfy the key properties:
1.  **Bounded Sigmoid Creation**: A base sigmoid is created that maps the interval
    `(0, 1)` to `(0, 1)` with zero derivatives at the endpoints. This is achieved
    by composing a standard `tanh` function with the inverse of a slower-growing
    `tanh` function.
2.  **Midpoint Control**: To ensure the function passes through the control point
    `(a_half, 0.5)`, the input `a` is first transformed by a specially designed
    power function (`1 - (1 - a)^k`) before being passed to the bounded sigmoid.
    This transformation maps `a_half` to `0.5` while preserving differentiability 
    at the boundaries.

Arguments:
- `a`: The input SGS area fraction (often approximated as `ρa / ρ`).
- `a_half`: The value of `a` at which the weight function should be 0.5, controlling
          the transition point of the sigmoid curve.

Returns:
- The computed weight, a value between 0 and 1.
"""
function sgs_weight_function(a, a_half)
    if a < 0
        zero(a)
    elseif a > min(1, 42 * a_half) # autodiff generates NaNs when a is large
        one(a)
    else
        (1 + tanh(2 * atanh(1 - 2 * (1 - a)^(-1 / log2(1 - a_half))))) / 2
    end
end

"""
    draft_sum(f, sgsʲs)

Computes the sum of a function `f` applied to each draft subdomain
state `sgsʲ` in the iterator `sgsʲs`.

Arguments:
- `f`: A function to apply to each element of `sgsʲs`.
- `sgsʲs`: An iterator over the draft subdomain states.
"""
draft_sum(f, sgsʲs) = mapreduce_with_init(f, +, sgsʲs)

"""
    env_value(grid_scale_value, f_draft, gs)    

Computes the value of a quantity `ρaχ` in the environment subdomain by subtracting 
the sum of its values in all draft subdomains from the grid-scale value. 

This is based on the domain decomposition principle for density-area weighted 
quantities: `GridMean(ρχ) = Env(ρaχ) + Sum(Drafts(ρaχ))`.

Arguments:
- `grid_scale_value`: The `ρa`-weighted grid-scale value of the quantity.
- `f_draft`: A function that extracts the corresponding value from a draft subdomain state.
- `gs`: The grid-scale state, which contains the draft subdomain states `gs.sgsʲs`.
"""
function env_value(grid_scale_value, f_draft, gs)
    return grid_scale_value - draft_sum(f_draft, gs.sgsʲs)
end

"""
    specific_env_value(χ_name::Symbol, gs, turbconv_model)

Calculates the specific value of a quantity `χ` in the environment (`χ⁰`).

This function uses the domain decomposition principle to first find the
density-area-weighted environment value (`ρa⁰χ⁰`) and the environment
density-area (`ρa⁰`). It then computes the specific value using the
regularized `specific` function, which provides a stable result even when the
environment area fraction is very small.

Arguments:
- `χ_name`: The `Symbol` for the specific quantity `χ` (e.g., `:h_tot`, `:q_tot`).
- `gs`: The grid-scale state, containing grid-mean and draft subdomain states.
- `turbconv_model`: The turbulence convection model, containing parameters for regularization.

Returns:
- The specific value of the quantity `χ` in the environment.
"""
function specific_env_value(χ_name::Symbol, gs, turbconv_model)
    # Grid-scale density-weighted variable name, e.g., :ρq_tot
    ρχ_name = Symbol(:ρ, χ_name)

    # Numerator: ρa⁰χ⁰ = (gs.ρχ) - (Σ sgsʲ.ρa * sgsʲ.χ)
    ρaχ⁰ = env_value(
        getproperty(gs, ρχ_name),
        sgsʲ -> getproperty(sgsʲ, :ρa) * getproperty(sgsʲ, χ_name),
        gs,
    )

    # Denominator: ρa⁰ = gs.ρ - Σ sgsʲ.ρa
    ρa⁰_val = env_value(gs.ρ, sgsʲ -> sgsʲ.ρa, gs)

    # Call the 5-argument specific function for regularized division
    return specific(
        ρaχ⁰,                      # ρaχ for environment
        ρa⁰_val,                   # ρa for environment
        getproperty(gs, ρχ_name),  # Fallback ρχ is the grid-mean value
        gs.ρ,                      # Fallback ρ is the grid-mean value
        turbconv_model,
    )
end

"""
    specific_env_mse(gs, p)

Computes the specific moist static energy (`mse`) in the environment (`mse⁰`).

This is a specialized helper function because `mse` is not a grid-scale prognostic
variable. It first computes the grid-scale moist static energy density (`ρmse`)
from other grid-scale quantities (`ρ`, total specific enthalpy `h_tot`, specific 
kinetic energy `K`). It then uses the `env_value` helper to compute the environment's 
portion of `ρmse` and `ρa` via domain decomposition, and finally calculates the specific 
value using the regularized `specific` function.

Arguments:
- `gs`: The grid-scale state (`Y.c`), containing `ρ` and `sgsʲs`.
- `p`: The cache, containing the `turbconv_model` 

Returns:
- A `ClimaCore.Fields.Field` containing the specific moist static energy of the
  environment (`mse⁰`).
"""
function specific_env_mse(gs, p)
    # Get necessary precomputed values from the cache `p`
    (; ᶜK, ᶜts) = p.precomputed  # TODO: replace by on-the-fly computation
    (; turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜh_tot = @. lazy(TD.total_specific_enthalpy(thermo_params, ᶜts, specific(gs.ρe_tot, gs.ρ)))

    # 1. Define the grid-scale moist static energy density `ρ * mse`.
    grid_scale_ρmse = gs.ρ .* (ᶜh_tot .- ᶜK)

    # 2. Compute the environment's density-area-weighted mse (`ρa⁰mse⁰`).
    ρa⁰mse⁰ = p.scratch.ᶜtemp_scalar
    @. ρa⁰mse⁰ = env_value(grid_scale_ρmse, sgsʲ -> sgsʲ.ρa * sgsʲ.mse, gs)

    # 3. Compute the environment's density-area product (`ρa⁰`).
    ρa⁰_val = @. lazy(ρa⁰(gs))

    # 4. Compute and return the final specific environment mse (`mse⁰`).
    return @. lazy(specific(
        ρa⁰mse⁰,
        ρa⁰_val,
        grid_scale_ρmse,
        gs.ρ,
        turbconv_model,
    ))
end

"""
    ρa⁰(gs)

Computes the environment area-weighted density (`ρa⁰`).

This function uses the `env_value` helper, which applies the domain
decomposition principle (`GridMean = Environment + Sum(Drafts)`) to calculate
the environment area-weighted density by subtracting the sum of all draft
subdomain area-weighted densities (`ρaʲ`) from the grid-mean density (`ρ`).

Arguments:
- `gs`: The grid-scale state, which contains the grid-mean density `gs.ρ` and
        the draft subdomain states `gs.sgsʲs`.

Returns:
- The area-weighted density (`ρa⁰`).
"""
ρa⁰(gs) = env_value(gs.ρ, sgsʲ -> sgsʲ.ρa, gs)

"""
    specific_tke(sgs⁰, gs, turbconv_model)

Computes the specific turbulent kinetic energy (`tke`) in the environment (`tke⁰`).

This is a specialized helper that encapsulates the call to the regularized
`specific` function for the TKE variable. It provides `0` as the grid-scale
fallback value (`ρχ_fallback`) in the limit of small environmental area
fraction.

Arguments:
- `sgs⁰`: The environment SGS state (`Y.c.sgs⁰`), containing `ρatke`.
- `gs`: The grid-scale state (`Y.c`), containing the grid-mean density `ρ`.
- `turbconv_model`: The turbulence convection model, for regularization parameters.

Returns:
- The specific TKE of the environment (`tke⁰`).
"""
function specific_tke(sgs⁰, gs, turbconv_model)
    ρa⁰_val = ρa⁰(gs)

    return specific(
        sgs⁰.ρatke,     # ρaχ for environment TKE
        ρa⁰_val,        # ρa for environment, now computed internally
        0,              # Fallback ρχ is zero for TKE
        gs.ρ,           # Fallback ρ
        turbconv_model,
    )
end

"""
    u₃⁰(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model)

Computes the environment vertical velocity `u₃⁰`.

This function calculates the environment's total vertical momentum (`ρa⁰u₃⁰`) and
its total area-weighted density (`ρa⁰`) using the domain decomposition principle 
(GridMean = Env + Sum(Drafts)). It then computes the final specific velocity `u₃⁰` 
using the regularized `specific` function to ensure numerical stability when the 
environment area fraction `a⁰` is small.

Arguments:
- `ρaʲs`: A tuple of area-weighted densities for each draft subdomain.
- `u₃ʲs`: A tuple of vertical velocities for each draft subdomain.
- `ρ`: The grid-mean air density.
- `u₃`: The grid-mean vertical velocity.
- `turbconv_model`: The turbulence convection model, containing regularization parameters.
"""
u₃⁰(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model) = specific(
    ρ * u₃ - unrolled_dotproduct(ρaʲs, u₃ʲs),
    ρ - reduce(+, ρaʲs),
    ρ * u₃,
    ρ,
    turbconv_model,
)

"""
    mapreduce_with_init(f, op, iter...)

A wrapper for Julia's `mapreduce` function that automatically determines
the initial value (`init`) for the reduction.

This is useful for iterators whose elements are custom structs or 
`ClimaCore.Geometry.AxisTensor`s, where the zero element cannot be inferred
as a simple scalar. It uses `ClimaCore.RecursiveApply` tools (`rzero`,
`rpromote_type`) to create a type-stable, correctly-structured zero element
based on the output of the function `f` applied to the first elements of the
iterators.

Arguments:
- `f`: The function to apply to each element.
- `op`: The reduction operator (e.g., `+`, `*`).
- `iter...`: One or more iterators.
"""
function mapreduce_with_init(f, op, iter...)
    r₀ = rzero(rpromote_type(typeof(f(map(first, iter)...))))
    mapreduce(f, op, iter...; init = r₀)
end

"""
    unrolled_dotproduct(a::Tuple, b::Tuple)

Computes the dot product of two `Tuple`s (`a` and `b`) using a recursive,
manually unrolled implementation.

This function is designed to be type-stable and efficient for CUDA kernels,
where standard `mapreduce` implementations can otherwise suffer from type-inference 
failures.

It uses `ClimaCore.RecursiveApply` operators (`⊞` for addition, `⊠` for
multiplication), which allows it to handle dot products of tuples containing
complex, nested types such as `ClimaCore.Geometry.AxisTensor`s.

Arguments:
- `a`: The first `Tuple`.
- `b`: The second `Tuple`, which must have the same length as `a`.

Returns:
- The result of the dot product, `Σᵢ a[i] * b[i]`.
"""
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
