import ClimaCore.MatrixFields: @name

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
    elseif a > 1
        one(a)
    else
        (1 + tanh(2 * atanh(1 - 2 * (1 - a)^(-1 / log2(1 - a_half))))) / 2
    end
end

"""
    tracer_names(field)

Filters and returns the names of the variables from a given state
vector component, excluding `ρ`, `ρe_tot`, and `uₕ` and SGS fields.

Arguments:

- `field`: A component of the state vector `Y.c`.

Returns:

- A `Tuple` of `ClimaCore.MatrixFields.FieldName`s corresponding to the tracers.
"""
tracer_names(field) =
    unrolled_filter(MatrixFields.top_level_names(field)) do name
        !(
            name in
            (@name(ρ), @name(ρe_tot), @name(uₕ), @name(sgs⁰), @name(sgsʲs))
        )
    end

"""
    foreach_gs_tracer(f::F, Yₜ, Y) where {F}

Applies a given function `f` to each grid-scale scalar variable (except `ρ` and  `ρe_tot`)
in the state `Y` and its corresponding tendency `Yₜ`.
This utility abstracts the process of iterating over all scalars. It uses
`tracer_names` to identify the relevant variables and `unrolled_foreach` to
ensure a performant loop. For each tracer, it calls the provided function `f`
with the tendency field, the state field, and a boolean flag indicating if
the current tracer is `ρq_tot` (to allow for special handling).

Arguments:

- `f`: A function to apply to each grid-scale scalar. It must have the signature `f
  (ᶜρχₜ, ᶜρχ, ρχ_name)`, where `ᶜρχₜ` is the tendency field, `ᶜρχ`
  is the state field, and `ρχ_name` is a `MatrixFields.@name` object.
- `Yₜ`: The tendency state vector.
- `Y`: The current state vector.

# Example

```julia
foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
    # Apply some operation, e.g., a sponge layer
    @. ᶜρχₜ += some_sponge_function(ᶜρχ)
    if ρχ_name == @name(ρq_tot)
        # Perform an additional operation only for ρq_tot
    end
end
```
"""
foreach_gs_tracer(f::F, Yₜ, Y) where {F} =
    unrolled_foreach(tracer_names(Y.c)) do scalar_name
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, scalar_name)
        ᶜρχ = MatrixFields.get_field(Y.c, scalar_name)
        f(ᶜρχₜ, ᶜρχ, scalar_name)
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
- `p`: The cache, containing the `turbconv_model` and precomputed grid-scale
       fields `ᶜh_tot` and `ᶜK`.

Returns:
- A `ClimaCore.Fields.Field` containing the specific moist static energy of the
  environment (`mse⁰`).
"""
function specific_env_mse(gs, p)
    # Get necessary precomputed values from the cache `p`
    (; ᶜh_tot, ᶜK) = p.precomputed  # TODO: replace by on-the-fly computation
    (; turbconv_model) = p.atmos

    # 1. Define the grid-scale moist static energy density `ρ * mse`.
    grid_scale_ρmse = gs.ρ .* (ᶜh_tot .- ᶜK)

    # 2. Compute the environment's density-area-weighted mse (`ρa⁰mse⁰`).
    ρa⁰mse⁰ = env_value(grid_scale_ρmse, sgsʲ -> sgsʲ.ρa * sgsʲ.mse, gs)

    # 3. Compute the environment's density-area product (`ρa⁰`).
    ρa⁰_val = ρa⁰(gs)

    # 4. Compute and return the final specific environment mse (`mse⁰`).
    return specific(
        ρa⁰mse⁰,
        ρa⁰_val,
        grid_scale_ρmse,
        gs.ρ,
        turbconv_model,
    )
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


# TODO everything below this line may not be needed in the end
"""
    specific_sgs(χ_name, sgs, gs, turbconv_model)

Computes a single specific quantity `χ` from a subgrid-scale (SGS) state `sgs`,
identified by `χ_name`.

This function uses `ClimaCore.MatrixFields` utilities to robustly handle field names. 
It constructs the required variable names (e.g., `@name(q_tot)`) and computes the specific 
value using the regularized `specific` function. When the SGS area fraction 
becomes small, to avoid division by zero, this includes a smooth fallback to grid-mean 
values when they are available, and to zero when they are not.

Arguments:
- `χ_name`: A `ClimaCore.MatrixFields.FieldName` representing the
          specific quantity to be calculated (e.g., `@name(q_tot)`).
- `sgs`: The SGS state (e.g., a draft subdomain or the environment).
- `gs`: The grid-scale state, used to provide fallback values.
- `turbconv_model`: The turbulence convection model, for regularization parameters.

Returns:
- The specific value of the requested SGS quantity `χ`.
"""
# TODO: Replace turbconv_model by passing parameters needed for sgs_weight_function
function specific_sgs(χ_name, sgs, gs, turbconv_model)
    # Deconstruct the name to prepend ρa or ρ to the top-level variable name
    first_name = ClimaCore.MatrixFields.extract_first(χ_name)
    last_name = ClimaCore.MatrixFields.drop_first(χ_name)
    sgs_first_name = Symbol(:ρa, first_name)
    gs_first_name = Symbol(:ρ, first_name)
    sgs_name = ClimaCore.MatrixFields.append_internal_name(sgs_first_name, last_name)
    gs_name = ClimaCore.MatrixFields.append_internal_name(gs_first_name, last_name)

    ρaχ = ClimaCore.MatrixFields.get_field(sgs, sgs_name)
    ρχ_fallback = if ClimaCore.MatrixFields.has_field(gs, gs_name)
        ClimaCore.MatrixFields.get_field(gs, gs_name)
    else
        # Fallback for variables that do not exist at grid scale (e.g., TKE)
        zero(ρaχ)
    end

    return specific(ρaχ, sgs.ρa, ρχ_fallback, gs.ρ, turbconv_model)
end



# Helper functions for manipulating symbols in the generated functions:
has_prefix(symbol, prefix_symbol) =
    startswith(string(symbol), string(prefix_symbol))
remove_prefix(symbol, prefix_symbol) =
    Symbol(string(symbol)[(ncodeunits(string(prefix_symbol)) + 1):end])
# Note that we need to use ncodeunits instead of length because prefix_symbol
# can contain non-ASCII characters like 'ρ'.

"""
    all_specific_gs(gs)

Computes all specific quantities (`χ`) from a grid-scale state `gs`.

This `@generated` function introspects the field names of `gs` at compile time.
It identifies all density-weighted fields (e.g., `:ρq_tot`, `:ρe_tot`), divides
them by the grid-scale density `gs.ρ`, and returns them in a new `NamedTuple`.
This provides a type-stable and performant way to convert all relevant state
variables to their specific counterparts at once.

Arguments:
- `gs`: The grid-scale state, which must contain a `:ρ` field and other fields 
    with a `:ρ` prefix.

Returns:
- A new `NamedTuple` containing only the specific quantities (e.g., `:q_tot`, `:e_tot`).
"""
@generated function all_specific_gs(gs)
    gs_names = Base._nt_names(gs)
    relevant_gs_names =
        filter(name -> has_prefix(name, :ρ) && name != :ρ, gs_names)
    all_specific_gs_names = map(name -> remove_prefix(name, :ρ), relevant_gs_names)
    all_specific_gs_values = map(name -> :(gs.$name / gs.ρ), relevant_gs_names)
    return :(NamedTuple{$all_specific_gs_names}(($(all_specific_gs_values...),)))
end

"""
    all_specific_sgs(sgs, gs, turbconv_model)

Computes all specific quantities (`χ`) from a subgrid-scale (SGS) state `sgs`.

This `@generated` function identifies all density-area-weighted fields in the SGS
state (e.g., `:ρaq_tot`) and generates code to compute their specific values by
calling the `specific_sgs` helper for each variable. This provides a performant,
type-stable method for converting all relevant SGS variables to specific quantities.

Arguments:
- `sgs`: A `NamedTuple`-like object for the SGS state (e.g., an updraft or environment).
- `gs`: The corresponding grid-scale state, used to provide fallback values.
- `turbconv_model`: The turbulence convection model, for regularization parameters.

Returns:
- A new `NamedTuple` containing only the specific SGS quantities (e.g., `:q_tot`).
"""
@generated function all_specific_sgs(sgs, gs, turbconv_model)
    sgs_names = Base._nt_names(sgs)
    relevant_sgs_names =
        filter(name -> has_prefix(name, :ρa) && name != :ρa, sgs_names)
    specific_sgs_names =
        map(name -> remove_prefix(name, :ρa), relevant_sgs_names)
    specific_sgs_values = map(
        name -> :(specific_sgs($(QuoteNode(name)), sgs, gs, turbconv_model)),
        specific_sgs_names,
    )
    return :(NamedTuple{$specific_sgs_names}(($(specific_sgs_values...),)))
end

"""
    remove_energy_var(specific_state)

Creates a copy of `specific_state` with the energy variable (`:e_tot`) removed, 
where `specific_state` is the result of calling, e.g., `all_specific_gs`, `all_specific_sgsʲs`, 
or `all_specific_sgs⁰`. This is a utility function used to isolate non-energy tracer variables, 
for example, to calculate diffusive fluxes (which, for energy, involve gradients of enthalpy, 
not energy, and hence are handled separateyly). 

It dispatches on the input type to handle either a single `NamedTuple` or a `Tuple` of them 
(such as a collection of draft states). 

Arguments:
- `specific_state`: A `NamedTuple` or a `Tuple` of `NamedTuple`s from which to
                    remove the `:e_tot` field.

Returns:
- A new `NamedTuple` or `Tuple` without the `:e_tot` field(s).
"""
remove_energy_var(specific_state::NamedTuple) =
    Base.structdiff(specific_state, NamedTuple{(:e_tot,)})
remove_energy_var(specific_state::Tuple) =
    map(remove_energy_var, specific_state)


"""
    matching_subfields(tendency_field, specific_field)

Given a field that contains the tendencies of variables of the form `ρχ` or
`ρaχ` and another field that contains the values of specific variables `χ`,
returns all tuples `(tendency_field.<ρχ or ρaχ>, specific_field.<χ>, :<χ>)`.
Variables in `tendency_field` that do not have matching variables in
`specific_field` are omitted, as are variables in `specific_field` that do not
have matching variables in `tendency_field`. This function is needed to avoid
allocations due to failures in type inference, which are triggered when the
`propertynames` of these fields are manipulated during runtime in order to pick
out the matching subfields (as of Julia 1.8).
"""
@generated function matching_subfields(tendency_field, specific_field)
    tendency_names = Base._nt_names(eltype(tendency_field))
    specific_names = Base._nt_names(eltype(specific_field))
    prefix = :ρa in tendency_names ? :ρa : :ρ
    relevant_specific_names =
        filter(name -> Symbol(prefix, name) in tendency_names, specific_names)
    subfield_tuples = map(
        name -> :((
            tendency_field.$(Symbol(prefix, name)),
            specific_field.$name,
            $(QuoteNode(name)),
        )),
        relevant_specific_names,
    )
    return :(($(subfield_tuples...),))
end

