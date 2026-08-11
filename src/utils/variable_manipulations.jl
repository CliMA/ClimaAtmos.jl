import ClimaCore.MatrixFields: @name
import ClimaCore.RecursiveApply: ⊞, ⊠, rzero, rpromote_type

"""
    specific(ρχ, ρ)
    specific(ρaχ, ρa, ρχ, ρ, turbconv_model)

Compute the specific quantity `χ` (per unit mass of moist air) from a
density-weighted quantity.

The two-argument method divides the grid-mean density-weighted quantity `ρχ` by
the grid-mean density `ρ`, which is always well defined and non-zero.

The five-argument method returns the specific quantity of a subgrid-scale (SGS)
subdomain whose density-area product `ρa` may vanish. It blends the SGS quotient
`ρaχ / ρa` with the grid-mean quotient `ρχ / ρ`,

    χ = w (ρaχ / ρa) + (1 - w) (ρχ / ρ),   w = sgs_weight_function(ρa / ρ, a_half),

and falls back to the grid-mean value outright when `ρa < eps(typeof(ρ))`, where
even a zero weight would leave a `0 / 0` (and autodiff would produce `NaN`s).
The blend keeps the result finite as the subdomain area fraction goes to zero,
at the price of breaking the domain decomposition (the SGS subdomains no longer
sum exactly to the grid mean) where the area fraction is small.

# Arguments

  - `ρχ`: Grid-mean density-weighted quantity, e.g. `ρe_tot` or `ρq_tot`
    [kg/m³ × units of `χ`].
  - `ρ`: Grid-mean density [kg/m³].
  - `ρaχ`: Density-area-weighted SGS quantity, e.g. `sgsʲ.ρa * sgsʲ.mse`
    [kg/m³ × units of `χ`].
  - `ρa`: Density-area product of the SGS subdomain [kg/m³].
  - `turbconv_model`: Turbulence-convection model; supplies the regularization
    parameter `a_half`.

# Returns

The specific quantity `χ` [units of `χ` per kg].

See also `sgs_weight_function` and `env_relaxation_feedback`.
"""
specific(ρχ, ρ) = ρχ / ρ

function specific(ρaχ, ρa, ρχ, ρ, turbconv_model)
    # TODO: Replace turbconv_model struct by parameters, and include a_half in
    # parameters, not in config
    weight = sgs_weight_function(ρa / ρ, turbconv_model.a_half)
    # If ρa is exactly zero, the weight is zero, making the first term NaN (0 * … / 0).
    # For negligible ρa the weight is also zero, and autodiff can still produce NaNs.
    # The ifelse handles both cases explicitly by returning the grid-mean value when ρa < eps.
    return ρa < eps(typeof(ρ)) ? ρχ / ρ : weight * ρaχ / ρa + (1 - weight) * ρχ / ρ
end

"""
    env_relaxation_feedback(ρaʲ, ρa⁰, ρ, turbconv_model)

Return the magnitude of `-∂χ⁰/∂χʲ` for an environment value diagnosed by
`specific` from the domain decomposition,

    χ⁰ = w · (ρχ - Σⱼ ρaʲ χʲ) / ρa⁰ + (1 - w) · ρχ / ρ,

differentiated at fixed grid-mean `ρχ` and fixed `ρaʲ`:

    ∂χ⁰/∂χʲ = -w · ρaʲ / ρa⁰,

with `w = sgs_weight_function(ρa⁰/ρ, a_half)` and the same `ρa⁰ < eps` fallback
branch as `specific` (where the derivative is zero). This is the exact
derivative of `specific`'s regularized quotient, so the gating keeps the
result finite as `ρa⁰ → 0`. Used by the entrainment-relaxation diagonals of
the implicit Jacobian in `manual_sparse_jacobian.jl`.
"""
@inline env_relaxation_feedback(ρaʲ, ρa⁰, ρ, turbconv_model) =
    ρa⁰ < eps(typeof(ρ)) ? zero(ρ) :
    sgs_weight_function(ρa⁰ / ρ, turbconv_model.a_half) * ρaʲ / ρa⁰

"""
    is_ρ_weighted_name(name)

Return `true` if `name` is a top-level `@name(ρχ)` for some variable `χ`.

The test is purely lexical: the name chain must have length 1 (so composite
names such as `@name(sgsʲs.:(1).q_tot)` are rejected) and its single symbol must
start with `ρ`. This is the auto-discovery predicate behind
`gs_tracer_names`; see the Passive Tracers page of the documentation.
"""
@generated is_ρ_weighted_name(
    ::MatrixFields.FieldName{name_chain},
) where {name_chain} =
    length(name_chain) == 1 && startswith(string(name_chain[1]), "ρ")

"""
    specific_tracer_name(ρχ_name)

Convert the density-weighted name `@name(ρχ)` to the specific name `@name(χ)`
by stripping the leading `ρ`.

Inverse of `get_ρχ_name` for top-level names.
"""
@generated function specific_tracer_name(
    ::MatrixFields.FieldName{ρχ_name_chain},
) where {ρχ_name_chain}
    χ_symbol = Symbol(string(ρχ_name_chain[1])[(ncodeunits("ρ") + 1):end])
    return :(@name($χ_symbol))
end

"""
    gs_tracer_names(Y)

Return a `Tuple` of the `@name`s of the grid-scale tracers in `Y.c`.

A grid-scale tracer is any top-level `ρ`-prefixed field of `Y.c` (see
`is_ρ_weighted_name`) other than `ρ`, `ρe_tot`, and `ρtke`. Velocities and SGS
fields are excluded automatically, because `uₕ` and `sgsʲs` are not
`ρ`-prefixed. Adding such a field to the state is all that is needed to opt a
tracer into the generic transport, diffusion, and hyperdiffusion loops.
"""
gs_tracer_names(Y) =
    unrolled_filter(MatrixFields.top_level_names(Y.c)) do name
        is_ρ_weighted_name(name) && !(name in (@name(ρ), @name(ρe_tot), @name(ρtke)))
    end

"""
    specific_gs_tracer_names(Y)

Return a `Tuple` of the specific tracer names `@name(χ)` corresponding to the
density-weighted tracer names `@name(ρχ)` in `gs_tracer_names(Y)`.
"""
specific_gs_tracer_names(Y) =
    unrolled_map(specific_tracer_name, gs_tracer_names(Y))

"""
    ᶜempty(Y)

Return a lazy center `Field` of empty `NamedTuple`s, used as the zero-tracer
result of `ᶜgs_tracers` and `ᶜspecific_gs_tracers`.
"""
ᶜempty(Y) = lazy.(Returns((;)).(Y.c))

"""
    ᶜgs_tracers(Y)

Return a lazy center `Field` of `NamedTuple`s holding the density-weighted
values of all grid-scale tracers given by `gs_tracer_names(Y)`, keyed by the
tracer symbols `ρχ`.
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

Return a lazy center `Field` of `NamedTuple`s holding the specific values
`χ = ρχ / ρ` of all grid-scale tracers, keyed by the specific tracer symbols
given by `specific_gs_tracer_names(Y)`.
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

Apply the function `f` to each grid-scale tracer of the state `Y` or of a
similar value such as the tendency `Yₜ`.

The loop over `gs_tracer_names(Y)` is unrolled, so it stays type-stable and
allocation-free on both CPU and GPU. Although the first value must be similar to
`Y`, the remaining values may also be center `Field`s similar to `Y.c`, and they
may carry the specific tracers of `specific_gs_tracer_names(Y)` instead of the
density-weighted ones; the corresponding subfield is looked up per value.

# Arguments

  - `f`: Function applied to each grid-scale tracer, with the signature
    `f(ρχ_or_χ_fields..., ρχ_name)`, where `ρχ_or_χ_fields` are the tracer
    subfields (density-weighted or specific) of each input value and `ρχ_name` is
    the `MatrixFields.FieldName` of the tracer.
  - `Y_or_similar_values`: The state `Y` or similar values such as `Yₜ`, or center
    `Field`s similar to `Y.c`.

# Returns

`nothing`.

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
    _is_sgs_tracer_name(name)

Return `true` if `name` is an SGS tracer name, i.e. anything other than the core
PROPHET variables `ρa`, `mse`, and `q_tot`, which receive physics-specific
treatment.
"""
_is_sgs_tracer_name(::MatrixFields.FieldName{name_chain}) where {name_chain} =
    !(last(name_chain) in (:ρa, :mse, :q_tot))

"""
    sgs_tracer_names(Y)

Return a `Tuple` of the `@name`s (relative to `Y.c.sgsʲs.:(1)`) of all SGS
tracers carried by the first updraft of `Y`.

"Tracer" means any scalar of `Y.c.sgsʲs.:(1)` that is not one of the core
PROPHET variables `ρa`, `mse`, or `q_tot`. Returns `()` when prognostic EDMF is
inactive, i.e. when `Y.c` has no `sgsʲs` field. Every updraft carries the same
set of tracers, so the first updraft is representative.
"""
sgs_tracer_names(Y) =
    _sgs_tracer_names(Val(hasproperty(Y.c, :sgsʲs)), Y)
_sgs_tracer_names(::Val{false}, Y) = ()
_sgs_tracer_names(::Val{true}, Y) =
    unrolled_filter(
        _is_sgs_tracer_name,
        MatrixFields.top_level_names(Y.c.sgsʲs.:(1)),
    )


"""
    sgs_weight_function(a, a_half)

Compute the smooth, monotonic weight `w(a) ∈ [0, 1]` used to blend a
subgrid-scale quantity with its grid-mean counterpart in `specific`.

The weight makes the transition between the two continuous and numerically
stable where the SGS area fraction `a` is small.

Properties:

  - `w(a) = 0` for `a ≤ 0`.
  - `w(a) = 1` for `a ≥ min(1, 42 a_half)`; the `42 a_half` cutoff short-circuits
    the sigmoid where it is indistinguishable from 1 and where autodiff of the
    closed form generates `NaN`s.
  - `w(a_half) = 1/2`.
  - `w` is continuously differentiable with vanishing derivatives at `a = 0` and
    `a = 1`, so the blend introduces no kinks.
  - `w` rises steeply near `a = a_half` and is nearly flat elsewhere, so for small
    `a_half` it is already close to 1 a few multiples of `a_half` above it.

On `0 < a < 1` the weight is a sigmoid built in two steps: a base sigmoid maps
`(0, 1)` onto `(0, 1)` with zero endpoint derivatives, by composing `tanh` with
the inverse of a slower-growing `tanh`; and the input is pre-transformed by
`1 - (1 - a)^k` so that `a_half` maps to `1/2` without spoiling the endpoint
derivatives.

# Arguments

  - `a`: SGS area fraction, in practice approximated by `ρa / ρ` [-].
  - `a_half`: Area fraction at which the weight equals `1/2`, i.e. the transition
    point of the sigmoid [-].

# Returns

The weight `w(a)` [-].
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

Sum the function `f` over the draft subdomain states `sgsʲ` in `sgsʲs`.

The sum is unrolled over the (statically known) number of drafts, so it is
type-stable inside broadcast kernels.

# Arguments

  - `f`: Function applied to each element of `sgsʲs`.
  - `sgsʲs`: Iterator over the draft subdomain states, e.g. `Y.c.sgsʲs`.
"""
draft_sum(f, sgsʲs) = unrolled_sum(f, sgsʲs)

"""
    ᶜenv_value(grid_scale_value, f_draft, gs)

Return a lazy center `Field` with the environment share of a density-area
weighted quantity, obtained by subtracting the draft sum from the grid-scale
value.

This applies the domain decomposition `ρχ = ρa⁰χ⁰ + Σⱼ ρaʲχʲ`, valid for the
`ρa`-weighted quantities of `PrognosticEDMFX`.

# Arguments

  - `grid_scale_value`: Grid-scale value `ρχ` of the quantity.
  - `f_draft`: Function extracting the corresponding `ρaʲχʲ` from a draft
    subdomain state.
  - `gs`: Iterator over the draft subdomain states, e.g. `Y.c.sgsʲs`.

See also `env_value` for the non-lazy, pointwise version.
"""
function ᶜenv_value(grid_scale_value, f_draft, gs)
    return @. lazy(grid_scale_value - draft_sum(f_draft, gs))
end

"""
    env_value(grid_scale_value, f_draft, gs)

Return the environment share of a density-area weighted quantity at a point.

Pointwise counterpart of `ᶜenv_value`, for use inside a broadcast expression
rather than as the producer of one.
"""
function env_value(grid_scale_value, f_draft, gs)
    return grid_scale_value - draft_sum(f_draft, gs)
end



"""
    ᶜspecific_env_value(χ_name, Y, p)

Compute the specific value `χ⁰` of a quantity `χ` in the environment.

Domain decomposition gives the environment numerator `ρa⁰χ⁰ = ρχ - Σⱼ ρaʲχʲ`
and denominator `ρa⁰ = ρ - Σⱼ ρaʲ`; the quotient is then formed with the
regularized `specific`, which stays finite as the environment area fraction goes
to zero. The grid-mean tracer name is derived from `χ_name` with
`get_ρχ_name`, so `χ` must have a grid-mean counterpart `ρχ` in `Y.c`.

Only `PrognosticEDMFX` is supported; `EDOnlyEDMFX` throws, since it has no draft
subdomains and its environment coincides with the grid mean.

# Arguments

  - `χ_name`: `MatrixFields.FieldName` of the specific quantity, e.g.
    `@name(q_tot)`.
  - `Y`: State, providing the grid mean `Y.c` and the drafts `Y.c.sgsʲs`.
  - `p`: Cache, providing `p.atmos.turbconv_model`.

# Returns

A lazy center `Field` with the environment value `χ⁰`.
"""
function ᶜspecific_env_value(χ_name, Y, p)
    turbconv_model = p.atmos.turbconv_model

    # Grid-scale density-weighted variable name, e.g., ρq_tot
    ρχ_name = get_ρχ_name(χ_name)

    ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)

    # environment density-area-weighted mse (`ρa⁰χ⁰`).
    # Numerator: ρa⁰χ⁰ = ρχ - (Σ ρaʲ * χʲ)
    if turbconv_model isa PrognosticEDMFX
        #Numerator: ρa⁰χ⁰ = ρχ - (Σ sgsʲ.ρa * sgsʲ.χ)

        ᶜρaχ⁰ = ᶜenv_value(
            ᶜρχ,
            sgsʲ ->
                MatrixFields.get_field(sgsʲ, @name(ρa)) *
                MatrixFields.get_field(sgsʲ, χ_name),
            Y.c.sgsʲs,
        )
        # Denominator: ρa⁰ = ρ - Σ ρaʲ
        ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))

    elseif turbconv_model isa EDOnlyEDMFX
        error("Not implemented. You should use grid mean values.")
    end

    return @. lazy(specific(
        ᶜρaχ⁰,                      # ρaχ for environment
        ᶜρa⁰,                   # ρa for environment
        ᶜρχ,               # Fallback ρχ is the grid-mean value
        Y.c.ρ,                      # Fallback ρ is the grid-mean value
        turbconv_model,
    ))
end

"""
    get_ρχ_name(χ_name)

Construct the `FieldName` of the density-weighted quantity `ρχ` from the
specific tracer name `χ_name`.

The construction is recursive on hierarchical names: a leaf name is returned
with a `ρ` prefix, e.g. `@name(q_rai)` becomes `@name(ρq_rai)`; a composite name
keeps its parent and has `ρ` prepended at the leaf, e.g.
`@name(sgsʲs.:(1).q_rai)` becomes `@name(sgsʲs.:(1).ρq_rai)`.

This is the pairing that lets an SGS tracer `χ` in `Y.c.sgsʲs.:(j)` find its
grid-mean counterpart `ρχ` in `Y.c`; see the Passive Tracers page of the
documentation.
"""
function get_ρχ_name(χ_name)
    parent_name = MatrixFields.FieldName(MatrixFields.extract_first(χ_name))
    child_name = MatrixFields.drop_first(χ_name)
    ρχ_name =
        (child_name == MatrixFields.@name()) ?
        MatrixFields.FieldName(Symbol(:ρ, MatrixFields.extract_first(χ_name))) :
        MatrixFields.append_internal_name(parent_name, get_ρχ_name(child_name))

    return ρχ_name
end

"""
    get_χʲ_name_from_ρχ_name(ρχ_name)

Construct the `FieldName` of the specific tracer in the first updraft (`χʲ` with
`j = 1`) that corresponds to the grid-mean density-weighted tracer `ρχ_name`.

The construction is recursive on hierarchical names: a leaf name has its `ρ`
prefix stripped and the updraft prefix prepended, e.g. `@name(ρq_rai)` becomes
`@name(sgsʲs.:(1).q_rai)`; a composite name keeps its parent and has the
transformation applied at the leaf.

Inverse of `get_ρχ_name` composed with the projection onto the first updraft.
"""
function get_χʲ_name_from_ρχ_name(ρχ_name)
    parent_name = MatrixFields.FieldName(MatrixFields.extract_first(ρχ_name))
    child_name = MatrixFields.drop_first(ρχ_name)
    χʲ_name =
        (child_name == MatrixFields.@name()) ?
        MatrixFields.append_internal_name(
            @name(sgsʲs.:(1)),
            specific_tracer_name(ρχ_name),
        ) :
        MatrixFields.append_internal_name(parent_name, get_χʲ_name_from_ρχ_name(child_name))
    return χʲ_name
end

"""
    ρa⁰(ρ, sgsʲs, turbconv_model)

Compute the area-weighted density of the environment, `ρa⁰ = ρ - Σⱼ ρaʲ`
[kg/m³].

Only `PrognosticEDMFX` carries draft subdomains; for every other
turbulence-convection model the environment is the whole grid box and the
grid-mean density `ρ` is returned unchanged.

# Arguments

  - `ρ`: Grid-mean density [kg/m³].
  - `sgsʲs`: Iterable of draft subdomain states, typically `Y.c.sgsʲs`.
  - `turbconv_model`: Turbulence-convection model.
"""
function ρa⁰(ρ, sgsʲs, turbconv_model)
    # ρ - Σ ρaʲ
    if turbconv_model isa PrognosticEDMFX
        return env_value(ρ, sgsʲ -> sgsʲ.ρa, sgsʲs)
    else
        return ρ
    end
end

"""
    a⁰(sgsʲs, ᶜρʲs, turbconv_model)

Compute the environment area fraction, `a⁰ = 1 - Σⱼ aʲ` with
`aʲ = draft_area(ρaʲ, ρʲ)` [-].

Only `PrognosticEDMFX` carries draft subdomains; for every other
turbulence-convection model the environment fills the grid box and `1` is
returned.

# Arguments

  - `sgsʲs`: Iterable of draft subdomain states, typically `Y.c.sgsʲs`.
  - `ᶜρʲs`: Iterable of draft densities, typically `p.precomputed.ᶜρʲs`
    [kg/m³].
  - `turbconv_model`: Turbulence-convection model.
"""
function a⁰(sgsʲs, ᶜρʲs, turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        return 1 - mapreduce_with_init(
            (sgsʲ, ᶜρʲ) -> draft_area(sgsʲ.ρa, ᶜρʲ),
            +,
            sgsʲs,
            ᶜρʲs,
        )
    else
        return 1
    end
end

"""
    ᶜspecific_env_mse(Y, p)

Compute the specific moist static energy of the environment, `mse⁰` [J/kg].

`mse` is not a grid-scale prognostic variable, so this needs its own helper
rather than `ᶜspecific_env_value`: the grid-scale density of moist static energy
is reconstructed as `ρ mse = ρ (h_tot - K)` from the precomputed total specific
enthalpy `ᶜh_tot` and specific kinetic energy `ᶜK`. Domain decomposition through
`ᶜenv_value` then supplies the environment numerator and `ρa⁰` the denominator,
and the quotient is formed with the regularized `specific`.

Only `PrognosticEDMFX` is supported; `EDOnlyEDMFX` throws, since its environment
coincides with the grid mean.

# Arguments

  - `Y`: State, providing `Y.c.ρ` and the drafts `Y.c.sgsʲs`.
  - `p`: Cache, providing `p.atmos.turbconv_model` and the precomputed `ᶜK` and
    `ᶜh_tot`.

# Returns

A lazy center `Field` with the environment moist static energy `mse⁰` [J/kg].
"""
function ᶜspecific_env_mse(Y, p)
    turbconv_model = p.atmos.turbconv_model
    (; ᶜK, ᶜh_tot) = p.precomputed

    # grid-scale moist static energy density `ρ * mse`.
    ᶜρmse = @. lazy(Y.c.ρ * (ᶜh_tot - ᶜK))

    # environment density-area-weighted mse (`ρa⁰mse⁰`).
    # Numerator: ρa⁰mse⁰ = ρmse - (Σ ρaʲ * mseʲ)

    if turbconv_model isa PrognosticEDMFX
        ρa⁰mse⁰ = ᶜenv_value(ᶜρmse, sgsʲ -> sgsʲ.ρa * sgsʲ.mse, Y.c.sgsʲs)
        ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    elseif turbconv_model isa EDOnlyEDMFX
        error("Not implemented. You should use grid mean values.")
    end

    return @. lazy(specific(ρa⁰mse⁰, ᶜρa⁰, ᶜρmse, Y.c.ρ, turbconv_model))
end

"""
    u₃⁰(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model)

Compute the covariant vertical velocity of the environment, `u₃⁰`.

Domain decomposition gives the environment momentum `ρa⁰u₃⁰ = ρu₃ - Σⱼ ρaʲu₃ʲ`
and area-weighted density `ρa⁰ = ρ - Σⱼ ρaʲ`; the quotient is formed with the
regularized `specific`, which stays finite as the environment area fraction goes
to zero.

# Arguments

  - `ρaʲs`: Tuple of the draft area-weighted densities `ρaʲ` [kg/m³].
  - `u₃ʲs`: Tuple of the draft covariant vertical velocities `u₃ʲ`.
  - `ρ`: Grid-mean air density [kg/m³].
  - `u₃`: Grid-mean covariant vertical velocity.
  - `turbconv_model`: Turbulence-convection model; supplies `a_half`.
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

Reduce `f` over `iter...` with `op`, inferring the `init` value automatically.

`mapreduce` needs an explicit `init` when the elements are custom structs or
`ClimaCore.Geometry.AxisTensor`s, whose zero is not a scalar. The zero is built
here with `rzero` and `rpromote_type` from `ClimaCore.RecursiveApply`, applied to
the result of `f` on the first elements, which keeps the reduction type-stable.

# Arguments

  - `f`: Function applied to each element.
  - `op`: Reduction operator, e.g. `+`.
  - `iter...`: One or more iterators, zipped elementwise by `f`.
"""
function mapreduce_with_init(f, op, iter...)
    r₀ = rzero(rpromote_type(typeof(f(map(first, iter)...))))
    mapreduce(f, op, iter...; init = r₀)
end

"""
    promote_type_mul(x, y)

Return the type of the product of a `Number` and a
`ClimaCore.Geometry.AxisTensor`, which is the type of the tensor.

Used by `unrolled_dotproduct` to build the zero element of the reduction.
"""
promote_type_mul(n::Number, x::Geometry.AxisTensor) = typeof(x)
promote_type_mul(x::Geometry.AxisTensor, n::Number) = typeof(x)

"""
    unrolled_dotproduct(a::Tuple, b::Tuple)

Compute the dot product `Σᵢ a[i] * b[i]` of two equal-length `Tuple`s.

The recursion is manually unrolled, which keeps the result type-stable in CUDA
kernels, where `mapreduce` can fail type inference. Products and sums go through
the `ClimaCore.RecursiveApply` operators `⊠` and `⊞`, so the tuples may hold
nested types such as `ClimaCore.Geometry.AxisTensor`s.

# Arguments

  - `a`: First `Tuple`.
  - `b`: Second `Tuple`, of the same length as `a`.
"""
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
