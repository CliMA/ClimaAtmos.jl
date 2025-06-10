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
- `ρχ`: The grid-mean density-weighted quantity (e.g., `ρe_tot`, `ρq_tot`).
- `ρ`: The grid-mean density.
- `ρaχ`: The density-area-weighted SGS quantity (e.g., `sgs.ρa * sgs.h_tot`).
- `ρa`: The density-area product of the SGS component.
- `ρχ_fallback`: The grid-mean density-weighted quantity used for the fallback value.
- `ρ_fallback`: The grid-mean density used for the fallback value.
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

# Helper functions for manipulating symbols in the generated functions:
has_prefix(symbol, prefix_symbol) =
    startswith(string(symbol), string(prefix_symbol))
remove_prefix(symbol, prefix_symbol) =
    Symbol(string(symbol)[(ncodeunits(string(prefix_symbol)) + 1):end])
# Note that we need to use ncodeunits instead of length because prefix_symbol
# can contain non-ASCII characters like 'ρ'.

"""
    specific_gs(gs)

Converts every variable of the form `ρχ` in the grid-scale state `gs` into the
specific variable `χ` by dividing it by `ρ`. All other variables in `gs` are
omitted from the result.
"""
@generated function specific_gs(gs)
    gs_names = Base._nt_names(gs)
    relevant_gs_names =
        filter(name -> has_prefix(name, :ρ) && name != :ρ, gs_names)
    specific_gs_names = map(name -> remove_prefix(name, :ρ), relevant_gs_names)
    specific_gs_values = map(name -> :(gs.$name / gs.ρ), relevant_gs_names)
    return :(NamedTuple{$specific_gs_names}(($(specific_gs_values...),)))
end

"""
    specific_sgs(sgs, gs, turbconv_model)

Converts every variable of the form `ρaχ` in the sub-grid-scale state `sgs` into
the specific variable `χ` by dividing it by `ρa`. All other variables in `sgs`
are omitted from the result. The division is computed as
`specific(ρaχ, ρa, ρχ, ρ, turbconv_model)`, which is preferable to simply
calling `ρaχ / ρa` because it avoids numerical issues that arise when `a` is
small. The values of `ρ` and `ρχ` are taken from `gs`, but, when `ρχ` is not
available in `gs` (e.g., when `χ` is a second moment variable like `tke`), its
value is assumed to be equal to the value of `ρaχ` in `sgs`.
"""
@generated function specific_sgs(sgs, gs, turbconv_model)
    sgs_names = Base._nt_names(sgs)
    gs_names = Base._nt_names(gs)
    relevant_sgs_names =
        filter(name -> has_prefix(name, :ρa) && name != :ρa, sgs_names)
    specific_sgs_names =
        map(name -> remove_prefix(name, :ρa), relevant_sgs_names)
    relevant_gs_names = map(name -> Symbol(:ρ, name), specific_sgs_names)
    specific_sgs_values = map(
        (sgs_name, gs_name) -> :(specific(
            sgs.$sgs_name,
            sgs.ρa,
            $(gs_name in gs_names ? :(gs.$gs_name) : :(sgs.$sgs_name)),
            gs.ρ,
            turbconv_model,
        )),
        relevant_sgs_names,
        relevant_gs_names,
    )
    return :(NamedTuple{$specific_sgs_names}(($(specific_sgs_values...),)))
end

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
u₃⁺(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model) = specific(
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
u₃⁰(ρaʲs, u₃ʲs, ρ, u₃, turbconv_model) = specific(
    ρ * u₃ - unrolled_dotproduct(ρaʲs, u₃ʲs),
    ρ - reduce(+, ρaʲs),
    ρ * u₃,
    ρ,
    turbconv_model,
)

"""
    remove_energy_var(specific_state)

Creates a copy of `specific_state` with the energy variable
removed, where `specific_state` is the result of calling, e.g., `specific_gs`,
`specific_sgsʲs`, or `specific_sgs⁰`.
"""
remove_energy_var(specific_state::NamedTuple) =
    Base.structdiff(specific_state, NamedTuple{(:e_tot,)})
remove_energy_var(specific_state::Tuple) =
    map(remove_energy_var, specific_state)


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
