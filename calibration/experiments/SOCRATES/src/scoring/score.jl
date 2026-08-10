"""
Score normalization.

Hydrometeor profiles span orders of magnitude and are zero over most of the column, so an unnormalized
least-squares misfit is dominated by whichever variable happens to be largest. Each variable's block is
divided by a per-variable scale `sqrt(pool_var)`, chosen so the *nonzero* part of the reference has
magnitude one, and a per-variable factor then expresses relative importance.
"""

using Statistics: Statistics

"""
    mean_nonzero_elements(x; all_zero = 0)

Mean of the nonzero elements of `x`, or `all_zero` when every element is zero.

For a hydrometeor field that is zero over most of the column this measures the magnitude *where there is
condensate*, rather than being dragged to zero by the empty part of the profile.
"""
function mean_nonzero_elements(x; all_zero = zero(eltype(x)))
    n = count(!iszero, x)
    n == 0 && return oftype(one(eltype(x)) * all_zero, all_zero)
    return sum(x) / n
end

"""
    ScoreTransform(; characteristic, obs_var_scaling, additional_uncertainty, uncertainty_floor, normalization)

Per-variable score normalization.

# Fields

  - `characteristic`: the magnitude a variable is measured against once `pool_var` floors at it, and the
    scale used when its reference block is entirely zero. Declared, never inferred.
  - `obs_var_scaling`: multiplies `pool_var`, so it scales the variable's contribution to the misfit. A
    value `(1/f)^2` makes the variable behave as if `f` times more important.
  - `additional_uncertainty`: relative observational uncertainty on the diagonal of `Γ`, as a fraction of
    the normalized field's time-mean magnitude.
  - `uncertainty_floor`: smallest observational standard deviation given to any entry, as a fraction of
    the variable's characteristic magnitude. It sets the uncertainty on entries the reference measures no
    variance for, where the profile is identically zero. Being tied to the characteristic rather than to
    `pool_var` is also what gives `obs_var_scaling` and `characteristic` any effect on the misfit: `Γ` is
    formed from the normalized series, so every term scaling with `pool_var` cancels out of
    `Δyᵀ Γ⁻¹ Δy` exactly.
  - `normalization`: `:pooled_nonzero_mean_to_value` normalizes so the mean of the nonzero reference
    values is one, floored at the characteristic value; `:pooled_variance` normalizes by the variance.
"""
struct ScoreTransform{C <: AbstractDict, S <: AbstractDict, U <: AbstractDict, F <: AbstractDict}
    characteristic::C
    obs_var_scaling::S
    additional_uncertainty::U
    uncertainty_floor::F
    normalization::Symbol
end

"""Characteristic magnitudes [kg/kg for profiles, kg/m² for paths]."""
const DEFAULT_CHARACTERISTIC = Dict{String, Float64}(
    "clw" => 1.0e-4,
    "cli" => 1.0e-8,
    "husra" => 1.0e-7,
    "hussn" => 1.0e-7,
    "lwp" => 1.0e-1,
    "iwp" => 1.0e-5,
    "rwp" => 3.0e-4,
    "swp" => 2.0e-4,
)

"""
Per-variable variance scaling. Because the normalization divides by `sqrt(pool_var)`, these also set
relative importance: `(1/f)^2` boosts a variable by a factor `f`.
"""
const DEFAULT_OBS_VAR_SCALING = Dict{String, Float64}(
    "clw" => (1.0 / 1.0)^2,
    "cli" => (1.0 / 2.5)^2,
    "husra" => 1.0,
    "hussn" => (1.0 / 2.0)^2,
    "lwp" => (1.0 / 0.5)^2,
    "iwp" => (1.0 / 3.0)^2,
    "rwp" => 1.0,
    "swp" => 1.0,
)

"""Relative observational uncertainty per variable, added to the diagonal of `Γ`."""
const DEFAULT_ADDITIONAL_UNCERTAINTY =
    Dict{String, Float64}(name => 0.1 for name in keys(DEFAULT_CHARACTERISTIC))

"""
Smallest observational standard deviation per variable, as a fraction of its characteristic magnitude.
An error the size of the characteristic value therefore always costs `(1/fraction)^2`.
"""
const DEFAULT_UNCERTAINTY_FLOOR =
    Dict{String, Float64}(name => 0.05 for name in keys(DEFAULT_CHARACTERISTIC))

function ScoreTransform(;
    characteristic = DEFAULT_CHARACTERISTIC,
    obs_var_scaling = DEFAULT_OBS_VAR_SCALING,
    additional_uncertainty = DEFAULT_ADDITIONAL_UNCERTAINTY,
    uncertainty_floor = DEFAULT_UNCERTAINTY_FLOOR,
    normalization::Symbol = :pooled_nonzero_mean_to_value,
)
    normalization in (:pooled_nonzero_mean_to_value, :pooled_variance) || error(
        "Unknown normalization `:$normalization`; expected `:pooled_nonzero_mean_to_value` or \
         `:pooled_variance`.",
    )
    return ScoreTransform(
        characteristic,
        obs_var_scaling,
        additional_uncertainty,
        uncertainty_floor,
        normalization,
    )
end

_entry(d, name, what) =
    get(d, name) do
        error("ScoreTransform has no $what for variable `$name`; add one to its table.")
    end

"""
    pool_var(transform, name, reference)

The squared normalization scale for variable `name`, from its reference values over the scoring window.

`:pooled_nonzero_mean_to_value` gives `obs_var_scaling * mean_nonzero_elements(reference)^2`, using the
declared `characteristic` when the block is entirely zero, then floored at
`obs_var_scaling * characteristic^2`. The floor matters because a *tiny but nonzero* mean would otherwise
give a normalizer far below the all-zero case and blow the normalized misfit up.

`:pooled_variance` gives `obs_var_scaling * var(reference)`, with no floor.

The result is never zero: a zero scale would divide by zero downstream, so it is raised to `eps`.
"""
function pool_var(transform::ScoreTransform, name::AbstractString, reference)
    scaling = _entry(transform.obs_var_scaling, name, "obs_var_scaling")
    FT = eltype(reference)
    value = if transform.normalization === :pooled_variance
        scaling * Statistics.var(reference)
    else
        cv = FT(_entry(transform.characteristic, name, "characteristic value"))
        μ = mean_nonzero_elements(reference; all_zero = cv)
        max(scaling * μ^2, scaling * cv^2)
    end
    return iszero(value) ? eps(FT) : FT(value)
end

"""
    normalizer(transform, name, reference)

`sqrt(pool_var)` — the factor a variable's block is divided by, in the variable's own units.
"""
normalizer(transform::ScoreTransform, name::AbstractString, reference) =
    sqrt(pool_var(transform, name, reference))

"""
    nanmean(x)

Mean of the finite entries of `x`, or zero when there are none. Absent reference data is carried as
`NaN` and dropped from the statistics rather than propagated.
"""
nanmean(x) = (f = filter(isfinite, x); isempty(f) ? 0.0 : sum(f) / length(f))

"""
    normalized_characteristic(transform, name)

`name`'s characteristic magnitude in normalized units, `1/sqrt(obs_var_scaling)`.
"""
normalized_characteristic(transform::ScoreTransform, name::AbstractString) =
    1 / sqrt(_entry(transform.obs_var_scaling, name, "obs_var_scaling"))

"""
    uncertainty_diagonal(transform, name, series)

Diagonal observational variance for one variable's block of the normalized `series` (rows × times):

```
D_i = (additional_uncertainty * mean_t|series_i|)^2 + (uncertainty_floor * normalized_characteristic)^2
```

The floor carries the entries the reference measures no variance for, of which there are hundreds per
case, and without which `Γ` is singular — the sample covariance of a few dozen snapshots cannot have
more rank than that however many rows it has.
"""
function uncertainty_diagonal(
    transform::ScoreTransform,
    name::AbstractString,
    series::AbstractMatrix,
)
    factor = _entry(transform.additional_uncertainty, name, "additional_uncertainty")
    fraction = _entry(transform.uncertainty_floor, name, "uncertainty_floor")
    floor = fraction * normalized_characteristic(transform, name)
    magnitude = [nanmean(abs.(view(series, i, :))) for i in axes(series, 1)]
    return @. (factor * magnitude)^2 + floor^2
end

"""
    normalize(transform, name, values, reference)

`values` divided by the scale derived from `reference`. Applied to the reference when building
observations and to the model output when building `G`, with the *same* `reference` both times, so the
two live in the same normalized space.
"""
normalize(transform::ScoreTransform, name::AbstractString, values, reference) =
    values ./ normalizer(transform, name, reference)
