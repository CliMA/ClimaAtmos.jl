# Aggregating the information about negative tracers at each vertical level

import Statistics: mean

"""
    compute_min_per_level!(out, field)

Computes the minimum value of a field at each vertical level and stores it in `out`.
"""
function compute_min_per_level!(out, field)
    out′ = isnothing(out) ? similar(Fields.column(field, 1, 1, 1)) : out
    for i in 1:Spaces.nlevels(axes(field))
        field_level = Fields.level(field, i)
        Fields.level(out′, i) .= minimum(field_level)
    end
    return out′
end

"""
    compute_negative_mean_per_level!(out, field)

Computes the mean of negative values of a field at each vertical level.
If no negative values exist at a level, result is 0.
"""
function compute_negative_mean_per_level!(out, field)
    out′ = isnothing(out) ? similar(Fields.column(field, 1, 1, 1)) : out
    for i in 1:Spaces.nlevels(axes(field))
        field_level = Fields.level(field, i)
        # Mask: 1 for negative values, 0 otherwise (excludes zero)
        negative_mask =
            @. lazy(ifelse(field_level < 0, one(field_level), zero(field_level)))
        # Sum of negative values only
        sum_negative = sum(@. lazy(field_level * negative_mask))
        # Count of negative values
        count_negative = sum(negative_mask)
        # Mean of negative values (0 if no negatives)
        Fields.level(out′, i) .=
            ifelse(count_negative > 0, sum_negative / count_negative, 0)
    end
    return out′
end

"""
    compute_negative_fraction_per_level!(out, field)

Computes the fraction of the level where `field` is negative.
"""
function compute_negative_fraction_per_level!(out, field)
    out′ = isnothing(out) ? similar(Fields.column(field, 1, 1, 1)) : out
    for i in 1:Spaces.nlevels(axes(field))
        field_level = Fields.level(field, i)
        negative_mask_field_level = @. lazy(-sign(min(field_level, 0)))
        # Weighted by the area
        Fields.level(out′, i) .=
            sum(negative_mask_field_level) / sum(@. lazy(one((field_level))))
    end
    return out′
end

# =============================================================================
# Code generation for tracer diagnostics
# =============================================================================

# Each entry: (short_name, long_name, state_field_symbol, model_types, model_accessor)
# - short_name: prefix for diagnostic names (e.g., "clw" → "clw_min", "clw_neg_mean", "clw_neg_frac")
# - long_name: human-readable tracer name for descriptions
# - state_field_symbol: symbol of the ρq field in state.c (e.g., :ρq_liq)
# - model_types: Union of model types that support this diagnostic
# - model_accessor: expression to get the model from cache.atmos

const MICROPHYSICS_MODELS =
    Union{Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3}
const MOISTURE_MODELS = Union{EquilMoistModel, NonEquilMoistModel}

const TRACER_DIAGNOSTICS = [
    # (short_name, long_name, ρq_field, model_types, model_accessor_field)
    ("clw", "Cloud Liquid Water", :ρq_liq, MICROPHYSICS_MODELS, :microphysics_model),
    ("cli", "Cloud Ice", :ρq_ice, MICROPHYSICS_MODELS, :microphysics_model),
    ("husra", "Rain Specific Humidity", :ρq_rai, MICROPHYSICS_MODELS, :microphysics_model),
    ("hussn", "Snow Specific Humidity", :ρq_sno, MICROPHYSICS_MODELS, :microphysics_model),
    ("hus", "Specific Humidity", :ρq_tot, MOISTURE_MODELS, :moisture_model),
]

# Diagnostic types: (suffix, compute_func, long_name_template, units, comments_template)
const DIAGNOSTIC_TYPES = [
    (
        "_min",
        compute_min_per_level!,
        "Vertical Profile of Minimal",
        "kg/kg",
        "Minimum value of",
    ),
    (
        "_neg_mean",
        compute_negative_mean_per_level!,
        "Vertical Profile of Mean Negative",
        "kg/kg",
        "Mean of negative",
    ),
    (
        "_neg_frac",
        compute_negative_fraction_per_level!,
        "Vertical Profile of Negative Fraction",
        "%",
        "Fraction of grid points with negative",
    ),
]

# Generate all diagnostic functions and registrations
for (short_name, long_name, ρq_field, model_types, model_accessor_field) in
    TRACER_DIAGNOSTICS
    for (suffix, compute_func, long_name_prefix, units, comments_prefix) in
        DIAGNOSTIC_TYPES
        # Construct names
        diag_name = Symbol(short_name, suffix)
        compute_name = Symbol("compute_", diag_name, "!")

        # Build the full diagnostic metadata
        full_long_name = "$long_name_prefix $long_name"
        full_comments = "$comments_prefix $long_name at each vertical level"

        # Generate the functions using @eval
        @eval begin
            # Dispatcher function
            $compute_name(out, state, cache, time) =
                $compute_name(out, state, cache, time, cache.atmos.$model_accessor_field)

            # Error fallback for unsupported models
            $compute_name(_, _, _, _, model::T) where {T} =
                error_diagnostic_variable($diag_name, model)

            # Implementation for supported model types
            function $compute_name(
                out,
                state,
                cache,
                time,
                ::$model_types,
            )
                # Use scratch space to avoid allocation
                q_field = cache.scratch.ᶜtemp_scalar
                @. q_field = state.c.$ρq_field / state.c.ρ
                $compute_func(out, q_field)
            end

            # Register the diagnostic variable
            add_diagnostic_variable!(
                short_name = $short_name * $suffix,
                long_name = $full_long_name,
                units = $units,
                comments = $full_comments,
                compute! = $compute_name,
            )
        end
    end
end
