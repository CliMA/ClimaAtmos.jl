# Aggregating the information about negative tracers at each vertical level

import Statistics: mean

"""
    compute_min_per_level!(out, field)

Computes the minimum value of a field at each vertical level and stores it in `out`.
"""
function compute_min_per_level!(out, field)
    out′ = isnothing(out) ? similar(Fields.column(field, 1,1,1)) : out
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
    out′ = isnothing(out) ? similar(Fields.column(field, 1,1,1)) : out
    for i in 1:Spaces.nlevels(axes(field))
        field_level = Fields.level(field, i)
        clipped_field_level = @. lazy(min(field_level, 0))
        Fields.level(out′, i) .= mean(clipped_field_level) 
    end
    return out′
end

"""
    compute_negative_fraction_per_level!(out, field)

Computes the fraction of the level where `field` is negative.
"""
function compute_negative_fraction_per_level!(out, field)
    out′ = isnothing(out) ? similar(Fields.column(field, 1,1,1)) : out
    for i in 1:Spaces.nlevels(axes(field))
        field_level = Fields.level(field, i)
        negative_mask_field_level = @. lazy(-sign(min(field_level, 0)))

        # Weighted by the area
        Fields.level(out′, i) .= sum(negative_mask_field_level) / sum(@. lazy(one((field_level))))
    
        # Unweighted by the area
        #Fields.level(out′, i) .= reduce(+, negative_mask_field_level) / reduce(+, (@. lazy(one((field_level)))))
    end
    return out′
end

# Wrappers and Definitions

# --- Cloud Liquid Water (clw) ---
compute_clw_min!(out, state, cache, time) =
    compute_clw_min!(out, state, cache, time, cache.atmos.microphysics_model)

compute_clw_min!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("clw_min", model)

function compute_clw_min!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_min_per_level!(out, state.c.ρq_liq ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "clw_min",
    long_name = "Vertical Profile of Minimal Cloud Liquid Water",
    units = "kg/kg",
    comments = "Minimum value of Cloud Liquid Water at each vertical level",
    compute! = compute_clw_min!,
)

compute_clw_negative_mean!(out, state, cache, time) =
    compute_clw_negative_mean!(out, state, cache, time, cache.atmos.microphysics_model)

compute_clw_negative_mean!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("clw_neg_mean", model)

function compute_clw_negative_mean!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_negative_mean_per_level!(out, state.c.ρq_liq ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "clw_neg_mean",
    long_name = "Vertical Profile of Mean Negative Cloud Liquid Water",
    units = "kg/kg",
    comments = "Mean of negative Cloud Liquid Water values at each vertical level",
    compute! = compute_clw_negative_mean!,
)

compute_clw_negative_fraction!(out, state, cache, time) =
    compute_clw_negative_fraction!(out, state, cache, time, cache.atmos.microphysics_model)

compute_clw_negative_fraction!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("clw_neg_frac", model)

function compute_clw_negative_fraction!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_negative_fraction_per_level!(out, state.c.ρq_liq ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "clw_neg_frac",
    long_name = "Vertical Profile of Negative Fraction Cloud Liquid Water",
    units = "%",
    comments = "Fraction of grid points with negative Cloud Liquid Water at each vertical level",
    compute! = compute_clw_negative_fraction!,
)

# --- Cloud Ice (cli) ---
compute_cli_min!(out, state, cache, time) =
    compute_cli_min!(out, state, cache, time, cache.atmos.microphysics_model)

compute_cli_min!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("cli_min", model)

function compute_cli_min!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_min_per_level!(out, state.c.ρq_ice ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "cli_min",
    long_name = "Vertical Profile of Minimal Cloud Ice",
    units = "kg/kg",
    comments = "Minimum value of Cloud Ice at each vertical level",
    compute! = compute_cli_min!,
)

compute_cli_negative_mean!(out, state, cache, time) =
    compute_cli_negative_mean!(out, state, cache, time, cache.atmos.microphysics_model)

compute_cli_negative_mean!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("cli_neg_mean", model)

function compute_cli_negative_mean!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_negative_mean_per_level!(out, state.c.ρq_ice ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "cli_neg_mean",
    long_name = "Vertical Profile of Mean Negative Cloud Ice",
    units = "kg/kg",
    comments = "Mean of negative Cloud Ice values at each vertical level",
    compute! = compute_cli_negative_mean!,
)

compute_cli_negative_fraction!(out, state, cache, time) =
    compute_cli_negative_fraction!(out, state, cache, time, cache.atmos.microphysics_model)

compute_cli_negative_fraction!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("cli_neg_frac", model)

function compute_cli_negative_fraction!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_negative_fraction_per_level!(out, state.c.ρq_ice ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "cli_neg_frac",
    long_name = "Vertical Profile of Negative Fraction Cloud Ice",
    units = "%",
    comments = "Fraction of grid points with negative Cloud Ice at each vertical level",
    compute! = compute_cli_negative_fraction!,
)

# --- Rain Specific Humidity (husra) ---
compute_husra_min!(out, state, cache, time) =
    compute_husra_min!(out, state, cache, time, cache.atmos.microphysics_model)

compute_husra_min!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("husra_min", model)

function compute_husra_min!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_min_per_level!(out, state.c.ρq_rai ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "husra_min",
    long_name = "Vertical Profile of Minimal Rain Specific Humidity",
    units = "kg/kg",
    comments = "Minimum value of Rain Specific Humidity at each vertical level",
    compute! = compute_husra_min!,
)

compute_husra_negative_mean!(out, state, cache, time) =
    compute_husra_negative_mean!(out, state, cache, time, cache.atmos.microphysics_model)

compute_husra_negative_mean!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("husra_neg_mean", model)

function compute_husra_negative_mean!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_negative_mean_per_level!(out, state.c.ρq_rai ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "husra_neg_mean",
    long_name = "Vertical Profile of Mean Negative Rain Specific Humidity",
    units = "kg/kg",
    comments = "Mean of negative Rain Specific Humidity values at each vertical level",
    compute! = compute_husra_negative_mean!,
)

compute_husra_negative_fraction!(out, state, cache, time) =
    compute_husra_negative_fraction!(out, state, cache, time, cache.atmos.microphysics_model)

compute_husra_negative_fraction!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("husra_neg_frac", model)

function compute_husra_negative_fraction!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_negative_fraction_per_level!(out, state.c.ρq_rai ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "husra_neg_frac",
    long_name = "Vertical Profile of Negative Fraction Rain Specific Humidity",
    units = "%",
    comments = "Fraction of grid points with negative Rain Specific Humidity at each vertical level",
    compute! = compute_husra_negative_fraction!,
)

# --- Snow Specific Humidity (hussn) ---
compute_hussn_min!(out, state, cache, time) =
    compute_hussn_min!(out, state, cache, time, cache.atmos.microphysics_model)

compute_hussn_min!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hussn_min", model)

function compute_hussn_min!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_min_per_level!(out, state.c.ρq_sno ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "hussn_min",
    long_name = "Vertical Profile of Minimal Snow Specific Humidity",
    units = "kg/kg",
    comments = "Minimum value of Snow Specific Humidity at each vertical level",
    compute! = compute_hussn_min!,
)

compute_hussn_negative_mean!(out, state, cache, time) =
    compute_hussn_negative_mean!(out, state, cache, time, cache.atmos.microphysics_model)

compute_hussn_negative_mean!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hussn_neg_mean", model)

function compute_hussn_negative_mean!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_negative_mean_per_level!(out, state.c.ρq_sno ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "hussn_neg_mean",
    long_name = "Vertical Profile of Mean Negative Snow Specific Humidity",
    units = "kg/kg",
    comments = "Mean of negative Snow Specific Humidity values at each vertical level",
    compute! = compute_hussn_negative_mean!,
)

compute_hussn_negative_fraction!(out, state, cache, time) =
    compute_hussn_negative_fraction!(out, state, cache, time, cache.atmos.microphysics_model)

compute_hussn_negative_fraction!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hussn_neg_frac", model)

function compute_hussn_negative_fraction!(
    out,
    state,
    cache,
    time,
    microphysics_model::Union{
        Microphysics1Moment, Microphysics2Moment, Microphysics2MomentP3,
    },
)
    compute_negative_fraction_per_level!(out, state.c.ρq_sno ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "hussn_neg_frac",
    long_name = "Vertical Profile of Negative Fraction Snow Specific Humidity",
    units = "%",
    comments = "Fraction of grid points with negative Snow Specific Humidity at each vertical level",
    compute! = compute_hussn_negative_fraction!,
)

# --- Specific Humidity (hus) ---
# Note: hus diagnostic usually depends on MoistureModel
compute_hus_min!(out, state, cache, time) =
    compute_hus_min!(out, state, cache, time, cache.atmos.moisture_model)

compute_hus_min!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hus_min", model)

function compute_hus_min!(
    out,
    state,
    cache,
    time,
    moisture_model::Union{EquilMoistModel, NonEquilMoistModel},
)
    compute_min_per_level!(out, state.c.ρq_tot ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "hus_min",
    long_name = "Vertical Profile of Minimal Specific Humidity",
    units = "kg/kg",
    comments = "Minimum value of Specific Humidity at each vertical level",
    compute! = compute_hus_min!,
)

compute_hus_negative_mean!(out, state, cache, time) =
    compute_hus_negative_mean!(out, state, cache, time, cache.atmos.moisture_model)

compute_hus_negative_mean!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hus_neg_mean", model)

function compute_hus_negative_mean!(
    out,
    state,
    cache,
    time,
    moisture_model::Union{EquilMoistModel, NonEquilMoistModel},
)
    compute_negative_mean_per_level!(out, state.c.ρq_tot ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "hus_neg_mean",
    long_name = "Vertical Profile of Mean Negative Specific Humidity",
    units = "kg/kg",
    comments = "Mean of negative Specific Humidity values at each vertical level",
    compute! = compute_hus_negative_mean!,
)

compute_hus_negative_fraction!(out, state, cache, time) =
    compute_hus_negative_fraction!(out, state, cache, time, cache.atmos.moisture_model)

compute_hus_negative_fraction!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("hus_neg_frac", model)

function compute_hus_negative_fraction!(
    out,
    state,
    cache,
    time,
    moisture_model::Union{EquilMoistModel, NonEquilMoistModel},
)
    compute_negative_fraction_per_level!(out, state.c.ρq_tot ./ state.c.ρ)
end
add_diagnostic_variable!(
    short_name = "hus_neg_frac",
    long_name = "Vertical Profile of Negative Fraction Specific Humidity",
    units = "%",
    comments = "Fraction of grid points with negative Specific Humidity at each vertical level",
    compute! = compute_hus_negative_fraction!,
)
