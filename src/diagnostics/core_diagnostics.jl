# This file is included in Diagnostics.jl

# Rho

# FIXME: Gabriele wrote this as an example. Gabriele doesn't know anything about the
# physics. Please fix this!
add_diagnostic_variable!(
    short_name = "air_density",
    long_name = "air_density",
    units = "kg m^-3",
    comments = "Density of air, a prognostic variable",
    compute! = (out, state, cache, time) -> begin
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        return copy(state.c.ρ)
    end,
)

# Relative humidity

function compute_relative_humidity!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T}
    error("Cannot compute relative_humidity with moisture_model = $T")
end

function compute_relative_humidity!(
    out,
    state,
    cache,
    time,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    thermo_params = CAP.thermodynamics_params(cache.params)
    return TD.relative_humidity.(thermo_params, cache.ᶜts)
end

compute_relative_humidity!(out, state, cache, time) =
    compute_relative_humidity!(
        out,
        state,
        cache,
        time,
        cache.atmos.moisture_model,
    )

# FIXME: Gabriele wrote this as an example. Gabriele doesn't know anything about the
# physics. Please fix this!
add_diagnostic_variable!(
    short_name = "relative_humidity",
    long_name = "relative_humidity",
    units = "",
    comments = "Relative Humidity",
    compute! = compute_relative_humidity!,
)
