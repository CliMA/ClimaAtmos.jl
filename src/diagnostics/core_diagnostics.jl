# This file is included in Diagnostics.jl

# FIXME: Gabriele wrote this as an example. Gabriele doesn't know anything about the
# physics. Please fix this!
add_diagnostic_variable!(
    short_name = "air_density",
    long_name = "air_density",
    units = "kg m^-3",
    comments = "Density of air, a prognostic variable",
    compute_from_integrator = (integrator, out) -> begin
        # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
        # We will want: out .= integrator.u.c.ρ
        return deepcopy(integrator.u.c.ρ)
    end,
)
