# This file is included in Diagnostics.jl

# TKE

# This is an example of how to compute the same diagnostic variable differently depending on
# the model. This is also exposed to the user, which could define their own
# compute_tke_from_integrator.

function compute_tke_from_integrator(integrator, out, ::EDMFX)
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    return deepcopy(integrator.p.ᶜspecific⁰.tke)
end

function compute_tke_from_integrator(integrator, out, ::DiagnosticEDMFX)
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    return deepcopy(integrator.p.tke⁰)
end

function compute_tke_from_integrator(
    integrator,
    out,
    turbconv_model::T,
) where {T}
    error("Cannot compute tke with turbconv_model = $T")
end

# FIXME: Gabriele wrote this as an example. Gabriele doesn't know anything about the
# physics. Please fix this!
add_diagnostic_variable!(
    short_name = "tke",
    long_name = "turbolent_kinetic_energy",
    units = "J",
    comments = "Turbolent Kinetic Energy",
    compute_from_integrator = (integrator, out) -> compute_tke_from_integrator(
        integrator,
        out,
        integrator.p.atmos.turbconv_model,
    ),
)
