# This file is included in Diagnostics.jl

# TKE

# This is an example of how to compute the same diagnostic variable differently depending on
# the model. This is also exposed to the user, which could define their own
# compute_tke_from_integrator.

function compute_tke_from_integrator!(out, integrator, ::EDMFX)
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    return copy(integrator.p.ᶜspecific⁰.tke)
end

function compute_tke_from_integrator!(out, integrator, ::DiagnosticEDMFX)
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    return copy(integrator.p.tke⁰)
end

compute_tke_from_integrator!(out, integrator) =
    compute_tke_from_integrator!(out, integrator, integrator.p.atmos)

function compute_tke_from_integrator!(
    out,
    integrator,
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
    compute_from_integrator! = (integrator, out) ->
        compute_tke_from_integrator!,
)
