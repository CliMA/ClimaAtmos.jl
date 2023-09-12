# This file is included in Diagnostics.jl

# TKE

# This is an example of how to compute the same diagnostic variable differently depending on
# the model. This is also exposed to the user, which could define their own
# compute_tke.

function compute_tke!(out, state, cache, time, ::EDMFX)
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    return copy(cache.ᶜspecific⁰.tke)
end

function compute_tke!(out, state, cache, time, ::DiagnosticEDMFX)
    # FIXME: Avoid extra allocations when ClimaCore overloads .= for this use case
    # We will want: out .= integrator.u.c.ρ
    return copy(cache.tke⁰)
end

compute_tke!(out, state, cache, time) = compute_tke!(out, state, cache.atmos)

function compute_tke!(out, state, cache, time, turbconv_model::T) where {T}
    error("Cannot compute tke with turbconv_model = $T")
end

# FIXME: Gabriele wrote this as an example. Gabriele doesn't know anything about the
# physics. Please fix this!
add_diagnostic_variable!(
    short_name = "tke",
    long_name = "turbolent_kinetic_energy",
    units = "J",
    comments = "Turbolent Kinetic Energy",
    compute! = compute_tke!,
)
