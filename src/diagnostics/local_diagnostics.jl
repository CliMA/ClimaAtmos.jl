# Opt-in local diagnostics: this file is not part of the ClimaAtmos build.
# To enable it, add `include("local_diagnostics.jl")` after the other
# diagnostics includes in `src/diagnostics/diagnostic.jl`. They read caches
# (`sslt_drydep_fluxes`, `sslt_wetdep_rates`) that exist only when prognostic
# sea salt and a wet-deposition microphysics are configured, so they cannot
# join the unconditionally registered shipped diagnostics.
import ..PrognosticSeaSalt
import ..WetDepositionMicrophysics

###
# Sea salt dry deposition rate (2d)
###

function compute_sslt_drydep_column!(out, state, cache, sslt_model)
    sslt_model isa PrognosticSeaSalt || error(
        "dryss requires `prognostic_aerosols` to include sea salt",
    )
    isnothing(out) && (out = zeros(axes(Fields.level(state.f, half))))
    fluxes = cache.tracers.sslt_drydep_fluxes
    (; surface_ct3_unit) = cache.core
    sfc_sinks = map(bin_names(sslt_model)) do bin_name
        flux = getproperty(fluxes, Symbol(:ρ, bin_name))
        # The cached flux points downward (negative), so flip the sign to
        # report deposition as positive toward the surface.
        @. lazy(-dot(flux, surface_ct3_unit))
    end
    out .= foldl((a, b) -> @.(lazy(a + b)), sfc_sinks)
    return out
end

add_diagnostic_variable!(
    short_name = "dryss",
    long_name = "Sea-Salt Aerosol Dry Deposition Rate",
    standard_name = "tendency_of_atmosphere_mass_content_of_sea_salt_dry_aerosol_particles_due_to_dry_deposition",
    units = "kg m^-2 s^-1",
    comments = "Turbulent dry-deposition surface flux of prognostic sea \
                salt, summed over bins, positive toward the surface. Reads \
                the cached fluxes actually applied by the tendency, so \
                tracer mass budgets close against this flux. The \
                gravitational (settling) part of deposition is not included \
                here; it exits through the settling term's free-outflow \
                surface boundary.",
    compute! = (out, u, p, t) ->
        compute_sslt_drydep_column!(out, u, p, p.atmos.seasalt),
)

###
# Sea salt wet deposition rate (2d)
###

function compute_sslt_wetdep_column!(out, state, cache, sslt_model)
    sslt_model isa PrognosticSeaSalt || error(
        "wetss requires `prognostic_aerosols` to include sea salt",
    )
    cache.atmos.microphysics_model isa WetDepositionMicrophysics || error(
        "wetss requires 0- or 1-moment microphysics (wet deposition is \
         otherwise off)",
    )
    isnothing(out) && (out = zeros(axes(Fields.level(state.f, half))))
    rates = cache.tracers.sslt_wetdep_rates
    dt = float(cache.dt)
    ᶜsink_rates = map(bin_names(sslt_model)) do bin_name
        ᶜρχ = getproperty(state.c, Symbol(:ρ, bin_name))
        ᶜk = getproperty(rates, Symbol(:ρ, bin_name))
        @. lazy(ᶜρχ * (-expm1(-(ᶜk * dt))) / dt)
    end
    ᶜtotal = foldl((ᶜa, ᶜb) -> @.(lazy(ᶜa + ᶜb)), ᶜsink_rates)
    Operators.column_integral_definite!(out, ᶜtotal)
    return out
end

add_diagnostic_variable!(
    short_name = "wetss",
    long_name = "Sea-Salt Aerosol Wet Deposition Rate",
    standard_name = "tendency_of_atmosphere_mass_content_of_sea_salt_dry_aerosol_particles_due_to_wet_deposition",
    units = "kg m^-2 s^-1",
    comments = "Column-integrated wet-removal rate of prognostic sea salt \
                (in-cloud scavenging + below-cloud washout), positive toward \
                the surface. Integrates the per-step exponential sink \
                actually applied by the tendency, so tracer mass budgets \
                close against this flux.",
    compute! = (out, u, p, t) ->
        compute_sslt_wetdep_column!(out, u, p, p.atmos.seasalt),
)

###
# Sea salt gravitational settling rate (2d)
###

function compute_sslt_settling_column!(out, state, cache, sslt_model)
    sslt_model isa PrognosticSeaSalt || error(
        "setss requires `prognostic_aerosols` to include sea salt",
    )
    isnothing(out) && (out = zeros(axes(Fields.level(state.f, half))))
    fluxes = cache.tracers.sslt_settling_fluxes
    # Each per-bin field already holds the grid-mean settling surface mass
    # flux (downward, positive) written by `aerosol_settling_tendency!`; sum
    # over bins.
    sfc_sinks = map(bin_names(sslt_model)) do bin_name
        getproperty(fluxes, Symbol(:ρ, bin_name))
    end
    out .= foldl((a, b) -> @.(lazy(a + b)), sfc_sinks)
    return out
end

add_diagnostic_variable!(
    short_name = "setss",
    long_name = "Sea-Salt Aerosol Gravitational Settling Rate",
    standard_name = "tendency_of_atmosphere_mass_content_of_sea_salt_dry_aerosol_particles_due_to_gravitational_settling",
    units = "kg m^-2 s^-1",
    comments = "Gravitational-settling surface mass flux of prognostic sea \
                salt, summed over size bins, positive toward the surface. \
                Reads the grid-mean settling flux (mass-weighted fall speed \
                times the lowest-level mass) actually applied by the tendency, \
                so tracer mass budgets close against emission minus dryss, \
                wetss, and setss. This is the gravitational surface outflow \
                that the settling term's free-outflow boundary removes; it is \
                the term dry deposition (turbulent) and wet deposition \
                (scavenging) do not include.",
    compute! = (out, u, p, t) ->
        compute_sslt_settling_column!(out, u, p, p.atmos.seasalt),
)
