# Opt-in local diagnostics: this file is not part of the ClimaAtmos build.
# To enable it, add `include("local_diagnostics.jl")` after the other
# diagnostics includes in `src/diagnostics/diagnostic.jl`. These recompute
# parameterization inputs from the current state, so they stay out of the
# shipped diagnostics (which only read model fields).
import ..PrognosticSeaSalt

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
