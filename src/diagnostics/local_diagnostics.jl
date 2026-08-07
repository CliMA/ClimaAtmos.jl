# Opt-in local diagnostics: this file is not part of the ClimaAtmos build.
# To enable it, add `include("local_diagnostics.jl")` after the other
# diagnostics includes in `src/diagnostics/diagnostic.jl`. These recompute
# parameterization inputs from the current state, so they stay out of the
# shipped diagnostics (which only read model fields).
import ..set_wind_at_height!
import ..AbstractPrognosticAerosol

###
# MOST surface-layer quantities (2d)
###

# Allocate on the surface space on first call, then reuse `out`.
function compute_most_wind!(out, u, p, writer!)
    field = isnothing(out) ? similar(p.precomputed.sfc_conditions.ustar) : out
    return writer!(field, u, p)
end

add_diagnostic_variable!(
    short_name = "u10sfc",
    long_name = "10 m Wind Speed (MOST)",
    units = "m s^-1",
    comments = "MOST point wind at 10 m recovered from (u*, L, z0); the value used for the sea salt emission flux.",
    compute! = (out, u, p, t) -> compute_most_wind!(
        out, u, p,
        (field, state, cache) ->
            set_wind_at_height!(field, eltype(state)(10), state, cache),
    ),
)

add_diagnostic_variable!(
    short_name = "ustar",
    long_name = "Surface Friction Velocity",
    units = "m s^-1",
    comments = "Friction velocity from the surface flux scheme, used in Monin-Obukhov wind reconstruction.",
    compute! = (out, u, p, t) -> begin
        isnothing(out) ? copy(p.precomputed.sfc_conditions.ustar) :
        (out .= p.precomputed.sfc_conditions.ustar)
    end,
)

add_diagnostic_variable!(
    short_name = "buoyflux",
    long_name = "Surface Buoyancy Flux",
    units = "m^2 s^-3",
    comments = "Surface buoyancy flux from the surface flux scheme. Positive values indicate an unstable surface layer.",
    compute! = (out, u, p, t) -> begin
        isnothing(out) ? copy(p.precomputed.sfc_conditions.buoyancy_flux) :
        (out .= p.precomputed.sfc_conditions.buoyancy_flux)
    end,
)

add_diagnostic_variable!(
    short_name = "obukhovlen",
    long_name = "Obukhov Length",
    units = "m",
    comments = "Monin-Obukhov length from the surface flux scheme. Positive over stable conditions, negative over unstable.",
    compute! = (out, u, p, t) -> begin
        isnothing(out) ? copy(p.precomputed.sfc_conditions.obukhov_length) :
        (out .= p.precomputed.sfc_conditions.obukhov_length)
    end,
)

add_diagnostic_variable!(
    short_name = "oceanfrac",
    long_name = "Ocean Fraction",
    units = "1",
    comments = "Fraction of each grid cell covered by ocean. Sea salt emission is weighted by this; non-zero values over land indicate a coupler masking issue.",
    compute! = (out, u, p, t) -> begin
        isnothing(out) ? copy(p.ocean_fraction) : (out .= p.ocean_fraction)
    end,
)

###
# Sea salt surface emission flux (2d) — total and per-bin
###

# Read the per-bin fluxes cached by `set_sea_salt_surface_fluxes!` (the exact
# values the emission tendency applies); `bin_indices` selects the bins to sum.
function compute_sea_salt_emission_flux!(out, state, cache, time, bin_indices)
    cache.atmos.seasalt isa AbstractPrognosticAerosol ||
        error("emission diagnostics require sea salt in `prognostic_aerosols`")
    fluxes = cache.tracers.seasalt_sfc_fluxes
    isnothing(out) && (out = similar(cache.ocean_fraction))
    out .= 0
    for bin_index in bin_indices
        flux = fluxes[bin_index]
        @. out += flux.components.data.:1
    end
    return out
end

add_diagnostic_variable!(
    short_name = "emiss",
    long_name = "Sea-Salt Aerosol Surface Emission Flux",
    units = "kg m^-2 s^-1",
    comments = "Total upward sea salt mass flux at the surface, summed over all bins.",
    compute! = (out, u, p, t) -> compute_sea_salt_emission_flux!(
        out, u, p, t, eachindex(bin_names(PrescribedSeaSalt)),
    ),
)

for (bin_index, bin) in enumerate(bin_names(PrescribedSeaSalt))
    add_diagnostic_variable!(
        short_name = "emi$(lowercase(string(bin)))",
        long_name = "Sea-Salt Aerosol Surface Emission Flux, bin $bin_index",
        units = "kg m^-2 s^-1",
        comments = "Upward sea salt mass flux at the surface for size bin \
                    $bin_index (edges set by `ssa_size_bin_divisions`).",
        compute! = (out, u, p, t) ->
            compute_sea_salt_emission_flux!(out, u, p, t, (bin_index,)),
    )
end
