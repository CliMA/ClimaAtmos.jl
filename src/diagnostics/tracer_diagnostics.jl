# This file is included in Diagnostics.jl

# Tracers

function compute_tracer!(out, state, cache, time, tracer_name)
    tracer_name in propertynames(cache.tracers) ||
        error("$tracer_name does not exist in the model")
    if isnothing(out)
        return copy(getproperty(cache.tracers, tracer_name))
    else
        out .= getproperty(cache.tracers, tracer_name)
    end
end

const DUST_BIN_NAMES = (:DST01, :DST02, :DST03, :DST04, :DST05)
const SEA_SALT_BIN_NAMES = (:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05)

_prescribed_aerosols_field(cache) =
    :prescribed_aerosols_field in propertynames(cache.tracers) ?
    cache.tracers.prescribed_aerosols_field : nothing

function compute_aerosol!(out, state, cache, time, aerosol_name)
    ρname = Symbol(:ρ, aerosol_name)
    prescribed = _prescribed_aerosols_field(cache)
    if ρname in propertynames(state.c)
        ρχ = getproperty(state.c, ρname)
        isnothing(out) && return @. specific(ρχ, state.c.ρ)
        @. out = specific(ρχ, state.c.ρ)
    elseif !isnothing(prescribed) && aerosol_name in propertynames(prescribed)
        χ = getproperty(prescribed, aerosol_name)
        isnothing(out) && return copy(χ)
        out .= χ
    else
        error("$aerosol_name does not exist in the model")
    end
end

# Sum aerosol mass over `bin_names` into `aero_conc`; per bin, interactive
# takes precedence over prescribed. With `density_weighted` 
# the result is in kg/m³ (for column integrals), otherwise kg/kg
function accumulate_aerosol_bins!(aero_conc, state, cache, bin_names, density_weighted)
    prescribed = _prescribed_aerosols_field(cache)
    found = false
    @. aero_conc = 0
    for name in bin_names
        ρname = Symbol(:ρ, name)
        if ρname in propertynames(state.c)
            ρχ = getproperty(state.c, ρname)
            if density_weighted
                @. aero_conc += ρχ
            else
                @. aero_conc += specific(ρχ, state.c.ρ)
            end
        elseif !isnothing(prescribed) && name in propertynames(prescribed)
            χ = getproperty(prescribed, name)
            if density_weighted
                @. aero_conc += χ * state.c.ρ
            else
                @. aero_conc += χ
            end
        else
            continue
        end
        found = true
    end
    return found
end

function compute_aerosol_sum!(out, state, cache, bin_names, species)
    aero_conc = cache.scratch.ᶜtemp_scalar
    accumulate_aerosol_bins!(aero_conc, state, cache, bin_names, false) ||
        error("$species does not exist in the model")
    isnothing(out) ? copy(aero_conc) : (out .= aero_conc)
end

compute_dust!(out, state, cache, time) =
    compute_aerosol_sum!(out, state, cache, DUST_BIN_NAMES, "Dust")

compute_sea_salt!(out, state, cache, time) =
    compute_aerosol_sum!(out, state, cache, SEA_SALT_BIN_NAMES, "Sea salt")

function compute_sea_salt_column!(out, state, cache, time)
    aero_conc = cache.scratch.ᶜtemp_scalar
    accumulate_aerosol_bins!(aero_conc, state, cache, SEA_SALT_BIN_NAMES, true) ||
        error("Sea salt does not exist in the model")
    isnothing(out) && (out = zeros(axes(Fields.level(state.f, half))))
    Operators.column_integral_definite!(out, aero_conc)
    return out
end

###
# Ozone concentration (3d)
###
add_diagnostic_variable!(
    short_name = "o3",
    long_name = "Mole Fraction of O3",
    standard_name = "mole_fraction_of_ozone_in_air",
    units = "mol mol^-1",
    compute! = (out, u, p, t) -> compute_tracer!(out, u, p, t, :o3),
)

###
# Dust concentration (3d) — total and per-bin
###
add_diagnostic_variable!(
    short_name = "mmrdust",
    long_name = "Dust Aerosol Mass Mixing Ratio",
    standard_name = "mass_fraction_of_dust_dry_aerosol_particles_in_air",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of dust aerosol particles in air.",
    compute! = (out, u, p, t) -> compute_dust!(out, u, p, t),
)

for (bin, long_bin) in (
    (:DST01, "bin 1 (0.1–1 μm)"),
    (:DST02, "bin 2 (1–2.5 μm)"),
    (:DST03, "bin 3 (2.5–5 μm)"),
    (:DST04, "bin 4 (5–10 μm)"),
    (:DST05, "bin 5 (10–20 μm)"),
)
    add_diagnostic_variable!(
        short_name = "mmr$(lowercase(string(bin)))",
        long_name = "Dust Aerosol Mass Mixing Ratio $long_bin",
        units = "kg kg^-1",
        comments = "Dry mass fraction of dust aerosol $long_bin.",
        compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, bin),
    )
end

###
# Sea salt concentration (3d) — total and per-bin
###
add_diagnostic_variable!(
    short_name = "mmrss",
    long_name = "Sea-Salt Aerosol Mass Mixing Ratio",
    standard_name = "mass_fraction_of_sea_salt_dry_aerosol_particles_in_air",
    units = "kg kg^-1",
    comments = "Dry mass fraction of sea salt aerosol particles in air.",
    compute! = (out, u, p, t) -> compute_sea_salt!(out, u, p, t),
)

for (bin, long_bin) in (
    (:SSLT01, "bin 1 (0.03–0.1 μm)"),
    (:SSLT02, "bin 2 (0.1–0.5 μm)"),
    (:SSLT03, "bin 3 (0.5–1.5 μm)"),
    (:SSLT04, "bin 4 (1.5–5 μm)"),
    (:SSLT05, "bin 5 (5–10 μm)"),
)
    add_diagnostic_variable!(
        short_name = "mmr$(lowercase(string(bin)))",
        long_name = "Sea-Salt Aerosol Mass Mixing Ratio $long_bin",
        units = "kg kg^-1",
        comments = "Dry mass fraction of sea salt aerosol $long_bin.",
        compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, bin),
    )
end

###
# Sulfate concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrso4",
    long_name = "Aerosol Sulfate Mass Mixing Ratio",
    standard_name = "mass_fraction_of_sulfate_dry_aerosol_particles_in_air",
    units = "kg kg^-1",
    comments = "Prescribed dry mass of sulfate (SO4) in aerosol particles as a fraction of air mass.",
    compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, :SO4),
)

###
# Hydrophobic black carbon concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrbcpo",
    long_name = "Hydrophobic Elemental Carbon Mass Mixing Ratio",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of hydrophobic black carbon aerosol particles in air.",
    compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, :CB1),
)

###
# Hydrophilic black carbon concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrbcpi",
    long_name = "Hydrophilic Elemental Carbon Mass Mixing Ratio",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of hydrophilic black carbon aerosol particles in air.",
    compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, :CB2),
)

###
# Hydrophobic organic carbon concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrocpo",
    long_name = "Hydrophobic Organic Carbon Mass Mixing Ratio",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of hydrophobic organic carbon aerosol particles in air.",
    compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, :OC1),
)

###
# Hydrophilic organic carbon concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrocpi",
    long_name = "Hydrophilic Organic Carbon Mass Mixing Ratio",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of hydrophilic organic carbon aerosol particles in air.",
    compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, :OC2),
)

###
# Sea salt column mass (2d)
###
add_diagnostic_variable!(
    short_name = "loadss",
    long_name = "Load of Sea-Salt Aerosol",
    standard_name = "atmosphere_mass_content_of_sea_salt_dry_aerosol _particles",
    units = "kg m^-2",
    comments = "The total dry mass of sea salt aerosol particles per unit area.",
    compute! = (out, u, p, t) -> compute_sea_salt_column!(out, u, p, t),
)

###
# Sea salt surface emission flux (2d) — total and per-bin
###

function compute_sea_salt_emission_flux!(out, state, cache, time)
    :interactive_aerosols_field in propertynames(cache.tracers) ||
        error(
            "interactive_aerosols_field not in cache — is sea_salt_emission_tendency! active?",
        )
    bins = cache.tracers.interactive_aerosols_field
    names = propertynames(bins)
    isnothing(out) && (out = similar(getproperty(bins, first(names))))
    out .= getproperty(bins, first(names))
    for n in Base.tail(names)
        out .+= getproperty(bins, n)
    end
    return out
end

add_diagnostic_variable!(
    short_name = "emiss",
    long_name = "Sea-Salt Aerosol Surface Emission Flux",
    units = "kg m^-2 s^-1",
    comments = "Total upward sea salt mass flux at the surface, summed over all bins.",
    compute! = (out, u, p, t) -> compute_sea_salt_emission_flux!(out, u, p, t),
)

function compute_sea_salt_emission_flux_bin!(out, state, cache, time, bin_name)
    :interactive_aerosols_field in propertynames(cache.tracers) ||
        error(
            "interactive_aerosols_field not in cache — is sea_salt_emission_tendency! active?",
        )
    flux = getproperty(cache.tracers.interactive_aerosols_field, bin_name)
    if isnothing(out)
        return copy(flux)
    else
        out .= flux
    end
end

for (bin, long_bin) in (
    (:SSLT01, "bin 1 (0.03–0.1 μm)"),
    (:SSLT02, "bin 2 (0.1–0.5 μm)"),
    (:SSLT03, "bin 3 (0.5–1.5 μm)"),
    (:SSLT04, "bin 4 (1.5–5 μm)"),
    (:SSLT05, "bin 5 (5–10 μm)"),
)
    add_diagnostic_variable!(
        short_name = "emi$(lowercase(string(bin)))",
        long_name = "Sea-Salt Aerosol Surface Emission Flux $long_bin",
        units = "kg m^-2 s^-1",
        comments = "Upward sea salt mass flux at the surface for $long_bin.",
        compute! = (out, u, p, t) -> compute_sea_salt_emission_flux_bin!(out, u, p, t, bin),
    )
end
