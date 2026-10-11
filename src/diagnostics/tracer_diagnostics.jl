# This file is included in Diagnostics.jl

# Tracers

"""
    compute_tracer!(out, state, cache, time, tracer_name)

Read a prescribed tracer field out of `cache.tracers` by name.

Mutates and returns `out`; when `out` is `nothing`, returns a copy of the cached field.
Errors if the model carries no tracer called `tracer_name`.
"""
function compute_tracer!(out, state, cache, time, tracer_name)
    tracer_name in propertynames(cache.tracers) ||
        error("$tracer_name does not exist in the model")
    if isnothing(out)
        return copy(getproperty(cache.tracers, tracer_name))
    else
        out .= getproperty(cache.tracers, tracer_name)
    end
end

# A species "exists" when its prognostic model carries every requested bin, or
# (without a model) at least one of its bins is prescribed from the MERRA-2
# climatology.
species_exists(cache, names, ::Nothing) =
    any(name -> has_prescribed_aerosol_bin(cache, name), names)
species_exists(cache, names, model::AbstractPrognosticAerosol) =
    all(name -> name in bin_names(model), names)

function compute_species_bin_mmr!(out, state, cache, bin_name, species_model)
    species_exists(cache, (bin_name,), species_model) ||
        error("$bin_name does not exist in the model")
    isnothing(out) && (out = zeros(axes(state.c)))
    out .= ᶜaerosol_bin_mmr(state, cache, bin_name, species_model)
    return out
end

function compute_species_mmr!(
    out,
    state,
    cache,
    bin_names,
    species_model,
    species_label,
)
    species_exists(cache, bin_names, species_model) ||
        error("$species_label does not exist in the model")
    isnothing(out) && (out = zeros(axes(state.c)))
    out .= ᶜaerosol_species_mmr(state, cache, bin_names, species_model)
    return out
end

function compute_species_column!(
    out,
    state,
    cache,
    bin_names,
    species_model,
    species_label,
)
    species_exists(cache, bin_names, species_model) ||
        error("$species_label does not exist in the model")
    isnothing(out) && (out = zeros(axes(Fields.level(state.f, half))))
    ᶜmmr = ᶜaerosol_species_mmr(state, cache, bin_names, species_model)
    Operators.column_integral_definite!(out, @. lazy(ᶜmmr * state.c.ρ))
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
# Dust concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrdust",
    long_name = "Dust Aerosol Mass Mixing Ratio",
    standard_name = "mass_fraction_of_dust_dry_aerosol_particles_in_air",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of dust aerosol particles in air.",
    compute! = (out, u, p, t) ->
        compute_species_mmr!(
            out,
            u,
            p,
            AEROSOL_SPECIES_BIN_NAMES.dust,
            p.atmos.dust,
            "Dust",
        ),
)

###
# Sea salt concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrss",
    long_name = "Sea-Salt Aerosol Mass Mixing Ratio",
    standard_name = "mass_fraction_of_sea_salt_dry_aerosol_particles_in_air",
    units = "kg kg^-1",
    comments = "Dry mass fraction of sea salt aerosol particles in air, summed \
                over size bins.",
    compute! = (out, u, p, t) ->
        compute_species_mmr!(
            out,
            u,
            p,
            species_bin_names(p.atmos.seasalt, AEROSOL_SPECIES_BIN_NAMES.seasalt),
            p.atmos.seasalt,
            "Sea salt",
        ),
)

###
# Sea salt per-bin concentrations (3d)
###
# Registered up to `MAX_SSLT_DIAGNOSTIC_BINS` so that a prognostic run with more
# bins than the 5 MERRA-2 ones (set by the ClimaParams arrays) can output each
# of them; a bin that the active model does not carry is an error at output.
const MAX_SSLT_DIAGNOSTIC_BINS = 12
for bin_index in 1:MAX_SSLT_DIAGNOSTIC_BINS
    bin_name = Symbol(:SSLT, lpad(bin_index, 2, '0'))
    add_diagnostic_variable!(
        short_name = "mmr$(lowercase(string(bin_name)))",
        long_name = "Sea-Salt Aerosol Mass Mixing Ratio, bin $bin_index",
        units = "kg kg^-1",
        comments = "Dry mass fraction of sea salt aerosol particles in air in \
                    size bin $bin_index; prognostic when sea salt is in \
                    `prognostic_aerosols`, otherwise the prescribed MERRA-2 \
                    climatology.",
        compute! = (out, u, p, t) ->
            compute_species_bin_mmr!(out, u, p, bin_name, p.atmos.seasalt),
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
    compute! = (out, u, p, t) ->
        compute_species_bin_mmr!(out, u, p, :SO4, p.atmos.sulfate),
)

###
# Hydrophobic black carbon concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrbcpo",
    long_name = "Hydrophobic Elemental Carbon Mass Mixing Ratio",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of hydrophobic black carbon aerosol particles in air.",
    compute! = (out, u, p, t) ->
        compute_species_bin_mmr!(out, u, p, :CB1, p.atmos.black_carbon),
)

###
# Hydrophilic black carbon concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrbcpi",
    long_name = "Hydrophilic Elemental Carbon Mass Mixing Ratio",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of hydrophilic black carbon aerosol particles in air.",
    compute! = (out, u, p, t) ->
        compute_species_bin_mmr!(out, u, p, :CB2, p.atmos.black_carbon),
)

###
# Hydrophobic organic carbon concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrocpo",
    long_name = "Hydrophobic Organic Carbon Mass Mixing Ratio",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of hydrophobic organic carbon aerosol particles in air.",
    compute! = (out, u, p, t) ->
        compute_species_bin_mmr!(out, u, p, :OC1, p.atmos.organic_carbon),
)

###
# Hydrophilic organic carbon concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrocpi",
    long_name = "Hydrophilic Organic Carbon Mass Mixing Ratio",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of hydrophilic organic carbon aerosol particles in air.",
    compute! = (out, u, p, t) ->
        compute_species_bin_mmr!(out, u, p, :OC2, p.atmos.organic_carbon),
)

###
# Sea salt emission wind of the standalone column driver (2d)
###
function compute_u10ss!(out, state, cache, time)
    cache.atmos.column_emission || error(
        "u10ss requires `prognostic_aerosol_column_emission: true` (in coupled \
         runs the emission wind lives in ClimaCoupler)",
    )
    isnothing(out) && (out = zeros(axes(Fields.level(state.f, half))))
    out .= sslt_column_u10(cache)
    return out
end

add_diagnostic_variable!(
    short_name = "u10ss",
    long_name = "Sea-Salt Emission 10 m Wind Speed",
    units = "m s^-1",
    comments = "MOST 10 m wind speed diagnosed from the model's own surface \
                conditions and fed to the sea-salt emission flux by the \
                standalone column driver (`prognostic_aerosol_column_emission`).",
    compute! = compute_u10ss!,
)

###
# Sea salt history tracers (3d): grid mean, first updraft, and the cached
# exposure rates of the environment. All `H/ρ` per kg of air; divide by
# `mmrss` offline for the mass-weighted mean (e.g. age) of the population.
###
const SSLT_HISTORY_DIAGNOSTICS = (
    (:SSHAGE, "Mass Age", "s", "mass × age: age of the population is this over mmrss"),
    (
        :SSHUEM,
        "Emission Wind",
        "m s^-1",
        "mass × 10 m emission wind: mean emission wind is this over mmrss",
    ),
    (
        :SSHSET,
        "Settling Exposure",
        "",
        "mass × settling exposure (`settling_exposure_rate` integrated along \
         the population's history)",
    ),
    (
        :SSHSET2,
        "Settling Exposure Squared",
        "",
        "mass × settling exposure squared; with mmrsshset gives the within-\
         cell variance of the exposure",
    ),
    (
        :SSHWET,
        "Below-Cloud Washout Exposure",
        "",
        "mass × unit-efficiency below-cloud washout exposure (the swept-\
         volume rate over the cloud-free area integrated, dimensionless; \
         under 0M the rain is the precipitation shadow inverted to a 1M \
         rain mixing ratio)",
    ),
    (
        :SSHINC,
        "In-Cloud Scavenging Exposure",
        "",
        "mass × in-cloud (nucleation) scavenging exposure, the integrated \
         condensate-to-precipitation conversion rate the population sat in",
    ),
)

function compute_sslt_history_updraft!(out, state, cache, tracer_name)
    isnothing(cache.atmos.seasalt_history) &&
        error("$tracer_name requires \"SSLTH\" in `prognostic_aerosols`")
    cache.atmos.turbconv_model isa PrognosticEDMFX || error(
        "updraft $tracer_name requires `turbconv: prognostic_edmfx`",
    )
    isnothing(out) && (out = zeros(axes(state.c)))
    out .= getproperty(state.c.sgsʲs.:1, tracer_name)
    return out
end

for (tracer_name, label, units, description) in SSLT_HISTORY_DIAGNOSTICS
    short = lowercase(string(tracer_name))
    add_diagnostic_variable!(
        short_name = "mmr$short",
        long_name = "Sea-Salt History: $label",
        units = isempty(units) ? "kg kg^-1" : "kg kg^-1 $units",
        comments = "Sea-salt history tracer $tracer_name per unit air mass: \
                    $description. Passive companion of the prognostic sea \
                    salt bins (\"SSLTH\" in `prognostic_aerosols`).",
        compute! = (out, u, p, t) -> compute_species_bin_mmr!(
            out,
            u,
            p,
            tracer_name,
            p.atmos.seasalt_history,
        ),
    )
    add_diagnostic_variable!(
        short_name = "mmr$(short)up",
        long_name = "Updraft Sea-Salt History: $label",
        units = isempty(units) ? "kg kg^-1" : "kg kg^-1 $units",
        comments = "Sea-salt history tracer $tracer_name of the first \
                    updraft, per unit updraft air mass: $description.",
        compute! = (out, u, p, t) ->
            compute_sslt_history_updraft!(out, u, p, tracer_name),
    )
end

add_diagnostic_variable!(
    short_name = "washrss",
    long_name = "Sea-Salt Below-Cloud Washout Exposure Rate",
    units = "s^-1",
    comments = "Unit-efficiency below-cloud washout exposure rate of the \
                environment driving the SSHWET history tracer: the swept-\
                volume collection rate of the rain over the cloud-free area. \
                Under 0M the rain is the precipitation shadow inverted to the \
                1M rain mixing ratio that carries the same mass flux, so the \
                rate has the same meaning under both microphysics.",
    compute! = (out, u, p, t) ->
        compute_tracer!(out, u, p, t, :sslt_washout_exposure_rate⁰),
)

add_diagnostic_variable!(
    short_name = "incrss",
    long_name = "Sea-Salt In-Cloud Scavenging Exposure Rate",
    units = "s^-1",
    comments = "In-cloud (nucleation) scavenging rate of the environment \
                driving the SSHINC history tracer: the cloud-fraction-\
                weighted condensate-to-precipitation conversion rate.",
    compute! = (out, u, p, t) ->
        compute_tracer!(out, u, p, t, :sslt_in_cloud_exposure_rate⁰),
)

###
# Sea salt column mass (2d)
###
add_diagnostic_variable!(
    short_name = "loadss",
    long_name = "Load of Sea-Salt Aerosol",
    units = "kg m^-2",
    comments = "Total dry mass of sea salt aerosol particles per unit area.",
    compute! = (out, u, p, t) ->
        compute_species_column!(
            out,
            u,
            p,
            species_bin_names(p.atmos.seasalt, AEROSOL_SPECIES_BIN_NAMES.seasalt),
            p.atmos.seasalt,
            "Sea salt",
        ),
)
