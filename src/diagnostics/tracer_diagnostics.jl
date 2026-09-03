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

# A species "exists" when it has a prognostic model or at least one of its
# bins is prescribed from the MERRA-2 climatology.
species_exists(cache, bin_names, species_model) =
    !isnothing(species_model) ||
    any(name -> has_prescribed_aerosol_bin(cache, name), bin_names)

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
            AEROSOL_SPECIES_BIN_NAMES.seasalt,
            p.atmos.seasalt,
            "Sea salt",
        ),
)

###
# Sea salt per-bin concentrations (3d)
###
for (bin_index, bin_name) in enumerate(AEROSOL_SPECIES_BIN_NAMES.seasalt)
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
            AEROSOL_SPECIES_BIN_NAMES.seasalt,
            p.atmos.seasalt,
            "Sea salt",
        ),
)
