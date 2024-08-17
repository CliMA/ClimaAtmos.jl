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

function compute_aerosol!(out, state, cache, time, aerosol_name)
    :prescribed_aerosols_field in propertynames(cache.tracers) ||
        error("Aerosols do not exist in the model")
    aerosol_name in propertynames(cache.tracers.prescribed_aerosols_field) ||
        error("$aerosol_name does not exist in the model")
    if isnothing(out)
        return copy(
            getproperty(cache.tracers.prescribed_aerosols_field, aerosol_name),
        )
    else
        out .=
            getproperty(cache.tracers.prescribed_aerosols_field, aerosol_name)
    end
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
    comments = "Prescribed dry mass fraction of dust aerosol particles in air. Only the smallest size is included.",
    compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, :DST01),
)

###
# Sea salt concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrss",
    long_name = "Sea-Salt Aerosol Mass Mixing Ratio",
    standard_name = "mass_fraction_of_sea_salt_dry_aerosol_particles_in_air",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of sea salt aerosol particles in air. Only the smallest size is included.",
    compute! = (out, u, p, t) -> compute_aerosol!(out, u, p, t, :SSLT01),
)

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
