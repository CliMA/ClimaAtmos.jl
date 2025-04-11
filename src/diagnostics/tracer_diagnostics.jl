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

function compute_dust!(out, state, cache, time)
    :prescribed_aerosols_field in propertynames(cache.tracers) ||
        error("Aerosols do not exist in the model")
    any(
        x -> x in propertynames(cache.tracers.prescribed_aerosols_field),
        [:DST01, :DST02, :DST03, :DST04, :DST05],
    ) || error("Dust does not exist in the model")
    if isnothing(out)
        aero_conc = cache.scratch.ᶜtemp_scalar
        @. aero_conc = 0
        for prescribed_aerosol_name in [:DST01, :DST02, :DST03, :DST04, :DST05]
            if prescribed_aerosol_name in
               propertynames(cache.tracers.prescribed_aerosols_field)
                aerosol_field = getproperty(
                    cache.tracers.prescribed_aerosols_field,
                    prescribed_aerosol_name,
                )
                @. aero_conc += aerosol_field
            end
        end
        return aero_conc
    else
        aero_conc = cache.scratch.ᶜtemp_scalar
        @. aero_conc = 0
        for prescribed_aerosol_name in [:DST01, :DST02, :DST03, :DST04, :DST05]
            if prescribed_aerosol_name in
               propertynames(cache.tracers.prescribed_aerosols_field)
                aerosol_field = getproperty(
                    cache.tracers.prescribed_aerosols_field,
                    prescribed_aerosol_name,
                )
                @. aero_conc += aerosol_field
            end
        end
        out .= aero_conc
    end
end

function compute_sea_salt!(out, state, cache, time)
    :prescribed_aerosols_field in propertynames(cache.tracers) ||
        error("Aerosols do not exist in the model")
    any(
        x -> x in propertynames(cache.tracers.prescribed_aerosols_field),
        [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05],
    ) || error("Sea salt does not exist in the model")
    if isnothing(out)
        aero_conc = cache.scratch.ᶜtemp_scalar
        @. aero_conc = 0
        for prescribed_aerosol_name in
            [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
            if prescribed_aerosol_name in
               propertynames(cache.tracers.prescribed_aerosols_field)
                aerosol_field = getproperty(
                    cache.tracers.prescribed_aerosols_field,
                    prescribed_aerosol_name,
                )
                @. aero_conc += aerosol_field
            end
        end
        return aero_conc
    else
        aero_conc = cache.scratch.ᶜtemp_scalar
        @. aero_conc = 0
        for prescribed_aerosol_name in
            [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
            if prescribed_aerosol_name in
               propertynames(cache.tracers.prescribed_aerosols_field)
                aerosol_field = getproperty(
                    cache.tracers.prescribed_aerosols_field,
                    prescribed_aerosol_name,
                )
                @. aero_conc += aerosol_field
            end
        end
        out .= aero_conc
    end
end

function compute_sea_salt_column!(out, state, cache, time)
    :prescribed_aerosols_field in propertynames(cache.tracers) ||
        error("Aerosols do not exist in the model")
    any(
        x -> x in propertynames(cache.tracers.prescribed_aerosols_field),
        [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05],
    ) || error("Sea salt does not exist in the model")
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        aero_conc = cache.scratch.ᶜtemp_scalar
        @. aero_conc = 0
        for prescribed_aerosol_name in
            [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
            if prescribed_aerosol_name in
               propertynames(cache.tracers.prescribed_aerosols_field)
                aerosol_field = getproperty(
                    cache.tracers.prescribed_aerosols_field,
                    prescribed_aerosol_name,
                )
                @. aero_conc += aerosol_field * state.c.ρ
            end
        end
        Operators.column_integral_definite!(out, aero_conc)
        return out
    else
        aero_conc = cache.scratch.ᶜtemp_scalar
        @. aero_conc = 0
        for prescribed_aerosol_name in
            [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
            if prescribed_aerosol_name in
               propertynames(cache.tracers.prescribed_aerosols_field)
                aerosol_field = getproperty(
                    cache.tracers.prescribed_aerosols_field,
                    prescribed_aerosol_name,
                )
                @. aero_conc += aerosol_field * state.c.ρ
            end
        end
        Operators.column_integral_definite!(out, aero_conc)
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
    comments = "Prescribed dry mass fraction of dust aerosol particles in air.",
    compute! = (out, u, p, t) -> compute_dust!(out, u, p, t),
)

###
# Sea salt concentration (3d)
###
add_diagnostic_variable!(
    short_name = "mmrss",
    long_name = "Sea-Salt Aerosol Mass Mixing Ratio",
    standard_name = "mass_fraction_of_sea_salt_dry_aerosol_particles_in_air",
    units = "kg kg^-1",
    comments = "Prescribed dry mass fraction of sea salt aerosol particles in air.",
    compute! = (out, u, p, t) -> compute_sea_salt!(out, u, p, t),
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
