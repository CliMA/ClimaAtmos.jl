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
    has_prescribed = :prescribed_aerosols_field in propertynames(cache.tracers) &&
        aerosol_name in propertynames(cache.tracers.prescribed_aerosols_field)
    ρname = Symbol(:ρ, aerosol_name)
    has_prognostic = ρname in propertynames(state.c)
    has_prescribed || has_prognostic ||
        error("$aerosol_name does not exist in the model")
    if isnothing(out)
        if has_prognostic
            ρχ = getproperty(state.c, ρname)
            return copy(@. specific(ρχ, state.c.ρ))
        else
            return copy(getproperty(cache.tracers.prescribed_aerosols_field, aerosol_name))
        end
    else
        if has_prognostic
            ρχ = getproperty(state.c, ρname)
            @. out = specific(ρχ, state.c.ρ)
        else
            aerosol_field = getproperty(cache.tracers.prescribed_aerosols_field, aerosol_name)
            out .= aerosol_field
        end
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
    has_prescribed = :prescribed_aerosols_field in propertynames(cache.tracers) &&
        any(
            x -> x in propertynames(cache.tracers.prescribed_aerosols_field),
            [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05],
        )
    has_prognostic = any(
        x -> x in propertynames(state.c),
        [:ρSSLT01, :ρSSLT02, :ρSSLT03, :ρSSLT04, :ρSSLT05],
    )
    has_prescribed || has_prognostic || error("Sea salt does not exist in the model")
    if isnothing(out)
        aero_conc = cache.scratch.ᶜtemp_scalar
        @. aero_conc = 0
        if has_prescribed
            for name in [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
                if name in propertynames(cache.tracers.prescribed_aerosols_field)
                    aerosol_field = getproperty(cache.tracers.prescribed_aerosols_field, name)
                    @. aero_conc += aerosol_field
                end
            end
        end
        if has_prognostic
            for ρname in [:ρSSLT01, :ρSSLT02, :ρSSLT03, :ρSSLT04, :ρSSLT05]
                if ρname in propertynames(state.c)
                    ρχ = getproperty(state.c, ρname)
                    @. aero_conc += specific(ρχ, state.c.ρ)
                end
            end
        end
        return aero_conc
    else
        aero_conc = cache.scratch.ᶜtemp_scalar
        @. aero_conc = 0
        if has_prescribed
            for name in [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
                if name in propertynames(cache.tracers.prescribed_aerosols_field)
                    aerosol_field = getproperty(cache.tracers.prescribed_aerosols_field, name)
                    @. aero_conc += aerosol_field
                end
            end
        end
        if has_prognostic
            for ρname in [:ρSSLT01, :ρSSLT02, :ρSSLT03, :ρSSLT04, :ρSSLT05]
                if ρname in propertynames(state.c)
                    ρχ = getproperty(state.c, ρname)
                    @. aero_conc += specific(ρχ, state.c.ρ)
                end
            end
        end
        out .= aero_conc
    end
end

function compute_sea_salt_column!(out, state, cache, time)
    has_prognostic =
        any(n -> hasproperty(state.c, Symbol(:ρ, n)), [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05])
    has_prescribed =
        :prescribed_aerosols_field in propertynames(cache.tracers) &&
        any(
            x -> x in propertynames(cache.tracers.prescribed_aerosols_field),
            [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05],
        )
    has_prognostic || has_prescribed || error("Sea salt does not exist in the model")

    # column_integral_definite! integrates kg/m³ over altitude to give kg/m²,
    # so aero_conc must be in kg/m³ (not mixing ratio kg/kg).
    # Prognostic: ρχ is already kg/m³. Prescribed: multiply mixing ratio by ρ.
    aero_conc = cache.scratch.ᶜtemp_scalar
    if isnothing(out)
        out = zeros(axes(Fields.level(state.f, half)))
        @. aero_conc = 0
        if has_prognostic
            for name in [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
                ρχ_name = Symbol(:ρ, name)
                if hasproperty(state.c, ρχ_name)
                    ρχ = getproperty(state.c, ρχ_name)
                    @. aero_conc += ρχ
                end
            end
        else
            for name in [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
                if name in propertynames(cache.tracers.prescribed_aerosols_field)
                    aerosol_field = getproperty(cache.tracers.prescribed_aerosols_field, name)
                    @. aero_conc += aerosol_field * state.c.ρ
                end
            end
        end
        Operators.column_integral_definite!(out, aero_conc)
        return out
    else
        @. aero_conc = 0
        if has_prognostic
            for name in [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
                ρχ_name = Symbol(:ρ, name)
                if hasproperty(state.c, ρχ_name)
                    ρχ = getproperty(state.c, ρχ_name)
                    @. aero_conc += ρχ
                end
            end
        else
            for name in [:SSLT01, :SSLT02, :SSLT03, :SSLT04, :SSLT05]
                if name in propertynames(cache.tracers.prescribed_aerosols_field)
                    aerosol_field = getproperty(cache.tracers.prescribed_aerosols_field, name)
                    @. aero_conc += aerosol_field * state.c.ρ
                end
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

# The total sea salt emission flux is computed each timestep by sea_salt_emission_tendency!
# and stored in cache.tracers.sea_salt_emission_flux_sfc. The diagnostic just reads it,
# so the time-average exactly matches what was applied to the tracers.
function compute_sea_salt_emission_flux!(out, state, cache, time)
    isempty(_aerosol_names(cache.atmos.prognostic_aerosols)) &&
        error("No prognostic sea salt bins in this run")
    flux = cache.tracers.sea_salt_emission_flux_sfc
    if isnothing(out)
        return copy(flux)
    else
        out .= flux
    end
end

add_diagnostic_variable!(
    short_name = "emiss",
    long_name = "Sea-Salt Aerosol Surface Emission Flux",
    units = "kg m^-2 s^-1",
    comments = "Total upward sea salt mass flux at the surface, summed over all bins.",
    compute! = (out, u, p, t) -> compute_sea_salt_emission_flux!(out, u, p, t),
)

function compute_sea_salt_emission_flux_bin!(out, state, cache, time, bin_name)
    isempty(_aerosol_names(cache.atmos.prognostic_aerosols)) &&
        error("No prognostic sea salt bins in this run")
    flux = getproperty(cache.tracers.sea_salt_emission_flux_bins_sfc, bin_name)
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

