# This file is included in Diagnostics.jl
import ClimaCore.MatrixFields: @name, has_field, get_field

# Tracers

function compute_tracer(state, cache, time, tracer_name)
    (; tracers) = cache
    @assert has_field(tracers, tracer_name) "tracer $tracer_name does not exist"
    return get_field(tracers, tracer_name)
end

function compute_aerosol(state, cache, time, aerosol_name)
    (; tracers) = cache
    @assert has_field(tracers, @name(prescribed_aerosols_field)) "Aerosols do not exist in the model"
    aerosols = tracers.prescribed_aerosols_field
    @assert has_field(aerosols, aerosol_name) "$aerosol_name does not exist in the model"
    return get_field(aerosols, aerosol_name)
end

function compute_dust(state, cache, time)
    (; tracers) = cache
    dust_names = (@name(DST01), @name(DST02), @name(DST03), @name(DST04), @name(DST05))
    @assert has_field(tracers, @name(prescribed_aerosols_field)) "Aerosols do not exist in the model"
    aerosols = tracers.prescribed_aerosols_field
    @assert any(x -> has_field(aerosols, x), dust_names) "Dust does not exist in the model"
    
    function dust_sum(aerosols)
        mapreduce(+, dust_names) do dust_name
            has_field(aerosols, dust_name) ? get_field(aerosols, dust_name) : 0
        end
    end
    return @. lazy(dust_sum(aerosols))
end

function compute_sea_salt(state, cache, time)
    (; tracers) = cache
    @assert has_field(tracers, @name(prescribed_aerosols_field)) "Aerosols do not exist in the model"
    aerosols = tracers.prescribed_aerosols_field
    sea_salt_names = (@name(SSLT01), @name(SSLT02), @name(SSLT03), @name(SSLT04), @name(SSLT05))
    @assert any(x -> has_field(aerosols, x), sea_salt_names) "Sea salt does not exist in the model"

    function sea_salt_sum(aerosols)
        mapreduce(+, sea_salt_names) do sea_salt_name
            has_field(aerosols, sea_salt_name) ? get_field(aerosols, sea_salt_name) : 0
        end
    end
    return @. lazy(sea_salt_sum(aerosols))
end

function compute_sea_salt_column!(out, state, cache, time)
    out′ = isnothing(out) ? zeros(axes(Fields.level(state.f, half))) : out
    sea_salt = @. lazy(compute_sea_salt(state, cache, time) * state.c.ρ)
    Operators.column_integral_definite!(out′, sea_salt)
    return out′
end

###
# Ozone concentration (3d)
###
add_diagnostic_variable!(short_name = "o3", units = "mol mol^-1",
    long_name = "Mole Fraction of O3",
    standard_name = "mole_fraction_of_ozone_in_air",
    compute = (u, p, t) -> compute_tracer(u, p, t, @name(o3)),
)

###
# Dust concentration (3d)
###
add_diagnostic_variable!(short_name = "mmrdust", units = "kg kg^-1",
    long_name = "Dust Aerosol Mass Mixing Ratio",
    standard_name = "mass_fraction_of_dust_dry_aerosol_particles_in_air",
    comments = "Prescribed dry mass fraction of dust aerosol particles in air.",
    compute = compute_dust,
)

###
# Sea salt concentration (3d)
###
add_diagnostic_variable!(short_name = "mmrss", units = "kg kg^-1",
    long_name = "Sea-Salt Aerosol Mass Mixing Ratio",
    standard_name = "mass_fraction_of_sea_salt_dry_aerosol_particles_in_air",
    comments = "Prescribed dry mass fraction of sea salt aerosol particles in air.",
    compute = compute_sea_salt,
)

###
# Sulfate concentration (3d)
###
add_diagnostic_variable!(short_name = "mmrso4", units = "kg kg^-1",
    long_name = "Aerosol Sulfate Mass Mixing Ratio",
    standard_name = "mass_fraction_of_sulfate_dry_aerosol_particles_in_air",
    comments = "Prescribed dry mass of sulfate (SO4) in aerosol particles as a fraction of air mass.",
    compute = (u, p, t) -> compute_aerosol(u, p, t, @name(SO4)),
)

###
# Hydrophobic black carbon concentration (3d)
###
add_diagnostic_variable!(short_name = "mmrbcpo", units = "kg kg^-1",
    long_name = "Hydrophobic Elemental Carbon Mass Mixing Ratio",
    comments = "Prescribed dry mass fraction of hydrophobic black carbon aerosol particles in air.",
    compute = (u, p, t) -> compute_aerosol(u, p, t, @name(CB1)),
)

###
# Hydrophilic black carbon concentration (3d)
###
add_diagnostic_variable!(short_name = "mmrbcpi", units = "kg kg^-1",
    long_name = "Hydrophilic Elemental Carbon Mass Mixing Ratio",
    comments = "Prescribed dry mass fraction of hydrophilic black carbon aerosol particles in air.",
    compute = (u, p, t) -> compute_aerosol(u, p, t, @name(CB2)),
)

###
# Hydrophobic organic carbon concentration (3d)
###
add_diagnostic_variable!(short_name = "mmrocpo", units = "kg kg^-1",
    long_name = "Hydrophobic Organic Carbon Mass Mixing Ratio",
    comments = "Prescribed dry mass fraction of hydrophobic organic carbon aerosol particles in air.",
    compute = (u, p, t) -> compute_aerosol(u, p, t, @name(OC1)),
)

###
# Hydrophilic organic carbon concentration (3d)
###
add_diagnostic_variable!(short_name = "mmrocpi", units = "kg kg^-1",
    long_name = "Hydrophilic Organic Carbon Mass Mixing Ratio",
    comments = "Prescribed dry mass fraction of hydrophilic organic carbon aerosol particles in air.",
    compute = (u, p, t) -> compute_aerosol(u, p, t, @name(OC2)),
)

###
# Sea salt column mass (2d)
###
add_diagnostic_variable!(short_name = "loadss", units = "kg m^-2",
    long_name = "Load of Sea-Salt Aerosol",
    standard_name = "atmosphere_mass_content_of_sea_salt_dry_aerosol _particles",
    comments = "The total dry mass of sea salt aerosol particles per unit area.",
    compute! = compute_sea_salt_column!,
)
