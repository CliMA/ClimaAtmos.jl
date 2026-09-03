import UnrolledUtilities: unrolled_foreach

"""
    has_prescribed_aerosol_bin(p, bin_name)

Whether `bin_name` is one of the MERRA-2 bins listed in the
`prescribed_aerosols` config (and therefore present in
`p.tracers.prescribed_aerosols_field`).
"""
has_prescribed_aerosol_bin(p, bin_name) =
    :prescribed_aerosols_field in propertynames(p.tracers) &&
    bin_name in propertynames(p.tracers.prescribed_aerosols_field)

"""
    ᶜaerosol_bin_mmr(u, p, bin_name, species_model)

Cell-center dry mass mixing ratio [kg/kg] of one aerosol bin, from the source
`species_model` selects: an [`AbstractPrognosticAerosol`](@ref) diagnoses it
lazily from `u.c.ρ<bin_name>`, while `nothing` (no prognostic model) falls
back to the prescribed MERRA-2 field in
`p.tracers.prescribed_aerosols_field`, or zero when the bin is not
prescribed either.
"""
function ᶜaerosol_bin_mmr(u, p, bin_name, ::Nothing)
    if has_prescribed_aerosol_bin(p, bin_name)
        return getproperty(p.tracers.prescribed_aerosols_field, bin_name)
    else
        return @. lazy(zero(u.c.ρ))
    end
end
function ᶜaerosol_bin_mmr(u, p, bin_name, ::AbstractPrognosticAerosol)
    ᶜρχ = getproperty(u.c, Symbol(:ρ, bin_name))
    return @. lazy(specific(ᶜρχ, u.c.ρ))
end

"""
    ᶜaerosol_species_mmr(u, p, bin_names, species_model)

Cell-center dry mass mixing ratio [kg/kg] of a species, summed over
`bin_names` (one species entry of `AEROSOL_SPECIES_BIN_NAMES`), with each
bin read through [`ᶜaerosol_bin_mmr`](@ref) from the source `species_model`
selects.
"""
function ᶜaerosol_species_mmr(u, p, bin_names::Tuple, species_model)
    ᶜbin_mmrs = map(
        bin_name -> ᶜaerosol_bin_mmr(u, p, bin_name, species_model),
        bin_names,
    )
    return foldl((ᶜa, ᶜb) -> @.(lazy(ᶜa + ᶜb)), ᶜbin_mmrs)
end

#####
##### Emission and deposition tendencies
#####

"""
    aerosol_emission_tendency!(Yₜ, Y, p, t)
    aerosol_emission_tendency!(Yₜ, Y, p, t, species_model)

Apply the surface emission tendency of every aerosol species, dispatching
to methods within `AbstractPrognosticAerosol` species models.
"""
aerosol_emission_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
aerosol_emission_tendency!(Yₜ, Y, p, t) = unrolled_foreach(
    model -> aerosol_emission_tendency!(Yₜ, Y, p, t, model),
    values(species_models(p.atmos.aerosols)),
)


"""
    aerosol_deposition_tendency!(Yₜ, Y, p, t)
    aerosol_deposition_tendency!(Yₜ, Y, p, t, species_model)

Apply the deposition tendency of every aerosol species, dispatching
to methods within `AbstractPrognosticAerosol` species models.
"""
aerosol_deposition_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
aerosol_deposition_tendency!(Yₜ, Y, p, t) = unrolled_foreach(
    model -> aerosol_deposition_tendency!(Yₜ, Y, p, t, model),
    values(species_models(p.atmos.aerosols)),
)

include("sea_salt.jl")
