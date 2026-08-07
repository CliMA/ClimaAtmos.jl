"""
    prescribed_bin_names(species_model)
    prescribed_bin_names(aerosols::AtmosAerosols)

Bin names read from the MERRA-2 climatology; empty for species that are off or
prognostic.
"""
prescribed_bin_names(::Union{Nothing, AbstractPrognosticAerosol}) = ()
prescribed_bin_names(m::AbstractPrescribedAerosol) = bin_names(m)
prescribed_bin_names(a::AtmosAerosols) = foldl(
    (names, m) -> (names..., prescribed_bin_names(m)...),
    values(species_models(a));
    init = (),
)

"""
    ᶜaerosol_bin_mmr(u, p, bin_name, species_model)

Cell-center dry mass mixing ratio [kg/kg] of one aerosol bin, from either
source: prescribed species read the MERRA-2 field in
`p.tracers.prescribed_aerosols_field`, prognostic species diagnose it lazily
from `u.c.ρ<bin_name>`.
"""
ᶜaerosol_bin_mmr(u, p, bin_name, ::AbstractPrescribedAerosol) =
    getproperty(p.tracers.prescribed_aerosols_field, bin_name)
function ᶜaerosol_bin_mmr(u, p, bin_name, ::AbstractPrognosticAerosol)
    ᶜρχ = getproperty(u.c, Symbol(:ρ, bin_name))
    return @. lazy(specific(ᶜρχ, u.c.ρ))
end

"""
    ᶜaerosol_species_mmr(u, p, species_model)

Cell-center dry mass mixing ratio [kg/kg] of a species, summed over
its [`bin_names`](@ref).
"""
ᶜaerosol_species_mmr(u, p, ::Nothing) = @. lazy(zero(u.c.ρ))
function ᶜaerosol_species_mmr(u, p, species_model)
    ᶜbin_mmrs = map(
        bin_name -> ᶜaerosol_bin_mmr(u, p, bin_name, species_model),
        bin_names(species_model),
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
aerosol_emission_tendency!(Yₜ, Y, p, t,
    ::Union{Nothing, AbstractPrescribedAerosol},
) = nothing
aerosol_emission_tendency!(Yₜ, Y, p, t) = foreach(
    model -> aerosol_emission_tendency!(Yₜ, Y, p, t, model),
    species_models(p.atmos.aerosols),
)


"""
    aerosol_deposition_tendency!(Yₜ, Y, p, t)
    aerosol_deposition_tendency!(Yₜ, Y, p, t, species_model)

Apply the deposition tendency of every aerosol species, dispatching
to methods within `AbstractPrognosticAerosol` species models.
"""
aerosol_deposition_tendency!(Yₜ, Y, p, t,
    ::Union{Nothing, AbstractPrescribedAerosol},
) = nothing
aerosol_deposition_tendency!(Yₜ, Y, p, t) = foreach(
    model -> aerosol_deposition_tendency!(Yₜ, Y, p, t, model),
    species_models(p.atmos.aerosols),
)

include("sea_salt.jl")
