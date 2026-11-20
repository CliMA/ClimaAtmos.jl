import UnrolledUtilities: unrolled_foreach, unrolled_flatmap

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
`species_model` selects: an `AbstractPrognosticAerosol` diagnoses it
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
    species_bin_names(prescribed_names, species_model)

Bins to sum for a species: the MERRA-2 `prescribed_names` when `species_model`
is `nothing`, otherwise the prognostic model's own `bin_names`.
"""
species_bin_names(prescribed_names, ::Nothing) = prescribed_names
species_bin_names(_, m::AbstractPrognosticAerosol) = bin_names(m)

"""
    ᶜaerosol_species_mmr(u, p, prescribed_names, species_model)

Cell-center dry mass mixing ratio [kg/kg] of a species, summed over
`species_bin_names`, with each bin read through
`ᶜaerosol_bin_mmr` from the source `species_model` selects.
`prescribed_names` is one species entry of `AEROSOL_SPECIES_BIN_NAMES`.
"""
function ᶜaerosol_species_mmr(u, p, prescribed_names::Tuple, species_model)
    ᶜbin_mmrs = map(
        bin_name -> ᶜaerosol_bin_mmr(u, p, bin_name, species_model),
        species_bin_names(prescribed_names, species_model),
    )
    return foldl((ᶜa, ᶜb) -> @.(lazy(ᶜa + ᶜb)), ᶜbin_mmrs)
end

#####
##### Emission and deposition tendencies
#####

"""
    aerosol_state_names(species_model)
    aerosol_state_names(aerosols)

`MatrixFields.FieldName`s of the grid-mean `ρ<bin>` tracers of a prognostic
species, one per `bin_names` entry, built at compile time so the
per-bin loops stay type stable. For an `AtmosAerosols`, those of every
species, in the order of `aerosol_surface_flux_names`.
"""
@generated aerosol_state_names(::T) where {T <: AbstractPrognosticAerosol} =
    :($(map(n -> MatrixFields.FieldName(Symbol(:ρ, n)), bin_names(T))))
aerosol_state_names(::Nothing) = ()
aerosol_state_names(a::AtmosAerosols) =
    unrolled_flatmap(aerosol_state_names, values(species_models(a)))

"""
    aerosol_surface_flux_names(aerosols)

Names of the `sfc_conditions` entries holding the surface flux of each
prognostic aerosol bin, `ρ_flux_<bin>`, following `ρ_flux_q_tot`.
"""
aerosol_surface_flux_names(::Nothing) = ()
aerosol_surface_flux_names(m::AbstractPrognosticAerosol) =
    map(n -> Symbol(:ρ_flux_, n), bin_names(m))
aerosol_surface_flux_names(a::AtmosAerosols) =
    unrolled_flatmap(aerosol_surface_flux_names, values(species_models(a)))

"""
    uniform_surface_fluxes(species_model, params)

Per-bin surface emission fluxes [kg m⁻² s⁻¹] applied uniformly in standalone
runs, which have no coupler to supply the surface state.
"""
uniform_surface_fluxes(::Nothing, params) = ()

"""
    init_aerosol_surface_fluxes!(p)

Write the `uniform_surface_fluxes` of each prognostic aerosol species
into its `ρ_flux_<bin>` entries of `p.precomputed.sfc_conditions`.
"""
function init_aerosol_surface_fluxes!(p)
    (; sfc_conditions) = p.precomputed
    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(p.scratch.ᶠtemp_scalar), half)
    for m in values(species_models(p.atmos.aerosols)),
        (name, flux) in
        zip(aerosol_surface_flux_names(m), uniform_surface_fluxes(m, p.params))

        getproperty(sfc_conditions, name) .=
            SurfaceConditions.vector_from_component.(flux, sfc_local_geometry)
    end
    return nothing
end

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
