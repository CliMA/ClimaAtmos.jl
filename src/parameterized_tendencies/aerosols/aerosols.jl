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
    aerosol_state_names(species_model)

`MatrixFields.FieldName`s of the grid-mean `ρ<bin>` tracers of a prognostic
species, one per [`bin_names`](@ref) entry, built at compile time so the
per-bin loops stay type stable.
"""
@generated aerosol_state_names(::T) where {T <: AbstractPrognosticAerosol} =
    :($(map(n -> MatrixFields.FieldName(Symbol(:ρ, n)), bin_names(T))))

"""
    aerosol_surface_flux_tendency!(Yₜ, Y, p, species_model, fluxes)

Apply cached per-bin surface fluxes (surface `C3` fields keyed by tracer
name, upward positive) as bottom boundary conditions on the grid-mean
`Y.c.ρ<bin>` tracers, using [`boundary_tendency_scalar`](@ref), and mirror
the specific tendency onto each updraft tracer so updraft and grid-mean
concentrations do not drift apart at the surface. Skipped entirely when
`disable_surface_flux_tendency` is set, so every aerosol surface exchange
follows the same switch as the momentum, energy, and water fluxes.
"""
function aerosol_surface_flux_tendency!(
    Yₜ,
    Y,
    p,
    species_model::AbstractPrognosticAerosol,
    fluxes,
)
    p.atmos.disable_surface_flux_tendency && return nothing
    n_updrafts = n_mass_flux_subdomains(p.atmos.turbconv_model)

    MatrixFields.unrolled_foreach(aerosol_state_names(species_model)) do ρχ_name
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        btt = boundary_tendency_scalar(
            ᶜχ,
            fluxes[MatrixFields.extract_first(ρχ_name)],
        )
        @. ᶜρχₜ -= btt

        for j in 1:n_updrafts
            ᶜχʲₜ = MatrixFields.get_field(
                Yₜ.c.sgsʲs.:($j),
                specific_tracer_name(ρχ_name),
            )
            @. ᶜχʲₜ -= specific(btt, p.precomputed.ᶜρʲs.:($$j))
        end
    end
    return nothing
end

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
