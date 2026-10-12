import UnrolledUtilities: unrolled_filter, unrolled_flatmap, unrolled_foreach

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
    prognostic_aerosol_state_names(aerosols)

The grid-mean `ρ<bin>` state names of every prognostic species in an
[`AtmosAerosols`](@ref) group, concatenated in species order; `()` when there
is no group or no prognostic species. Used by the implicit solver to drop the
aerosol blocks from the Jacobian (`implicit_aerosol_jacobian: false`).
"""
prognostic_aerosol_state_names(::Nothing) = ()
prognostic_aerosol_state_names(aerosols::AtmosAerosols) = unrolled_flatmap(
    aerosol_state_names,
    unrolled_filter(
        model -> model isa AbstractPrognosticAerosol,
        values(species_models(aerosols)),
    ),
)

"""
    aerosol_surface_flux_tendency!(Yₜ, Y, p, species_model, fluxes)

Apply cached per-bin surface fluxes (surface `C3` fields keyed by tracer
name, upward positive) as bottom boundary conditions on the grid-mean
`Y.c.ρ<bin>` tracers, using [`boundary_tendency_scalar`](@ref), and mirror
the specific tendency onto each updraft tracer so updraft and grid-mean
concentrations do not drift apart at the surface. The species methods of
[`aerosol_emission_tendency!`](@ref) (upward fluxes, a source) and
[`aerosol_dry_deposition_tendency!`](@ref) (downward fluxes, a sink) hand
their cached fluxes to this one function. Skipped entirely when
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
    aerosol_settling_tendency!(Yₜ, Y, p, t)
    aerosol_settling_tendency!(Yₜ, Y, p, t, species_model)

Apply the gravitational settling tendency of every aerosol species,
dispatching to methods within `AbstractPrognosticAerosol` species models.
Together with [`aerosol_dry_deposition_tendency!`](@ref) this makes up dry
removal: settling deposits the gravitational flux through its free-outflow
bottom boundary, and the surface flux adds only the turbulent part.
"""
aerosol_settling_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
aerosol_settling_tendency!(Yₜ, Y, p, t) = unrolled_foreach(
    model -> aerosol_settling_tendency!(Yₜ, Y, p, t, model),
    values(species_models(p.atmos.aerosols)),
)

"""
    aerosol_dry_deposition_tendency!(Yₜ, Y, p, t)
    aerosol_dry_deposition_tendency!(Yₜ, Y, p, t, species_model)

Apply the turbulent dry-deposition surface sink of every aerosol species,
dispatching to methods within `AbstractPrognosticAerosol` species models.
"""
aerosol_dry_deposition_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
aerosol_dry_deposition_tendency!(Yₜ, Y, p, t) = unrolled_foreach(
    model -> aerosol_dry_deposition_tendency!(Yₜ, Y, p, t, model),
    values(species_models(p.atmos.aerosols)),
)

"""
    aerosol_wet_deposition_tendency!(Yₜ, Y, p, t)
    aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, species_model)

Apply the wet-removal (in-cloud scavenging + below-cloud washout) tendency of
every aerosol species, dispatching to methods within
`AbstractPrognosticAerosol` species models.
"""
aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
aerosol_wet_deposition_tendency!(Yₜ, Y, p, t) = unrolled_foreach(
    model -> aerosol_wet_deposition_tendency!(Yₜ, Y, p, t, model),
    values(species_models(p.atmos.aerosols)),
)

"""
    aerosol_deposition_tendency!(Yₜ, Y, p, t)

Apply every aerosol removal tendency that is not transport: the turbulent
dry-deposition surface sink ([`aerosol_dry_deposition_tendency!`](@ref)) and
wet removal ([`aerosol_wet_deposition_tendency!`](@ref)). The gravitational
part of dry deposition leaves through the bottom boundary of
[`aerosol_settling_tendency!`](@ref).
"""
function aerosol_deposition_tendency!(Yₜ, Y, p, t)
    aerosol_dry_deposition_tendency!(Yₜ, Y, p, t)
    aerosol_wet_deposition_tendency!(Yₜ, Y, p, t)
    return nothing
end

"""
    aerosol_nonnegativity_constraint!(Y, p, t)
    aerosol_nonnegativity_constraint!(Y, p, t, species_model)

Remove negative prognostic aerosol tracer masses from the state `Y` in place, dispatching to
every active `AbstractPrognosticAerosol` species model the same way the other aerosol
tendencies do (`values(species_models(p.atmos.aerosols))`).

WHY THIS EXISTS. `settling_courant_max` (`aerosol_settling_tendency!`,
`set_sslt_dry_deposition_fluxes!`) caps the EXPLICIT removal velocity of settling and dry
deposition so that no single stage evaluation removes more than one cell's worth of mass in
that stage alone. That is necessary but not sufficient: ARS222's explicit Butcher tableau has a
negative third-stage coefficient (`δ = 1 - 1/(2γ) ≈ -0.707` with `γ = 1 - √2/2`), so the scheme
is not strong-stability-preserving for an explicitly-treated sink. A per-stage Courant cap
bounds each stage's own contribution, but cannot bound what the stages' non-convex
recombination (weights `δ` and `1 - δ ≈ 1.707`, not both in `[0, 1]`) does to a tracer that
starts the step close to zero — the recombination can still drive it negative regardless of how
tightly the cap is set. This is why tightening `settling_courant_max` alone only delays, rather
than eliminates, an eventual blow-up once the settled bin (in practice the largest, fastest-
falling one) carries real mass: a tighter cap shrinks each stage's contribution and so slows how
fast the recombination error compounds, but does not touch the mechanism.

This mirrors `tracer_nonnegativity_constraint!`'s **`TracerNonnegativityVaporConstraint`**
treatment of the water tracers — a plain pointwise `max(0, ρχ)`, not the mass-conserving
`TracerNonnegativityElementConstraint` variant (`ClimaCore.Limiters.QuasiMonotoneLimiter`-based
clip that redistributes the deficit horizontally instead of creating or destroying mass).

THAT WAS THE FIRST VERSION OF THIS FIX, AND IT DOES NOT WORK ON THIS GPU+MPI COMBINATION.
`QuasiMonotoneLimiter` carries a `ghost_buffer` field for cross-element MPI communication which
is not a GPU-kernel-compatible bitstype under `ClimaComms.MPICommsContext` (the context this
experiment runs under via `CLIMACOMMS_CONTEXT=MPI`, even at `nprocs=1`); calling
`Limiters.compute_bounds!` from this function crashed `KernelError: passing non-bitstype
argument` in production (Slurm 258172, day 1). The water-tracer path that uses this same limiter
is disabled in this experiment's config (`tracer_nonnegativity_method: nothing`), so nothing had
ever actually exercised `QuasiMonotoneLimiter` on the GPU here before this function did. A plain
elementwise `max(0, ρχ)` is trivially GPU-safe (no limiter object, no MPI-carrying struct, no
kernel launch beyond ordinary broadcast) and sidesteps the incompatibility entirely, at the cost
of the mass-conserving redistribution property: a triggered clamp is now honest, bounded mass
creation (the size of whatever deficit it corrected) rather than a horizontal transfer within
the element. That trade is unambiguously worth it here: unconditional, always-correct GPU safety
beats a "nicer" fix that crashes the moment it actually runs. Restoring the
element-conserving version is future work, gated on either a ClimaCore/GPUCompiler fix for
`QuasiMonotoneLimiter` under an MPI context, or constructing it against a `SingletonCommsContext`
specifically for this call regardless of the run's overall communications context.

Unconditional (not gated by a config option) whenever `p.atmos.aerosols` has at least one
`AbstractPrognosticAerosol` species active, because it guards against a specific, verified
integrator defect rather than expressing a tunable physics choice.
"""
aerosol_nonnegativity_constraint!(Y, p, t) = unrolled_foreach(
    model -> aerosol_nonnegativity_constraint!(Y, p, t, model),
    values(species_models(p.atmos.aerosols)),
)
aerosol_nonnegativity_constraint!(Y, p, t, ::Nothing) = nothing
function aerosol_nonnegativity_constraint!(
    Y, p, t, species_model::AbstractPrognosticAerosol,
)
    for ρχ_name in aerosol_state_names(species_model)
        MatrixFields.has_field(Y.c, ρχ_name) || continue
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        @. ᶜρχ = max(0, ᶜρχ)
    end
    return nothing
end

###
### Helpers
###

"""
    _aerosol_air_state(thp, T, p, q_tot, q_liq, q_ice, ρ_air, R_d, ap)

Cell relative humidity, viscosity, and mean free path, pre-computed
to avoid per bin computation. Evaluated per
subdomain into `p.tracers.sslt_air_state⁰` / `sslt_air_stateʲs`.
"""
function _aerosol_air_state(thp, T, p, q_tot, q_liq, q_ice, ρ_air, R_d, ap)
    FT = typeof(T)

    RH = TD.relative_humidity(thp, T, p, q_tot, q_liq, q_ice)

    μ = air_dynamic_viscosity(T, ap)
    v̄ = sqrt(8 * R_d * T / FT(π))
    λ = 2 * μ / (ρ_air * v̄)

    return (; RH, μ, λ)
end


include("lognormal_moments.jl")
include("hygroscopic_growth.jl")
include("settling.jl")
include("dry_deposition.jl")
include("sea_salt.jl")
include("column_emission.jl")
include("wet_deposition.jl")
# After every sink it mirrors (settling, dry and wet deposition).
include("history_tracers.jl")
