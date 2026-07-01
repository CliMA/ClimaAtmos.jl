###
### Chemistry Module
###

# Gas-phase chemistry for ClimaAtmos.
# The MUSICA backend is provided by the ClimaAtmosMusica extension;
# this file defines only the fallback for when no chemistry is loaded.

"""
    chemistry_tendency!(Yₜ, Y, p, t, ::Nothing)

No chemistry model is active.
"""
chemistry_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    chemistry_tendency!(Yₜ, Y, p, t, ::AbstractChemistryModel)

Generic fallback: no-op for passive tracers, since transport is handled by
the auto-discovery machinery (`gs_tracer_names` / `sgs_tracer_names`), and
there are no chemical sources or sinks unless a more specific method is
provided (e.g. by the ClimaAtmosMusica extension for `GasPhaseChem`).
"""
chemistry_tendency!(Yₜ, Y, p, t, ::AbstractChemistryModel) = nothing

"""
    update_chemistry!(Y, p, t, chemistry_model)

Apply in-place chemistry updates to `Y`. The MUSICA-backed implementation is
provided by the ClimaAtmosMusica extension when `Musica` is loaded.
"""
update_chemistry!(Y, p, t, ::Nothing) = nothing
update_chemistry!(Y, p, t, ::AbstractChemistryModel) = nothing

"""
    chemistry_cache(Y, atmos)

Return a named tuple of chemistry runtime objects (e.g. MICM solver and state)
that should be created once and reused across timesteps. The MUSICA-backed
implementation is provided by the ClimaAtmosMusica extension.
"""
chemistry_cache(_, ::Nothing) = (;)
chemistry_cache(_, ::AbstractChemistryModel) = (;)

function musica_species_names end
