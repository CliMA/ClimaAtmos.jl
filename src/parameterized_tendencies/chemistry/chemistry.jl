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

"""
    chemistry_species_names(config_path)

Return the ordered `Tuple` of species `Symbol`s that defines the tracer set for a
gas-phase chemistry mechanism. Species are read from the top-level `species` list
of the MICM configuration file at `config_path` (the same file consumed by
Musica/MICM), so the mechanism-driven set of `ρq_gas_<species>` tracers can be
determined *without* Musica being loaded. Returns `()` for `nothing`.

This is the single source of truth for *which* tracers exist; everything
downstream (transport, the implicit Jacobian, the MICM solve) is agnostic to the
particular species and discovers them from the state or the mechanism.
"""
chemistry_species_names(::Nothing) = ()
function chemistry_species_names(config_path::String)
    data = YAML.load_file(config_path)
    haskey(data, "species") || error(
        "Chemistry config `$config_path` has no top-level `species` list.",
    )
    return Tuple(Symbol(species["name"]) for species in data["species"])
end
