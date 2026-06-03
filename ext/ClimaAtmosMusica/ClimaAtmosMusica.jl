module ClimaAtmosMusica

import ClimaAtmos
import Musica


"""
    ClimaAtmos.musica_species_names(path::String)

Get the names of the chemical species from a MUSICA configuration file.
"""
const _musica_species_cache = Dict{String, Tuple{Vararg{Symbol}}}()
function ClimaAtmos.musica_species_names(path::String)
    if haskey(_musica_species_cache, path)
        return _musica_species_cache[path]
    end
    micm = Musica.MICM(; config_path = path)
    st = Musica.create_state(micm)
    sp_map = Musica.get_species_ordering(st)

    n = maximum(values(sp_map)) + 1
    names = Vector{Symbol}(undef, n)
    for (s, idx) in sp_map
        names[idx + 1] = Symbol(s)
    end
    species = Tuple(names)
    _musica_species_cache[path] = species
    return species
end

"""
    ClimaAtmos.GasPhaseChem(config_path::String)
Construct a `GasPhaseChem` chemistry model with the given MUSICA configuration file.
"""
function ClimaAtmos.GasPhaseChem(config_path::String)
    names = ClimaAtmos.musica_species_names(config_path)
    return ClimaAtmos.GasPhaseChem(config_path, names)
end


"""
    ClimaAtmos.chemistry_tendency!(Yₜ, Y, p, t, ::ClimaAtmos.GasPhaseChem)

MUSICA-backed gas-phase chemistry tendency.
Loaded automatically when `Musica` is imported alongside `ClimaAtmos`.
"""
function ClimaAtmos.chemistry_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::ClimaAtmos.GasPhaseChem,
)
    @info "MUSICA version: $(Musica.get_version())"
    return nothing
end

end # module ClimaAtmosMusica
