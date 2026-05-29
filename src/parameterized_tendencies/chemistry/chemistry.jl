# ============================================================================
# Chemistry Module
# ============================================================================
# Gas-phase chemistry for ClimaAtmos.

using Musica

const _musica_species_cache = Dict{String, Tuple{Vararg{Symbol}}}()

function musica_species_names(path::String)
    if haskey(_musica_species_cache, path)
        return _musica_species_cache[path]
    end

    micm = MICM(config_path=path)
    st = Musica.create_state(micm)
    sp_map = Musica.get_species_ordering(st)

    n = maximum(values(sp_map)) + 1
    names = Vector{Symbol}(undef, n)
    for (s, idx) in sp_map
        names[idx + 1] = Symbol(s)
    end

    species_names = Tuple(names)
    _musica_species_cache[path] = species_names
    return species_names
end

include("tendency.jl")
