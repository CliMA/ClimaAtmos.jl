module ClimaAtmosMusica

import ClimaAtmos
import Musica

"""
    ClimaAtmos.chemistry_tendency!(Yₜ, Y, p, t, ::ClimaAtmos.GasPhaseChem)

No-op in the extension: transport is handled by the auto-discovery machinery.
Chemistry sources/sinks are applied via `update_chemistry!`.
"""
function ClimaAtmos.chemistry_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::ClimaAtmos.GasPhaseChem,
)
    return nothing
end

"""
    ClimaAtmos.update_chemistry!(Yₜ, Y, p, t, chemistry_model::ClimaAtmos.GasPhaseChem)

MUSICA-backed in-place chemistry update. Iterates over all grid cells and applies
MICM kinetics for the species from the config file.
Only runs when `chemistry_model.config_path` is set.
"""
function ClimaAtmos.update_chemistry!(
    Y,
    p,
    t,
    chemistry_model::ClimaAtmos.GasPhaseChem,
)
    isnothing(chemistry_model.config_path) && return nothing

    micm = Musica.MICM(; config_path = chemistry_model.config_path)
    state = Musica.create_state(micm)
    Musica.set_conditions!(state; temperatures=298.15, pressures=101325)
    Musica.set_user_defined_rate_parameters!(
        state,
        Dict(
            "USER.forward_AB_to_A_B" => 2.0e-3,
            "USER.reverse_A_B_to_AB" => 1.0e-3,
        ),
    )
    species = ClimaAtmos.musica_species_names(chemistry_model.config_path)
    n_cells = length(parent(Y.c.ρ))

    for i in 1:n_cells
        concs = Dict(
            String(s) => Float64(parent(getproperty(Y.c, Symbol(:ρ, s)))[i])
            for s in species
        )
        Musica.set_concentrations!(state, concs)
        # println("Running MUSICA chemistry for cell $i with concentrations: ", concs)
        Musica.solve!(micm, state, Float64(p.dt))
        updated = Musica.get_concentrations(state)
        for s in species
            parent(getproperty(Y.c, Symbol(:ρ, s)))[i] = Float32(only(updated[String(s)]))
        end
    end
    return nothing
end

const _musica_species_cache = Dict{String, Tuple{Vararg{Symbol}}}()

"""
    ClimaAtmos.musica_species_names(path::String)

Read species names from a MUSICA config file, cached after the first call.
"""
function ClimaAtmos.musica_species_names(path::String)
    haskey(_musica_species_cache, path) && return _musica_species_cache[path]
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

end # module ClimaAtmosMusica
