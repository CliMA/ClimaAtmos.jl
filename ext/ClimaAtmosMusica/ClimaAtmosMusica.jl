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
"""
    ClimaAtmos.update_chemistry!(Yₜ, Y, p, t, ::ClimaAtmos.GasPhaseChem)

MUSICA-backed gas-phase chemistry update function. Updates Y directly in-place 
using MICM rather than returning a tendency. Currently just for one grid cell.
"""
# Musica only version (not using tracers)
function ClimaAtmos.update_chemistry!(
    Yₜ,
    Y,
    p,
    t,
    ::ClimaAtmos.GasPhaseChem,
)
    micm = Musica.MICM(; config_path = p.atmos.chemistry_model.config_path)
    state = Musica.create_state(micm)
    # print the initial concentrations of these species

    Musica.set_conditions!(state; temperatures=298.0, pressures=101325.0)
    Musica.set_concentrations!(state, Dict(
        "A" => 1.0,
        "B" => 0.0,
        "C" => 0.0,
    ))
    # print the initial concentrations
    println("Initial concentrations:")
    for (s, conc) in Musica.get_concentrations(state)
        println("  $s = $conc")
    end
    result = Musica.solve!(micm, state, Float64(p.dt))
    updated_concs = Musica.get_concentrations(state)
    println("Updated concentrations after chemistry update:")
    for (s, conc) in updated_concs
        println("  $s = $conc")
    end

end
# SINGLE CELL UPDATE
# function ClimaAtmos.update_chemistry!(
#     Yₜ,
#     Y,
#     p,
#     t,
#     chemistry_model::ClimaAtmos.GasPhaseChem,
# )
#     micm = Musica.MICM(; config_path = chemistry_model.config_path)
#     state = Musica.create_state(micm)
#     names = ClimaAtmos.species_names(chemistry_model)
#     n_cells = length(parent(Y.c.ρ))

#     for i in 1:n_cells
#         concs = Dict(String(s) => Float64(parent(getproperty(Y.c, Symbol(:ρ, s)))[i]) for s in names)
#         Musica.set_concentrations!(state, concs)
#         if i == 1
#              println("Initial concentrations for cell $i:")
#              for s in names
#                  println("  $(String(s)) = $(concs[String(s)])")
#              end
#         end

#         Musica.solve!(micm, state, Float64(p.dt))
#         updated = Musica.get_concentrations(state)
#         if i == 1
#             println("Updated concentrations for cell $i after chemistry update:")
#             for s in names
#                 println("  $(String(s)) = $(updated[String(s)])")
#             end
#         end

#         for s in names
#             parent(getproperty(Y.c, Symbol(:ρ, s)))[i] = Float32(only(updated[String(s)]))
#         end
#     end
# end
# SKETCH OF VECTORIZED UPDATE
# function ClimaAtmos.update_chemistry!(
#     Yₜ,
#     Y,
#     p,
#     t,
#     chemistry_model::ClimaAtmos.GasPhaseChem,
# )   
#     micm = Musica.MICM(; config_path = chemistry_model.config_path)
#     n_cells = length(parent(Y.c.ρ))
#     state = Musica.create_state(micm; number_of_grid_cells = n_cells)

#     names = ClimaAtmos.species_names(chemistry_model)
#     concs = Dict{String,Vector{Float64}}()
#     for s in names
#         concs[String(s)] = Float64.(vec(parent(getproperty(Y.c, Symbol(:ρ, s)))))
#     end
#     Musica.set_concentrations!(state, concs)

#     Musica.solve!(micm, state, Float64(p.dt))

#     updated = Musica.get_concentrations(state)
#     for s in names
#         parent(getproperty(Y.c, Symbol(:ρ, s))) .= updated[String(s)]
#     end
# end

end # module ClimaAtmosMusica
