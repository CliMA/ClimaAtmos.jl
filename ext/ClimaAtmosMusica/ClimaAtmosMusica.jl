module ClimaAtmosMusica

import ClimaAtmos
import Musica
import Musica.MechanismConfiguration:
    Species, PhaseSpecies, Phase, ReactionComponent, Arrhenius, Troe, Surface, Mechanism


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
# SOLVE CHEMISTRY FOR UPDRAFTS
function ClimaAtmos.update_chemistry!(
    Y,
    p,
    t,
    chemistry_model::ClimaAtmos.GasPhaseChem,
)
    isnothing(chemistry_model.config_path) && return nothing

    micm = Musica.MICM(; config_path = chemistry_model.config_path) # chemistry model should have a micm
    state = Musica.create_state(micm)                               # also make the state outside of the solve set
    Musica.set_conditions!(state; temperatures = 298.15, pressures = 101325)
    Musica.set_user_defined_rate_parameters!(
        state,
        Dict(
            "USER.forward_AB_to_A_B" => 1.0,
            "USER.reverse_A_B_to_AB" => 1.0,
        ),
    )
    mw_A = 0.029
    mw_B = 0.029
    mw_AB = 0.058
    species = ClimaAtmos.musica_species_names(chemistry_model.config_path)
    n_cells = length(parent(Y.c.ρ))

    # print the number of cells
    # println("Running MUSICA chemistry for $n_cells cells")

    for i in 1:n_cells
        # concs = Dict(
        #     String(s) => Float64(parent(getproperty(Y.c, Symbol(:ρ, s)))[i])
        #     for s in species
        # )
        # hard code the concentrations
        concs = Dict(
            "A" => Float64(parent(getproperty(Y.c, Symbol(:ρ, :A)))[i]) / mw_A,
            "B" => Float64(parent(getproperty(Y.c, Symbol(:ρ, :B)))[i]) / mw_B,
            "AB" => Float64(parent(getproperty(Y.c, Symbol(:ρ, :AB)))[i]) / mw_AB,
        )

        Musica.set_concentrations!(state, concs)
        # println("Running MUSICA chemistry for cell $i with concentrations: ", concs)
        Musica.solve!(micm, state, p.dt)
        updated = Musica.get_concentrations(state)
        # for s in species
        #     parent(getproperty(Y.c, Symbol(:ρ, s)))[i] = Float64(only(updated[String(s)]))
        # end
        parent(getproperty(Y.c, Symbol(:ρ, :A)))[i] = Float64(only(updated["A"])) * mw_A
        parent(getproperty(Y.c, Symbol(:ρ, :B)))[i] = Float64(only(updated["B"])) * mw_B
        parent(getproperty(Y.c, Symbol(:ρ, :AB)))[i] = Float64(only(updated["AB"])) * mw_AB
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




"""
    build_nighttime_no3()

Nighttime NOx / N2O5 chemistry mechanism:

    (1) NO + O3      → NO2 + O2        k = 1.9e-14 * exp(-1400/T)
    (2) NO2 + O3     → NO3 + O2        k = 1.4e-13 * exp(-2470/T)
    (3) NO3 + NO     → 2 NO2           k = 2.6e-11 * exp(+380/T)
    (4) NO3 + NO2 +M → N2O5 + M        Troe falloff (IUPAC 2013)
    (5) N2O5 + M     → NO3 + NO2 + M   Troe falloff, thermal decomposition
    (6) N2O5         → 2 HNO3          heterogeneous uptake, γ * ω/4 * [N2O5]

`M` is not a tracked species: Troe falloff already accounts for the bath-gas
dependence internally, so reactions (4)/(5) only list the chemically active
reactants/products.

The (4)/(5) Troe parameters and the (6) uptake coefficient below are
representative literature values, not a verified data-sheet transcription —
check them against the current IUPAC/JPL evaluation before using this for a
science run.
"""
function build_nighttime_no3()
    NO = Species(name = "NO", molecular_weight = 0.0300)
    O3 = Species(name = "O3", molecular_weight = 0.0480)
    NO2 = Species(name = "NO2", molecular_weight = 0.0460)
    O2 = Species(name = "O2", molecular_weight = 0.0320)
    NO3 = Species(name = "NO3", molecular_weight = 0.0620)
    N2O5 = Species(name = "N2O5", molecular_weight = 0.1080)
    HNO3 = Species(name = "HNO3", molecular_weight = 0.0630)

    # N2O5 needs an explicit diffusion coefficient: it's the only species used as
    # a `Surface` reactant, and MICM's surface-reaction rate law requires it.
    gas = Phase(
        name = "gas",
        species = [
            NO,
            O3,
            NO2,
            O2,
            NO3,
            PhaseSpecies(name = "N2O5", diffusion_coefficient = 1.0e-5),
            HNO3,
        ],
    )

    no_o3 = Arrhenius(
        name = "NO_O3",
        gas_phase = "gas",
        A = 1.9e-14,
        C = -1400.0,
        reactants = [
            ReactionComponent(species_name = "NO"),
            ReactionComponent(species_name = "O3"),
        ],
        products = [
            ReactionComponent(species_name = "NO2"),
            ReactionComponent(species_name = "O2"),
        ],
    )

    no2_o3 = Arrhenius(
        name = "NO2_O3",
        gas_phase = "gas",
        A = 1.4e-13,
        C = -2470.0,
        reactants = [
            ReactionComponent(species_name = "NO2"),
            ReactionComponent(species_name = "O3"),
        ],
        products = [
            ReactionComponent(species_name = "NO3"),
            ReactionComponent(species_name = "O2"),
        ],
    )

    no3_no = Arrhenius(
        name = "NO3_NO",
        gas_phase = "gas",
        A = 2.6e-11,
        C = 380.0,
        reactants = [
            ReactionComponent(species_name = "NO3"),
            ReactionComponent(species_name = "NO"),
        ],
        products = [ReactionComponent(species_name = "NO2", coefficient = 2.0)],
    )

    # IUPAC-style falloff: k0(T) = k0_A * exp(k0_C/T) * (T/300)^k0_B, same form for kinf.
    n2o5_formation = Troe(
        name = "NO2_NO3_M_to_N2O5",
        gas_phase = "gas",
        k0_A = 3.6e-30,
        k0_B = -4.1,
        kinf_A = 1.9e-12,
        kinf_B = 0.2,
        Fc = 0.35,
        reactants = [
            ReactionComponent(species_name = "NO2"),
            ReactionComponent(species_name = "NO3"),
        ],
        products = [ReactionComponent(species_name = "N2O5")],
    )

    n2o5_decomposition = Troe(
        name = "N2O5_M_decomposition",
        gas_phase = "gas",
        k0_A = 1.3e-3,
        k0_B = -3.5,
        k0_C = -11000.0,
        kinf_A = 9.7e14,
        kinf_B = 0.1,
        kinf_C = -11080.0,
        Fc = 0.35,
        reactants = [ReactionComponent(species_name = "N2O5")],
        products = [
            ReactionComponent(species_name = "NO3"),
            ReactionComponent(species_name = "NO2"),
        ],
    )

    n2o5_uptake = Surface(
        name = "N2O5_heterogeneous_uptake",
        gas_phase = "gas",
        reaction_probability = 0.02,
        gas_phase_species = ReactionComponent(species_name = "N2O5"),
        gas_phase_products = [ReactionComponent(species_name = "HNO3", coefficient = 2.0)],
    )

    return Mechanism(
        name = "NOx_N2O5",
        version = "1.0.0",
        species = [NO, O3, NO2, O2, NO3, N2O5, HNO3],
        phases = [gas],
        reactions = [
            no_o3,
            no2_o3,
            no3_no,
            n2o5_formation,
            n2o5_decomposition,
            n2o5_uptake,
        ],
    )
end



end # module ClimaAtmosMusica
