module ClimaAtmosMusica

import ClimaAtmos
import Musica

"""
    ClimaAtmos.chemistry_cache(Y, chemistry_model::ClimaAtmos.GasPhaseChem)

Create the MICM solver and its associated state once at cache-build time so
they are not re-allocated on every chemistry timestep.
"""
function ClimaAtmos.chemistry_cache(_, chemistry_model::ClimaAtmos.GasPhaseChem)
    isnothing(chemistry_model.config_path) && return (;)
    micm = Musica.MICM(; config_path = chemistry_model.config_path)
    state = Musica.create_state(micm)
    return (; micm, state)
end

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
    ClimaAtmos.update_chemistry!(Y, p, t, chemistry_model::ClimaAtmos.GasPhaseChem)

MUSICA-backed in-place chemistry update. Iterates over all grid cells and applies
MICM kinetics for the mechanism's species. The species set comes from the
`GasPhaseChem{N, names}` type parameters (see `chemistry_species_names`), and each
species' molar mass is read from the mechanism via `Musica.get_species_property`,
so nothing here is specific to a particular mechanism. Only runs when
`chemistry_model.config_path` is set.

MICM works in molar concentrations while ClimaAtmos stores mass mixing ratios, so
we divide by the molar mass on the way in and multiply on the way out.
"""
#TODO SOLVE CHEMISTRY FOR UPDRAFTS
function ClimaAtmos.update_chemistry!(
    Y,
    p,
    t,
    chemistry_model::ClimaAtmos.GasPhaseChem{N, names},
) where {N, names}
    isnothing(chemistry_model.config_path) && return nothing

    (; micm, state) = p.chemistry
    Musica.set_conditions!(state; temperatures = 298.15, pressures = 101325)

    # Activate every user-defined reaction rate parameter. Setting each to 1 makes a 
    # USER_DEFINED reaction's
    # config `scaling factor` its effective rate. This targets only the runtime-
    # rate reactions (USER_DEFINED, photolysis, emission); Arrhenius-type rates
    # are computed internally by MICM and never appear here.
    rate_param_names = keys(Musica.get_user_defined_rate_parameters_ordering(state))
    Musica.set_user_defined_rate_parameters!(
        state,
        Dict(name => 1.0 for name in rate_param_names),
    )

    # Molar masses (kg mol⁻¹), read once from the mechanism.
    molar_masses = map(
        s -> Musica.get_species_property(
            micm,
            String(s),
            "molecular weight [kg mol-1]",
            Float64,
        ),
        names,
    )

    n_cells = length(parent(Y.c.ρ))
    for i in 1:n_cells
        concs = Dict(
            String(names[j]) =>
                Float64(parent(getproperty(Y.c, Symbol(:ρ, names[j])))[i]) /
                molar_masses[j] for j in 1:N
        )
        Musica.set_concentrations!(state, concs)
        Musica.solve!(micm, state, p.dt)
        updated = Musica.get_concentrations(state)
        for j in 1:N
            parent(getproperty(Y.c, Symbol(:ρ, names[j])))[i] =
                Float64(only(updated[String(names[j])])) * molar_masses[j]
        end
    end
    return nothing
end

end # module ClimaAtmosMusica
