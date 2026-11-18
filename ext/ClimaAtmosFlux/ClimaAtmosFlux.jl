module ClimaAtmosFlux

import ClimaAtmos
import Flux
import JLD2

"""
    ClimaAtmos.build_mlcloud(nn_filepath, nn_param_vec)

Flux-backed `MLCloud` constructor. Loaded automatically when `Flux` is imported
alongside `ClimaAtmos`.
"""
function ClimaAtmos.build_mlcloud(nn_filepath, nn_param_vec)
    # The file stores the network's Flux restructure function, so Flux must be loaded
    nn_architecture = JLD2.load(nn_filepath)["re"]
    return ClimaAtmos.MLCloud_constructor(nn_architecture(nn_param_vec))
end

end # module ClimaAtmosFlux
