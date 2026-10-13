module ClimaAtmosMusica

import ClimaAtmos
import Musica

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
    @info "MUSICA version: $(pkgversion(Musica))"
    return nothing
end

end # module ClimaAtmosMusica
