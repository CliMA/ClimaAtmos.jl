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
    chemistry_tendency!(Yₜ, Y, p, t, ::GasPhaseChem)

Source terms are provided by Musica extension.
"""
function chemistry_tendency!(Yₜ, Y, p, t, ::GasPhaseChem) end
