# ============================================================================
# Chemistry Module
# ============================================================================
# Gas-phase chemistry for ClimaAtmos.
# The MUSICA backend is provided by the ClimaAtmosMusica extension;
# this file defines only the no-op fallback for when no chemistry is loaded.

"""
    chemistry_tendency!(Yₜ, Y, p, t, ::Nothing)

No-op: no chemistry model is active.
"""
chemistry_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
