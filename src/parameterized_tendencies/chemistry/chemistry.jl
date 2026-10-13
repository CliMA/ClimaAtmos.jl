###
### Chemistry Module
###

# Gas-phase chemistry for ClimaAtmos.
# The MUSICA backend is provided by the ClimaAtmosMusica extension;
# this file defines only the fallback for when no chemistry is loaded.

"""
    chemistry_tendency!(Yₜ, Y, p, t, chemistry_model)

Add gas-phase chemistry source terms to `Yₜ` in place; return `nothing`.

Dispatches on the chemistry model in `p.atmos`:

  - `::Nothing`: no chemistry is active; the tendency is a no-op.
  - `::GasPhaseChem`: gas-phase chemistry. The fallback defined here is a no-op; the
    MUSICA-backed method is provided by the `ClimaAtmosMusica` extension, which is loaded
    automatically when `Musica` is imported alongside `ClimaAtmos`.
"""
chemistry_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
function chemistry_tendency!(Yₜ, Y, p, t, ::GasPhaseChem) end
