# ============================================================================
# Unified Chemistry Tendencies
# ============================================================================
#
# Single entry point for all chemistry tendency calculations. Uses a MUSICA
# backend to calculate gas-phase chemistry tendencies.

import Musica as MSC

function chemistry_tendency!(Yₜ, Y, p, t, ::GasPhaseChem)
    println("MUSICA version: ", MSC.get_version())
    return nothing
end