using LinearAlgebra: ×, norm, dot

import .Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

import ClimaCore.Fields: ColumnField

NVTX.@annotate function dss!(Y, p, t)
    if do_dss(Y)
        Spaces.weighted_dss_start2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_start2!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_internal2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_internal2!(Y.f, p.ghost_buffer.f)
        Spaces.weighted_dss_ghost2!(Y.c, p.ghost_buffer.c)
        Spaces.weighted_dss_ghost2!(Y.f, p.ghost_buffer.f)
    end
end
