using LinearAlgebra: ×, norm, dot

import .Parameters as CAP
using ClimaCore: Operators, Fields, Limiters, Geometry, Spaces

import ClimaComms
using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

import ClimaCore.Fields: ColumnField

NVTX.@annotate function dss!(Y, p, t)
    if p.do_dss
        Spaces.weighted_dss!(Y.c => p.ghost_buffer.c, Y.f => p.ghost_buffer.f)
    end
end
