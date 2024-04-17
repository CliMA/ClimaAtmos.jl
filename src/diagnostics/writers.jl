# Writers.jl
#
# This file defines generic writers for diagnostics with opinionated defaults.

import ClimaCore: Hypsography
import ClimaCore.Remapping: Remapper, interpolate, interpolate!

import NCDatasets

include("hdf5_writer.jl")
include("netcdf_writer.jl")
