using Revise, Infiltrator
import ClimaAtmos as CA
import SciMLBase: step!
import ClimaComms
import ClimaCore: Fields, Geometry, Operators, Spaces, Grids, Utilities
ENV["CLIMACOMMS_DEVICE"]="CPU"
