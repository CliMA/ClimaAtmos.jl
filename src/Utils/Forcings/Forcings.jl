module Forcings

abstract type AbstractForcing end
abstract type AbstractAtmosForcing end

include("forcing_held_suarez.jl")

end # module