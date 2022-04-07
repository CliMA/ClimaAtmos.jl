module Forcings

using UnPack

using ...Models

export AbstractForcing, AbstractAtmosForcing


export HeldSuarezForcing

abstract type AbstractForcing end
abstract type AbstractAtmosForcing <: AbstractForcing end

include("forcing_held_suarez.jl")

end # module