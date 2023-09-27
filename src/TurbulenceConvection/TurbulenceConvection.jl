module TurbulenceConvection

include("Parameters.jl")
import .Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters

Base.broadcastable(param_set::APS) = tuple(param_set)

end
