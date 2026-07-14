module COSP

include("subcol.jl")
import .COSPSubcolumns

include("prec_subcol.jl")
import .COSPPrecipSubcolumns

include("hydrometeor_subcol.jl")
import .COSPHydrometeorSubcolumns

include("cloudsat_optics.jl")
import .COSPCloudSatOptics

include("cloudsat_reflectivity.jl")
import .COSPCloudSatReflectivity

end
