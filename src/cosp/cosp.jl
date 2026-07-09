module COSP

include("subcol.jl")
import .COSPSubcolumns

include("prec_subcol.jl")
import .COSPPrecipSubcolumns

include("hydrometeor_subcol.jl")
import .COSPHydrometeorSubcolumns

include("reff_np_1m.jl")
import .COSP1MReffNpDiagnostics

include("cloudsat_optics.jl")
import .COSPCloudSatOptics

include("cloudsat_reflectivity.jl")
import .COSPCloudSatReflectivity

end
