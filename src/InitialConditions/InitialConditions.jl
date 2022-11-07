module InitialConditions

import LinearAlgebra: norm_sqr
import Thermodynamics as TD
import ClimaCore.Geometry as Geometry
import ..TurbulenceConvection as TC
import ..Parameters as CAP

# Types
import ..DryModel
import ..PotentialTemperature
import ..TotalEnergy
import ..InternalEnergy
import ..EquilMoistModel
import ..NonEquilMoistModel

function face_initial_condition(local_geometry, params, turbconv_model)
    z = local_geometry.coordinates.z
    FT = eltype(z)
    tc_kwargs = if turbconv_model isa Nothing
        NamedTuple()
    else
        TC.face_prognostic_vars_edmf(FT, local_geometry, turbconv_model)
    end
    (; w = Geometry.Covariant3Vector(FT(0)), tc_kwargs...)
end

include("baro_wave.jl")
include("box.jl")
include("sphere.jl")
include("single_column.jl")

end
