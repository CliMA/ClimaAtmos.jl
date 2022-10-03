module ClimaAtmos

include("Parameters.jl")
import .Parameters

# import ClimaCore: Fields

# function error_on_nan(cache::Fields.FieldVectors)
#     for pn in propertynames(cache)
#         error_on_nan(getproperty(cache, pn))
#     end
# end

# function error_on_nan(Y::Fields.FieldVectors, p::NamedTuple)
#     error_on_nan(Y)
#     error_on_nan(p)
# end

# function error_on_nan(cache::NamedTuple)
#     for pn in propertynames(cache)
#         error_on_nan(getproperty(cache, pn))
#     end
# end

# function error_on_nan(Y::Fields.Field)
#     if any(isnan, Y)
#         props_with_nan = filter(
#             prop -> any(isnan, parent(Fields.single_field(Y, prop))),
#             Fields.property_chains(Y),
#         )
#         error("NaN detected in $(join(map(prop_string, props_with_nan), ", "))")
#     end
# end


include("Experimental/Experimental.jl")

include("TurbulenceConvection/TurbulenceConvection.jl")

end # module
