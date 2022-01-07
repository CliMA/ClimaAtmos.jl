@inline function rhs_moisture!(dY, Y, Ya, t, _...)
    error("not implemented for this model configuration.")
end

@inline rhs_moisture!(dY, Y, Ya, t, p, ::Dry, params, FT) = nothing
