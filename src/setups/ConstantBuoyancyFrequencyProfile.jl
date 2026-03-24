"""
    ConstantBuoyancyFrequencyProfile()

A setup with a constant Brunt-Väisälä frequency (N = 0.01 s⁻¹), a surface
temperature of 288 K, and a uniform horizontal wind of 10 m/s. The temperature
is capped by an isothermal layer to avoid unreasonable values at high altitudes.

Used for topography test cases.

## Example
```julia
setup = ConstantBuoyancyFrequencyProfile()
```
"""
struct ConstantBuoyancyFrequencyProfile end

function center_initial_condition(
    ::ConstantBuoyancyFrequencyProfile,
    local_geometry,
    params,
)
    FT = eltype(params)
    (; z) = local_geometry.coordinates
    (p, T) = background_p_and_T(params, z)
    return physical_state(; T, p, u = background_u(FT))
end
