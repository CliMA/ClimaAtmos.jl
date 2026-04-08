"""
    IsothermalProfile(; temperature = 300)

A setup with a uniform temperature and barometric pressure profile.

## Example
```julia
setup = IsothermalProfile(; temperature = 300)
```
"""
struct IsothermalProfile{FT}
    temperature::FT
end

IsothermalProfile(; temperature = 300) = IsothermalProfile(temperature)

function center_initial_condition(setup::IsothermalProfile, local_geometry, params)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    MSLP = CAP.MSLP(params)
    grav = CAP.grav(params)
    T = FT(setup.temperature)

    (; z) = local_geometry.coordinates
    p = MSLP * exp(-z * grav / (R_d * T))

    return physical_state(; T, p)
end
