"""
    SimplePlume(; prognostic_tke = false)

A simple plume setup using a `DryAdiabaticProfile` with T_surface=310K and
T_min=290K. No moisture. Used for testing EDMFX plume dynamics.

## Example
```julia
setup = SimplePlume(; prognostic_tke = true)
```
"""
struct SimplePlume
    prognostic_tke::Bool
end

SimplePlume(; prognostic_tke::Bool = false) = SimplePlume(prognostic_tke)

function center_initial_condition(setup::SimplePlume, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    temp_profile = DryAdiabaticProfile{FT}(thermo_params, FT(310), FT(290))

    (; z) = local_geometry.coordinates
    T, p = temp_profile(thermo_params, z)

    return physical_state(; T, p)
end

function surface_condition(::SimplePlume, params)
    FT = eltype(params)
    parameterization = MoninObukhov(;
        z0 = FT(1e-4), θ_flux = FT(8e-2), q_flux = FT(0), ustar = FT(0.28),
    )
    return SurfaceState(;
        parameterization, T = FT(310), p = FT(101500), q_vap = FT(0.02245),
    )
end
