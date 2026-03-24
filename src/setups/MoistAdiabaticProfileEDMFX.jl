"""
    MoistAdiabaticProfileEDMFX(; perturb = false)

A moist adiabatic profile for testing EDMFX advection. Uses a
`DryAdiabaticProfile` with T_surface=330K and T_min=200K, combined with
Gaussian moisture and draft area profiles centered at z=4km.

The face initial condition sets `w_draft = 1.0` (non-zero updraft velocity).

## Example
```julia
setup = MoistAdiabaticProfileEDMFX(; perturb = true)
```
"""
struct MoistAdiabaticProfileEDMFX
    perturb::Bool
end

MoistAdiabaticProfileEDMFX(; perturb::Bool = false) =
    MoistAdiabaticProfileEDMFX(perturb)

function center_initial_condition(setup::MoistAdiabaticProfileEDMFX, local_geometry, params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    temp_profile = DryAdiabaticProfile{FT}(thermo_params, FT(330), FT(200))

    (; z) = local_geometry.coordinates
    T, p = temp_profile(thermo_params, z)
    if setup.perturb
        T += _temperature_perturbation(local_geometry.coordinates)
    end

    q_tot = z < FT(0.7e4) ? FT(1e-3) * exp(-(z - FT(4e3))^2 / 2 / FT(1e3)^2) : FT(0)
    draft_area = z < FT(0.7e4) ? FT(0.5) * exp(-(z - FT(4e3))^2 / 2 / FT(1e3)^2) : FT(0)

    return physical_state(; T, p, q_tot, draft_area)
end

function face_initial_condition(::MoistAdiabaticProfileEDMFX, local_geometry, params)
    FT = eltype(params)
    return (; w = FT(0), w_draft = FT(1))
end
