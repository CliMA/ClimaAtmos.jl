# ============================================================================
# MoistBaroclinicWave and MoistBaroclinicWaveWithEDMF
# ============================================================================

"""
    MoistBaroclinicWave(; perturb = true, deep_atmosphere = false)

A moist baroclinic wave setup. Uses the same dynamical core as
[`DryBaroclinicWave`](@ref), but adds a moisture profile and converts
virtual temperature to temperature.

## Example
```julia
setup = MoistBaroclinicWave(; perturb = true, deep_atmosphere = false)
```
"""
struct MoistBaroclinicWave
    perturb::Bool
    deep_atmosphere::Bool
end

MoistBaroclinicWave(; perturb::Bool = true, deep_atmosphere::Bool = false) =
    MoistBaroclinicWave(perturb, deep_atmosphere)

"""
    MoistBaroclinicWaveWithEDMF(; perturb = true, deep_atmosphere = false)

The same setup as [`MoistBaroclinicWave`](@ref), but with an initial TKE of 0
and an initial draft area fraction of 0.2.

## Example
```julia
setup = MoistBaroclinicWaveWithEDMF(; perturb = true, deep_atmosphere = false)
```
"""
struct MoistBaroclinicWaveWithEDMF
    perturb::Bool
    deep_atmosphere::Bool
end

MoistBaroclinicWaveWithEDMF(; perturb::Bool = true, deep_atmosphere::Bool = false) =
    MoistBaroclinicWaveWithEDMF(perturb, deep_atmosphere)

# ============================================================================
# Shared moist baroclinic wave helper
# ============================================================================

const MoistBaroclinic = Union{MoistBaroclinicWave, MoistBaroclinicWaveWithEDMF}

function _moist_baroclinic_wave_values(z, ϕ, λ, params, perturb, deep_atmosphere)
    FT = eltype(params)
    MSLP = CAP.MSLP(params)

    p_w = FT(3.4e4)
    p_t = FT(1e4)
    q_t = FT(1e-12)
    q_0 = FT(0.018)
    ϕ_w = FT(40)
    ε = FT(0.608)

    if deep_atmosphere
        (; p, T, u, v) =
            deep_atmos_barowave_values(z, ϕ, λ, params, perturb)
    else
        (; p, T, u, v) =
            shallow_atmos_barowave_values(z, ϕ, λ, params, perturb)
    end

    q_tot =
        (p <= p_t) ? q_t : q_0 * exp(-(ϕ / ϕ_w)^4) * exp(-((p - MSLP) / p_w)^2)
    T = T / (1 + ε * q_tot)

    return (; T, p, q_tot, u, v)
end

# ============================================================================
# center_initial_condition
# ============================================================================

function center_initial_condition(setup::MoistBaroclinicWave, local_geometry, params)
    (; z, lat, long) = local_geometry.coordinates
    (; T, p, q_tot, u, v) = _moist_baroclinic_wave_values(
        z, lat, long, params, setup.perturb, setup.deep_atmosphere,
    )
    return physical_state(; T, p, q_tot, u, v)
end

function center_initial_condition(
    setup::MoistBaroclinicWaveWithEDMF,
    local_geometry,
    params,
)
    FT = eltype(params)
    (; z, lat, long) = local_geometry.coordinates
    (; T, p, q_tot, u, v) = _moist_baroclinic_wave_values(
        z, lat, long, params, setup.perturb, setup.deep_atmosphere,
    )
    return physical_state(; T, p, q_tot, u, v, tke = FT(0), draft_area = FT(0.2))
end
