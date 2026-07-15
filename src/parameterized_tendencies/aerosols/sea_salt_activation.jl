# Activation seam for prognostic sea salt (plan §6, Phase 2).
#
# This provides the bridge from the prognostic per-bin dry-mass tracers
# (ρSSLTxx) to a CloudMicrophysics `AerosolDistribution`, plus a thin wrapper
# around `AerosolActivation`. It is the drop-in point for coupling interactive
# sea salt to droplet activation.
#
# It is INERT by default: nothing here is called from any tendency, so runs are
# bit-for-bit identical to the pre-activation baseline. Wiring it into the
# 2-moment activation source (behind an `enable_aerosol_activation` config flag,
# default `false`) is the remaining step; it belongs next to the prescribed-
# aerosol activation path in microphysics_wrappers.jl, feeding these prognostic
# masses instead of `prescribed_aerosols_field`.

"""
    sea_salt_number_concentration(M, ρ_s, r_dry, σ)

Number concentration `N` [# m⁻³] implied by the dry-mass concentration `M`
(`ρSSLTxx`, [kg m⁻³]) for a lognormal mode of dry-salt density `ρ_s`,
number-median radius `r_dry`, and width `σ`:

    N = M / (ρ_s · v̄),   v̄ = sea_salt_mean_particle_volume(r_dry, σ)

Linear in `M` (so timestep-invariant given fixed size params), and inverts
[`sea_salt_mean_particle_volume`](@ref) exactly (round-trip `N → M → N`).
"""
function sea_salt_number_concentration(M, ρ_s, r_dry, σ)
    FT = typeof(M)
    v̄ = sea_salt_mean_particle_volume(r_dry, σ)
    return max(M, zero(FT)) / (ρ_s * v̄)
end

"""
    sea_salt_mode_kappa(r_dry, σ, N, κ)

Single-component (κ-hygroscopicity) `CloudMicrophysics.AerosolModel.Mode_κ` for
one sea-salt bin. Field order (CM 0.36):
`(r_dry, stdev, N, vol_mix_ratio, mass_mix_ratio, molar_mass, kappa)`. For a
pure component the mixing ratios are `(1,)`; `molar_mass` is `(0,)` because only
number activation (which uses `κ` and `vol_mix_ratio`) is used here — mirroring
the prescribed-aerosol path. Add the real molar mass (NaCl ≈ 0.0584 kg mol⁻¹)
before using `M_activated`.
"""
function sea_salt_mode_kappa(r_dry, σ, N, κ)
    FT = typeof(r_dry)
    return CMAM.Mode_κ(r_dry, σ, N, (FT(1),), (FT(1),), (FT(0),), (κ,))
end

"""
    bins_to_aerosol_distribution(bin_masses, r_drys, σ, κ, ρ_s)

Bridge the prognostic per-bin dry-mass concentrations `bin_masses`
(`ρSSLT01…`, an `NTuple`) to a `CloudMicrophysics.AerosolModel.AerosolDistribution`
of κ-Köhler modes, using per-bin dry radii `r_drys` (matching `bin_masses`
length), shared width `σ`, hygroscopicity `κ`, and dry-salt density `ρ_s`. Pure
and allocation-free (tuple-based), so it can be called pointwise.
"""
function bins_to_aerosol_distribution(
    bin_masses::NTuple{n, FT},
    r_drys::NTuple{n, FT},
    σ,
    κ,
    ρ_s,
) where {n, FT}
    modes = ntuple(n) do k
        N = sea_salt_number_concentration(bin_masses[k], ρ_s, r_drys[k], σ)
        sea_salt_mode_kappa(r_drys[k], σ, N, FT(κ))
    end
    return CMAM.AerosolDistribution(modes)
end

"""
    sea_salt_activated_number(
        dist, act_params, air_params, thermo_params, T, p, w,
        q_tot, q_liq, q_ice, N_liq, N_ice,
    )

Total activated number concentration [# m⁻³] for the sea-salt distribution
`dist` via `CloudMicrophysics.AerosolActivation.total_N_activated`, using the
local-supersaturation-with-preexisting-hydrometeors variant (`N_liq`, `N_ice`
sinks) so it is valid at and above cloud base. Provided for the activation
wiring; not called by any tendency yet.
"""
function sea_salt_activated_number(
    dist, act_params, air_params, thermo_params, T, p, w,
    q_tot, q_liq, q_ice, N_liq, N_ice,
)
    return CMAA.total_N_activated(
        act_params, dist, air_params, thermo_params,
        T, p, w, q_tot, q_liq, q_ice, N_liq, N_ice,
    )
end
