# Activation seam for prognostic sea salt (plan §6, Phase 2).
#
# The bridge from the prognostic per-bin dry-mass tracers (ρSSLTxx) to a
# CloudMicrophysics `AerosolDistribution`, plus a thin wrapper around
# `AerosolActivation` — the drop-in point for coupling interactive sea salt to
# droplet activation. Per-bin sizes are host-precomputed moments of the
# emitted Gong spectrum (`sea_salt_particle_masses` for the number↔mass
# bridge, `sea_salt_bin_lognormal_fits` for the mode shape), so emitted mass
# and diagnosed number stay consistent.
#
# INERT by default: nothing here is called from any tendency, so runs are
# bit-for-bit identical to the pre-activation baseline. Wiring it into the
# 2-moment activation source (behind an `enable_aerosol_activation` flag,
# default off) belongs next to the prescribed-aerosol activation path in
# microphysics_wrappers.jl, feeding these prognostic masses instead of
# `prescribed_aerosols_field`.

"""
    sea_salt_number_concentration(M, particle_mass)

Number concentration `N` (# m⁻³) implied by the dry-mass concentration `M`
(`ρSSLTxx`, kg m⁻³) for a bin with mean emitted particle mass `particle_mass`
(kg, from [`sea_salt_particle_masses`](@ref) — the same moment emission uses,
so the round trip `N → M → N` is exact). Linear in `M`.
"""
function sea_salt_number_concentration(M, particle_mass)
    return max(M, zero(M)) / particle_mass
end

"""
    sea_salt_mode_kappa(r_dry, σ, N, κ)

Single-component κ-hygroscopicity `CloudMicrophysics.AerosolModel.Mode_κ` for
one sea salt bin, with `(r_dry, σ)` the bin's dry-basis lognormal fit
([`sea_salt_bin_lognormal_fits`](@ref)). For a pure component the mixing
ratios are `(1,)`; `molar_mass` is `(0,)` because only number activation is
used here, mirroring the prescribed-aerosol path (add NaCl ≈ 0.0584 kg mol⁻¹
before using `M_activated`).
"""
function sea_salt_mode_kappa(r_dry, σ, N, κ)
    FT = typeof(r_dry)
    return CMAM.Mode_κ(r_dry, σ, N, (FT(1),), (FT(1),), (FT(0),), (κ,))
end

"""
    bins_to_aerosol_distribution(bin_masses, κ, particle_masses, lognormal_fits)

Bridge the prognostic per-bin dry-mass concentrations `bin_masses` (`ρSSLT01…`,
an `NTuple`) to a `CloudMicrophysics.AerosolModel.AerosolDistribution` of
κ-Köhler modes. `particle_masses` and `lognormal_fits` are the host-side
per-bin moments of the emitted Gong spectrum
([`sea_salt_particle_masses`](@ref), [`sea_salt_bin_lognormal_fits`](@ref)).
Pure and allocation-free (tuple-based), so it can be called pointwise.
"""
function bins_to_aerosol_distribution(
    bin_masses::NTuple{N, FT},
    κ,
    particle_masses,
    lognormal_fits,
) where {N, FT}
    modes = ntuple(Val(N)) do k
        number =
            sea_salt_number_concentration(bin_masses[k], particle_masses[k])
        r_g, σ_g = lognormal_fits[k]
        sea_salt_mode_kappa(r_g, σ_g, number, FT(κ))
    end
    return CMAM.AerosolDistribution(modes)
end

"""
    sea_salt_activated_number(
        dist, act_params, air_params, thermo_params, T, p, w,
        q_tot, q_liq, q_ice, N_liq, N_ice,
    )

Total activated number concentration (# m⁻³) for the sea salt distribution
`dist` via `CloudMicrophysics.AerosolActivation.total_N_activated`, in the
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
