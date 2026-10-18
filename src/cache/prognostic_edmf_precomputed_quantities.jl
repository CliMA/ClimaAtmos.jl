#####
##### Precomputed quantities
#####
import NVTX
import Thermodynamics as TD
import ClimaCore: DataLayouts, Spaces, Fields

"""
    saturation_adjustment_tuple(thermo_params, ::TD.ph, p, h, q_tot)

Return `(; T, q_liq, q_ice)` from a pressure-enthalpy saturation adjustment.

This wrapper keeps the SGS subdomain thermodynamics to a single
`TD.saturation_adjustment` call per grid point, and caps the Newton iteration at
`maxiter = 4`. The `TD.ρe` counterpart used for the grid mean lives in
`precomputed_quantities.jl`.

# Arguments

  - `thermo_params`: Thermodynamics parameter set.
  - `p`: Air pressure [Pa].
  - `h`: Specific enthalpy [J/kg].
  - `q_tot`: Total water specific humidity [kg/kg].
"""
function saturation_adjustment_tuple(thermo_params, ::TD.ph, p, h, q_tot)
    sa_result = TD.saturation_adjustment(
        thermo_params,
        TD.ph(),
        p,
        h,
        q_tot;
        maxiter = 4,
    )
    return (; T = sa_result.T, q_liq = sa_result.q_liq, q_ice = sa_result.q_ice)
end

"""
    set_prognostic_edmf_precomputed_quantities_environment!(Y, p, ᶠuₕ³, t)

Update the EDMFX environment precomputed quantities in `p` for `PrognosticEDMFX`.
Returns `nothing`.

Mutates in `p.precomputed`: the environment velocities and kinetic energy
(`ᶠu₃⁰`, `ᶜu⁰`, `ᶠu³⁰`, `ᶜK⁰`) and the environment thermodynamic state (`ᶜT⁰`,
`ᶜq_tot_nonneg⁰`, `ᶜq_liq⁰`, `ᶜq_ice⁰`); for `EquilibriumMicrophysics0M` it also
reuses the shared `ᶜsa_result` field as a saturation-adjustment workspace.

The environment values are reconstructed as the grid mean minus the updrafts
(`ᶜspecific_env_value`, `ᶜspecific_env_mse`). For
`NonEquilibriumMicrophysics1M`/`2M`, the condensate is taken directly from the
prognostic tracers and `ᶜT⁰` is obtained from the enthalpy, floored at
`T_min_sgs`; for `EquilibriumMicrophysics0M`, `ᶜT⁰` and the phase partition come
from a pressure-enthalpy saturation adjustment. Reads `ᶜΦ` from `p.core` and
`ᶜp` from `p.precomputed`.

# Arguments

  - `ᶠuₕ³`: Third contravariant component of the horizontal velocity, interpolated
    to faces [1/s].
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_environment!(
    Y,
    p,
    ᶠuₕ³,
    t,
)

    thermo_params = CAP.thermodynamics_params(p.params)
    (; turbconv_model) = p.atmos
    (; ᶜΦ,) = p.core
    (; ᶜp) = p.precomputed
    (; ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) =
        p.precomputed

    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    T_min_sgs = CAP.T_min_sgs(p.params)

    if p.atmos.microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), Y, p)
        ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), Y, p)
        ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
        ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
        # One fused column loop: environment velocities and kinetic energy,
        # then the environment thermodynamic state from primitives.
        DataLayouts.foreach_column(
            ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
            Y.c, Y.f, ᶠuₕ³, ᶜΦ, ᶜq_tot⁰, ᶜmse⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰,
        ) do ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
        Yc, Yf, ᶠuₕ³, ᶜΦ, ᶜq_tot⁰, ᶜmse⁰, ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰
            set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Yc, Yf, turbconv_model)
            set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Yc.uₕ, ᶠuₕ³)
            @. ᶜq_liq⁰ = max(0, ᶜq_lcl⁰ + ᶜq_rai⁰)
            @. ᶜq_ice⁰ = max(0, ᶜq_icl⁰ + ᶜq_sno⁰)
            # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
            @. ᶜq_tot_nonneg⁰ = max(ᶜq_liq⁰ + ᶜq_ice⁰, ᶜq_tot⁰)
            ᶜh⁰ = @. lazy(ᶜmse⁰ - ᶜΦ)  # specific enthalpy
            @. ᶜT⁰ = max(
                T_min_sgs,
                TD.air_temperature(
                    thermo_params,
                    TD.ph(),
                    ᶜh⁰,
                    ᶜq_tot_nonneg⁰,
                    ᶜq_liq⁰,
                    ᶜq_ice⁰,
                ),
            )
            return nothing
        end
    else
        # EquilibriumMicrophysics0M: use saturation adjustment to get T and phase partition
        DataLayouts.foreach_column(
            ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜq_tot_nonneg⁰, Y.c, Y.f, ᶠuₕ³, ᶜq_tot⁰,
        ) do ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜq_tot_nonneg⁰, Yc, Yf, ᶠuₕ³, ᶜq_tot⁰
            set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Yc, Yf, turbconv_model)
            set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Yc.uₕ, ᶠuₕ³)
            @. ᶜq_tot_nonneg⁰ = max(0, ᶜq_tot⁰)
            return nothing
        end
        (; ᶜsa_result) = p.precomputed
        h⁰ = @. lazy(ᶜmse⁰ - ᶜΦ)
        @. ᶜsa_result =
            saturation_adjustment_tuple(thermo_params, TD.ph(), ᶜp, h⁰, ᶜq_tot_nonneg⁰)
        @. ᶜT⁰ = ᶜsa_result.T
        @. ᶜq_liq⁰ = ᶜsa_result.q_liq
        @. ᶜq_ice⁰ = ᶜsa_result.q_ice
    end
    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_draft!(Y, p, ᶠuₕ³, t)

Update the velocity and thermodynamic quantities of each SGS draft (updraft) for
`PrognosticEDMFX`. Returns `nothing`.

Mutates in `p.precomputed`, for every mass-flux subdomain `j`: `ᶜuʲs`, `ᶠu³ʲs`,
`ᶜKʲs`, `ᶠKᵥʲs`, `ᶜTʲs`, `ᶜq_tot_nonnegʲs`, `ᶜq_liqʲs`, `ᶜq_iceʲs`, and `ᶜρʲs`;
for `EquilibriumMicrophysics0M` it also reuses the shared `ᶜsa_result` field as
a saturation-adjustment workspace.

For `NonEquilibriumMicrophysics1M`/`2M`, the draft condensate is taken from the
prognostic subdomain tracers and `ᶜTʲ` follows from the enthalpy `mse - Φ`,
floored at `T_min_sgs`; for `EquilibriumMicrophysics0M`, `ᶜTʲ` and the phase
partition come from a pressure-enthalpy saturation adjustment. The draft density
`ᶜρʲ` is diagnosed at the grid-mean pressure `ᶜp`, which is shared across all
subdomains.

# Arguments

  - `ᶠuₕ³`: Third contravariant component of the horizontal velocity, interpolated
    to faces [1/s].
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_draft!(
    Y,
    p,
    ᶠuₕ³,
    t,
)
    (; turbconv_model) = p.atmos
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_params = CAP.thermodynamics_params(p.params)

    (; ᶜΦ) = p.core
    (;
        ᶜp,
        ᶜuʲs,
        ᶠu³ʲs,
        ᶜKʲs,
        ᶠKᵥʲs,
        ᶜTʲs,
        ᶜq_tot_nonnegʲs,
        ᶜq_liqʲs,
        ᶜq_iceʲs,
        ᶜρʲs,
    ) = p.precomputed

    T_min_sgs = CAP.T_min_sgs(p.params)
    for j in 1:n
        ᶜuʲ = ᶜuʲs.:($j)
        ᶠu³ʲ = ᶠu³ʲs.:($j)
        ᶜKʲ = ᶜKʲs.:($j)
        ᶠKᵥʲ = ᶠKᵥʲs.:($j)
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜq_tot_nonnegʲ = ᶜq_tot_nonnegʲs.:($j)
        ᶜq_liqʲ = ᶜq_liqʲs.:($j)
        ᶜq_iceʲ = ᶜq_iceʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜsgsʲ = Y.c.sgsʲs.:($j)
        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃

        if p.atmos.microphysics_model isa Union{
            NonEquilibriumMicrophysics1M,
            NonEquilibriumMicrophysics2M,
        }
            # One fused column loop per draft: velocities and kinetic energies,
            # then the draft thermodynamic state from the prognostic tracers.
            DataLayouts.foreach_column(
                ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠKᵥʲ, ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ,
                ᶜsgsʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³, ᶜΦ, ᶜp,
            ) do ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠKᵥʲ, ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ,
            ᶜsgsʲ, ᶠu₃ʲ, ᶜuₕ, ᶠuₕ³, ᶜΦ, ᶜp
                set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, ᶜuₕ, ᶠuₕ³)
                @. ᶠKᵥʲ = (adjoint(CT3(ᶠu₃ʲ)) * ᶠu₃ʲ) / 2
                @. ᶜq_liqʲ = max(0, ᶜsgsʲ.q_lcl + ᶜsgsʲ.q_rai)
                @. ᶜq_iceʲ = max(0, ᶜsgsʲ.q_icl + ᶜsgsʲ.q_sno)
                # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
                @. ᶜq_tot_nonnegʲ = max(ᶜq_liqʲ + ᶜq_iceʲ, ᶜsgsʲ.q_tot)
                ᶜhʲ = @. lazy(ᶜsgsʲ.mse - ᶜΦ)
                @. ᶜTʲ = max(
                    T_min_sgs,
                    TD.air_temperature(
                        thermo_params,
                        TD.ph(),
                        ᶜhʲ,
                        ᶜq_tot_nonnegʲ,
                        ᶜq_liqʲ,
                        ᶜq_iceʲ,
                    ),
                )
                @. ᶜρʲ = TD.air_density(
                    thermo_params,
                    ᶜTʲ,
                    ᶜp,
                    ᶜq_tot_nonnegʲ,
                    ᶜq_liqʲ,
                    ᶜq_iceʲ,
                )
                return nothing
            end
        else
            # EquilibriumMicrophysics0M: use saturation adjustment
            DataLayouts.foreach_column(
                ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠKᵥʲ, ᶜq_tot_nonnegʲ, ᶜsgsʲ, ᶠu₃ʲ, Y.c.uₕ, ᶠuₕ³,
            ) do ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠKᵥʲ, ᶜq_tot_nonnegʲ, ᶜsgsʲ, ᶠu₃ʲ, ᶜuₕ, ᶠuₕ³
                set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, ᶜuₕ, ᶠuₕ³)
                @. ᶠKᵥʲ = (adjoint(CT3(ᶠu₃ʲ)) * ᶠu₃ʲ) / 2
                @. ᶜq_tot_nonnegʲ = max(0, ᶜsgsʲ.q_tot)
                return nothing
            end
            (; ᶜsa_result) = p.precomputed
            @. ᶜsa_result = saturation_adjustment_tuple(
                thermo_params,
                TD.ph(),
                ᶜp,
                ᶜsgsʲ.mse - ᶜΦ,
                ᶜq_tot_nonnegʲ,
            )
            @. ᶜTʲ = ᶜsa_result.T
            @. ᶜq_liqʲ = ᶜsa_result.q_liq
            @. ᶜq_iceʲ = ᶜsa_result.q_ice
            @. ᶜρʲ =
                TD.air_density(
                    thermo_params,
                    ᶜTʲ,
                    ᶜp,
                    ᶜq_tot_nonnegʲ,
                    ᶜq_liqʲ,
                    ᶜq_iceʲ,
                )
        end
    end
    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_draft_thermo!(Y, p)

Update the per-updraft thermodynamic precomputed quantities — for every
subdomain `j`, `ᶜq_tot_nonnegʲs.:(j)`, `ᶜq_liqʲs.:(j)`, `ᶜq_iceʲs.:(j)`,
`ᶜTʲs.:(j)`, `ᶜρʲs.:(j)` — from `Y`, the cached pressure `ᶜp`, and the
geopotential `ᶜΦ`. Does not touch velocity, kinetic energy, or the
subdomain-density difference `ᶜρ_diffʲs`.

For `NonEquilibriumMicrophysics1M`/`2M`, draft condensate is taken from the
prognostic subdomain tracers and `ᶜTʲ` follows from the enthalpy `mse - Φ`,
floored at `T_min_sgs`; for `EquilibriumMicrophysics0M`, `ᶜTʲ` and the phase
partition come from a pressure-enthalpy saturation adjustment that reuses the
shared `ᶜsa_result` field as scratch. `ᶜρʲ` is diagnosed at the shared `ᶜp`.

Called from `set_prognostic_edmf_precomputed_quantities_draft!` after the
velocity block, and from the EDMFX implicit-stage Picard refresh in
`initialize_implicit_stage_problem!`.
"""
function set_prognostic_edmf_precomputed_quantities_draft_thermo!(Y, p)
    (; microphysics_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core
    (; ᶜp, ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs, ᶜρʲs) = p.precomputed
    T_min_sgs = CAP.T_min_sgs(p.params)

    for j in 1:n_mass_flux_subdomains(turbconv_model)
        ᶜTʲ = ᶜTʲs.:($j)
        ᶜq_tot_nonnegʲ = ᶜq_tot_nonnegʲs.:($j)
        ᶜq_liqʲ = ᶜq_liqʲs.:($j)
        ᶜq_iceʲ = ᶜq_iceʲs.:($j)
        ᶜρʲ = ᶜρʲs.:($j)
        ᶜsgsʲ = Y.c.sgsʲs.:($j)

        if microphysics_model isa Union{
            NonEquilibriumMicrophysics1M,
            NonEquilibriumMicrophysics2M,
        }
            DataLayouts.foreach_column(
                ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ,
                ᶜsgsʲ, ᶜΦ, ᶜp,
            ) do ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ,
            ᶜsgsʲ, ᶜΦ, ᶜp
                @. ᶜq_liqʲ = max(0, ᶜsgsʲ.q_lcl + ᶜsgsʲ.q_rai)
                @. ᶜq_iceʲ = max(0, ᶜsgsʲ.q_icl + ᶜsgsʲ.q_sno)
                # Clamp q_tot ≥ q_cond to keep q_vap = q_tot - q_cond non-negative.
                @. ᶜq_tot_nonnegʲ = max(ᶜq_liqʲ + ᶜq_iceʲ, ᶜsgsʲ.q_tot)
                ᶜhʲ = @. lazy(ᶜsgsʲ.mse - ᶜΦ)
                @. ᶜTʲ = max(
                    T_min_sgs,
                    TD.air_temperature(
                        thermo_params,
                        TD.ph(),
                        ᶜhʲ,
                        ᶜq_tot_nonnegʲ,
                        ᶜq_liqʲ,
                        ᶜq_iceʲ,
                    ),
                )
                @. ᶜρʲ = TD.air_density(
                    thermo_params,
                    ᶜTʲ,
                    ᶜp,
                    ᶜq_tot_nonnegʲ,
                    ᶜq_liqʲ,
                    ᶜq_iceʲ,
                )
                return nothing
            end
        else
            # EquilibriumMicrophysics0M: pressure-enthalpy saturation adjustment.
            @. ᶜq_tot_nonnegʲ = max(0, ᶜsgsʲ.q_tot)
            (; ᶜsa_result) = p.precomputed
            @. ᶜsa_result = saturation_adjustment_tuple(
                thermo_params,
                TD.ph(),
                ᶜp,
                ᶜsgsʲ.mse - ᶜΦ,
                ᶜq_tot_nonnegʲ,
            )
            @. ᶜTʲ = ᶜsa_result.T
            @. ᶜq_liqʲ = ᶜsa_result.q_liq
            @. ᶜq_iceʲ = ᶜsa_result.q_ice
            @. ᶜρʲ = TD.air_density(
                thermo_params,
                ᶜTʲ,
                ᶜp,
                ᶜq_tot_nonnegʲ,
                ᶜq_liqʲ,
                ᶜq_iceʲ,
            )
        end
    end
    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_explicit_closures!(Y, p, t)

Update the precomputed quantities of the `PrognosticEDMFX` explicit closures.
Returns `nothing`.

Mutates in `p.precomputed`, per mass-flux subdomain `j`: the entrainment velocity
scale `ᶜentr_vel_scaleʲs` [1/m], which multiplies a velocity to give a rate, the
non-velocity entrainment rate `ᶜentr_nonvel_rateʲs` [1/s], the turbulent
entrainment rate `ᶜturb_entrʲs` [1/s], the signed area-bounding
entrainment/detrainment rate
`ᶜarea_bounding_entr_detrʲs` [1/s], and the relative density excess `ᶜρ_diffʲs`
[-]. It also writes the squared norm of the vertical shear of the horizontal
wind `ᶜstrain_rate_norm` [1/s²], set by `set_strain_rate_norm!`, the
surface TKE flux `ρtke_flux`, and (via `set_edmfx_surface_conditions!`) the
per-updraft surface payload at level 1.

The total entrainment rate is not stored: it is assembled at each tendency call
site by `compute_entrainment` from `ᶜentr_vel_scaleʲs` and the then-current
updraft velocity. The environment velocity is passed to the coefficient model as
`w⁰ = 0`; see the comment at the call site for why.

Reads `ᶜp`, `ᶠu³`, the environment state (`ᶜT⁰`, `ᶜq_tot_nonneg⁰`, `ᶜq_liq⁰`,
`ᶜq_ice⁰`), the draft state (`ᶜuʲs`, `ᶜTʲs`, `ᶜq_*ʲs`, `ᶜρʲs`), and `ustar` from
`p.precomputed.sfc_conditions`, so it must run after
`set_prognostic_edmf_precomputed_quantities_draft!` and
`..._environment!`. Uses `p.scratch.ᶠtemp_C123`.
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_explicit_closures!(
    Y,
    p,
    t,
)

    (; turbconv_model) = p.atmos

    (; params) = p
    (; ᶜgradᵥ_ᶠΦ) = p.core
    thermo_params = CAP.thermodynamics_params(params)
    turbconv_params = CAP.turbconv_params(params)

    FT = eltype(params)
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜp) = p.precomputed
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; ρtke_flux) = p.precomputed
    (;
        ᶜuʲs,
        ᶜTʲs,
        ᶜq_tot_nonnegʲs,
        ᶜq_liqʲs,
        ᶜq_iceʲs,
        ᶜρʲs,
        ᶜentr_vel_scaleʲs,
        ᶜentr_nonvel_rateʲs,
        ᶜarea_bounding_entr_detrʲs,
        ᶜturb_entrʲs,
        ᶜρ_diffʲs,
    ) = p.precomputed
    (; ustar) = p.precomputed.sfc_conditions

    ᶜz = Fields.coordinate_field(Y.c).z
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))

    entr_model = p.atmos.edmfx_model.entr_model
    zero_FT = FT(0) # (a scalar, not the type: loop bodies must capture isbits values only)
    for j in 1:n
        # Compute the entrainment velocity scale, non-velocity entrainment rate,
        # and the signed area-bounding rate, in one fused column loop per draft
        # (the draft area and buoyancy are loop temporaries shared by the
        # closures).
        # The environment velocity is passed as w⁰ = 0 to the coefficient model;
        # using the true (ᶜwʲ - ᶜw⁰) difference would introduce residual forcing
        # when ᶜwʲ ≈ 0, which can spuriously grow the area fraction and destabilize
        # otherwise trivial updrafts. The environment buoyancy is passed as b⁰ = 0.
        # The total entrainment rate is then assembled
        # at every tendency call site (`edmfx_entr_detr_tendency!` and the
        # implicit ρa solve) via `compute_entrainment` using the
        # (then-updated) updraft velocity |wʲ|, together with the
        # non-velocity-proportional rate computed just below.
        DataLayouts.foreach_column(
            ᶜentr_vel_scaleʲs.:($j), ᶜentr_nonvel_rateʲs.:($j), ᶜturb_entrʲs.:($j),
            ᶜarea_bounding_entr_detrʲs.:($j), ᶜρ_diffʲs.:($j),
            Y.c, Y.c.sgsʲs.:($j), ᶜz, z_sfc, ᶜp, ᶜρʲs.:($j), ᶜuʲs.:($j), ᶜlg,
            ᶜTʲs.:($j), ᶜq_tot_nonnegʲs.:($j), ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j),
            ᶜgradᵥ_ᶠΦ, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜtke,
        ) do ᶜentr_vel_scaleʲ, ᶜentr_nonvel_rateʲ, ᶜturb_entrʲ,
        ᶜarea_bounding_entr_detrʲ, ᶜρ_diffʲ,
        Yc, ᶜsgsʲ, ᶜz, z_sfc, ᶜp, ᶜρʲ, ᶜuʲ, ᶜlg, ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ,
        ᶜq_iceʲ, ᶜgradᵥ_ᶠΦ, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜtke
            ᶜaʲ = @. draft_area(ᶜsgsʲ.ρa, ᶜρʲ)
            ᶜwʲ = @. get_physical_w(ᶜuʲ, ᶜlg)
            ᶜbuoyʲ = @. vertical_buoyancy_acceleration(Yc.ρ, ᶜρʲ, ᶜgradᵥ_ᶠΦ, ᶜlg)
            @. ᶜentr_vel_scaleʲ = entrainment_velocity_scale(
                thermo_params,
                turbconv_params,
                ᶜz,
                z_sfc,
                ᶜp,
                Yc.ρ,
                ᶜaʲ,
                ᶜwʲ,
                TD.relative_humidity(
                    thermo_params,
                    ᶜTʲ,
                    ᶜp,
                    ᶜq_tot_nonnegʲ,
                    ᶜq_liqʲ,
                    ᶜq_iceʲ,
                ),
                ᶜbuoyʲ,
                zero_FT,
                TD.relative_humidity(
                    thermo_params,
                    ᶜT⁰,
                    ᶜp,
                    ᶜq_tot_nonneg⁰,
                    ᶜq_liq⁰,
                    ᶜq_ice⁰,
                ),
                zero_FT,
                max(ᶜtke, 0),
                entr_model,
            )
            @. ᶜentr_nonvel_rateʲ = entrainment_nonvel_rate(
                turbconv_params,
                ᶜaʲ,
                ᶜwʲ,
                ᶜbuoyʲ,
                zero_FT,
                zero_FT,
                entr_model,
            )
            @. ᶜturb_entrʲ = turbulent_entrainment(turbconv_params, ᶜaʲ)

            @. ᶜarea_bounding_entr_detrʲ = area_bounding_entr_detr(turbconv_params, ᶜaʲ)

            @. ᶜρ_diffʲ = (ᶜρʲ - Yc.ρ) / ᶜρʲ
            return nothing
        end
    end

    # Surface BC payload per updraft at level 1 (capped mass flux +
    # buoyant-air mse/q_tot values).
    set_edmfx_surface_conditions!(Y, p)

    set_strain_rate_norm!(Y, p)

    ρtke_flux_values = Fields.field_values(ρtke_flux)
    ρ_sfc_values = Fields.field_values(Fields.level(Y.c.ρ, 1)) # TODO: replace by surface value
    ustar_values = Fields.field_values(ustar)
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), half),
    )
    @. ρtke_flux_values = surface_flux_tke(
        turbconv_params,
        ρ_sfc_values,
        ustar_values,
        sfc_local_geometry_values,
    )

    return nothing
end
