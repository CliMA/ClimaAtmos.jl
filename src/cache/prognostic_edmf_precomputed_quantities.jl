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
    (; ᶜΦ) = p.core
    (; ᶜp, ᶜK, ᶜh_tot) = p.precomputed
    (; ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) =
        p.precomputed

    T_min_sgs = CAP.T_min_sgs(p.params)

    if p.atmos.microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        # One fused column loop: environment velocities and kinetic energy,
        # then the environment thermodynamic state from primitives.
        DataLayouts.foreach_column(
            ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
            Y.c, Y.f, ᶠuₕ³, ᶜΦ, ᶜK, ᶜh_tot,
        ) do ᶠu₃⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰,
        Yc, Yf, ᶠuₕ³, ᶜΦ, ᶜK, ᶜh_tot
            set_sgs_ᶠu₃!(u₃⁰, ᶠu₃⁰, Yc, Yf, turbconv_model)
            set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Yc.uₕ, ᶠuₕ³)
            ᶜρa⁰ = @inline @. lazy(ρa⁰(Yc.ρ, Yc.sgsʲs, turbconv_model))
            @inline function ᶜχ⁰(name) # the environment value of a specific tracer
                ᶜρχ = MatrixFields.get_field(Yc, get_ρχ_name(name))
                return @inline @. lazy(
                    specific_env_value(name, ᶜρχ, Yc.ρ, Yc.sgsʲs, ᶜρa⁰, turbconv_model),
                )
            end
            @. ᶜq_liq⁰ = max(0, $(ᶜχ⁰(@name(q_lcl))) + $(ᶜχ⁰(@name(q_rai))))
            @. ᶜq_ice⁰ = max(0, $(ᶜχ⁰(@name(q_icl))) + $(ᶜχ⁰(@name(q_sno))))
            # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
            @. ᶜq_tot_nonneg⁰ = max(ᶜq_liq⁰ + ᶜq_ice⁰, $(ᶜχ⁰(@name(q_tot))))
            ᶜh⁰ = @inline @. lazy(
                specific_env_mse(Yc.ρ, Yc.sgsʲs, ᶜh_tot, ᶜK, ᶜρa⁰, turbconv_model) - ᶜΦ,
            )
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
        ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
        ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
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
`ᶜKʲs`, `ᶠKᵥʲs`, `ᶜTʲs`, `ᶜq_tot_nonnegʲs`, `ᶜq_liqʲs`, `ᶜq_iceʲs`, and `ᶜρʲs`.

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
    (; microphysics_model, turbconv_model) = p.atmos
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
    drafts = unrolled_map(StaticOneTo(n)) do j
        @inline
        (
            ᶜuʲs.:($j), ᶠu³ʲs.:($j), ᶜKʲs.:($j), ᶠKᵥʲs.:($j), ᶜTʲs.:($j),
            ᶜq_tot_nonnegʲs.:($j), ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j), ᶜρʲs.:($j),
            Y.c.sgsʲs.:($j), Y.f.sgsʲs.:($j).u₃, Y.c.uₕ, ᶠuₕ³, ᶜΦ, ᶜp,
        )
    end
    # One fused column loop for every draft: velocities and kinetic energies,
    # then the draft thermodynamic state.
    DataLayouts.foreach_column(drafts) do drafts
        for (
                ᶜuʲ,
                ᶠu³ʲ,
                ᶜKʲ,
                ᶠKᵥʲ,
                ᶜTʲ,
                ᶜq_tot_nonnegʲ,
                ᶜq_liqʲ,
                ᶜq_iceʲ,
                ᶜρʲ,
                ᶜsgsʲ,
                ᶠu₃ʲ,
                ᶜuₕ,
                ᶠuₕ³,
                ᶜΦ,
                ᶜp,
            ) in drafts

            (ᶜsgsʲ, ᶠu₃ʲ, ᶜuₕ, ᶠuₕ³, ᶜΦ, ᶜp) =
                register_copy((ᶜsgsʲ, ᶠu₃ʲ, ᶜuₕ, ᶠuₕ³, ᶜΦ, ᶜp))
            set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠu₃ʲ, ᶜuₕ, ᶠuₕ³)
            @. ᶠKᵥʲ = (adjoint(CT3(ᶠu₃ʲ)) * ᶠu₃ʲ) / 2
            set_draft_thermo_column!(
                ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ, ᶜp,
                microphysics_model, thermo_params, T_min_sgs,
            )
        end
    end
    return nothing
end

"""
    set_draft_thermo_column!(
        ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ, ᶜp,
        microphysics_model, thermo_params, T_min_sgs,
    )

Column body of the draft thermodynamic state. For `NonEquilibriumMicrophysics1M`/`2M`,
the condensate comes from the prognostic draft tracers, `q_tot` is clamped to at least
the condensate, and `T` follows from the enthalpy `mse - Φ`, floored at `T_min_sgs`;
otherwise, `T` and the phase partition come from a pressure-enthalpy saturation
adjustment. The density is diagnosed at the grid-mean pressure.
"""
@inline function set_draft_thermo_column!(
    ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ, ᶜp,
    microphysics_model, thermo_params, T_min_sgs,
)
    if microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
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
    else
        @. ᶜq_tot_nonnegʲ = max(0, ᶜsgsʲ.q_tot)
        ᶜsa_resultʲ = @. saturation_adjustment_tuple(
            thermo_params,
            TD.ph(),
            ᶜp,
            ᶜsgsʲ.mse - ᶜΦ,
            ᶜq_tot_nonnegʲ,
        )
        @. ᶜTʲ = ᶜsa_resultʲ.T
        @. ᶜq_liqʲ = ᶜsa_resultʲ.q_liq
        @. ᶜq_iceʲ = ᶜsa_resultʲ.q_ice
    end
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
partition come from a pressure-enthalpy saturation adjustment. `ᶜρʲ` is
diagnosed at the shared `ᶜp`.

Called from the Picard loop of `initialize_implicit_stage_problem!` after each
tracer sweep, and from `solve_sgs_updraft_tracers_implicit_stage_analytic!`
after its transport sweep. `set_prognostic_edmf_precomputed_quantities_draft!`
computes the same thermodynamics (`set_draft_thermo_column!`) in its fused
column loop.
"""
function set_prognostic_edmf_precomputed_quantities_draft_thermo!(Y, p)
    (; microphysics_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core
    (; ᶜp, ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs, ᶜρʲs) = p.precomputed
    T_min_sgs = CAP.T_min_sgs(p.params)
    drafts = unrolled_map(StaticOneTo(n_mass_flux_subdomains(turbconv_model))) do j
        @inline
        (
            ᶜTʲs.:($j), ᶜq_tot_nonnegʲs.:($j), ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j),
            ᶜρʲs.:($j), Y.c.sgsʲs.:($j),
        )
    end
    DataLayouts.foreach_column(drafts, ᶜΦ, ᶜp) do drafts, ᶜΦ, ᶜp
        for (ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ, ᶜsgsʲ) in drafts
            set_draft_thermo_column!(
                ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ, ᶜp,
                microphysics_model, thermo_params, T_min_sgs,
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
wind `ᶜstrain_rate_norm` [1/s²], set by `set_strain_rate_norm_column!`, the
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
`..._environment!`.
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

    (; ᶜp, ᶠu³, ᶜstrain_rate_norm) = p.precomputed
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
    zero_FT = FT(0)
    drafts = unrolled_map(StaticOneTo(n)) do j
        @inline
        (
            ᶜentr_vel_scaleʲs.:($j), ᶜentr_nonvel_rateʲs.:($j), ᶜturb_entrʲs.:($j),
            ᶜarea_bounding_entr_detrʲs.:($j), ᶜρ_diffʲs.:($j), Y.c.sgsʲs.:($j),
            ᶜρʲs.:($j), ᶜuʲs.:($j), ᶜTʲs.:($j), ᶜq_tot_nonnegʲs.:($j),
            ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j),
        )
    end
    # The entrainment velocity scale, non-velocity entrainment rate, and signed
    # area-bounding rate of every draft, and the vertical strain-rate norm, in
    # one fused column loop.
    DataLayouts.foreach_column(
        drafts, ᶜstrain_rate_norm, Y.c, ᶜz, z_sfc, ᶜp, ᶠu³, ᶜlg, ᶜgradᵥ_ᶠΦ, ᶜT⁰,
        ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜtke,
    ) do drafts, ᶜstrain_rate_norm, Yc, ᶜz, z_sfc, ᶜp, ᶠu³, ᶜlg, ᶜgradᵥ_ᶠΦ, ᶜT⁰,
    ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜtke
        for (
                ᶜentr_vel_scaleʲ,
                ᶜentr_nonvel_rateʲ,
                ᶜturb_entrʲ,
                ᶜarea_bounding_entr_detrʲ,
                ᶜρ_diffʲ,
                ᶜsgsʲ,
                ᶜρʲ,
                ᶜuʲ,
                ᶜTʲ,
                ᶜq_tot_nonnegʲ,
                ᶜq_liqʲ,
                ᶜq_iceʲ,
            ) in drafts

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
        end
        set_strain_rate_norm_column!(ᶜstrain_rate_norm, Yc.uₕ, ᶠu³)
    end

    # Surface BC payload per updraft at level 1 (capped mass flux +
    # buoyant-air mse/q_tot values).
    set_edmfx_surface_conditions!(Y, p)

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
