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
    env_thermo_state(
        microphysics_model, c, h_tot, K, Φ, p, turbconv_model, thermo_params, T_min_sgs,
    )

Return the thermodynamic state of the EDMFX environment at one point,
`(; T, q_tot_nonneg, q_liq, q_ice)`, from the grid-mean state `c` (a point of
`Y.c`, including the drafts `c.sgsʲs`), the grid-mean total specific enthalpy
`h_tot` and kinetic energy `K`, and the geopotential `Φ`. The environment values
are the grid mean minus the drafts (`specific_env_value`, `specific_env_mse`).

  - `NonEquilibriumMicrophysics1M`/`2M`: the condensate is the environment cloud
    plus precipitating species, `q_tot` is clamped to at least the condensate, and
    `T` follows from the environment enthalpy, floored at `T_min_sgs`; `p` is unused.
  - `EquilibriumMicrophysics0M`: `q_tot` is clamped to be non-negative, and `T` and
    the phase partition come from a pressure-enthalpy saturation adjustment at the
    grid-mean pressure `p`.
"""
@inline function env_thermo_state(
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
    c, h_tot, K, Φ, p, turbconv_model, thermo_params, T_min_sgs,
)
    ρa⁰_val = ρa⁰(c.ρ, c.sgsʲs, turbconv_model)
    χ⁰(name) = specific_env_value(
        name, MatrixFields.get_field(c, get_ρχ_name(name)), c.ρ, c.sgsʲs, ρa⁰_val,
        turbconv_model,
    )
    q_liq = max(0, χ⁰(@name(q_lcl)) + χ⁰(@name(q_rai)))
    q_ice = max(0, χ⁰(@name(q_icl)) + χ⁰(@name(q_sno)))
    # Clamp q_tot ≥ q_cond to ensure non-negative vapor (q_vap = q_tot - q_cond)
    q_tot_nonneg = max(q_liq + q_ice, χ⁰(@name(q_tot)))
    h = specific_env_mse(c.ρ, c.sgsʲs, h_tot, K, ρa⁰_val, turbconv_model) - Φ
    T = max(
        T_min_sgs,
        TD.air_temperature(thermo_params, TD.ph(), h, q_tot_nonneg, q_liq, q_ice),
    )
    return (; T, q_tot_nonneg, q_liq, q_ice)
end
@inline function env_thermo_state(
    ::EquilibriumMicrophysics0M,
    c, h_tot, K, Φ, p, turbconv_model, thermo_params, T_min_sgs,
)
    ρa⁰_val = ρa⁰(c.ρ, c.sgsʲs, turbconv_model)
    q_tot = specific_env_value(
        @name(q_tot), c.ρq_tot, c.ρ, c.sgsʲs, ρa⁰_val, turbconv_model,
    )
    q_tot_nonneg = max(0, q_tot)
    h = specific_env_mse(c.ρ, c.sgsʲs, h_tot, K, ρa⁰_val, turbconv_model) - Φ
    (; T, q_liq, q_ice) =
        saturation_adjustment_tuple(thermo_params, TD.ph(), p, h, q_tot_nonneg)
    return (; T, q_tot_nonneg, q_liq, q_ice)
end

@inline thermo_T(state) = state.T
@inline thermo_q_tot_nonneg(state) = state.q_tot_nonneg
@inline thermo_q_liq(state) = state.q_liq
@inline thermo_q_ice(state) = state.q_ice

@inline thermo_state(T, q_tot_nonneg, q_liq, q_ice) = (; T, q_tot_nonneg, q_liq, q_ice)

"""
    ᶜenv_thermo_state(Y, p)

Return a lazy center field of the EDMFX environment thermodynamic state,
`(; T, q_tot_nonneg, q_liq, q_ice)` at each point, from the precomputed
environment fields (`ᶜenv_thermo_fields`). A column loop that reads several
components copies it into registers once (`register_copy`). Requires
`PrognosticEDMFX`.
"""
function ᶜenv_thermo_state(Y, p)
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = ᶜenv_thermo_fields(Y, p)
    return @. lazy(thermo_state(ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰))
end

"""
    ᶜenv_thermo_fields(Y, p)

Return the precomputed EDMFX environment temperature and specific humidities,
`(; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)`, which `set_environment_column!` sets
from `Y` and the grid-mean `ᶜh_tot`, `ᶜK`, and `ᶜp` (`env_thermo_state`).
Requires `PrognosticEDMFX`.
"""
function ᶜenv_thermo_fields(Y, p)
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    return (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
end

"""
    set_environment_column!(
        environment, Yc, Yf, ᶠuₕ³, ᶜΦ, ᶜK, ᶜh_tot, ᶜp,
        microphysics_model, turbconv_model, thermo_params, T_min_sgs,
    )

Column body of the EDMFX environment quantities, which writes into the column
fields `environment = (ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)`:
the environment velocity, contravariant vertical velocity, and kinetic energy,
from its vertical velocity (a loop temporary) and `ᶠuₕ³`, and the environment
thermodynamic state (`env_thermo_state`), from the grid-mean `ᶜh_tot`, `ᶜK`, and
`ᶜp`. Runs inside fused column loops, so it is inlined.
"""
@inline function set_environment_column!(
    environment, Yc, Yf, ᶠuₕ³, ᶜΦ, ᶜK, ᶜh_tot, ᶜp,
    microphysics_model, turbconv_model, thermo_params, T_min_sgs,
)
    (ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = environment
    ᶠu₃⁰ = Base.materialize(ᶠsgs_u₃(u₃⁰, Yc, Yf, turbconv_model))
    set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠu₃⁰, Yc.uₕ, ᶠuₕ³)
    ᶜstate⁰ = @. env_thermo_state(
        microphysics_model, Yc, ᶜh_tot, ᶜK, ᶜΦ, ᶜp, turbconv_model, thermo_params,
        T_min_sgs,
    )
    @. ᶜT⁰ = thermo_T(ᶜstate⁰)
    @. ᶜq_tot_nonneg⁰ = thermo_q_tot_nonneg(ᶜstate⁰)
    @. ᶜq_liq⁰ = thermo_q_liq(ᶜstate⁰)
    @. ᶜq_ice⁰ = thermo_q_ice(ᶜstate⁰)
    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_environment!(Y, p, ᶠuₕ³)

Update the EDMFX environment velocities and kinetic energy (`ᶜu⁰`, `ᶠu³⁰`, `ᶜK⁰`)
and thermodynamic state (`ᶜT⁰`, `ᶜq_tot_nonneg⁰`, `ᶜq_liq⁰`, `ᶜq_ice⁰`) in one
column loop (`set_environment_column!`). Returns `nothing`.

# Arguments

  - `ᶠuₕ³`: Third contravariant component of the horizontal velocity, interpolated
    to faces [1/s].
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_environment!(
    Y,
    p,
    ᶠuₕ³,
)
    pc = p.precomputed
    (; microphysics_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    T_min_sgs = CAP.T_min_sgs(p.params)
    environment =
        (pc.ᶜu⁰, pc.ᶠu³⁰, pc.ᶜK⁰, pc.ᶜT⁰, pc.ᶜq_tot_nonneg⁰, pc.ᶜq_liq⁰, pc.ᶜq_ice⁰)
    DataLayouts.foreach_column(
        environment, Y.c, Y.f, ᶠuₕ³, p.core.ᶜΦ, pc.ᶜK, pc.ᶜh_tot, pc.ᶜp,
    ) do environment, Yc, Yf, ᶠuₕ³, ᶜΦ, ᶜK, ᶜh_tot, ᶜp
        set_environment_column!(
            environment, Yc, Yf, ᶠuₕ³, ᶜΦ, ᶜK, ᶜh_tot, ᶜp,
            microphysics_model, turbconv_model, thermo_params, T_min_sgs,
        )
    end
    return nothing
end

"""
    draft_thermo_state(microphysics_model, sgsʲ, Φ, p, thermo_params, T_min_sgs)

Return the thermodynamic state of an EDMFX draft at one point,
`(; T, q_tot_nonneg, q_liq, q_ice, ρ)`, from its prognostic variables `sgsʲ` (a
point of `Y.c.sgsʲs.:(j)`), the geopotential `Φ`, and the grid-mean pressure `p`,
at which the draft density is diagnosed.

  - `NonEquilibriumMicrophysics1M`/`2M`: the condensate is the draft cloud plus
    precipitating species, `q_tot` is clamped to at least the condensate, and `T`
    follows from the enthalpy `mse - Φ`, floored at `T_min_sgs`.
  - `EquilibriumMicrophysics0M`: `q_tot` is clamped to be non-negative, and `T` and
    the phase partition come from a pressure-enthalpy saturation adjustment.
"""
@inline function draft_thermo_state(
    ::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
    sgsʲ, Φ, p, thermo_params, T_min_sgs,
)
    q_liq = max(0, sgsʲ.q_lcl + sgsʲ.q_rai)
    q_ice = max(0, sgsʲ.q_icl + sgsʲ.q_sno)
    # Clamp q_tot ≥ q_cond to keep q_vap = q_tot - q_cond non-negative.
    q_tot_nonneg = max(q_liq + q_ice, sgsʲ.q_tot)
    T = max(
        T_min_sgs,
        TD.air_temperature(
            thermo_params, TD.ph(), sgsʲ.mse - Φ, q_tot_nonneg, q_liq, q_ice,
        ),
    )
    ρ = TD.air_density(thermo_params, T, p, q_tot_nonneg, q_liq, q_ice)
    return (; T, q_tot_nonneg, q_liq, q_ice, ρ)
end
@inline function draft_thermo_state(
    ::EquilibriumMicrophysics0M, sgsʲ, Φ, p, thermo_params, T_min_sgs,
)
    q_tot_nonneg = max(0, sgsʲ.q_tot)
    (; T, q_liq, q_ice) = saturation_adjustment_tuple(
        thermo_params, TD.ph(), p, sgsʲ.mse - Φ, q_tot_nonneg,
    )
    ρ = TD.air_density(thermo_params, T, p, q_tot_nonneg, q_liq, q_ice)
    return (; T, q_tot_nonneg, q_liq, q_ice, ρ)
end

@inline thermo_ρ(state) = state.ρ

"""
    draft_density(microphysics_model, sgsʲ, Φ, p, thermo_params, T_min_sgs)

Density of an EDMFX draft at one point, at the grid-mean pressure `p`
(`draft_thermo_state`).
"""
@inline draft_density(microphysics_model, sgsʲ, Φ, p, thermo_params, T_min_sgs) =
    thermo_ρ(
        draft_thermo_state(microphysics_model, sgsʲ, Φ, p, thermo_params, T_min_sgs),
    )

"""
    set_draft_temperature_and_density!(
        ᶜTʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ, ᶜp, microphysics_model, thermo_params, T_min_sgs,
    )

Column body that writes the temperature and density of an EDMFX draft
(`draft_thermo_state`), evaluated once per point.
"""
@inline function set_draft_temperature_and_density!(
    ᶜTʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ, ᶜp, microphysics_model, thermo_params, T_min_sgs,
)
    ᶜstateʲ = @. draft_thermo_state(
        microphysics_model, ᶜsgsʲ, ᶜΦ, ᶜp, thermo_params, T_min_sgs,
    )
    @. ᶜTʲ = thermo_T(ᶜstateʲ)
    @. ᶜρʲ = thermo_ρ(ᶜstateʲ)
    return nothing
end

"""
    ᶜdraft_thermo_state(Y, p, j)

Return a lazy center field of the thermodynamic state of EDMFX draft `j`
(`draft_thermo_state`), from `Y.c.sgsʲs.:(j)` and the precomputed grid-mean `ᶜp`.
A column loop that reads several components copies it into registers once
(`register_copy`).
"""
function ᶜdraft_thermo_state(Y, p, j)
    f = DraftThermoState(
        p.atmos.microphysics_model,
        CAP.thermodynamics_params(p.params),
        CAP.T_min_sgs(p.params),
    )
    ᶜsgsʲ = Y.c.sgsʲs.:($j)
    return @. lazy(f(ᶜsgsʲ, p.core.ᶜΦ, p.precomputed.ᶜp))
end

"""
    DraftThermoState(microphysics_model, thermo_params, T_min_sgs)

`draft_thermo_state` as a function of the draft prognostic variables, the
geopotential, and the grid-mean pressure, with the microphysics model and the
parameters stored in the function, so that the arguments of its lazy broadcast
(`ᶜdraft_thermo_state`) are all fields.
"""
struct DraftThermoState{M, P, FT}
    microphysics_model::M
    thermo_params::P
    T_min_sgs::FT
end
@inline (f::DraftThermoState)(sgsʲ, Φ, p) =
    draft_thermo_state(f.microphysics_model, sgsʲ, Φ, p, f.thermo_params, f.T_min_sgs)

"""
    ᶜdraft_thermo_fields(Y, p, j)

Return the temperature and specific humidities of EDMFX draft `j`,
`(; ᶜTʲ, ᶜq_tot_nonnegʲ, ᶜq_liqʲ, ᶜq_iceʲ)`: the cached `ᶜTʲs`, and lazy center
fields of the humidities, evaluated pointwise (`ᶜdraft_thermo_state`). The draft
temperature and density stay cached (`ᶜTʲs`, `ᶜρʲs`): their readers run in the
heaviest kernels, and `initialize_implicit_stage_problem!` keeps them at the Picard
iterate while it restores the draft scalars.
"""
function ᶜdraft_thermo_fields(Y, p, j)
    ᶜstateʲ = ᶜdraft_thermo_state(Y, p, j)
    return (;
        ᶜTʲ = p.precomputed.ᶜTʲs.:($j),
        ᶜq_tot_nonnegʲ = (@. lazy(thermo_q_tot_nonneg(ᶜstateʲ))),
        ᶜq_liqʲ = (@. lazy(thermo_q_liq(ᶜstateʲ))),
        ᶜq_iceʲ = (@. lazy(thermo_q_ice(ᶜstateʲ))),
    )
end

"""
    ᶜdraft_thermo_tuples(Y, p)

Return `(; ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs)`, each a tuple over the
EDMFX drafts of the lazy fields of `ᶜdraft_thermo_fields`, so that `ᶜTʲs.:(j)`
selects draft `j` as for a field of tuples.
"""
function ᶜdraft_thermo_tuples(Y, p)
    n_val = Val(n_mass_flux_subdomains(p.atmos.turbconv_model))
    drafts = ntuple(j -> ᶜdraft_thermo_fields(Y, p, j), n_val)
    return (;
        ᶜTʲs = map(draft -> draft.ᶜTʲ, drafts),
        ᶜq_tot_nonnegʲs = map(draft -> draft.ᶜq_tot_nonnegʲ, drafts),
        ᶜq_liqʲs = map(draft -> draft.ᶜq_liqʲ, drafts),
        ᶜq_iceʲs = map(draft -> draft.ᶜq_iceʲ, drafts),
    )
end

"""
    ᶜdraft_velocities(Y, p)

Return a tuple over the EDMFX drafts of lazy center fields of their velocities,
the horizontal velocity plus the interpolated draft vertical velocity, so that
`ᶜuʲs.:(j)` selects draft `j` as for a field of tuples. The horizontal operators of
`horizontal_advection_tendency!` and `prep_hyperdiffusion_tendency!` read them
from `p.scratch.ᶜtemp_C123s` (`set_draft_velocities!`).
"""
function ᶜdraft_velocities(Y, p)
    n_val = Val(n_mass_flux_subdomains(p.atmos.turbconv_model))
    return ntuple(n_val) do j
        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
        @. lazy(C123(Y.c.uₕ) + ᶜinterp(C123(ᶠu₃ʲ)))
    end
end

"""
    set_draft_velocities!(ᶜuʲs, Y, p)

Evaluate the draft velocities (`ᶜdraft_velocities`) into the field of tuples
`ᶜuʲs`, for readers that apply horizontal operators to them. Returns `ᶜuʲs`.
"""
function set_draft_velocities!(ᶜuʲs, Y, p)
    MatrixFields.unrolled_foreach(
        StaticOneTo(n_mass_flux_subdomains(p.atmos.turbconv_model)),
    ) do j
        @. ᶜuʲs.:($$j) = C123(Y.c.uₕ) + ᶜinterp(C123(Y.f.sgsʲs.:($$j).u₃))
    end
    return ᶜuʲs
end

"""
    ᶜenv_velocity(Y, p)

Return the precomputed EDMFX environment velocity `ᶜu⁰`, the horizontal velocity
plus the interpolated environment vertical velocity (`set_environment_column!`).
Requires `PrognosticEDMFX`.
"""
ᶜenv_velocity(Y, p) = p.precomputed.ᶜu⁰

"""
    set_prognostic_edmf_nonequilibrium_quantities!(Y, p, ᶠuₕ³)

Update, for `PrognosticEDMFX` with `NonEquilibriumMicrophysics1M`/`2M`, the
grid-mean velocity and thermodynamic quantities, the draft densities `ᶜρʲs`, and
the environment quantities in one column loop. Writes the same `p.precomputed`
fields, with the same values, as the grid-mean loop of
`set_implicit_precomputed_quantities_core!`,
`set_prognostic_edmf_precomputed_quantities_draft!`, and
`set_prognostic_edmf_precomputed_quantities_environment!`. Returns `nothing`.
"""
NVTX.@annotate function set_prognostic_edmf_nonequilibrium_quantities!(Y, p, ᶠuₕ³)
    (; turbconv_model) = p.atmos
    n_val = Val(n_mass_flux_subdomains(turbconv_model))
    thermo_params = CAP.thermodynamics_params(p.params)
    T_min_sgs = CAP.T_min_sgs(p.params)
    pc = p.precomputed
    # (outputs in containers: a loop over more than 31 arguments would not infer)
    grid_mean = (
        pc.ᶜu, pc.ᶠu³, pc.ᶜK, pc.ᶜT, pc.ᶜq_tot_nonneg, pc.ᶜq_liq, pc.ᶜq_ice,
        pc.ᶜh_tot, pc.ᶜp,
    )
    drafts = ntuple(
        j -> (pc.ᶜTʲs.:($j), pc.ᶜρʲs.:($j), Y.c.sgsʲs.:($j)),
        n_val,
    )
    environment =
        (pc.ᶜu⁰, pc.ᶠu³⁰, pc.ᶜK⁰, pc.ᶜT⁰, pc.ᶜq_tot_nonneg⁰, pc.ᶜq_liq⁰, pc.ᶜq_ice⁰)
    mm = p.atmos.microphysics_model
    DataLayouts.foreach_column(
        grid_mean, drafts, environment, Y.c, Y.f, ᶠuₕ³, p.core.ᶜΦ,
    ) do grid_mean, drafts, environment, Yc, Yf, ᶠuₕ³, ᶜΦ
        set_grid_mean_column!(
            grid_mean..., Yc, Yf.u₃, ᶠuₕ³, ᶜΦ, mm, thermo_params, T_min_sgs,
        )
        (_, _, ᶜK, _, _, _, _, ᶜh_tot, ᶜp) = grid_mean
        for (ᶜTʲ, ᶜρʲ, ᶜsgsʲ) in drafts
            set_draft_temperature_and_density!(
                ᶜTʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ, ᶜp, mm, thermo_params, T_min_sgs,
            )
        end
        set_environment_column!(
            environment, Yc, Yf, ᶠuₕ³, ᶜΦ, ᶜK, ᶜh_tot, ᶜp,
            mm, turbconv_model, thermo_params, T_min_sgs,
        )
    end
    return nothing
end

"""
    set_prognostic_edmf_precomputed_quantities_draft!(Y, p)

Update the temperature `ᶜTʲs` and density `ᶜρʲs` of each SGS draft (updraft) for
`PrognosticEDMFX` (`set_draft_temperatures_and_densities!`). The other draft
quantities are not stored: their readers evaluate them from `Y`
(`ᶜdraft_thermo_fields`, `ᶜdraft_velocities`).
Returns `nothing`.
"""
NVTX.@annotate function set_prognostic_edmf_precomputed_quantities_draft!(Y, p)
    set_draft_temperatures_and_densities!(Y, p)
    return nothing
end

"""
    set_draft_temperatures_and_densities!(Y, p)

Set the temperature `ᶜTʲs` and density `ᶜρʲs` of each EDMFX draft from its
prognostic variables and the grid-mean pressure `ᶜp`
(`set_draft_temperature_and_density!`). Returns `nothing`.
"""
function set_draft_temperatures_and_densities!(Y, p)
    (; ᶜTʲs, ᶜρʲs, ᶜp) = p.precomputed
    mm = p.atmos.microphysics_model
    thermo_params = CAP.thermodynamics_params(p.params)
    T_min_sgs = CAP.T_min_sgs(p.params)
    drafts = unrolled_map(StaticOneTo(n_mass_flux_subdomains(p.atmos.turbconv_model))) do j
        @inline
        (ᶜTʲs.:($j), ᶜρʲs.:($j), Y.c.sgsʲs.:($j))
    end
    DataLayouts.foreach_column(drafts, p.core.ᶜΦ, ᶜp) do drafts, ᶜΦ, ᶜp
        for (ᶜTʲ, ᶜρʲ, ᶜsgsʲ) in drafts
            set_draft_temperature_and_density!(
                ᶜTʲ, ᶜρʲ, ᶜsgsʲ, ᶜΦ, ᶜp, mm, thermo_params, T_min_sgs,
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

Reads `ᶜp`, `ᶠu³`, the environment and draft states (`ᶜenv_thermo_fields`,
`ᶜdraft_thermo_tuples`, `ᶜdraft_velocities`), and `ustar` from
`p.precomputed.sfc_conditions`, so it must run after
`set_prognostic_edmf_precomputed_quantities_draft!`.
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
    (; ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = ᶜenv_thermo_fields(Y, p)
    (; ρtke_flux) = p.precomputed
    (; ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = ᶜdraft_thermo_tuples(Y, p)
    ᶜuʲs = ᶜdraft_velocities(Y, p)
    (;
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
