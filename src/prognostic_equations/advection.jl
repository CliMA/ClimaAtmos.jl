#####
##### Advection and dynamics tendencies
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

"""
    horizontal_dynamics_tendency!(Yₜ, Y, p, t)

Add the explicit horizontal dynamics tendencies: horizontal advection of grid-mean
and EDMFX-subdomain prognostic variables, plus the horizontal pressure-gradient,
kinetic-energy-gradient, and geopotential-gradient forces on `uₕ`.

Increments (all with a minus sign, i.e., flux divergences and gradients are sinks):

  - `Yₜ.c.ρ`: horizontal mass-flux divergence `∇ₕ⋅(ρu)`.
  - `Yₜ.c.ρe_tot`: horizontal total-enthalpy-flux divergence `∇ₕ⋅(ρu h_tot)`.
  - `Yₜ.c.ρtke`: horizontal TKE-flux divergence, when the turbulence-convection model
    carries prognostic TKE.
  - `Yₜ.c.uₕ`: gradients of kinetic energy and geopotential (relative to a reference
    state) plus the pressure gradient in a split form,
    `cp_d/2 [θᵥ′ ∇Π + ∇(θᵥ′ Π) - Π ∇θᵥ′]`, where `θᵥ′` is the virtual potential
    temperature minus its reference profile and `Π` is the Exner function.
  - `Yₜ.c.sgsʲs.:(j).ρa` and `Yₜ.c.sgsʲs.:(j).mse` for each updraft `j`, when
    `p.atmos.turbconv_model isa PrognosticEDMFX` (advective form for `mse`).

Reads `Y.c` and, from the cache `p`, the precomputed fields `ᶜu`, `ᶜK`, `ᶜp`, `ᶜT`,
`ᶜq_liq`, `ᶜq_ice`, `ᶜq_tot_nonneg`, `ᶜh_tot` (and `ᶜuʲs` for `PrognosticEDMFX`), the
geopotential `ᶜΦ` from `p.core`, and scratch space. Called from
`remaining_tendency!`, i.e., placed in the explicit part of the IMEX splitting.
Returns `nothing`. The continuous-form equations are documented on the "Equations"
page of the docs (`docs/src/equations.md`).
"""
NVTX.@annotate function horizontal_dynamics_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜΦ) = p.core
    (; ᶜu, ᶜK, ᶜp, ᶜT, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = thermo_params.cp_d

    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜuʲs) = p.precomputed
    end

    @. Yₜ.c.ρ -= split_divₕ(Y.c.ρ * ᶜu, 1)
    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).ρa -= split_divₕ(
                Y.c.sgsʲs.:($$j).ρa * ᶜuʲs.:($$j),
                1,
            )
        end
    end

    (; ᶜh_tot) = p.precomputed
    @. Yₜ.c.ρe_tot -= split_divₕ(Y.c.ρ * ᶜu, ᶜh_tot)

    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).mse -=
                split_divₕ(ᶜuʲs.:($$j), Y.c.sgsʲs.:($$j).mse) -
                Y.c.sgsʲs.:($$j).mse * split_divₕ(ᶜuʲs.:($$j), 1)
        end
    end

    if use_prognostic_tke(p.atmos.turbconv_model)
        ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
        @. Yₜ.c.ρtke -= split_divₕ(Y.c.ρ * ᶜu, ᶜtke)
    end

    (; ᶜq_tot_nonneg) = p.precomputed
    ᶜΦ_r = @. lazy(phi_r(thermo_params, ᶜp))
    ᶜθ_v = p.scratch.ᶜtemp_scalar
    @. ᶜθ_v = theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
    ᶜθ_vr = @. lazy(theta_vr(thermo_params, ᶜp))
    ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
    ᶜθ_v_diff = @. lazy(ᶜθ_v - ᶜθ_vr)
    # split form pressure gradient: 0.5 * cp_d * [θv ∇Π + ∇(θv Π) - Π∇θv]
    @. Yₜ.c.uₕ -= C12(
        gradₕ(ᶜK + ᶜΦ - ᶜΦ_r) +
        cp_d *
        (
            ᶜθ_v_diff * gradₕ(ᶜΠ) + gradₕ(ᶜθ_v_diff * ᶜΠ) - ᶜΠ * gradₕ(ᶜθ_v_diff)
        ) / 2,
    )
    # Without the C12(), the right-hand side would be a C1 or C2 in 2D space.
    return nothing
end

"""
    horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)

Add the explicit horizontal advection tendencies for grid-mean tracers and for
EDMFX-subdomain tracers.

Increments (with a minus sign on each flux divergence):

  - Every grid-mean tracer `Yₜ.c.ρχ`: flux-form advection `∇ₕ⋅(ρu χ)` for all
    prognostic tracer variables in `Y.c` (identified by `is_tracer_var`).
  - `Yₜ.c.sgsʲs.:(j).q_tot`, when `p.atmos.turbconv_model isa PrognosticEDMFX`:
    advective-form advection with the updraft velocity `ᶜuʲs`.
  - Every auto-discovered SGS tracer in `Yₜ.c.sgsʲs.:(j)` (microphysics species and
    user-defined passive tracers, from `sgs_tracer_names`), also in advective form.

Reads `Y.c`, the precomputed velocities `ᶜu` (and `ᶜuʲs` for `PrognosticEDMFX`) from
`p`. `t` is unused. Called from `remaining_tendency!` with the limited tendency
vector `Yₜ_lim`, so these tendencies pass through the tracer limiters. Returns
`nothing`.
"""
NVTX.@annotate function horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜu) = p.precomputed

    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜuʲs) = p.precomputed
    end

    for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
        ᶜχ = @. lazy(specific(Y.c.:($$ρχ_name), Y.c.ρ))
        @. Yₜ.c.:($$ρχ_name) -= split_divₕ(Y.c.ρ * ᶜu, ᶜχ)
    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).q_tot -=
                split_divₕ(ᶜuʲs.:($$j), Y.c.sgsʲs.:($$j).q_tot) -
                Y.c.sgsʲs.:($$j).q_tot * split_divₕ(ᶜuʲs.:($$j), 1)
            # Auto-discovered SGS tracers (microphysics species and any
            # user-defined passive tracers)
            for χ_name in sgs_tracer_names(Y)
                ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
                ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:(1), χ_name)
                @. ᶜχʲₜ -=
                    split_divₕ(ᶜuʲs.:($$j), ᶜχʲ) -
                    ᶜχʲ * split_divₕ(ᶜuʲs.:($$j), 1)
            end
        end
    end
    return nothing
end

"""
    ᶜρq_tot_vertical_transport_bc(flow, thermo_params, t, ᶠu³)

Return the `ρq_tot` tendency from the prescribed surface moisture flux of a
`PrescribedFlow` configuration.

Builds a face divergence whose only nonzero contribution is the bottom-boundary
flux `ρu₃qₜ` prescribed by `get_ρu₃qₜ_surface`, so only the cell adjacent to the
surface is affected.

# Arguments

  - `flow`: The prescribed flow model (`PrescribedFlow`), or `nothing` for no effect.
  - `thermo_params`: Thermodynamic parameters, needed to compute surface air density.
  - `t`: Current simulation time.
  - `ᶠu³`: Face contravariant vertical velocity field (used only for its space and
    element type).

# Returns

A lazy broadcast with the surface vertical-transport tendency of `ρq_tot`, or a
`NullBroadcasted()` when `flow` is `nothing`.
"""
ᶜρq_tot_vertical_transport_bc(::Nothing, _, _, _) = NullBroadcasted()
function ᶜρq_tot_vertical_transport_bc(flow::PrescribedFlow, thermo_params, t, ᶠu³)
    ρu₃qₜ_sfc_bc = get_ρu₃qₜ_surface(flow, thermo_params, t)
    ᶜadvdivᵥ = Operators.DivergenceF2C(; bottom = Operators.SetValue(ρu₃qₜ_sfc_bc))
    return @. lazy(-(ᶜadvdivᵥ(zero(ᶠu³))))
end

"""
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)

Add the explicitly treated vertical advection tendencies: vector-invariant momentum
advection (with Coriolis and vorticity terms) and full vertical transport of
tracers and, optionally, TKE.

Increments:

  - Every grid-mean tracer `Yₜ.c.ρχ` except `ρe_tot` and `ρq_tot` (those are advected
    implicitly): flux-form `vertical_transport` with the grid-mean flow `ᶠu³` and the
    `p.atmos.numerics.tracer_upwinding` scheme. When EDMFX SGS mass flux is active,
    difference-form SGS corrections are added on top of this in
    `edmfx_sgs_mass_flux_tendency!`.
  - `Yₜ.c.ρq_tot`: the prescribed-flow surface flux from
    `ᶜρq_tot_vertical_transport_bc`, when moisture is present.
  - `Yₜ.c.uₕ` and `Yₜ.f.u₃`: vector-invariant vertical advection built from the
    vorticities `ᶜω³` and `ᶠω¹²`, the Coriolis fields `ᶜf³` and `ᶠf¹²`, and the
    kinetic-energy gradient `∇ᵥK`; the shallow-atmosphere branch (`ᶠf¹² === nothing`)
    omits the horizontal Coriolis components.
  - `Yₜ.f.sgsʲs.:(j).u₃` for each prognostic updraft: the analogous vector-invariant
    terms with updraft vorticities and kinetic energies.
  - `Yₜ.c.ρtke`: `vertical_transport` with the
    `p.atmos.numerics.edmfx_mse_q_tot_upwinding` scheme, when the
    turbulence-convection model carries prognostic TKE.

Reads `Y`, the core fields `ᶜf³` and `ᶠf¹²`, the precomputed `ᶜu`, `ᶠu³`, `ᶜK` (and
`ᶜuʲs`, `ᶜKʲs`, `ᶠKᵥʲs` for prognostic updrafts), and scratch space. Called from
`remaining_tendency!`. Returns `nothing`.
"""
NVTX.@annotate function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; turbconv_model, prescribed_flow) = p.atmos
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    advect_tke = use_prognostic_tke(turbconv_model)
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜf³, ᶠf¹²) = p.core
    (; ᶜu, ᶠu³, ᶜK) = p.precomputed
    (; edmfx_mse_q_tot_upwinding) = n > 0 || advect_tke ? p.atmos.numerics : all_nothing
    (; ᶜuʲs, ᶜKʲs, ᶠKᵥʲs) = n > 0 ? p.precomputed : all_nothing
    (; tracer_upwinding) = p.atmos.numerics
    thermo_params = CAP.thermodynamics_params(p.params)

    ᶜtke =
        advect_tke ?
        (@. lazy(specific(Y.c.ρtke, Y.c.ρ))) :
        nothing
    ᶜω³ = p.scratch.ᶜtemp_CT3
    ᶠω¹² = p.scratch.ᶠtemp_CT12
    ᶠω¹²ʲs = p.scratch.ᶠtemp_CT12ʲs

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = wcurlₕ(Y.c.uₕ)
    else
        @. ᶜω³ = zero(ᶜω³)
    end

    @. ᶠω¹² = ᶠcurlᵥ(Y.c.uₕ)
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) = ᶠω¹²
    end
    @. ᶠω¹² += CT12(wcurlₕ(Y.f.u₃))
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) += CT12(wcurlₕ(Y.f.sgsʲs.:($$j).u₃))
    end
    # Without the CT12(), the right-hand side would be a CT1 or CT2 in 2D space.

    ᶜρ = Y.c.ρ

    # Full vertical advection of passive tracers (such as liq, rai, etc) with the
    # grid-mean flow. When EDMFX sgs_mass_flux is active, difference-form SGS
    # corrections ρᵏaᵏ(u³ᵏ - u³)(χᵏ - χ) are added on top of this in
    # edmfx_sgs_mass_flux_tendency!.
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        if !(ρχ_name in (@name(ρe_tot), @name(ρq_tot)))
            ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
            vtt = vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, tracer_upwinding)
            @. ᶜρχₜ += vtt
        end
    end
    if !(p.atmos.microphysics_model isa DryModel)
        vtt_bc =
            ᶜρq_tot_vertical_transport_bc(prescribed_flow, thermo_params, t, ᶠu³)
        @. Yₜ.c.ρq_tot += vtt_bc
    end

    if isnothing(ᶠf¹²)
        # shallow atmosphere
        @. Yₜ.c.uₕ -=
            ᶜinterp(ᶠω¹² × (ᶠinterp(Y.c.ρ * ᶜJ) * ᶠu³)) / (Y.c.ρ * ᶜJ) +
            (ᶜf³ + ᶜω³) × CT12(ᶜu)
        @. Yₜ.f.u₃ -= ᶠω¹² × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                ᶠω¹²ʲs.:($$j) × ᶠinterp(CT12(ᶜuʲs.:($$j))) +
                ᶠgradᵥ(ᶜKʲs.:($$j) - ᶜinterp(ᶠKᵥʲs.:($$j)))
        end
    else
        # deep atmosphere
        @. Yₜ.c.uₕ -=
            ᶜinterp((ᶠf¹² + ᶠω¹²) × (ᶠinterp(Y.c.ρ * ᶜJ) * ᶠu³)) /
            (Y.c.ρ * ᶜJ) + (ᶜf³ + ᶜω³) × CT12(ᶜu)
        @. Yₜ.f.u₃ -= (ᶠf¹² + ᶠω¹²) × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                (ᶠf¹² + ᶠω¹²ʲs.:($$j)) × ᶠinterp(CT12(ᶜuʲs.:($$j))) +
                ᶠgradᵥ(ᶜKʲs.:($$j) - ᶜinterp(ᶠKᵥʲs.:($$j)))
        end
    end

    if use_prognostic_tke(turbconv_model) # advect_tke triggers allocations
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜtke, dt, edmfx_mse_q_tot_upwinding)
        @. Yₜ.c.ρtke += vtt
    end
end

"""
    edmfx_sgs_vertical_advection_tendency!(Yₜ, Y, p, t, turbconv_model)
    edmfx_sgs_vertical_advection_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

Add the explicit vertical advection, buoyancy, and sedimentation tendencies for the
EDMFX updraft prognostic variables.

The fallback method is a no-op for turbulence-convection models other than
`PrognosticEDMFX`. For `PrognosticEDMFX`, this increments, for each updraft `j`:

  - `Yₜ.c.sgsʲs.:(j).mse`: the buoyancy source (geopotential gradient times the
    updraft density anomaly `ᶜρ_diffʲs`) plus advective-form vertical advection with
    the updraft velocity, using the `p.atmos.numerics.edmfx_mse_q_tot_upwinding`
    scheme.
  - `Yₜ.c.sgsʲs.:(j).q_tot`: advective-form vertical advection (same scheme), plus
    the total-water contribution of condensate and precipitation sedimentation.
  - Every auto-discovered SGS tracer (from `sgs_tracer_names`): advective-form
    vertical advection with the `p.atmos.numerics.edmfx_tracer_upwinding` scheme.
  - For `NonEquilibriumMicrophysics1M`/`NonEquilibriumMicrophysics2M`, the updraft
    microphysics species (`q_lcl`, `q_icl`, `q_rai`, `q_sno`, and for 2M also `n_lcl`
    and `n_rai`): within-updraft sedimentation with lateral entrainment corrections
    via `updraft_sedimentation`. This path currently supports a single updraft.

Reads `Y`, the precomputed `ᶠu³ʲs`, `ᶜρʲs`, `ᶜρ_diffʲs`, and sedimentation
velocities, the core field `ᶜgradᵥ_ᶠΦ`, and scratch space. `t` is unused. See the
"PROPHET: Overview and Equations" page (`docs/src/prophet.md`) for the continuous
equations. Returns `nothing`.
"""
edmfx_sgs_vertical_advection_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_vertical_advection_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    (; edmfx_mse_q_tot_upwinding, edmfx_tracer_upwinding) = p.atmos.numerics
    (; ᶠu³ʲs, ᶜρʲs, ᶜρ_diffʲs) = p.precomputed
    (; ᶜgradᵥ_ᶠΦ) = p.core
    (; microphysics_model) = p.atmos

    FT = eltype(p.params)
    (one_FT, zero_FT, ϵ_FT) = (FT(1), FT(0), eps(FT)) # scalars for the loop bodies (isbits captures only)
    χ_names = sgs_tracer_names(Y)

    # Sedimenting updraft species with their updraft and grid-mean terminal
    # velocities: (species name, updraft velocity name, grid-mean velocity
    # name). The mass species also change the updraft total water; the number
    # concentrations of 2M microphysics do not.
    # TODO - add precipitation and cloud sedimentation in implicit solver/tendency with if/else
    # TODO - lazify ᶜwₗʲs computation. No need to cache it.
    mass_sedimentation =
        if microphysics_model isa
           Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
            unrolled_filter(
                ((q_name, _, _),) -> MatrixFields.has_field(Y.c.sgsʲs.:(1), q_name),
                (
                    (@name(q_lcl), @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
                    (@name(q_icl), @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
                    (@name(q_rai), @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
                    (@name(q_sno), @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
                ),
            )
        else
            ()
        end
    number_sedimentation = if microphysics_model isa NonEquilibriumMicrophysics2M
        unrolled_filter(
            ((n_name, _, _),) -> MatrixFields.has_field(Y.c.sgsʲs.:(1), n_name),
            (
                (@name(n_lcl), @name(ᶜwₙₗʲs.:(1)), @name(ᶜwₙₗ)),
                (@name(n_rai), @name(ᶜwₙᵣʲs.:(1)), @name(ᶜwₙᵣ)),
            ),
        )
    else
        ()
    end
    sedimentation = (mass_sedimentation..., number_sedimentation...)
    n_sed = length(sedimentation)
    sed_names = unrolled_map(first, sedimentation)
    sed_ρχ_names = unrolled_map(get_ρχ_name, sed_names)
    ᶜwʲs = unrolled_map(
        ((_, wʲ_name, _),) -> MatrixFields.get_field(p.precomputed, wʲ_name),
        sedimentation,
    )
    ᶜw_gss = unrolled_map(
        ((_, _, w_gs_name),) -> MatrixFields.get_field(p.precomputed, w_gs_name),
        sedimentation,
    )

    for j in 1:n
        # TODO - make sedimentation work for multiple updrafts
        if n_sed > 0 && j > 1
            error("Below code doesn't work for multiple updrafts")
        end

        # One fused column loop per updraft: buoyancy, vertical advection of
        # mse, q_tot, and the tracers, and the sedimentation of the microphysics
        # species. The updraft area, the inverse updraft density, the face
        # density, and the per-species fluxes are loop temporaries (registers
        # on GPUs) rather than scratch fields.
        DataLayouts.foreach_column(
            Yₜ.c.sgsʲs.:($j), Y.c.sgsʲs.:($j), Y.f.sgsʲs.:($j).u₃, ᶠu³ʲs.:($j),
            ᶜρʲs.:($j), ᶜρ_diffʲs.:($j), ᶜgradᵥ_ᶠΦ, Y.c, ᶜwʲs..., ᶜw_gss...,
        ) do ᶜsgsʲₜ, ᶜsgsʲ, ᶠu₃ʲ, ᶠu³ʲ, ᶜρʲ, ᶜρ_diffʲ, ᶜgradᵥ_ᶠΦ, Yc, ᶜws...
            @. ᶜsgsʲₜ.mse += adjoint(CT3(ᶜinterp(ᶠu₃ʲ))) * ᶜρ_diffʲ * ᶜgradᵥ_ᶠΦ

            va = vertical_advection(ᶠu³ʲ, ᶜsgsʲ.mse, edmfx_mse_q_tot_upwinding)
            @. ᶜsgsʲₜ.mse += va

            va = vertical_advection(ᶠu³ʲ, ᶜsgsʲ.q_tot, edmfx_mse_q_tot_upwinding)
            @. ᶜsgsʲₜ.q_tot += va

            MatrixFields.unrolled_foreach(χ_names) do χ_name
                ᶜχʲ = MatrixFields.get_field(ᶜsgsʲ, χ_name)
                ᶜχʲₜ = MatrixFields.get_field(ᶜsgsʲₜ, χ_name)
                # (not `va`: assigning the enclosing body's name here would box it)
                va_χ = vertical_advection(ᶠu³ʲ, ᶜχʲ, edmfx_tracer_upwinding)
                @. ᶜχʲₜ += va_χ
            end

            if !(sed_names isa Tuple{})
                ᶜJ = Fields.local_geometry_field(axes(ᶜρʲ)).J
                ᶠJ = Fields.local_geometry_field(axes(ᶠu³ʲ)).J
                ᶜa = @. draft_area(ᶜsgsʲ.ρa, ᶜρʲ)
                ᶜinv_ρ̂ = @. specific(one_FT, ᶜsgsʲ.ρa, zero_FT, ᶜρʲ, turbconv_model)
                ∂a∂z = @. ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠtop_bias(Geometry.WVector(ᶜa)))
                ᶠρ = @. ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ
                ᶜwʲs_col, ᶜw_gss_col = _split_halves(ᶜws)
                MatrixFields.unrolled_foreach(
                    _unrolled_zip(
                        _unrolled_zip(sed_names, sed_ρχ_names),
                        _unrolled_zip(ᶜwʲs_col, ᶜw_gss_col),
                    ),
                ) do ((name, ρχ_name), (ᶜwʲ, ᶜw_gs))
                    ᶜχʲ = MatrixFields.get_field(ᶜsgsʲ, name)
                    ᶜχʲₜ = MatrixFields.get_field(ᶜsgsʲₜ, name)
                    # Environment sedimentation flux density: ρ⁰w⁰χ⁰
                    # Reconstructed from grid-mean: ρ̂⁰w⁰χ⁰ = w_GS·ρχ_GS − ρ̂¹w¹χ¹
                    # Then ρ⁰w⁰χ⁰ = ρ̂⁰w⁰χ⁰ / a⁰ = ρ̂⁰w⁰χ⁰ / (1 − a)
                    ᶜρχ_gs = MatrixFields.get_field(Yc, ρχ_name)
                    ᶜρ⁰w⁰χ⁰ = @. (ᶜw_gs * ᶜρχ_gs - ᶜsgsʲ.ρa * ᶜwʲ * ᶜχʲ) /
                                 max(1 - ᶜa, ϵ_FT)
                    # Flux form sedimentation of tracers
                    vtt = updraft_sedimentation(ᶠρ, ∂a∂z, ᶜwʲ, ᶜa, ᶜχʲ, ᶜρ⁰w⁰χ⁰)
                    @. ᶜχʲₜ += ᶜinv_ρ̂ * vtt
                    if name in (@name(q_lcl), @name(q_icl), @name(q_rai), @name(q_sno))
                        @. ᶜsgsʲₜ.q_tot += ᶜinv_ρ̂ * vtt
                    end
                end
            end
            return nothing
        end
    end
end

"""
    updraft_sedimentation(ᶠρ, ∂a∂z, ᶜw, ᶜa, ᶜχ, ᶜρ⁰w⁰χ⁰)

Return the flux-form sedimentation tendency of an updraft tracer `ᶜχ` as a
column loop temporary (see `edmfx_sgs_vertical_advection_tendency!`), computed
from the face updraft density `ᶠρ`, the updraft area `ᶜa` and its divergence
`∂a∂z = ᶜprecipdivᵥ(ᶠtop_bias(a))`, the (downward) terminal velocity `ᶜw`, and
the environment sedimentation flux density `ᶜρ⁰w⁰χ⁰`.

The flux through each face is carried by the updraft fraction `a` of the
column. Where the updraft area shrinks with height (`∂a∂z < 0`) the air that
leaves the updraft also leaves its sedimenting mass behind, so the divergence
of the area-weighted flux `ρ w a χ` is corrected by the environment flux
density times `∂a∂z`; otherwise the updraft tendency is `a` times the
divergence of the flux `ρ w χ`. Called inside a fused column loop, so it must
stay inlined.
"""
@inline function updraft_sedimentation(ᶠρ, ∂a∂z, ᶜw, ᶜa, ᶜχ, ᶜρ⁰w⁰χ⁰)
    ᶠwχ = @. ᶠtop_bias(-(ᶜw) * ᶜχ)
    ᶠwaχ = @. ᶠtop_bias(-(ᶜw) * ᶜa * ᶜχ)
    return @. ifelse(
        ∂a∂z < 0,
        -(ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwaχ)) - ∂a∂z * ᶜρ⁰w⁰χ⁰),
        -(ᶜa * ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwχ))),
    )
end
