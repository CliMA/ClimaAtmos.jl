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

    # DG spaces: complete the element-local tendencies with interface terms
    dg_horizontal_dynamics_completion!(Yₜ, Y, p, t)
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

    # DG spaces: complete the grid-mean tracer advection with interface fluxes
    dg_horizontal_tracer_completion!(Yₜ, Y, p, t)
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
        if is_dg_horizontal(axes(Y.c))
            dg_ω³!(ᶜω³, Y)
        else
            @. ᶜω³ = wcurlₕ(Y.c.uₕ)
        end
    else
        @. ᶜω³ = zero(ᶜω³)
    end

    @. ᶠω¹² = ᶠcurlᵥ(Y.c.uₕ)
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) = ᶠω¹²
    end
    if is_dg_horizontal(axes(Y.f))
        dg_ω¹²_horizontal!(ᶠω¹², Y)
    else
        @. ᶠω¹² += CT12(wcurlₕ(Y.f.u₃))
    end
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
    via `updraft_sedimentation!`. This path currently supports a single updraft.

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

    FT = eltype(p.params)
    α_lat = CAP.sedimentation_lateral_coeff(p.params)
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J

    for j in 1:n
        ᶜa = (@. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))))

        # buoyancy term in mse equation
        @. Yₜ.c.sgsʲs.:($$j).mse +=
            adjoint(CT3(ᶜinterp(Y.f.sgsʲs.:($$j).u₃))) *
            ᶜρ_diffʲs.:($$j) * ᶜgradᵥ_ᶠΦ

        # Advective form advection of mse and q_tot with the updraft velocity
        # Note: This allocates because the function is too long
        va = vertical_advection(
            ᶠu³ʲs.:($j),
            Y.c.sgsʲs.:($j).mse,
            edmfx_mse_q_tot_upwinding,
        )
        @. Yₜ.c.sgsʲs.:($$j).mse += va

        va = vertical_advection(
            ᶠu³ʲs.:($j),
            Y.c.sgsʲs.:($j).q_tot,
            edmfx_mse_q_tot_upwinding,
        )
        @. Yₜ.c.sgsʲs.:($$j).q_tot += va

        # Advective form advection of auto-discovered SGS tracers
        # (microphysics species and any user-defined passive tracers)
        # with the updraft velocity
        for χ_name in sgs_tracer_names(Y)
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
            va = vertical_advection(
                ᶠu³ʲs.:($j),
                ᶜχʲ,
                edmfx_tracer_upwinding,
            )
            @. ᶜχʲₜ += va
        end

        if p.atmos.microphysics_model isa
           Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
            # TODO - add precipitation and cloud sedimentation in implicit solver/tendency with if/else
            # TODO - make it work for multiple updrafts
            if j > 1
                error("Below code doesn't work for multiple updrafts")
            end
            ᶜinv_ρ̂ = p.scratch.ᶜtemp_scalar_3
            @. ᶜinv_ρ̂ = specific(
                FT(1),
                Y.c.sgsʲs.:($$j).ρa,
                FT(0),
                ᶜρʲs.:($$j),
                turbconv_model,
            )
            # Sedimentation
            # TODO - lazify ᶜwₗʲs computation. No need to cache it.
            # Tuples: (updraft Y-path, species name, updraft vel, GS vel)
            sgs_microphysics_tracers = (
                (@name(c.sgsʲs.:(1).q_lcl), @name(q_lcl), @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
                (@name(c.sgsʲs.:(1).q_icl), @name(q_icl), @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
                (@name(c.sgsʲs.:(1).q_rai), @name(q_rai), @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
                (@name(c.sgsʲs.:(1).q_sno), @name(q_sno), @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
            )

            MatrixFields.unrolled_foreach(
                sgs_microphysics_tracers,
            ) do (qʲ_name, name, wʲ_name, w_gs_name)
                MatrixFields.has_field(Y, qʲ_name) || return

                ᶜqʲ = MatrixFields.get_field(Y, qʲ_name)
                ᶜqʲₜ = MatrixFields.get_field(Yₜ, qʲ_name)
                ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)
                # Environment sedimentation flux density: ρ⁰w⁰q⁰
                # Reconstructed from grid-mean: ρ̂⁰w⁰q⁰ = w_GS·ρq_GS − ρ̂¹w¹q¹
                # Then ρ⁰w⁰q⁰ = ρ̂⁰w⁰q⁰ / a⁰ = ρ̂⁰w⁰q⁰ / (1 − a)
                ᶜw_gs = MatrixFields.get_field(p.precomputed, w_gs_name)
                ᶜρq_gs = MatrixFields.get_field(Y.c, get_ρχ_name(name))
                ᶜρ⁰w⁰χ⁰ = @. lazy(
                    (ᶜw_gs * ᶜρq_gs - Y.c.sgsʲs.:($$j).ρa * ᶜwʲ * ᶜqʲ) /
                    max(1 - ᶜa, eps(FT)),
                )

                # Flux form sedimentation of tracers
                vtt = p.scratch.ᶜtemp_scalar_4
                updraft_sedimentation!(
                    vtt,
                    p,
                    ᶜρʲs.:($j),
                    ᶜwʲ,
                    ᶜa,
                    ᶜqʲ,
                    ᶠJ,
                    ᶜρ⁰w⁰χ⁰,
                    α_lat,
                )
                @. ᶜqʲₜ += ᶜinv_ρ̂ * vtt
                @. Yₜ.c.sgsʲs.:($$j).q_tot += ᶜinv_ρ̂ * vtt
            end
        end

        # Sedimentation of number concentrations for 2M microphysics
        if p.atmos.microphysics_model isa NonEquilibriumMicrophysics2M

            # TODO - add precipitation and cloud sedimentation in implicit solver/tendency with if/else
            # TODO - make it work for multiple updrafts
            if j > 1
                error("Below code doesn't work for multiple updrafts")
            end

            # Sedimentation velocities for microphysics number concentrations
            # (or any tracers that does not directly participate in variations of q_tot and mse)
            # Tuples: (updraft Y-path, species name, updraft vel, GS vel)
            sgs_microphysics_tracers = (
                (@name(c.sgsʲs.:(1).n_lcl), @name(n_lcl), @name(ᶜwₙₗʲs.:(1)), @name(ᶜwₙₗ)),
                (@name(c.sgsʲs.:(1).n_rai), @name(n_rai), @name(ᶜwₙᵣʲs.:(1)), @name(ᶜwₙᵣ)),
            )

            MatrixFields.unrolled_foreach(
                sgs_microphysics_tracers,
            ) do (χʲ_name, name, wʲ_name, w_gs_name)
                MatrixFields.has_field(Y, χʲ_name) || return

                ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                ᶜχʲₜ = MatrixFields.get_field(Yₜ, χʲ_name)
                ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)
                # Environment sedimentation flux density: ρ⁰w⁰χ⁰
                # Reconstructed from grid-mean: ρ̂⁰w⁰χ⁰ = w_GS·ρχ_GS − ρ̂¹w¹χ¹
                # Then ρ⁰w⁰χ⁰ = ρ̂⁰w⁰χ⁰ / a⁰ = ρ̂⁰w⁰χ⁰ / (1 − a)
                ᶜw_gs = MatrixFields.get_field(p.precomputed, w_gs_name)
                ᶜρχ_gs = MatrixFields.get_field(Y.c, get_ρχ_name(name))
                ᶜρ⁰w⁰χ⁰ = @. lazy(
                    (ᶜw_gs * ᶜρχ_gs - Y.c.sgsʲs.:($$j).ρa * ᶜwʲ * ᶜχʲ) /
                    max(1 - ᶜa, eps(FT)),
                )

                # Flux form sedimentation of tracers
                vtt = p.scratch.ᶜtemp_scalar_4
                updraft_sedimentation!(
                    vtt,
                    p,
                    ᶜρʲs.:($j),
                    ᶜwʲ,
                    ᶜa,
                    ᶜχʲ,
                    ᶠJ,
                    ᶜρ⁰w⁰χ⁰,
                    α_lat,
                )
                @. ᶜχʲₜ += ᶜinv_ρ̂ * vtt
            end
        end
    end
end

"""
    updraft_sedimentation!(vtt, p, ᶜρ, ᶜw, ᶜa, ᶜχ, ᶠJ, ᶜρ⁰w⁰χ⁰, α_lat)

Compute the sedimentation tendency of tracer `χ` within an updraft, including
lateral transfer (detrainment and entrainment) across tilted updraft boundaries,
and store it in `vtt`.

Sedimenting particles fall with velocity `w` through an updraft of fractional area
`a(z)`. The base within-updraft tendency is `a · ∂_z(ρ w χ)`, the vertical flux
convergence through the updraft cross-section with no lateral effects. When
`∂a/∂z > 0` (updraft narrows downward), sedimenting mass exits laterally through
the tilted boundary (detrainment); this is already excluded from `a · ∂_z(ρ w χ)`
and is captured by the grid-scale residual. When `∂a/∂z < 0` (updraft widens
downward), environment condensate enters through the tilted boundary (entrainment);
following the upwind (donor-cell) principle, the entrained sedimentation flux has
environment properties, not updraft properties. The combined tendency is

```math
\\text{tend} = a \\, \\partial_z(\\rho^{(1)} w^{(1)} \\chi^{(1)})
            + \\alpha_{lat} \\, \\min(\\partial_z a,\\, 0) \\,
              (\\rho^{(1)} w^{(1)} \\chi^{(1)} - \\rho^{(0)} w^{(0)} \\chi^{(0)})
```

where the second term vanishes when `∂a/∂z ≥ 0` or when updraft and environment
carry the same sedimentation flux.

# Arguments

  - `vtt`: Output field, overwritten with the tendency (also used as scratch).
  - `p`: Cache providing scratch fields.
  - `ᶜρ`: Updraft air density [kg/m³].
  - `ᶜw`: Updraft sedimentation velocity, positive downward [m/s].
  - `ᶜa`: Updraft area fraction [-].
  - `ᶜχ`: Updraft tracer specific quantity.
  - `ᶠJ`: Face Jacobian (grid geometry).
  - `ᶜρ⁰w⁰χ⁰`: Environment sedimentation flux density `ρ⁰ w⁰ χ⁰`.
  - `α_lat`: Lateral correction scaling; 0 disables, 1 is the full correction [-].

Called from `edmfx_sgs_vertical_advection_tendency!`. Returns `nothing`.
"""
function updraft_sedimentation!(
    vtt,
    p,
    ᶜρ,
    ᶜw,
    ᶜa,
    ᶜχ,
    ᶠJ,
    ᶜρ⁰w⁰χ⁰,
    α_lat,
)
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    # use output as a scratch field
    ∂a∂z = vtt
    @. ∂a∂z = ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠright_bias(Geometry.WVector(ᶜa)))
    ᶠρ = @. p.scratch.ᶠtemp_scalar = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ
    ᶠwχ = @. p.scratch.ᶠtemp_scalar_2 = ᶠright_bias(-(ᶜw) * ᶜχ)
    ᶠwaχ = @. p.scratch.ᶠtemp_scalar_3 = ᶠright_bias(-(ᶜw) * ᶜa * ᶜχ)
    # Base: within-updraft flux convergence a · ∂_z(ρ w χ)
    # Lateral correction: α_lat · min(∂a/∂z, 0) · ρ⁰w⁰χ⁰
    @. vtt = ifelse(
        ∂a∂z < 0,
        -(ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwaχ)) - α_lat * ∂a∂z * ᶜρ⁰w⁰χ⁰),
        -(ᶜa * ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwχ))),
    )
    return
end
