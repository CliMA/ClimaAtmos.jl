#####
##### Advection and dynamics tendencies
#####

using LinearAlgebra: ×, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

"""
    horizontal_dynamics_tendency!(Yₜ, Y, p, t)

Computes tendencies due to horizontal advection for prognostic variables of the
grid mean and EDMFX subdomains, and also applies horizontal pressure gradient and
gravitational acceleration terms for horizontal momentum.

Specifically, this function calculates:

  - Horizontal advection of density (`ρ`).
  - Horizontal advection of EDMFX updraft density-area product (`ρaʲ`).
  - Horizontal advection of total energy (`ρe_tot`) using total enthalpy flux.
  - Horizontal advection of EDMFX updraft moist static energy (`mseʲ`).
  - Horizontal advection of turbulent kinetic energy (`ρtke`) if used.
  - Horizontal pressure gradient, kinetic energy gradient, and geopotential gradient
    forces for horizontal momentum (`uₕ`).

Arguments:

  - `Yₜ`: The tendency state vector, modified in place.
  - `Y`: The current state vector.
  - `p`: Cache containing parameters, precomputed fields (e.g., velocities `ᶜu`,
    `ᶜu⁰`, `ᶜuʲs`; pressure `ᶜp`; kinetic energy `ᶜK`; total enthalpy `ᶜh_tot`),
    and core components (e.g., geopotential `ᶜΦ`).
  - `t`: Current simulation time (not directly used in calculations).

Modifies `Yₜ.c.ρ`, `Yₜ.c.ρe_tot`, `Yₜ.c.uₕ`, and EDMFX-related fields in
`Yₜ.c.sgsʲs` and `Yₜ.c.ρtke` if applicable.
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

Computes tendencies due to horizontal advection for tracers in the grid mean
and for specific humidity species within EDMFX subdomains.

Specifically, this function calculates:

  - Horizontal advection for all prognostic tracer variables (`ρχ_name`) in `Y.c`.
  - Horizontal advection for EDMFX updraft total specific humidity (`q_totʲ`).
  - Horizontal advection for other EDMFX updraft moisture species (`q_lclʲ`, `q_iclʲ`,
    `q_raiʲ`, `q_snoʲ`) if using a `NonEquilibriumMicrophysics1M` or
    `NonEquilibriumMicrophysics2M` microphysics model. If the `NonEquilibriumMicrophysics2M`
    model is used instead, `n_liqʲ` and `n_raiʲ` are also advected.

Arguments:

  - `Yₜ`: The tendency state vector, modified in place.
  - `Y`: The current state vector.
  - `p`: Cache containing parameters and precomputed fields (e.g., velocities `ᶜu`, `ᶜuʲs`).
  - `t`: Current simulation time (not directly used in calculations).

Modifies tracer fields in `Yₜ.c` (e.g., `Yₜ.c.ρq_tracer`) and EDMFX moisture fields
in `Yₜ.c.sgsʲs` if applicable.
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

Computes the vertical transport of `ρq_tot` at the surface due to prescribed flow.

If the flow is not prescribed, this has no effect.

# Arguments

  - `flow`: The prescribed flow model, see [`PrescribedFlow`](@ref).

      + If `flow` is `nothing`, this has no effect.

  - `thermo_params`: The thermodynamic parameters, needed to compute surface air density.

  - `t`: The current time.

  - `ᶠu³`: The vertical velocity field.

# Returns

  - The vertical transport of `ρq_tot` at the surface due to prescribed flow.
"""
ᶜρq_tot_vertical_transport_bc(::Nothing, _, _, _) = NullBroadcasted()
function ᶜρq_tot_vertical_transport_bc(flow::PrescribedFlow, thermo_params, t, ᶠu³)
    ρu₃qₜ_sfc_bc = get_ρu₃qₜ_surface(flow, thermo_params, t)
    ᶜadvdivᵥ = Operators.DivergenceF2C(; bottom = Operators.SetValue(ρu₃qₜ_sfc_bc))
    return @. lazy(-(ᶜadvdivᵥ(zero(ᶠu³))))
end

"""
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)

Computes tendencies due to explicit vertical advection for various grid-mean
prognostic variables, including passive tracers, energy, total water, momentum (using
a vector invariant form), and optionally TKE.

This function handles:

  - Calculation of vorticity components (`ᶜω³`, `ᶠω¹²`).
  - Vertical advection of passive tracers using `vertical_transport` with specified upwinding.
  - Upwinding corrections for vertical advection of energy and total water, assuming
    their central advection might be handled elsewhere or implicitly.
  - Vertical advection terms for horizontal and vertical momentum, differing for
    shallow and deep atmosphere approximations, incorporating Coriolis and vorticity effects.
  - Vertical advection of grid-mean TKE (`ρtke`) if `use_prognostic_tke` is true.

Arguments:

  - `Yₜ`: The tendency state vector, modified in place.
  - `Y`: The current state vector.
  - `p`: Cache containing parameters, core fields (e.g., `ᶜf³`, `ᶠf¹²`, `ᶜΦ`),
    precomputed fields (e.g., `ᶜu`, `ᶠu³`, `ᶜK`, EDMF velocities/TKE if applicable),
    atmospheric model settings (`p.atmos.numerics` for upwinding schemes),
    and scratch space.
  - `t`: Current simulation time (not directly used in calculations).

Modifies `Yₜ.c` (various tracers, `ρe_tot`, `ρq_tot`, `uₕ`), `Yₜ.f.u₃`,
`Yₜ.f.sgsʲs` (updraft `u₃`), and `Yₜ.c.ρtke` as applicable.
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
    edmfx_sgs_vertical_advection_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

Computes tendencies due to vertical advection and buoyancy for EDMFX subgrid-scale
(SGS) updraft prognostic variables.

This function handles:

  - Vertical advection of updraft density-area product (`ρaʲ`).
  - Vertical advection of updraft moist static energy (`mseʲ`) and total specific humidity (`q_totʲ`).
  - Vertical advection of other updraft moisture species (`q_lclʲ`, `q_iclʲ`, `q_raiʲ`, `q_snoʲ`)
    if using a `NonEquilibriumMicrophysics1M` or `NonEquilibriumMicrophysics2M` microphysics
    model. If the `NonEquilibriumMicrophysics2M` model is used, `n_liqʲ` and `n_raiʲ` are also advected.
  - Buoyancy source term in the updraft `mseʲ` equation (geopotential work done against
    the density anomaly).

Arguments:

  - `Yₜ`: The tendency state vector, modified in place.
  - `Y`: The current state vector.
  - `p`: Cache containing parameters (`p.params`), time step `dt`, core fields (`ᶠgradᵥ_ᶜΦ`),
    precomputed EDMF fields (e.g., `ᶠu³ʲs`, `ᶜρʲs`), atmospheric model settings
    (`p.atmos.numerics.edmfx_mse_q_tot_upwinding`), and scratch space.
  - `t`: Current simulation time (not directly used in calculations).
  - `turbconv_model`: The `PrognosticEDMFX` turbulence convection model instance.

Modifies EDMF updraft fields in `Yₜ.c.sgsʲs` and `Yₜ.f.sgsʲs`.
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
                )
                @. ᶜχʲₜ += ᶜinv_ρ̂ * vtt
            end
        end
    end
end

"""
    updraft_sedimentation!(vtt, p, ᶜρ, ᶜw, ᶜa, ᶜχ, ᶠJ, ᶜρ⁰w⁰χ⁰)

Compute the sedimentation tendency of tracer `χ` within an updraft, including
lateral transfer (detrainment and entrainment) across tilted updraft boundaries.

# Description

Sedimenting particles fall with velocity `w` through an updraft of fractional area `a(z)`.
The base within-updraft tendency is `a · ∂_z(ρ w χ)`, which captures the vertical flux
convergence through the updraft cross-section with no lateral effects.

When `∂a/∂z > 0` (updraft narrows downward), sedimenting mass exits laterally through
the tilted boundary (detrainment). This is already excluded from `a · ∂_z(ρ w χ)` and
is captured by the grid-scale residual.

When `∂a/∂z < 0` (updraft widens downward), environment condensate enters through the
tilted boundary (entrainment). The correction term uses the upwind (donor-cell) principle:
the entrained sedimentation flux has **environment** properties, not updraft properties.

# Equation

The combined tendency is:

```math
\\text{tend} = a \\cdot \\partial_z(\\rho^{(1)} w^{(1)} \\chi^{(1)})
            + \\min(\\partial_z a,\\, 0) \\cdot
              (\\rho^{(1)} w^{(1)} \\chi^{(1)} - \\rho^{(0)} w^{(0)} \\chi^{(0)})
```

The second term is zero when `∂a/∂z ≥ 0` and adds the net lateral transfer
when the updraft widens. If updraft and environment have the same sedimentation
flux, no lateral mixing occurs.

# Arguments

  - `vtt` : output field
  - `p`: cache containing scratch spaces
  - `ᶜρ`: updraft air density
  - `ᶜw`: updraft sedimentation velocity (positive downward)
  - `ᶜa`: updraft area fraction
  - `ᶜχ`: updraft tracer mixing ratio
  - `ᶠJ`: face Jacobian (grid geometry)
  - `ᶜρ⁰w⁰χ⁰`: environment sedimentation flux density (ρ⁰ · w⁰ · χ⁰)

`vtt` gets filled with tracer tendency due to sedimentation and lateral transfer.
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
)
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    # use output as a scratch field
    ∂a∂z = vtt
    @. ∂a∂z = ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠright_bias(Geometry.WVector(ᶜa)))
    ᶠρ = @. p.scratch.ᶠtemp_scalar = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ
    ᶠwχ = @. p.scratch.ᶠtemp_scalar_2 = ᶠright_bias(-(ᶜw) * ᶜχ)

    # TODO - testing if stable with older version
    ᶠwaχ = @. p.scratch.ᶠtemp_scalar_3 = ᶠright_bias(-(ᶜw) * ᶜa * ᶜχ)
    @. vtt = ifelse(
        ∂a∂z < 0,
        -(ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwaχ))),
        -(ᶜa * ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwχ))),
    )
#=
    # Base: within-updraft flux convergence a · ∂_z(ρ w χ)
    # Entrainment correction: min(∂a/∂z, 0) · (ρ¹w¹χ¹ − ρ⁰w⁰χ⁰)
    @. vtt =
        -(ᶜa * ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwχ))) +
        min(∂a∂z, 0) * (ᶜρ * ᶜw * ᶜχ - ᶜρ⁰w⁰χ⁰)
=#
    return
end
