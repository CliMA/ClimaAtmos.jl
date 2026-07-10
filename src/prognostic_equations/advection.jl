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
  - Sedimentation of the updraft condensate species (and number concentrations
    for 2M microphysics), including the cross-subdomain transfer of condensate
    falling through tilted subdomain boundaries: each updraft receives a share
    of the grid-mean sedimentation flux divergence (mass fraction for gains,
    holdings fraction for losses; see the inline comment for the derivation
    and the positivity clip).
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
            ᶜJ = Fields.local_geometry_field(axes(Y.c)).J

            # Updraft sedimentation with cross-subdomain (updraft <->
            # environment) transfer.
            #
            # When subdomain area fractions vary with height, condensate
            # falling through the tilted subdomain side boundaries is
            # transferred between subdomains. With the antisymmetric
            # cross-boundary transfer of the PROPHET formulation
            # ("Sedimentation across Subdomain Boundaries"),
            #
            #   𝒥⁽ᵐⁿ⁾ = (1/ρ) [ρaⁿ ∂z(ρaᵐ χᵐ wᵐ) − ρaᵐ ∂z(ρaⁿ χⁿ wⁿ)],
            #
            # the sum of a subdomain's own sedimentation flux divergence and
            # its cross-boundary transfers telescopes to a share of the
            # grid-mean sedimentation flux divergence:
            #
            #   ∂t(ρaʲ χʲ)|sed = ωʲ ∂z(ρ χ w),   ωʲ = ρaʲ/ρ.
            #
            # Physically: condensate crossing a level is redistributed among
            # the subdomains below in proportion to their masses (random
            # horizontal overlap), so every subdomain feels the same
            # per-unit-mass sedimentation tendency. The grid-mean flux
            # divergence is computed with the same stencil as the grid-mean
            # sedimentation tendency (implicit_vertical_advection_tendency!
            # and vertical_advection_of_water_tendency!), so the subdomain
            # shares sum to the grid-mean tendency discretely, with the
            # environment as the residual.
            #
            # Caveat — how negative values can arise, and the clip applied
            # here: with the pure mass-fraction weight ωʲ = ρaʲ/ρ, a net
            # *loss* at a level (∂z(ρχw) < 0: condensate leaving downward
            # faster than it arrives from above) is also charged to each
            # subdomain by its mass fraction, including subdomains that hold
            # none of the species. For example, grid-mean rain falling
            # through the levels below the updraft's condensate base would
            # drive q_raiʲ negative there. Since condensate can only fall
            # out of a subdomain that actually holds it, net losses are
            # instead apportioned by holdings,
            #
            #   ωʲ = clamp(ρaʲ χʲ / (ρ χ), 0, 1)   where ∂z(ρ χ w) < 0.
            #
            # Holdings shares also sum to one across subdomains, so
            # conservation with the environment-as-residual is preserved;
            # the loss is proportional to χʲ, so it cannot create negative
            # values; and the two weights coincide when the subdomains are
            # materially identical, recovering the formula above. Gains keep
            # the mass-fraction weight.
            #
            # Bookkeeping approximations (deliberate, to keep this change
            # small): the transferred condensate also moves moist-air mass
            # and energy between subdomains, but ρaʲ is stepped by the
            # analytic implicit stage solve
            # (solve_sgs_ρa_implicit_stage_analytic!), which we do not
            # modify here — so the ρaʲ source, the paired (1 − q_totʲ)
            # dilution factor on the q_totʲ source, and the mseʲ source are
            # all omitted. As in the previous formulation, sedimentation
            # changes the updraft water content at fixed ρaʲ.
            MatrixFields.unrolled_foreach(
                sedimenting_sgs_tracer_names(Y),
            ) do χ_name
                ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
                ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:(1), χ_name)
                ρχ_name = get_ρχ_name(χ_name)
                ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
                ᶜw = MatrixFields.get_field(
                    p.precomputed,
                    sedimentation_velocity_name(ρχ_name),
                )

                # Grid-mean sedimentation flux divergence ∂z(ρ χ w).
                vtt = p.scratch.ᶜtemp_scalar_4
                @. vtt = -(ᶜprecipdivᵥ(
                    ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                        Geometry.WVector(-(ᶜw)) * specific(ᶜρχ, Y.c.ρ),
                    ),
                ))
                # Updraft share: mass fraction for gains, holdings fraction
                # for losses (see comment above).
                @. vtt *= ifelse(
                    vtt < 0,
                    ifelse(
                        ᶜρχ > FT(0),
                        clamp(
                            Y.c.sgsʲs.:(1).ρa * ᶜχʲ / ᶜρχ,
                            FT(0),
                            FT(1),
                        ),
                        FT(0),
                    ),
                    Y.c.sgsʲs.:(1).ρa / Y.c.ρ,
                )
                @. ᶜχʲₜ += ᶜinv_ρ̂ * vtt
                # Sedimenting condensate masses are part of the updraft
                # total water; number concentrations are not.
                if !isnothing(condensate_phase(χ_name))
                    @. Yₜ.c.sgsʲs.:(1).q_tot += ᶜinv_ρ̂ * vtt
                end
            end
        end
    end
end
