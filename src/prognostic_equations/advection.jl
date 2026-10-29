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
                ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
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
    n_val = Val(n)
    advect_tke_val = Val(use_prognostic_tke(turbconv_model))
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    (; ᶜf³, ᶠf¹²) = p.core
    (; ᶜu, ᶠu³, ᶜK) = p.precomputed
    (; edmfx_mse_q_tot_upwinding) =
        n > 0 || advect_tke_val isa Val{true} ? p.atmos.numerics : all_nothing
    (; ᶜuʲs, ᶜKʲs, ᶠKᵥʲs) = n > 0 ? p.precomputed : all_nothing
    (; tracer_upwinding) = p.atmos.numerics
    thermo_params = CAP.thermodynamics_params(p.params)

    ᶜω³ = p.scratch.ᶜtemp_CT3
    ᶠω¹² = p.scratch.ᶠtemp_CT12
    ᶠω¹²ʲs = p.scratch.ᶠtemp_CT12ʲs

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = wcurlₕ(Y.c.uₕ)
    else
        @. ᶜω³ = zero(ᶜω³)
    end

    # The vertical curl and its copies for the drafts in one column loop.
    if n > 0
        DataLayouts.foreach_column(ᶠω¹², ᶠω¹²ʲs, Y.c.uₕ) do ᶠω¹², ᶠω¹²ʲs, ᶜuₕ
            @. ᶠω¹² = ᶠcurlᵥ(ᶜuₕ)
            ntuple(n_val) do j
                ᶠω¹²ʲ = getproperty(ᶠω¹²ʲs, j)
                @. ᶠω¹²ʲ = ᶠω¹²
                return nothing
            end
            return nothing
        end
    else
        @. ᶠω¹² = ᶠcurlᵥ(Y.c.uₕ)
    end
    @. ᶠω¹² += CT12(wcurlₕ(Y.f.u₃))
    for j in 1:n
        @. ᶠω¹²ʲs.:($$j) += CT12(wcurlₕ(Y.f.sgsʲs.:($$j).u₃))
    end
    # Without the CT12(), the right-hand side would be a CT1 or CT2 in 2D space.

    if !(p.atmos.microphysics_model isa DryModel)
        vtt_bc =
            ᶜρq_tot_vertical_transport_bc(prescribed_flow, thermo_params, t, ᶠu³)
        @. Yₜ.c.ρq_tot += vtt_bc
    end

    # Full vertical advection of passive tracers (such as liq, rai, etc) with the
    # grid-mean flow. When EDMFX sgs_mass_flux is active, difference-form SGS
    # corrections ρᵏaᵏ(u³ᵏ - u³)(χᵏ - χ) are added on top of this in
    # edmfx_sgs_mass_flux_tendency!.
    tracer_names = unrolled_filter(
        ρχ_name -> !(ρχ_name in (@name(ρe_tot), @name(ρq_tot))),
        gs_tracer_names(Y),
    )
    draft_fields =
        n > 0 ?
        (
            ᶠω¹²ʲs,
            ntuple(j -> getproperty(ᶜuʲs, j), n)...,
            ntuple(j -> getproperty(ᶜKʲs, j), n)...,
            ntuple(j -> getproperty(ᶠKᵥʲs, j), n)...,
        ) : ()
    deep_val = Val(!isnothing(ᶠf¹²)) # the Coriolis faces exist in a deep atmosphere
    deep_fields = deep_val isa Val{true} ? (ᶠf¹²,) : ()

    # The tracer transports, the vector-invariant momentum advection and the TKE
    # transport in one fused column loop: the face density is a loop temporary
    # shared by every transport, and the vorticity is read from scratch once.
    DataLayouts.foreach_column(
        Yₜ.c, Yₜ.f, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶜu, ᶜK, ᶜf³, ᶜω³, ᶠω¹²,
        draft_fields..., deep_fields...,
    ) do Yₜc, Yₜf, Yc, ᶜJ, ᶠJ, ᶠu³, ᶜu, ᶜK, ᶜf³, ᶜω³, ᶠω¹², rest...
        n_drafts = val_value(n_val) # compile-time tuple offsets
        ᶜρ = Yc.ρ
        ᶠρ = @. ᶠinterp(ᶜρ * ᶜJ) / ᶠJ

        MatrixFields.unrolled_foreach(tracer_names) do ρχ_name
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
            add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, tracer_upwinding)
            return nothing
        end

        # In a deep atmosphere, the vorticity carries the horizontal Coriolis term.
        ᶠω¹²_tot =
            deep_val isa Val{true} ? (@. lazy(rest[end] + ᶠω¹²)) : ᶠω¹²
        @. Yₜc.uₕ -=
            ᶜinterp(ᶠω¹²_tot × (ᶠinterp(ᶜρ * ᶜJ) * ᶠu³)) / (ᶜρ * ᶜJ) +
            (ᶜf³ + ᶜω³) × CT12(ᶜu)
        @. Yₜf.u₃ -= ᶠω¹²_tot × ᶠinterp(CT12(ᶜu)) + ᶠgradᵥ(ᶜK)
        MatrixFields.unrolled_foreach(ntuple(Val, n_val)) do j_val
            j = val_value(j_val)
            ᶠω¹²ʲ = getproperty(rest[1], j)
            ᶠω¹²ʲ_tot =
                deep_val isa Val{true} ? (@. lazy(rest[end] + ᶠω¹²ʲ)) : ᶠω¹²ʲ
            ᶜuʲ = rest[1 + j]
            ᶜKʲ = rest[1 + n_drafts + j]
            ᶠKᵥʲ = rest[1 + 2 * n_drafts + j]
            ᶠu₃ʲₜ = getproperty(Yₜf.sgsʲs, j).u₃
            @. ᶠu₃ʲₜ -=
                ᶠω¹²ʲ_tot × ᶠinterp(CT12(ᶜuʲ)) + ᶠgradᵥ(ᶜKʲ - ᶜinterp(ᶠKᵥʲ))
            return nothing
        end

        if advect_tke_val isa Val{true} # advect_tke triggers allocations
            ᶜtke = @. lazy(specific(Yc.ρtke, Yc.ρ))
            add_vertical_transport!(Yₜc.ρtke, ᶠρ, ᶠu³, ᶜtke, dt, edmfx_mse_q_tot_upwinding)
        end
        return nothing
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
    updraft density anomaly `ᶜρ_diffʲs`), advective-form vertical advection with
    the updraft velocity using the `p.atmos.numerics.edmfx_mse_q_tot_upwinding`
    scheme, and the static energy `h_μ(Tʲ) + Φ` carried by condensate and
    precipitation sedimentation (with laterally entrained environment condensate
    at `h_μ(T⁰) + Φ`).
  - `Yₜ.c.sgsʲs.:(j).q_tot`: advective-form vertical advection (same scheme), plus
    the total-water contribution of condensate and precipitation sedimentation.
  - Every auto-discovered SGS tracer (from `sgs_tracer_names`): advective-form
    vertical advection with the `p.atmos.numerics.edmfx_tracer_upwinding` scheme.
  - For `NonEquilibriumMicrophysics1M`/`NonEquilibriumMicrophysics2M`, the updraft
    microphysics species (`q_lcl`, `q_icl`, `q_rai`, `q_sno`, and for 2M also `n_lcl`
    and `n_rai`): within-updraft sedimentation with lateral entrainment corrections
    via `updraft_sedimentation`. The condensate mass tendencies are a source
    of `ρaʲ` (`solve_sgs_ρa_implicit_stage_analytic!`), so `mse`, `q_tot`, and
    every SGS tracer also receive the dilution `-χ Σ_μ vtt_μ / ρaʲ` by the
    sedimented mass. This path supports a single updraft.

Reads `Y`, the precomputed `ᶠu³ʲs`, `ᶜρʲs`, `ᶜρ_diffʲs`, `ᶜTʲs`, `ᶜT⁰`, and
sedimentation velocities, and the core fields `ᶜgradᵥ_ᶠΦ` and `ᶜΦ`, in one
column loop per updraft. `t` is unused. See the "PROPHET: Overview and
Equations" page (`docs/src/prophet.md`) for the continuous equations. Returns
`nothing`.
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
    (; ᶠu³ʲs, ᶜρʲs, ᶜρ_diffʲs, ᶜTʲs, ᶜT⁰) = p.precomputed
    (; ᶜgradᵥ_ᶠΦ, ᶜΦ) = p.core
    (; microphysics_model) = p.atmos

    FT = eltype(p.params)
    (one_FT, zero_FT, ε_area) = (one(FT), zero(FT), eps(FT)) # isbits captures for the loop body
    thp = CAP.thermodynamics_params(p.params)
    χ_names = sgs_tracer_names(Y)

    # Sedimenting updraft species with their updraft and grid-mean terminal
    # velocities: (species name, updraft velocity name, grid-mean velocity name).
    # The condensate masses are a source of ρaʲ and dilute the updraft scalars;
    # the number concentrations of 2M microphysics are not.
    # TODO - add precipitation and cloud sedimentation in implicit solver/tendency with if/else
    # TODO - lazify ᶜwₗʲs computation. No need to cache it.
    sediments_val = Val(
        microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
    )
    mass_sedimentation =
        sediments_val isa Val{true} ?
        unrolled_filter(
            ((name, _, _),) -> MatrixFields.has_field(Y.c.sgsʲs.:(1), name),
            (
                (@name(q_lcl), @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
                (@name(q_icl), @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
                (@name(q_rai), @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
                (@name(q_sno), @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
            ),
        ) : ()
    number_sedimentation =
        microphysics_model isa NonEquilibriumMicrophysics2M ?
        unrolled_filter(
            ((name, _, _),) -> MatrixFields.has_field(Y.c.sgsʲs.:(1), name),
            (
                (@name(n_lcl), @name(ᶜwₙₗʲs.:(1)), @name(ᶜwₙₗ)),
                (@name(n_rai), @name(ᶜwₙᵣʲs.:(1)), @name(ᶜwₙᵣ)),
            ),
        ) : ()
    sedimentation = (mass_sedimentation..., number_sedimentation...)
    n_mass_val = Val(length(mass_sedimentation))
    mass_names = unrolled_map(first, mass_sedimentation)
    number_names = unrolled_map(first, number_sedimentation)
    ᶜwʲs = unrolled_map(
        ((_, wʲ_name, _),) -> MatrixFields.get_field(p.precomputed, wʲ_name),
        sedimentation,
    )
    ᶜw_gss = unrolled_map(
        ((_, _, w_gs_name),) -> MatrixFields.get_field(p.precomputed, w_gs_name),
        sedimentation,
    )

    for j in 1:n
        # TODO - make it work for multiple updrafts
        if sediments_val isa Val{true} && j > 1
            error("Below code doesn't work for multiple updrafts")
        end

        # One fused column loop per updraft: buoyancy, vertical advection of mse,
        # q_tot, and the tracers, and the sedimentation of the microphysics species.
        # The updraft area, the inverse updraft density, the face density, and the
        # per-species fluxes are loop temporaries rather than scratch fields.
        DataLayouts.foreach_column(
            Yₜ.c.sgsʲs.:($j), Y.c.sgsʲs.:($j), Y.f.sgsʲs.:($j).u₃, ᶠu³ʲs.:($j),
            ᶜρ_diffʲs.:($j), ᶜgradᵥ_ᶠΦ, ᶜρʲs.:($j), ᶜTʲs.:($j), ᶜT⁰, ᶜΦ, Y.c,
            ᶜwʲs..., ᶜw_gss...,
        ) do ᶜsgsʲₜ, ᶜsgsʲ, ᶠu₃ʲ, ᶠu³ʲ, ᶜρ_diffʲ, ᶜgradᵥ_ᶠΦ, ᶜρʲ, ᶜTʲ, ᶜT⁰, ᶜΦ, Yc,
        ᶜws...
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

            sediments_val isa Val{true} || return nothing

            ᶜJ = Fields.local_geometry_field(axes(ᶜρʲ)).J
            ᶠJ = Fields.local_geometry_field(axes(ᶠu³ʲ)).J
            ᶜa = @. draft_area(ᶜsgsʲ.ρa, ᶜρʲ)
            ᶜinv_ρ̂ = @. specific(one_FT, ᶜsgsʲ.ρa, zero_FT, ᶜρʲ, turbconv_model)
            ∂a∂z = @. ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠtop_bias(Geometry.WVector(ᶜa)))
            ᶠρ = @. ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ
            ᶜwʲs_col, ᶜw_gss_col = _split_halves(ᶜws)
            ᶜw_massʲs, ᶜw_numberʲs = _split_at(n_mass_val, ᶜwʲs_col)
            ᶜw_mass_gss, ᶜw_number_gss = _split_at(n_mass_val, ᶜw_gss_col)

            # Sedimentation of the condensate masses. Each species' flux
            # convergence `vtt` is a source of ρaʲ (in the analytic ρa solve), so an
            # intensive updraft scalar χ receives its own source minus the dilution
            # χ Σ_μ vtt_μ / ρaʲ by the sedimented mass. The mse source is the flux
            # convergence of q_μ (h_μ(Tʲ) + Φ), with laterally entrained environment
            # condensate carrying h_μ(T⁰) + Φ.
            ᶜsed_mass_source = @. zero(ᶜρʲ)
            MatrixFields.unrolled_foreach(
                _unrolled_zip(mass_names, _unrolled_zip(ᶜw_massʲs, ᶜw_mass_gss)),
            ) do (name, (ᶜwʲ, ᶜw_gs))
                ᶜqʲ = MatrixFields.get_field(ᶜsgsʲ, name)
                ᶜqʲₜ = MatrixFields.get_field(ᶜsgsʲₜ, name)
                ᶜρq_gs = MatrixFields.get_field(Yc, get_ρχ_name(name))
                # Environment sedimentation flux density, the clamped residual of
                # the grid-mean flux: ρ⁰w⁰q⁰ = max((w ρq - ρa wʲ qʲ) / (1 - a), 0)
                ᶜρ⁰w⁰q⁰ = @. max(
                    (ᶜw_gs * ᶜρq_gs - ᶜsgsʲ.ρa * ᶜwʲ * ᶜqʲ) / max(1 - ᶜa, ε_area),
                    zero_FT,
                )
                vtt_q = updraft_sedimentation(ᶠρ, ∂a∂z, ᶜwʲ, ᶜa, ᶜqʲ, ᶜρ⁰w⁰q⁰)
                @. ᶜqʲₜ += ᶜinv_ρ̂ * vtt_q
                @. ᶜsgsʲₜ.q_tot += ᶜinv_ρ̂ * vtt_q
                @. ᶜsed_mass_source += vtt_q
                h_cond_func = enthalpy_function(condensate_phase(name))
                ᶜqhʲ = @. ᶜqʲ * (h_cond_func(thp, ᶜTʲ) + ᶜΦ)
                ᶜρ⁰w⁰qh⁰ = @. ᶜρ⁰w⁰q⁰ * (h_cond_func(thp, ᶜT⁰) + ᶜΦ)
                vtt_h = updraft_sedimentation(ᶠρ, ∂a∂z, ᶜwʲ, ᶜa, ᶜqhʲ, ᶜρ⁰w⁰qh⁰)
                @. ᶜsgsʲₜ.mse += ᶜinv_ρ̂ * vtt_h
                return nothing
            end
            @. ᶜsgsʲₜ.mse -= ᶜinv_ρ̂ * ᶜsed_mass_source * ᶜsgsʲ.mse
            @. ᶜsgsʲₜ.q_tot -= ᶜinv_ρ̂ * ᶜsed_mass_source * ᶜsgsʲ.q_tot
            MatrixFields.unrolled_foreach(χ_names) do χ_name
                ᶜχʲ = MatrixFields.get_field(ᶜsgsʲ, χ_name)
                ᶜχʲₜ = MatrixFields.get_field(ᶜsgsʲₜ, χ_name)
                @. ᶜχʲₜ -= ᶜinv_ρ̂ * ᶜsed_mass_source * ᶜχʲ
            end

            # Sedimentation of the 2M number concentrations, which do not change
            # the updraft mass.
            MatrixFields.unrolled_foreach(
                _unrolled_zip(
                    number_names,
                    _unrolled_zip(ᶜw_numberʲs, ᶜw_number_gss),
                ),
            ) do (name, (ᶜwʲ, ᶜw_gs))
                ᶜχʲ = MatrixFields.get_field(ᶜsgsʲ, name)
                ᶜχʲₜ = MatrixFields.get_field(ᶜsgsʲₜ, name)
                ᶜρχ_gs = MatrixFields.get_field(Yc, get_ρχ_name(name))
                ᶜρ⁰w⁰χ⁰ = @. max(
                    (ᶜw_gs * ᶜρχ_gs - ᶜsgsʲ.ρa * ᶜwʲ * ᶜχʲ) / max(1 - ᶜa, ε_area),
                    zero_FT,
                )
                vtt_n = updraft_sedimentation(ᶠρ, ∂a∂z, ᶜwʲ, ᶜa, ᶜχʲ, ᶜρ⁰w⁰χ⁰)
                @. ᶜχʲₜ += ᶜinv_ρ̂ * vtt_n
                return nothing
            end
            return nothing
        end
    end
end

"""
    updraft_sedimentation!(vtt, p, ᶜρ, ᶜw, ᶜa, ᶜχ, ᶠJ, ᶜρ⁰w⁰χ⁰)

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
            + \\min(\\partial_z a,\\, 0) \\,
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

Called from `foreach_updraft_sedimentation`; the column loop of
`edmfx_sgs_vertical_advection_tendency!` uses its column form,
`updraft_sedimentation`. Returns `nothing`.
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
    @. ∂a∂z = ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠtop_bias(Geometry.WVector(ᶜa)))
    ᶠρ = @. p.scratch.ᶠtemp_scalar = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ
    ᶠwχ = @. p.scratch.ᶠtemp_scalar_2 = ᶠtop_bias(-(ᶜw) * ᶜχ)
    ᶠwaχ = @. p.scratch.ᶠtemp_scalar_3 = ᶠtop_bias(-(ᶜw) * ᶜa * ᶜχ)
    # ∂a/∂z ≥ 0 (updraft narrows downward): a · ∂_z(ρ w χ).
    # ∂a/∂z < 0 (updraft widens downward): a · ∂_z(ρwχ) + ∂a/∂z · (ρ¹w¹χ¹ − ρ⁰w⁰χ⁰)
    #   = ∂_z(a ρ w χ) − ∂a/∂z · ρ⁰w⁰χ⁰: the environment flux entering through the
    #   tilted boundary is a gain for the updraft (ρ⁰w⁰χ⁰ ≥ 0).
    @. vtt = ifelse(
        ∂a∂z < 0,
        -(ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwaχ)) + ∂a∂z * ᶜρ⁰w⁰χ⁰),
        -(ᶜa * ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwχ))),
    )
    return
end

"""
    updraft_sedimentation(ᶠρ, ∂a∂z, ᶜw, ᶜa, ᶜχ, ᶜρ⁰w⁰χ⁰)

Column-loop form of `updraft_sedimentation!`: return the flux-form
sedimentation tendency of the updraft tracer `ᶜχ` as a loop temporary, given
the face updraft density `ᶠρ = ᶠinterp(ρʲ J) / ᶠJ` and the area gradient
`∂a∂z = ᶜprecipdivᵥ(ᶠinterp(J) / ᶠJ ᶠtop_bias(a))`, which do not depend on the
tracer and are computed once per column. Called from the fused column loop of
`edmfx_sgs_vertical_advection_tendency!`, so it must stay inlined.
"""
@inline function updraft_sedimentation(ᶠρ, ∂a∂z, ᶜw, ᶜa, ᶜχ, ᶜρ⁰w⁰χ⁰)
    ᶠwχ = @. ᶠtop_bias(-(ᶜw) * ᶜχ)
    ᶠwaχ = @. ᶠtop_bias(-(ᶜw) * ᶜa * ᶜχ)
    return @. ifelse(
        ∂a∂z < 0,
        -(ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwaχ)) + ∂a∂z * ᶜρ⁰w⁰χ⁰),
        -(ᶜa * ᶜprecipdivᵥ(ᶠρ * Geometry.WVector(ᶠwχ))),
    )
end

"""
    foreach_updraft_sedimentation(f, Y, p, j)

Call `f(vtt, name, ᶜqʲ, ᶜwʲ, ᶜa, ᶠJ, ᶜρ⁰w⁰χ⁰)` for each sedimenting condensate
mass species present in updraft `j` (`q_lcl`, `q_icl`, `q_rai`, `q_sno`), with
`vtt` the within-updraft sedimentation tendency of the species from
`updraft_sedimentation!` [kg m⁻³ s⁻¹], `name` the species `@name` relative to
`Y.c.sgsʲs.:(j)`, `ᶜqʲ` its updraft specific humidity, `ᶜwʲ` its updraft
terminal velocity, `ᶜa` the updraft area fraction, `ᶠJ` the face Jacobian, and
`ᶜρ⁰w⁰χ⁰` the environment sedimentation flux density.

The environment flux density `ρ⁰w⁰q⁰` of the lateral correction is
reconstructed from the grid-mean flux, `ρ̂⁰w⁰q⁰ = w ρq - ρ̂ʲwʲqʲ` and
`ρ⁰w⁰q⁰ = max(ρ̂⁰w⁰q⁰ / (1 - a), 0)`. `vtt` is `p.scratch.ᶜtemp_scalar_4`,
overwritten for each species, and `updraft_sedimentation!` writes
`ᶠtemp_scalar`, `ᶠtemp_scalar_2`, and `ᶠtemp_scalar_3`. Supports a single
updraft. Returns `nothing`. Called from `updraft_sedimentation_rate!`.
"""
function foreach_updraft_sedimentation(f, Y, p, j)
    j == 1 || error("Updraft sedimentation supports a single updraft")
    FT = eltype(p.params)
    # A closure stores a captured type as a `DataType`, so `eps(FT)` inside it
    # dispatches at run time and allocates.
    ε_area = eps(FT)
    zero_FT = zero(FT)
    (; ᶜρʲs) = p.precomputed
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    ᶜa = (@. lazy(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1))))
    # Tuples: (species name, updraft vel, GS vel)
    sgs_microphysics_tracers = (
        (@name(q_lcl), @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
        (@name(q_icl), @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
        (@name(q_rai), @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
        (@name(q_sno), @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
    )
    MatrixFields.unrolled_foreach(
        sgs_microphysics_tracers,
    ) do (name, wʲ_name, w_gs_name)
        MatrixFields.has_field(Y.c.sgsʲs.:(1), name) || return

        ᶜqʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), name)
        ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)
        ᶜw_gs = MatrixFields.get_field(p.precomputed, w_gs_name)
        ᶜρq_gs = MatrixFields.get_field(Y.c, get_ρχ_name(name))
        ᶜρ⁰w⁰χ⁰ = @. lazy(
            max(
                (ᶜw_gs * ᶜρq_gs - Y.c.sgsʲs.:(1).ρa * ᶜwʲ * ᶜqʲ) /
                max(1 - ᶜa, ε_area),
                zero_FT,
            ),
        )
        vtt = p.scratch.ᶜtemp_scalar_4
        updraft_sedimentation!(
            vtt,
            p,
            ᶜρʲs.:(1),
            ᶜwʲ,
            ᶜa,
            ᶜqʲ,
            ᶠJ,
            ᶜρ⁰w⁰χ⁰,
        )
        f(vtt, name, ᶜqʲ, ᶜwʲ, ᶜa, ᶠJ, ᶜρ⁰w⁰χ⁰)
    end
    return nothing
end

"""
    updraft_sedimentation_matrix!(ᶜM, Y, p, dtγ, χ_name, ᶜ∂a∂z)

Overwrite `ᶜM` with `dtγ ∂vtt/∂χʲ`, the derivative of the within-updraft
sedimentation tendency `vtt` of `updraft_sedimentation!` for the tracer
`χ_name` of updraft 1 (relative to `Y.c.sgsʲs.:(1)`) with respect to that
tracer, and return the lateral diagonal that `ᶜM` includes as a lazy
broadcast.

Where `∂a/∂z ≥ 0`, `vtt = a ∂_z(ρwχ)`; where `∂a/∂z < 0`,
`vtt = ∂_z(aρwχ) - ∂a/∂z ρ⁰w⁰χ⁰` with
`ρ⁰w⁰χ⁰ = max((w ρχ - ρa wʲ χʲ) / (1 - a), 0)`. Where the residual is
positive, `∂(ρ⁰w⁰χ⁰)/∂χʲ = -ρa wʲ / (1 - a)`, so the lateral term contributes
`dtγ ∂a/∂z a ρʲ wʲ / (1 - a)` to the diagonal; where the clamp is active, its
derivative is zero. The flux through each face comes from the cell above, so
`ᶜM` is upper bidiagonal. `ᶜ∂a∂z` is the area gradient of
`updraft_sedimentation!`. Writes `ᶠsed_tracer_advection` in `p.scratch`. Called from
`updraft_sedimentation_rate!`; the column loop of `update_sgs_advection_jacobian!`
builds the same operator.
"""
function updraft_sedimentation_matrix!(ᶜM, Y, p, dtγ, χ_name, ᶜ∂a∂z)
    (; ᶜρʲs) = p.precomputed
    (; ᶠsed_tracer_advection) = p.scratch
    FT = Spaces.undertype(axes(Y.c))
    # Type-valued calls such as eps(eltype(ᶜa)) inside nested lazy
    # broadcasts leave the matrix eltype uninferable, so the constants are
    # computed here.
    ε_area = eps(FT)
    zero_FT = zero(FT)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜa = (@. lazy(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1))))
    ᶜwʲ = MatrixFields.get_field(
        p.precomputed,
        sgs_sedimentation_velocity_name(χ_name),
    )
    ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
    ρχ_name = get_ρχ_name(χ_name)
    ᶜρχ_gs = MatrixFields.get_field(Y.c, ρχ_name)
    ᶜw_gs = MatrixFields.get_field(
        p.precomputed,
        sedimentation_velocity_name(ρχ_name),
    )
    ᶜlateral_coeff = @. lazy(
        dtγ * ifelse(
            (ᶜ∂a∂z < 0) &
            (ᶜw_gs * ᶜρχ_gs - Y.c.sgsʲs.:(1).ρa * ᶜwʲ * ᶜχʲ > 0),
            ᶜ∂a∂z * ᶜa * ᶜρʲs.:(1) * ᶜwʲ / max(1 - ᶜa, ε_area),
            zero_FT,
        ),
    )
    @. ᶠsed_tracer_advection =
        DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) *
        ᶠtop_bias_matrix() *
        DiagonalMatrixRow(-Geometry.WVector(ᶜwʲ))
    @. ᶜM =
        dtγ * ifelse(ᶜ∂a∂z < 0,
            -(ᶜprecipdivᵥ_matrix()) * ᶠsed_tracer_advection *
            DiagonalMatrixRow(ᶜa),
            -DiagonalMatrixRow(ᶜa) * ᶜprecipdivᵥ_matrix() * ᶠsed_tracer_advection,
        ) + DiagonalMatrixRow(ᶜlateral_coeff)
    return ᶜlateral_coeff
end

"""
    updraft_sedimentation_rate!(ᶜr, Y, p, j, dtγ)

Overwrite `ᶜr` with the sedimentation mass source rate [s⁻¹] of updraft `j`
over an implicit stage of length `dtγ`, `r = Σ_μ Δq_μ / dtγ`. Here `Δq_μ` is
the change of the updraft condensate mass fraction `q_μ` by sedimentation in
one linearized backward-Euler step from the current state,

    (I - ρ̂⁻¹ M_μ) Δq_μ = dtγ ρ̂⁻¹ vtt_μ,

with `vtt_μ` the within-updraft sedimentation tendency from
`foreach_updraft_sedimentation`, `M_μ = dtγ ∂vtt_μ/∂q_μ` from
`updraft_sedimentation_matrix!`, and `ρ̂⁻¹` the regularized inverse
`specific(1, ρa, 0, ρʲ, turbconv_model)`. `M_μ` is upper bidiagonal, so each
column is solved in one top-to-bottom sweep. The Newton solve takes the same
linear step for the sedimentation of the condensate, so `r ρaʲ` is the
sedimented mass that dilutes the updraft scalars there, at any Courant number.
`solve_sgs_ρa_implicit_stage_analytic!` applies it in the `ρa` recurrence.

Writes `ᶜtemp_scalar_6`, `ᶜtemp_scalar_7`, and `ᶜtridiagonal_matrix_scalar`
in `p.scratch`, besides the scratch fields of `foreach_updraft_sedimentation`
and `updraft_sedimentation_matrix!`, so `ᶜr` must be a different field.
Returns `nothing`.
"""
function updraft_sedimentation_rate!(ᶜr, Y, p, j, dtγ)
    (; ᶜρʲs) = p.precomputed
    (; ᶜtridiagonal_matrix_scalar) = p.scratch
    FT = eltype(p.params)
    one_FT = one(FT)
    zero_FT = zero(FT)
    ᶜΔq = p.scratch.ᶜtemp_scalar_6
    ᶜ∂a∂z = p.scratch.ᶜtemp_scalar_7
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜa = (@. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))))
    @. ᶜ∂a∂z = ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠtop_bias(Geometry.WVector(ᶜa)))
    ᶜinv_ρ̂ = @. lazy(
        specific(
            one_FT,
            Y.c.sgsʲs.:($$j).ρa,
            zero_FT,
            ᶜρʲs.:($$j),
            p.atmos.turbconv_model,
        ),
    )
    @. ᶜr = 0
    foreach_updraft_sedimentation(Y, p, j) do vtt, name, _, _, _, _, _
        updraft_sedimentation_matrix!(
            ᶜtridiagonal_matrix_scalar,
            Y,
            p,
            dtγ,
            name,
            ᶜ∂a∂z,
        )
        # Row i of I - ρ̂⁻¹ M_μ has the diagonal 1 - ρ̂⁻¹ M[i, i] and the
        # superdiagonal -ρ̂⁻¹ M[i, i+1]; the sweep starts with Δq = 0 above
        # the top cell.
        input = @. lazy(
            tuple(
                dtγ * ᶜinv_ρ̂ * vtt,
                1 - ᶜinv_ρ̂ * getindex(ᶜtridiagonal_matrix_scalar, 0),
                -(ᶜinv_ρ̂ * getindex(ᶜtridiagonal_matrix_scalar, 1)),
            ),
        )
        Operators.column_accumulate!(
            ᶜΔq,
            input;
            init = zero_FT,
            reverse = true,
        ) do Δq_above, (rhs, diagonal, superdiagonal)
            return (rhs - superdiagonal * Δq_above) / diagonal
        end
        @. ᶜr += ᶜΔq / dtγ
    end
    return nothing
end
