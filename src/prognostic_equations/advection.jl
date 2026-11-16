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
    temperature minus its reference profile and `Π` is the Exner function, plus
    the vector-invariant term `(f³ + ω³) × u` with the vertical Coriolis
    parameter `ᶜf³` and the vertical vorticity `ω³ = wcurlₕ(uₕ)` (0 on 2D
    spaces).
  - `Yₜ.c.sgsʲs.:(j).ρa` and `Yₜ.c.sgsʲs.:(j).mse` for each updraft `j`, when
    `p.atmos.turbconv_model isa PrognosticEDMFX` (advective form for `mse`).

Reads `Y.c` and, from the cache `p`, the precomputed fields `ᶜu`, `ᶜK`, `ᶜp`, `ᶜT`,
`ᶜq_liq`, `ᶜq_ice`, `ᶜq_tot_nonneg`, `ᶜh_tot` (and `ᶜuʲs` for `PrognosticEDMFX`) and
the geopotential `ᶜΦ` and Coriolis parameter `ᶜf³` from `p.core`, in the slab loop of
`horizontal_advection_tendency!`. Called from `remaining_tendency!` (through
`horizontal_advection_tendency!`), i.e., placed in the explicit part of the IMEX
splitting. Returns `nothing`. The continuous-form equations are documented on the
"Equations" page of the docs (`docs/src/equations.md`).
"""
NVTX.@annotate function horizontal_dynamics_tendency!(Yₜ, Y, p, t)
    horizontal_advection_tendency!(Yₜ, nothing, Y, p, t)
    return nothing
end

"""
    horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)

Add the explicit horizontal advection tendencies for grid-mean tracers and for
EDMFX-subdomain tracers.

Increments (with a minus sign on each flux divergence):

  - Every grid-mean tracer `Yₜ.c.ρχ`: flux-form advection `∇ₕ⋅(ρu χ)` for all
    prognostic tracer variables in `Y.c` (identified by `gs_tracer_names`).
  - `Yₜ.c.sgsʲs.:(j).q_tot`, when `p.atmos.turbconv_model isa PrognosticEDMFX`:
    advective-form advection with the updraft velocity `ᶜuʲs`.
  - Every auto-discovered SGS tracer in `Yₜ.c.sgsʲs.:(j)` (microphysics species and
    user-defined passive tracers, from `sgs_tracer_names`), also in advective form.

Reads `Y.c`, the precomputed velocities `ᶜu` (and `ᶜuʲs` for `PrognosticEDMFX`) from
`p`, in the slab loop of `horizontal_advection_tendency!`. `t` is unused. Called
from `remaining_tendency!` (through `horizontal_advection_tendency!`) with the
limited tendency vector `Yₜ_lim`, so these tendencies pass through the tracer
limiters. Returns `nothing`.
"""
NVTX.@annotate function horizontal_tracer_advection_tendency!(Yₜ, Y, p, t)
    horizontal_advection_tendency!(nothing, Yₜ, Y, p, t)
    return nothing
end

"""
    horizontal_advection_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Add the tendencies of `horizontal_dynamics_tendency!` to `Yₜ` and those of
`horizontal_tracer_advection_tendency!` to `Yₜ_lim` in one fused slab loop, which
reads `Y.c` and the precomputed fields once. Either tendency vector can be
`nothing`, which skips its tendencies. `t` is unused. Returns `nothing`.
"""
NVTX.@annotate function horizontal_advection_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    (; turbconv_model) = p.atmos
    (; ᶜu) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    cp_d = thermo_params.cp_d
    # The vertical vorticity vanishes on 2D spaces.
    is_3d = Val(eltype(Fields.coordinate_field(Y.c)) <: Geometry.Abstract3DPoint)
    ᶜuʲs = turbconv_model isa PrognosticEDMFX ? p.precomputed.ᶜuʲs : ()
    # An absent tendency vector, and the inputs that only its tendencies read,
    # are empty tuples.
    Yₜc = isnothing(Yₜ) ? () : Yₜ.c
    Yₜ_limc = isnothing(Yₜ_lim) ? () : Yₜ_lim.c
    ᶜdynamics_inputs =
        isnothing(Yₜ) ? () :
        (
            p.precomputed.ᶜK, p.precomputed.ᶜp, p.precomputed.ᶜT,
            p.precomputed.ᶜq_tot_nonneg, p.precomputed.ᶜq_liq,
            p.precomputed.ᶜq_ice, p.precomputed.ᶜh_tot, p.core.ᶜΦ, p.core.ᶜf³,
        )
    ρχ_names = isnothing(Yₜ_lim) ? () : gs_tracer_names(Y)
    χ_names =
        isnothing(Yₜ_lim) || !(turbconv_model isa PrognosticEDMFX) ? () :
        (@name(q_tot), sgs_tracer_names(Y)...)

    DataLayouts.foreach_slab(
        Yₜc, Yₜ_limc, Y.c, ᶜu, ᶜuʲs, ᶜdynamics_inputs,
    ) do Yₜc, Yₜ_limc, Yc, ᶜu, ᶜuʲs, ᶜdynamics_inputs
        (ᶜu, ᶜuʲs, ᶜdynamics_inputs) = register_copy((ᶜu, ᶜuʲs, ᶜdynamics_inputs))
        ᶜρu = @. Yc.ρ * ᶜu
        MatrixFields.unrolled_foreach(ρχ_names) do ρχ_name
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜ_limc, ρχ_name)
            ᶜχ = @. specific(ᶜρχ, Yc.ρ)
            ᶜdivₕρuχ = @. split_divₕ(ᶜρu, ᶜχ)
            @. ᶜρχₜ -= ᶜdivₕρuχ
        end
        if ᶜdynamics_inputs !== ()
            (ᶜK, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜh_tot, ᶜΦ, ᶜf³) =
                ᶜdynamics_inputs
            @. Yₜc.ρ -= split_divₕ(ᶜρu, 1)
            ᶜh = register_copy(ᶜh_tot)
            ᶜdivₕρuh = @. split_divₕ(ᶜρu, ᶜh)
            @. Yₜc.ρe_tot -= ᶜdivₕρuh
            if use_prognostic_tke(turbconv_model)
                ᶜtke = @. specific(Yc.ρtke, Yc.ρ)
                ᶜdivₕρutke = @. split_divₕ(ᶜρu, ᶜtke)
                @. Yₜc.ρtke -= ᶜdivₕρutke
            end
            # split form pressure gradient: 0.5 * cp_d * [θv ∇Π + ∇(θv Π) - Π∇θv]
            ᶜΦ_r = @inline @. lazy(phi_r(thermo_params, ᶜp))
            ᶜθ_v = @. theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
            ᶜθ_v_diff = @. ᶜθ_v - theta_vr(thermo_params, ᶜp)
            ᶜΠ = @. TD.exner_given_pressure(thermo_params, ᶜp)
            # The vertical Coriolis parameter and vorticity
            ᶜfω³ = is_3d isa Val{true} ? (@. ᶜf³ + wcurlₕ(Yc.uₕ)) : ᶜf³
            @. Yₜc.uₕ -=
                C12(
                    gradₕ(ᶜK + ᶜΦ - ᶜΦ_r) +
                    cp_d *
                    (
                        ᶜθ_v_diff * gradₕ(ᶜΠ) + gradₕ(ᶜθ_v_diff * ᶜΠ) -
                        ᶜΠ * gradₕ(ᶜθ_v_diff)
                    ) / 2,
                ) + ᶜfω³ × CT12(ᶜu)
            # Without the C12(), the right-hand side would be a C1 or C2 in 2D space.
        end
        turbconv_model isa PrognosticEDMFX || return nothing
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            ᶜsgsʲ = Yc.sgsʲs.:($j)
            ᶜuʲ = ᶜuʲs.:($j)
            ᶜdivₕuʲ = @. split_divₕ(ᶜuʲ, 1)
            if ᶜdynamics_inputs !== ()
                ᶜsgsʲₜ = Yₜc.sgsʲs.:($j)
                @. ᶜsgsʲₜ.ρa -= split_divₕ(ᶜsgsʲ.ρa * ᶜuʲ, 1)
                # Advective form, sharing the divergence of the updraft velocity
                ᶜmseʲ = register_copy(ᶜsgsʲ.mse)
                ᶜmse_advʲ = @. split_divₕ(ᶜuʲ, ᶜmseʲ) - ᶜmseʲ * ᶜdivₕuʲ
                @. ᶜsgsʲₜ.mse -= ᶜmse_advʲ
            end
            # Total water, then the auto-discovered SGS tracers (microphysics
            # species and any user-defined passive tracers)
            MatrixFields.unrolled_foreach(χ_names) do χ_name
                @inline
                ᶜχʲ = register_copy(MatrixFields.get_field(ᶜsgsʲ, χ_name))
                ᶜχʲₜ = MatrixFields.get_field(Yₜ_limc.sgsʲs.:($j), χ_name)
                ᶜadvʲ = @. split_divₕ(ᶜuʲ, ᶜχʲ) - ᶜχʲ * ᶜdivₕuʲ
                @. ᶜχʲₜ -= ᶜadvʲ
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
@inline function ᶜρq_tot_vertical_transport_bc(flow::PrescribedFlow, thermo_params, t, ᶠu³)
    ρu₃qₜ_sfc_bc = get_ρu₃qₜ_surface(flow, thermo_params, t)
    ᶜadvdivᵥ = Operators.DivergenceF2C(; bottom = Operators.SetValue(ρu₃qₜ_sfc_bc))
    return @. lazy(-(ᶜadvdivᵥ(zero(ᶠu³))))
end

"""
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)

Add the explicitly treated vertical advection tendencies: vector-invariant momentum
advection (with Coriolis and vorticity terms) and the prescribed-flow moisture
boundary condition.

Increments:

  - `Yₜ.c.ρq_tot`: the prescribed-flow surface flux from
    `ᶜρq_tot_vertical_transport_bc`, when moisture is present.
  - `Yₜ.c.uₕ` and `Yₜ.f.u₃`: vector-invariant vertical advection built from the
    vorticity `ᶠω¹²`, the Coriolis field `ᶠf¹²`, and the kinetic-energy gradient
    `∇ᵥK`; the shallow-atmosphere branch (`ᶠf¹² === nothing`) omits the horizontal
    Coriolis components. The vertical vorticity and Coriolis terms of `Yₜ.c.uₕ`
    are added by `horizontal_dynamics_tendency!`.
  - `Yₜ.f.sgsʲs.:(j).u₃` for each prognostic updraft: the analogous vector-invariant
    terms with updraft vorticities and kinetic energies.

Reads `Y`, the core field `ᶠf¹²`, the precomputed `ᶜu`, `ᶠu³`, `ᶜK` (and
`ᶜuʲs`, `ᶜKʲs`, `ᶠKᵥʲs` for prognostic updrafts), and scratch space. Called from
`remaining_tendency!`. Returns `nothing`.
"""
NVTX.@annotate function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; turbconv_model, prescribed_flow) = p.atmos
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    n_val = Val(n)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    (; ᶠf¹²) = p.core
    (; ᶜu, ᶠu³, ᶜK) = p.precomputed
    (; ᶜuʲs, ᶜKʲs, ᶠKᵥʲs) = n > 0 ? p.precomputed : all_nothing
    thermo_params = CAP.thermodynamics_params(p.params)

    ᶠω¹² = p.scratch.ᶠtemp_CT12
    ᶠω¹²ʲs = p.scratch.ᶠtemp_CT12ʲs

    # The vertical curl and its copies for the drafts in one column loop.
    if n > 0
        DataLayouts.foreach_column(ᶠω¹², ᶠω¹²ʲs, Y.c.uₕ) do ᶠω¹², ᶠω¹²ʲs, ᶜuₕ
            @. ᶠω¹² = ᶠcurlᵥ(ᶜuₕ)
            ntuple(n_val) do j
                @inline
                ᶠω¹²ʲ = getproperty(ᶠω¹²ʲs, j)
                @. ᶠω¹²ʲ = ᶠω¹²
            end
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

    # Passive non-water tracer vertical advection runs post-ARK.
    drafts = ntuple(
        j -> (ᶠω¹²ʲs.:($j), ᶜuʲs.:($j), ᶜKʲs.:($j), ᶠKᵥʲs.:($j), Yₜ.f.sgsʲs.:($j).u₃),
        n_val,
    )

    DataLayouts.foreach_column(
        Yₜ.c, Yₜ.f, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶜu, ᶜK, ᶠω¹², drafts,
        something(ᶠf¹², ()),
    ) do Yₜc, Yₜf, Yc, ᶜJ, ᶠJ, ᶠu³, ᶜu, ᶜK, ᶠω¹², drafts, ᶠf¹²
        ᶜρ = Yc.ρ

        # In a deep atmosphere, the vorticity carries the horizontal Coriolis term.
        ᶠω¹²_tot = ᶠf¹² === () ? ᶠω¹² : (@. lazy(ᶠf¹² + ᶠω¹²))
        ᶜu¹² = @. CT12(ᶜu)
        @. Yₜc.uₕ -= ᶜinterp(ᶠω¹²_tot × (ᶠinterp(ᶜρ * ᶜJ) * ᶠu³)) / (ᶜρ * ᶜJ)
        @. Yₜf.u₃ -= ᶠω¹²_tot × ᶠinterp(ᶜu¹²) + ᶠgradᵥ(ᶜK)
        MatrixFields.unrolled_foreach(drafts) do (ᶠω¹²ʲ, ᶜuʲ, ᶜKʲ, ᶠKᵥʲ, ᶠu₃ʲₜ)
            @inline
            ᶠω¹²ʲ_tot = ᶠf¹² === () ? ᶠω¹²ʲ : (@. lazy(ᶠf¹² + ᶠω¹²ʲ))
            @. ᶠu₃ʲₜ -=
                ᶠω¹²ʲ_tot × ᶠinterp(CT12(ᶜuʲ)) + ᶠgradᵥ(ᶜKʲ - ᶜinterp(ᶠKᵥʲ))
        end
        return nothing
    end
end
