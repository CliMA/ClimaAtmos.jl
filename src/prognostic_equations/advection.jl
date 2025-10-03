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
- Horizontal advection of turbulent kinetic energy (`ρatke⁰`) if used.
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
`Yₜ.c.sgsʲs` and `Yₜ.c.sgs⁰` if applicable.
"""
NVTX.@annotate function horizontal_dynamics_tendency!(Yₜ, Y, p, t)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜΦ) = p.core
    (; ᶜu, ᶜK, ᶜp, ᶜts) = p.precomputed
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)

    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜuʲs) = p.precomputed
    end

    @. Yₜ.c.ρ -= wdivₕ(Y.c.ρ * ᶜu)
    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).ρa -= wdivₕ(Y.c.sgsʲs.:($$j).ρa * ᶜuʲs.:($$j))
        end
    end

    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(
            thermo_params,
            ᶜts,
            specific(Y.c.ρe_tot, Y.c.ρ),
        ),
    )
    @. Yₜ.c.ρe_tot -= wdivₕ(Y.c.ρ * ᶜh_tot * ᶜu)

    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).mse -=
                wdivₕ(Y.c.sgsʲs.:($$j).mse * ᶜuʲs.:($$j)) -
                Y.c.sgsʲs.:($$j).mse * wdivₕ(ᶜuʲs.:($$j))
        end
    end

    if use_prognostic_tke(p.atmos.turbconv_model)
        if p.atmos.turbconv_model isa EDOnlyEDMFX
            ᶜu_for_tke_advection = ᶜu
        elseif p.atmos.turbconv_model isa AbstractEDMF
            ᶜu_for_tke_advection = p.precomputed.ᶜu⁰
        else
            error(
                "Unsupported turbconv_model type for TKE advection: $(typeof(p.atmos.turbconv_model))",
            )
        end
        @. Yₜ.c.sgs⁰.ρatke -= wdivₕ(Y.c.sgs⁰.ρatke * ᶜu_for_tke_advection)

    end

    @. Yₜ.c.uₕ -= C12(gradₕ(ᶜp) / Y.c.ρ + gradₕ(ᶜK + ᶜΦ))
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
- Horizontal advection for other EDMFX updraft moisture species (`q_liqʲ`, `q_iceʲ`,
  `q_raiʲ`, `q_snoʲ`) if using a `NonEquilMoistModel` and `Microphysics1Moment`
  precipitation model. If the `Microphysics2Moment` model is used instead, `n_liqʲ`` 
  and `n_raiʲ` are also advected.

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
        @. Yₜ.c.:($$ρχ_name) -= wdivₕ(Y.c.:($$ρχ_name) * ᶜu)
    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).q_tot -=
                wdivₕ(Y.c.sgsʲs.:($$j).q_tot * ᶜuʲs.:($$j)) -
                Y.c.sgsʲs.:($$j).q_tot * wdivₕ(ᶜuʲs.:($$j))
            if p.atmos.moisture_model isa NonEquilMoistModel && (
                p.atmos.microphysics_model isa Microphysics1Moment ||
                p.atmos.microphysics_model isa Microphysics2Moment
            )
                @. Yₜ.c.sgsʲs.:($$j).q_liq -=
                    wdivₕ(Y.c.sgsʲs.:($$j).q_liq * ᶜuʲs.:($$j)) -
                    Y.c.sgsʲs.:($$j).q_liq * wdivₕ(ᶜuʲs.:($$j))
                @. Yₜ.c.sgsʲs.:($$j).q_ice -=
                    wdivₕ(Y.c.sgsʲs.:($$j).q_ice * ᶜuʲs.:($$j)) -
                    Y.c.sgsʲs.:($$j).q_ice * wdivₕ(ᶜuʲs.:($$j))
                @. Yₜ.c.sgsʲs.:($$j).q_rai -=
                    wdivₕ(Y.c.sgsʲs.:($$j).q_rai * ᶜuʲs.:($$j)) -
                    Y.c.sgsʲs.:($$j).q_rai * wdivₕ(ᶜuʲs.:($$j))
                @. Yₜ.c.sgsʲs.:($$j).q_sno -=
                    wdivₕ(Y.c.sgsʲs.:($$j).q_sno * ᶜuʲs.:($$j)) -
                    Y.c.sgsʲs.:($$j).q_sno * wdivₕ(ᶜuʲs.:($$j))
            end
            if p.atmos.moisture_model isa NonEquilMoistModel &&
               p.atmos.microphysics_model isa Microphysics2Moment
                @. Yₜ.c.sgsʲs.:($$j).n_liq -=
                    wdivₕ(Y.c.sgsʲs.:($$j).n_liq * ᶜuʲs.:($$j)) -
                    Y.c.sgsʲs.:($$j).n_liq * wdivₕ(ᶜuʲs.:($$j))
                @. Yₜ.c.sgsʲs.:($$j).n_rai -=
                    wdivₕ(Y.c.sgsʲs.:($$j).n_rai * ᶜuʲs.:($$j)) -
                    Y.c.sgsʲs.:($$j).n_rai * wdivₕ(ᶜuʲs.:($$j))
            end
        end
    end
    return nothing
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
- Vertical advection of grid-mean TKE (`ρatke⁰`) if `use_prognostic_tke` is true.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters, core fields (e.g., `ᶜf³`, `ᶠf¹²`, `ᶜΦ`),
       precomputed fields (e.g., `ᶜu`, `ᶠu³`, `ᶜK`, EDMF velocities/TKE if applicable),
       atmospheric model settings (`p.atmos.numerics` for upwinding schemes),
       and scratch space.
- `t`: Current simulation time (not directly used in calculations).

Modifies `Yₜ.c` (various tracers, `ρe_tot`, `ρq_tot`, `uₕ`), `Yₜ.f.u₃`,
`Yₜ.f.sgsʲs` (updraft `u₃`), and `Yₜ.c.sgs⁰.ρatke` as applicable.
"""
NVTX.@annotate function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; turbconv_model) = p.atmos
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    advect_tke = use_prognostic_tke(turbconv_model)
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜf³, ᶠf¹², ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶜK, ᶜts) = p.precomputed
    (; edmfx_upwinding) = n > 0 || advect_tke ? p.atmos.numerics : all_nothing
    (; ᶜuʲs, ᶜKʲs, ᶠKᵥʲs) = n > 0 ? p.precomputed : all_nothing
    (; energy_upwinding, tracer_upwinding) = p.atmos.numerics
    thermo_params = CAP.thermodynamics_params(p.params)

    ᶠu³⁰ =
        advect_tke ?
        (
            turbconv_model isa EDOnlyEDMFX ? p.precomputed.ᶠu³ :
            p.precomputed.ᶠu³⁰
        ) : nothing
    ᶜρa⁰ =
        advect_tke ?
        (
            turbconv_model isa PrognosticEDMFX ?
            (@. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))) : Y.c.ρ
        ) : nothing
    ᶜρ⁰ = if advect_tke
        if n > 0
            (; ᶜts⁰) = p.precomputed
            @. lazy(TD.air_density(thermo_params, ᶜts⁰))
        else
            Y.c.ρ
        end
    else
        nothing
    end
    ᶜtke⁰ =
        advect_tke ?
        (@. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))) :
        nothing
    ᶜa_scalar = p.scratch.ᶜtemp_scalar
    ᶜω³ = p.scratch.ᶜtemp_CT3
    ᶠω¹² = p.scratch.ᶠtemp_CT12
    ᶠω¹²ʲs = p.scratch.ᶠtemp_CT12ʲs

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = wcurlₕ(Y.c.uₕ)
    elseif point_type <: Geometry.Abstract2DPoint
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

    # Full vertical advection of passive tracers (like liq, rai, etc) ...
    # If sgs_mass_flux is true, the advection term is computed from the sum of SGS fluxes
    if p.atmos.edmfx_model.sgs_mass_flux isa Val{false}
        foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
            if !(ρχ_name in (@name(ρe_tot), @name(ρq_tot)))
                ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
                vtt = vertical_transport(ᶜρ, ᶠu³, ᶜχ, float(dt), tracer_upwinding)
                @. ᶜρχₜ += vtt
            end
        end
    end
    # ... and upwinding correction of energy and total water.
    # (The central advection of energy and total water is done implicitly.)
    if energy_upwinding != Val(:none)
        ᶜh_tot = @. lazy(
            TD.total_specific_enthalpy(
                thermo_params,
                ᶜts,
                specific(Y.c.ρe_tot, Y.c.ρ),
            ),
        )
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, float(dt), energy_upwinding)
        vtt_central = vertical_transport(ᶜρ, ᶠu³, ᶜh_tot, float(dt), Val(:none))
        @. Yₜ.c.ρe_tot += vtt - vtt_central
    end

    if !(p.atmos.moisture_model isa DryModel) && tracer_upwinding != Val(:none)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        vtt = vertical_transport(ᶜρ, ᶠu³, ᶜq_tot, float(dt), tracer_upwinding)
        vtt_central = vertical_transport(ᶜρ, ᶠu³, ᶜq_tot, float(dt), Val(:none))
        @. Yₜ.c.ρq_tot += vtt - vtt_central
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
        @. ᶜa_scalar = ᶜtke⁰ * draft_area(ᶜρa⁰, ᶜρ⁰)
        vtt = vertical_transport(ᶜρ⁰, ᶠu³⁰, ᶜa_scalar, dt, edmfx_upwinding)
        @. Yₜ.c.sgs⁰.ρatke += vtt
    end
end

"""
    edmfx_sgs_vertical_advection_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

Computes tendencies due to vertical advection and buoyancy for EDMFX subgrid-scale
(SGS) updraft prognostic variables.

This function handles:
- Vertical advection of updraft density-area product (`ρaʲ`).
- Vertical advection of updraft moist static energy (`mseʲ`) and total specific humidity (`q_totʲ`).
- Vertical advection of other updraft moisture species (`q_liqʲ`, `q_iceʲ`, `q_raiʲ`, `q_snoʲ`)
  if using a `NonEquilMoistModel` and `Microphysics1Moment` precipitation model. If the `Microphysics2Moment` 
  model is used instead, `n_liqʲ` and `n_raiʲ` are also advected.
- Buoyancy forcing terms in the updraft vertical momentum (`u₃ʲ`) equation, including
  adjustments for non-hydrostatic pressure.
- Buoyancy production/conversion terms in the updraft `mseʲ` equation.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters (`p.params`), time step `dt`, core fields (`ᶠgradᵥ_ᶜΦ`),
       precomputed EDMF fields (e.g., `ᶠu³ʲs`, `ᶜρʲs`), atmospheric model settings
       (`p.atmos.numerics.edmfx_upwinding`), and scratch space.
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
    (; params) = p
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    (; dt) = p
    (; edmfx_upwinding, tracer_upwinding) = p.atmos.numerics
    (; ᶠu³ʲs, ᶠKᵥʲs, ᶜρʲs) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core

    turbconv_params = CAP.turbconv_params(params)
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶜu₃ʲ = p.scratch.ᶜtemp_C3
    ᶜKᵥʲ = p.scratch.ᶜtemp_scalar_2
    ᶠρ_diff = p.scratch.ᶠtemp_scalar
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J

    for j in 1:n
        # TODO: Add a biased GradientF2F operator in ClimaCore
        @. ᶜu₃ʲ = ᶜinterp(Y.f.sgsʲs.:($$j).u₃)
        @. ᶜKᵥʲ = ifelse(
            ᶜu₃ʲ.components.data.:1 > 0,
            ᶜleft_bias(ᶠKᵥʲs.:($$j)),
            ᶜright_bias(ᶠKᵥʲs.:($$j)),
        )
        # Use downwind density for buoyancy forcing.
        # This ensures that the velocity perturbation responds with the correct phase
        # to oppose density anomalies, preventing spurious growth of grid-scale modes.
        # (Upwind choice would reinforce the anomaly and cause instability.)
        @. ᶠρ_diff = ifelse(
            Y.f.sgsʲs.:($$j).u₃.components.data.:1 < 0,
            ᶠleft_bias((ᶜρʲs.:($$j) - Y.c.ρ) / ᶜρʲs.:($$j)),
            ᶠright_bias((ᶜρʲs.:($$j) - Y.c.ρ) / ᶜρʲs.:($$j)),
        )
        # For the updraft u_3 equation, we assume the grid-mean to be hydrostatic
        # and calcuate the buoyancy term relative to the grid-mean density.
        # We also include the buoyancy term in the nonhydrostatic pressure closure here.
        @. Yₜ.f.sgsʲs.:($$j).u₃ -=
            (1 - α_b) * (ᶠρ_diff * ᶠgradᵥ_ᶜΦ) + ᶠgradᵥ(ᶜKᵥʲ)

        # buoyancy term in mse equation
        @. Yₜ.c.sgsʲs.:($$j).mse +=
            adjoint(CT3(ᶜinterp(Y.f.sgsʲs.:($$j).u₃))) *
            (ᶜρʲs.:($$j) - Y.c.ρ) *
            ᶜgradᵥ(CAP.grav(params) * ᶠz) / ᶜρʲs.:($$j)
    end

    for j in 1:n
        ᶜa = (@. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))))

        # Flux form vertical advection of area farction with the grid mean velocity
        ᶜ∂ρ∂t = vertical_transport(ᶜρʲs.:($j), ᶠu³ʲs.:($j), ᶜa, dt, edmfx_upwinding)
        @. Yₜ.c.sgsʲs.:($$j).ρa += ᶜ∂ρ∂t

        # Advective form advection of mse and q_tot with the grid mean velocity
        # Note: This allocates because the function is too long
        va = vertical_advection(
            ᶠu³ʲs.:($j),
            Y.c.sgsʲs.:($j).mse,
            edmfx_upwinding,
        )
        @. Yₜ.c.sgsʲs.:($$j).mse += va

        va = vertical_advection(
            ᶠu³ʲs.:($j),
            Y.c.sgsʲs.:($j).q_tot,
            edmfx_upwinding,
        )
        @. Yₜ.c.sgsʲs.:($$j).q_tot += va

        if p.atmos.moisture_model isa NonEquilMoistModel && (
            p.atmos.microphysics_model isa Microphysics1Moment ||
            p.atmos.microphysics_model isa Microphysics2Moment
        )
            # TODO - add precipitation and cloud sedimentation in implicit solver/tendency with if/else
            # TODO - make it work for multiple updrafts
            if j > 1
                error("Below code doesn't work for multiple updrafts")
            end
            FT = eltype(p.params)
            thp = CAP.thermodynamics_params(params)
            (; ᶜΦ) = p.core
            (; ᶜtsʲs) = p.precomputed
            ᶜ∂ρ∂t_sed = p.scratch.ᶜtemp_scalar_3
            @. ᶜ∂ρ∂t_sed = 0

            ᶜinv_ρ̂ = (@. lazy(
                specific(
                    FT(1),
                    Y.c.sgsʲs.:($$j).ρa,
                    FT(0),
                    ᶜρʲs.:($$j),
                    turbconv_model,
                ),
            ))

            # Sedimentation
            # TODO - lazify ᶜwₗʲs computation. No need to cache it.
            sgs_microphysics_tracers = (
                (@name(c.sgsʲs.:(1).q_liq), @name(q_liq), @name(ᶜwₗʲs.:(1))),
                (@name(c.sgsʲs.:(1).q_ice), @name(q_ice), @name(ᶜwᵢʲs.:(1))),
                (@name(c.sgsʲs.:(1).q_rai), @name(q_rai), @name(ᶜwᵣʲs.:(1))),
                (@name(c.sgsʲs.:(1).q_sno), @name(q_sno), @name(ᶜwₛʲs.:(1))),
            )

            MatrixFields.unrolled_foreach(
                sgs_microphysics_tracers,
            ) do (qʲ_name, name, wʲ_name)
                MatrixFields.has_field(Y, qʲ_name) || return

                ᶜqʲ = MatrixFields.get_field(Y, qʲ_name)
                ᶜqʲₜ = MatrixFields.get_field(Yₜ, qʲ_name)
                ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)
                ᶜaqʲ = (@. lazy(ᶜa * ᶜqʲ))

                # Flux form advection of tracers with updraft velocity
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³ʲs.:($j),
                    ᶜaqʲ,
                    dt,
                    tracer_upwinding,
                )
                @. ᶜqʲₜ += ᶜinv_ρ̂ * (vtt - ᶜqʲ * ᶜ∂ρ∂t)

                # Flux form sedimentation of tracers
                vtt = vertical_transport_sedimentation(
                    ᶜρʲs.:($j),
                    ᶜwʲ,
                    ᶜaqʲ,
                    ᶠJ,
                )
                @. ᶜqʲₜ += ᶜinv_ρ̂ * vtt
                @. Yₜ.c.sgsʲs.:($$j).q_tot += ᶜinv_ρ̂ * vtt
                @. ᶜ∂ρ∂t_sed += vtt

                # Flux form sedimentation of energy
                if name in (@name(q_liq), @name(q_rai))
                    ᶜmse_li = (@. lazy(
                        TD.internal_energy_liquid(thp, ᶜtsʲs.:($$j)) + ᶜΦ,
                    ))
                elseif name in (@name(q_ice), @name(q_sno))
                    ᶜmse_li = (@. lazy(
                        TD.internal_energy_ice(thp, ᶜtsʲs.:($$j)) + ᶜΦ,
                    ))
                else
                    error("Unsupported moisture tracer variable")
                end
                vtt = vertical_transport_sedimentation(
                    ᶜρʲs.:($j),
                    ᶜwʲ,
                    ᶜaqʲ .* ᶜmse_li,
                    ᶠJ,
                )
                @. Yₜ.c.sgsʲs.:($$j).mse += ᶜinv_ρ̂ * vtt
            end

            # Contribution of density variation due to sedimentation
            @. Yₜ.c.sgsʲs.:($$j).ρa += ᶜ∂ρ∂t_sed
            @. Yₜ.c.sgsʲs.:($$j).mse -= ᶜinv_ρ̂ * Y.c.sgsʲs.:($$j).mse * ᶜ∂ρ∂t_sed
            @. Yₜ.c.sgsʲs.:($$j).q_tot -= ᶜinv_ρ̂ * Y.c.sgsʲs.:($$j).q_tot * ᶜ∂ρ∂t_sed
            @. Yₜ.c.sgsʲs.:($$j).q_liq -= ᶜinv_ρ̂ * Y.c.sgsʲs.:($$j).q_liq * ᶜ∂ρ∂t_sed
            @. Yₜ.c.sgsʲs.:($$j).q_ice -= ᶜinv_ρ̂ * Y.c.sgsʲs.:($$j).q_ice * ᶜ∂ρ∂t_sed
            @. Yₜ.c.sgsʲs.:($$j).q_rai -= ᶜinv_ρ̂ * Y.c.sgsʲs.:($$j).q_rai * ᶜ∂ρ∂t_sed
            @. Yₜ.c.sgsʲs.:($$j).q_sno -= ᶜinv_ρ̂ * Y.c.sgsʲs.:($$j).q_sno * ᶜ∂ρ∂t_sed

        end

        # Sedimentation of number concentrations for 2M microphysics
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.microphysics_model isa Microphysics2Moment

            # TODO - add precipitation and cloud sedimentation in implicit solver/tendency with if/else
            # TODO - make it work for multiple updrafts
            if j > 1
                error("Below code doesn't work for multiple updrafts")
            end

            # Sedimentation velocities for microphysics number concentrations
            # (or any tracers that does not directly participate in variations of q_tot and mse)
            sgs_microphysics_tracers = (
                (@name(c.sgsʲs.:(1).n_liq), @name(ᶜwₙₗʲs.:(1))),
                (@name(c.sgsʲs.:(1).n_rai), @name(ᶜwₙᵣʲs.:(1))),
            )

            MatrixFields.unrolled_foreach(
                sgs_microphysics_tracers,
            ) do (χʲ_name, wʲ_name)
                MatrixFields.has_field(Y, χʲ_name) || return

                ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                ᶜχʲₜ = MatrixFields.get_field(Yₜ, χʲ_name)
                ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)
                ᶜaχʲ = (@. lazy(ᶜa * ᶜχʲ))

                # Flux form advection of tracers with updraft velocity
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³ʲs.:($j),
                    ᶜaχʲ,
                    dt,
                    tracer_upwinding,
                )
                @. ᶜχʲₜ += ᶜinv_ρ̂ * (vtt - ᶜχʲ * ᶜ∂ρ∂t)

                # Flux form sedimentation of tracers
                vtt = vertical_transport_sedimentation(
                    ᶜρʲs.:($j),
                    ᶜwʲ,
                    ᶜaχʲ,
                    ᶠJ,
                )
                @. ᶜχʲₜ += ᶜinv_ρ̂ * vtt

                # Contribution of density variation due to sedimentation
                @. ᶜχʲₜ -= ᶜinv_ρ̂ * ᶜχʲ * ᶜ∂ρ∂t_sed
            end

        end

    end
end
