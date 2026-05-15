#####
##### Tendencies applied to the grid-mean atmospheric state due to subgrid-scale (SGS)
##### fluxes computed by the EDMFX scheme
#####

# Multi-shot counter for SSLT diagnostic 4.1.
# Counts calls to `edmfx_sgs_mass_flux_tendency!`. The SSLT block dumps stats
# only when the post-increment value is in SSLT_DIAG_FIRE_AT. Skipping count=1
# avoids the t=0 build_cache call where all SSLT state is still zero.
# Sampling at counts 2, 6, 12, 30 catches early-step evolution (the blowup has
# an e-fold of ~1 h ≈ tens of tendency calls) so we can see whether (χʲ − χ̄)
# stays O(χ̄) or blows up.
const SSLT_DIAG_COUNTER = Ref(0)
const SSLT_DIAG_FIRE_AT = (2, 6, 12, 30, 100)

"""
    edmf_sgs_advection_handles(atmos, ρχ_name) -> Bool

Return `true` if mean-flow vertical advection should be **skipped** for
`ρχ_name` because another mechanism already provides the full vertical flux.

Two cases:

1. `ρq_tot` is always skipped — its vertical advection is handled implicitly
   (central scheme) in `implicit_vertical_advection_tendency!`; adding the
   explicit upwinding on top would double-count it.

2. `PrognosticEDMFX` with `sgs_mass_flux = true` replaces the grid-mean flux
   with a full updraft + environment decomposition, so the standard
   `vertical_transport` call must be suppressed for those tracers.

`DiagnosticEDMFX` is intentionally **not** in this list: it uses a deviatoric
(`wʲ − w̄`) × (`χʲ − χ̄`) SGS correction that is additive to the mean-flow
flux, so mean-flow advection must continue running for all tracers under
DiagnosticEDMFX — including microphysics and sea salt aerosols.
"""
function edmf_sgs_advection_handles(atmos, ρχ_name)
    # ρq_tot vertical advection is always handled implicitly — never add
    # the explicit upwinding version on top regardless of EDMF state.
    ρχ_name == @name(ρq_tot) && return true

    # Only PrognosticEDMFX replaces (rather than augments) grid-mean advection.
    atmos.turbconv_model isa PrognosticEDMFX || return false
    atmos.edmfx_model isa EDMFXModel || return false
    atmos.edmfx_model.sgs_mass_flux isa Val{true} || return false

    # Microphysics tracers carried by PrognosticEDMFX full-transport updrafts.
    if atmos.microphysics_model isa Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        ρχ_name in (
            @name(ρq_lcl), @name(ρq_icl), @name(ρq_rai), @name(ρq_sno),
            @name(ρn_lcl), @name(ρn_rai),
        ) && return true
    end

    # Sea salt aerosols are NOT listed here: under DiagnosticEDMFX (the current
    # config) the SGS flux is additive, so mean-flow advection must stay on.
    # If PrognosticEDMFX sea salt is ever implemented, add the bin names here.

    return false
end

"""
    edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Computes and applies tendencies to the grid-mean prognostic variables due to the
divergence of subgrid-scale (SGS) mass fluxes from EDMFX updrafts and the environment.

This involves terms of the form `- ∂(ρₖ aₖ w′ₖ ϕ′ₖ)/∂z`, where `k` denotes
an SGS component (updraft `j` or environment `0`), `aₖ` is the area fraction,
`w′ₖ` is the vertical velocity deviation from the grid mean, and `ϕ′ₖ` is the
deviation of a conserved variable `ϕ` (such as total enthalpy or specific humidity)
from its grid-mean value. These terms represent the redistribution of energy and tracers
by the resolved SGS circulations relative to the grid mean flow.

The specific implementation depends on the `turbconv_model` (e.g., `PrognosticEDMFX`
or `DiagnosticEDMFX`). A generic fallback doing nothing is also provided.
The function modifies `Yₜ.c` (grid-mean tendencies) in place.

Arguments:
- `Yₜ`: The tendency state vector for grid-mean variables.
- `Y`: The current state vector (used for grid-mean and SGS properties).
- `p`: Cache containing parameters, precomputed fields, atmospheric model settings,
       and scratch space.
- `t`: Current simulation time.
- `turbconv_model`: The turbulence convection model instance.
"""
edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding, edmfx_tracer_upwinding) = p.atmos.numerics
    (; ᶜp, ᶠu³) = p.precomputed
    (; ᶠu³ʲs, ᶜKʲs, ᶜρʲs) = p.precomputed
    (; ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; dt) = p

    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜρ⁰ = @. lazy(
        TD.air_density(
            thermo_params,
            ᶜT⁰,
            ᶜp,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
        ),
    )
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))

    if p.atmos.edmfx_model.sgs_mass_flux isa Val{true}

        # Enthalpy fluxes. First sum up the draft fluxes
        # TODO: Isolate assembly of flux term pattern to a function and
        # reuse (both in prognostic and diagnostic EDMFX)
        # [best after removal of precomputed quantities]
        ᶠu³_diff = p.scratch.ᶠtemp_CT3
        ᶜa_scalar = p.scratch.ᶜtemp_scalar
        (; ᶜh_tot) = p.precomputed
        for j in 1:n
            @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
            @. ᶜa_scalar =
                (Y.c.sgsʲs.:($$j).mse + ᶜKʲs.:($$j) - ᶜh_tot) *
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
            vtt = vertical_transport(
                ᶜρʲs.:($j),
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρe_tot += vtt
        end
        # Add the environment fluxes
        @. ᶠu³_diff = ᶠu³⁰ - ᶠu³
        ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
        @. ᶜa_scalar = (ᶜmse⁰ + ᶜK⁰ - ᶜh_tot) * draft_area(ᶜρa⁰, ᶜρ⁰)
        vtt = vertical_transport(
            ᶜρ⁰,
            ᶠu³_diff,
            ᶜa_scalar,
            dt,
            edmfx_sgsflux_upwinding,
        )
        @. Yₜ.c.ρe_tot += vtt

        if !(p.atmos.microphysics_model isa DryModel)
            # Specific humidity fluxes: First sum up the draft fluxes
            for j in 1:n
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
                @. ᶜa_scalar =
                    (Y.c.sgsʲs.:($$j).q_tot - specific(Y.c.ρq_tot, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_tot += vtt
            end
            # Add the environment fluxes
            ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
            @. ᶠu³_diff = ᶠu³⁰ - ᶠu³
            @. ᶜa_scalar =
                (ᶜq_tot⁰ - specific(Y.c.ρq_tot, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρq_tot += vtt
        end

        # Microphysics tracers fluxes
        if p.atmos.microphysics_model isa
           Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
            microphysics_tracers = (
                (@name(c.ρq_lcl), @name(c.sgsʲs.:(1).q_lcl), @name(q_lcl)),
                (@name(c.ρq_icl), @name(c.sgsʲs.:(1).q_icl), @name(q_icl)),
                (@name(c.ρq_rai), @name(c.sgsʲs.:(1).q_rai), @name(q_rai)),
                (@name(c.ρq_sno), @name(c.sgsʲs.:(1).q_sno), @name(q_sno)),
                (@name(c.ρn_lcl), @name(c.sgsʲs.:(1).n_lcl), @name(n_lcl)),
                (@name(c.ρn_rai), @name(c.sgsʲs.:(1).n_rai), @name(n_rai)),
            )
            # TODO using unrolled_foreach here allocates! (breaks the flame tests
            # even though they use 0M microphysics)
            # MatrixFields.unrolled_foreach(
            #     microphysics_tracers,
            # ) do (ρχ_name, χʲ_name, _)
            for (ρχ_name, χʲ_name, _) in microphysics_tracers
                MatrixFields.has_field(Y, ρχ_name) || continue

                ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                @. ᶜa_scalar =
                    ᶜχʲ *
                    draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1))
                vtt = vertical_transport(
                    ᶜρʲs.:(1),
                    ᶠu³ʲs.:(1),
                    ᶜa_scalar,
                    dt,
                    edmfx_tracer_upwinding,
                )
                ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                @. ᶜρχₜ += vtt
            end
            # MatrixFields.unrolled_foreach(
            #     microphysics_tracers,
            # ) do (ρχ_name, _, χ_name)
            for (ρχ_name, _, χ_name) in microphysics_tracers
                MatrixFields.has_field(Y, ρχ_name) || continue

                ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
                @. ᶜa_scalar = ᶜχ⁰ * draft_area(ᶜρa⁰, ᶜρ⁰)
                vtt = vertical_transport(
                    ᶜρ⁰,
                    ᶠu³⁰,
                    ᶜa_scalar,
                    dt,
                    edmfx_tracer_upwinding,
                )
                ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                @. ᶜρχₜ += vtt
            end
        end
    end
    # TODO - add vertical momentum fluxes
    return nothing
end

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::DiagnosticEDMFX,
)

    turbconv_params = CAP.turbconv_params(p.params)
    a_max = CAP.max_area(turbconv_params)
    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding, edmfx_tracer_upwinding) = p.atmos.numerics
    (; ᶠu³) = p.precomputed
    (; ᶜρaʲs, ᶜρʲs, ᶠu³ʲs, ᶜKʲs, ᶜmseʲs, ᶜq_totʲs) = p.precomputed
    (; dt) = p
    ᶜJ = Fields.local_geometry_field(Y.c).J
    FT = eltype(Y)

    if p.atmos.edmfx_model.sgs_mass_flux isa Val{true}
        thermo_params = CAP.thermodynamics_params(p.params)
        # energy
        (; ᶜh_tot) = p.precomputed
        ᶠu³_diff = p.scratch.ᶠtemp_CT3
        ᶜa_scalar = p.scratch.ᶜtemp_scalar
        for j in 1:n
            @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
            @. ᶜa_scalar =
                (ᶜmseʲs.:($$j) + ᶜKʲs.:($$j) - ᶜh_tot) * min(
                    min(draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)), a_max),
                    FT(0.02) / max(
                        Geometry.WVector(ᶜinterp(ᶠu³_diff)).components.data.:1,
                        eps(FT),
                    ),
                )
            vtt = vertical_transport(
                ᶜρʲs.:($j),
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρe_tot += vtt
        end
        # TODO: add environment flux?

        if !(p.atmos.microphysics_model isa DryModel)
            # Specific humidity fluxes
            for j in 1:n
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
                # @. ᶜa_scalar =
                #     (ᶜq_totʲs.:($$j) - specific(Y.c.ρq_tot, Y.c.ρ) *
                #     draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j))
                # TODO: remove this filter when mass flux is treated implicitly
                @. ᶜa_scalar =
                    (ᶜq_totʲs.:($$j) - specific(Y.c.ρq_tot, Y.c.ρ)) * min(
                        min(draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)), a_max),
                        FT(0.02) / max(
                            Geometry.WVector(
                                ᶜinterp(ᶠu³_diff),
                            ).components.data.:1,
                            eps(FT),
                        ),
                    )
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_tot += vtt
            end
        end

        # Microphysics tracers fluxes
        if p.atmos.microphysics_model isa
           Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
            microphysics_tracers = (
                (@name(c.ρq_lcl), @name(ᶜq_lclʲs.:(1))),
                (@name(c.ρq_icl), @name(ᶜq_iclʲs.:(1))),
                (@name(c.ρq_rai), @name(ᶜq_raiʲs.:(1))),
                (@name(c.ρq_sno), @name(ᶜq_snoʲs.:(1))),
                (@name(c.ρn_lcl), @name(ᶜn_lclʲs.:(1))),
                (@name(c.ρn_rai), @name(ᶜn_raiʲs.:(1))),
            )
            # TODO using unrolled_foreach here allocates! (breaks the flame tests
            # even though they use 0M microphysics)
            # MatrixFields.unrolled_foreach(
            #     microphysics_tracers,
            # ) do (ρχ_name, χʲ_name)
            @. ᶠu³_diff = ᶠu³ʲs.:(1) - ᶠu³
            for (ρχ_name, χʲ_name) in microphysics_tracers
                MatrixFields.has_field(Y, ρχ_name) || continue

                ᶜχʲ = MatrixFields.get_field(p.precomputed, χʲ_name)
                ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
                # @. ᶜa_scalar =
                #     (ᶜχʲ - ᶜχ) *
                #     draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j))
                # TODO: remove this filter when mass flux is treated implicitly
                @. ᶜa_scalar =
                    (ᶜχʲ - ᶜχ) * min(
                        min(draft_area(ᶜρaʲs.:(1), ᶜρʲs.:(1)), a_max),
                        FT(0.02) / max(
                            Geometry.WVector(
                                ᶜinterp(ᶠu³_diff),
                            ).components.data.:1,
                            eps(FT),
                        ),
                    )
                vtt = vertical_transport(
                    ᶜρʲs.:(1),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                @. ᶜρχₜ += vtt
            end
        end

        # Sea salt aerosol tracers (passive — no in-updraft source)
        if p.atmos.edmfx_model.prognostic_aerosols isa Val{true}
            aerosol_tracers = (
                (@name(c.ρSSLT01), @name(ᶜSSLT01ʲs.:(1))),
                (@name(c.ρSSLT02), @name(ᶜSSLT02ʲs.:(1))),
                (@name(c.ρSSLT03), @name(ᶜSSLT03ʲs.:(1))),
                (@name(c.ρSSLT04), @name(ᶜSSLT04ʲs.:(1))),
                (@name(c.ρSSLT05), @name(ᶜSSLT05ʲs.:(1))),
            )
            @. ᶠu³_diff = ᶠu³ʲs.:(1) - ᶠu³
            sslt_diag_fire =
                p.atmos.edmfx_model.prognostic_aerosols isa Val{true} &&
                ((SSLT_DIAG_COUNTER[] + 1) in SSLT_DIAG_FIRE_AT)
            for (ρχ_name, χʲ_name) in aerosol_tracers
                MatrixFields.has_field(Y, ρχ_name) || continue
                MatrixFields.has_field(p.precomputed, χʲ_name) || continue
                ᶜχʲ = MatrixFields.get_field(p.precomputed, χʲ_name)
                ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                ᶜχ  = (@. lazy(specific(ᶜρχ, Y.c.ρ)))

                # Diagnostic 4.1: at a few call counts (skipping t=0 init),
                # dump (χʲ − χ̄) magnitude vs χ̄ at boundary-layer levels 5/10/20.
                # ratio_max ≈ O(1) → column-march producing physical updraft values;
                # ratio_max ≫ 1 → χʲ is leaving the {χ̄, χʲ_prev} hull (column-march bug).
                # Track ratio_max across the firing counts to see whether the blowup
                # is born in column-march or accumulates via SGS-flux feedback.
                if sslt_diag_fire
                    ᶜdiff = p.scratch.ᶜtemp_scalar_2
                    ᶜchi  = p.scratch.ᶜtemp_scalar_3
                    @. ᶜdiff = ᶜχʲ - ᶜχ
                    @. ᶜchi  = specific(ᶜρχ, Y.c.ρ)
                    for i_lvl in (5, 10, 20)
                        diff_lvl = Array(parent(Fields.level(ᶜdiff, i_lvl)))
                        chi_lvl  = Array(parent(Fields.level(ᶜchi,  i_lvl)))
                        chiʲ_lvl = Array(parent(Fields.level(ᶜχʲ,   i_lvl)))
                        χ_scale  = max(maximum(abs, chi_lvl), eps(FT))
                        @info "[SSLT-diag-4.1] (χʲ−χ̄) vs χ̄" call=SSLT_DIAG_COUNTER[]+1 bin=ρχ_name level=i_lvl t=t diff_min=minimum(diff_lvl) diff_max=maximum(diff_lvl) χ_min=minimum(chi_lvl) χ_max=maximum(chi_lvl) χʲ_min=minimum(chiʲ_lvl) χʲ_max=maximum(chiʲ_lvl) ratio_max=maximum(abs, diff_lvl) / χ_scale
                    end
                end

                @. ᶜa_scalar =
                    (ᶜχʲ - ᶜχ) * min(
                        min(draft_area(ᶜρaʲs.:(1), ᶜρʲs.:(1)), a_max),
                        FT(0.02) / max(
                            Geometry.WVector(ᶜinterp(ᶠu³_diff)).components.data.:1,
                            eps(FT),
                        ),
                    )
                vtt = vertical_transport(
                    ᶜρʲs.:(1), ᶠu³_diff, ᶜa_scalar, dt, edmfx_tracer_upwinding,
                )
                ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                @. ᶜρχₜ += vtt
            end
            SSLT_DIAG_COUNTER[] += 1
        end

        # TODO: the following adds the environment flux to the tendency
        # Make active and test later
        # @. ᶠu³_diff = p.precomputed.ᶠu³⁰ - ᶠu³
        # ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model)
        # ᶜρ⁰ = p.scratch.ᶜtemp_scalar_2
        # @. ᶜρ⁰ = TD.air_density(
        #     CAP.thermodynamics_params(p.params),
        #     p.precomputed.ᶜts⁰,
        # )
        # ᶜmse⁰ = @.lazy(ᶜspecific_env_mse(Y, p))
        # @. ᶜa_scalar =
        #     (ᶜmse⁰ + p.precomputed.ᶜK⁰ - ᶜh_tot) * draft_area(ᶜρa⁰, ᶜρ⁰)
        # vtt = vertical_transport(
        #     ᶜρ⁰,
        #     ᶠu³_diff,
        #     ᶜa_scalar,
        #     dt,
        #     edmfx_sgsflux_upwinding,
        # )
        # @. Yₜ.c.ρe_tot += vtt
        # if !(p.atmos.microphysics_model isa DryModel)
        #     ᶜq_tot⁰ = @specific_env_value(:q_tot, Y.c, turbconv_model))
        #     @. ᶜa_scalar =
        #         (ᶜq_tot⁰ - specific(Y.c.ρq_tot, Y.c.ρ)) *
        #         draft_area(ᶜρa⁰, ᶜρ⁰)
        #     vtt = vertical_transport(
        #         ᶜρ⁰,
        #         ᶠu³_diff,
        #         ᶜa_scalar,
        #         dt,
        #         edmfx_sgsflux_upwinding,
        #     )
        #     @. Yₜ.c.ρq_tot += vtt
        # end
    end

end

"""
    edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Computes and applies the tendency to the grid-mean state `Y` due to SGS
diffusive fluxes from the EDMFX environment. This involves calculating the
divergence of turbulent fluxes, which are parameterized using eddy diffusivity
and viscosity closures.

This function parameterizes these fluxes using an eddy-diffusivity/viscosity
approach (K-theory) for the grid-mean. Tendencies are calculated for
total energy, moisture species, momentum, and optionally TKE.
The form is typically `- ∂/∂z(-D ∂ϕ/∂z)`, where `D` is an effective SGS eddy
diffusivity for the quantity `ϕ`.

The specific implementation depends on the `turbconv_model`. A generic fallback
doing nothing is also provided. The function modifies `Yₜ.c` (grid-mean tendencies)
in place.

Arguments:
- `Yₜ`: The tendency state vector for grid-mean variables.
- `Y`: The current state vector (used for grid-mean and SGS properties).
- `p`: Cache containing parameters, precomputed fields, atmospheric model settings,
       and scratch space.
- `t`: Current simulation time.
- `turbconv_model`: The turbulence convection model instance.
"""
edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::Union{EDOnlyEDMFX, DiagnosticEDMFX, PrognosticEDMFX},
)

    FT = Spaces.undertype(axes(Y.c))
    (; dt, params) = p
    turbconv_params = CAP.turbconv_params(params)
    (; ᶜu) = p.precomputed
    (; ρtke_flux) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))

    if p.atmos.edmfx_model.sgs_diffusive_flux isa Val{true}

        (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
        # scratch to prevent GPU Kernel parameter memory error
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_2
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜK_u = @. lazy(
            eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field),
        )
        ᶜprandtl_nvec = @. lazy(
            turbulent_prandtl_number(
                params,
                ᶜlinear_buoygrad,
                ᶜstrain_rate_norm,
            ),
        )
        ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

        ᶠρaK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_h = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h)
        ᶠρaK_u = p.scratch.ᶠtemp_scalar_2
        @. ᶠρaK_u = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_u)

        # Total enthalpy diffusion
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        (; ᶜh_tot) = p.precomputed
        @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(-(ᶠρaK_h * ᶠgradᵥ(ᶜh_tot)))

        if use_prognostic_tke(turbconv_model)
            # Turbulent TKE transport (diffusion)
            ᶜdivᵥ_ρtke = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(ρtke_flux),
            )
            # Add flux divergence and dissipation term, relaxing TKE to zero
            # in one time step if tke < 0
            @. Yₜ.c.ρtke -=
                ᶜdivᵥ_ρtke(-(ᶠρaK_u * ᶠgradᵥ(ᶜtke))) + ifelse(
                    ᶜtke >= FT(0),
                    tke_dissipation(
                        turbconv_params,
                        Y.c.ρtke,
                        ᶜtke,
                        ᶜmixing_length_field,
                    ),
                    Y.c.ρtke / dt,
                )
        end

        if !(p.atmos.microphysics_model isa DryModel)
            # Specific humidity diffusion
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )
            @. ᶜρχₜ_diffusion =
                ᶜdivᵥ_ρq_tot(-(ᶠρaK_h * ᶠgradᵥ(specific(Y.c.ρq_tot, Y.c.ρ))))
            @. Yₜ.c.ρq_tot -= ᶜρχₜ_diffusion
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion  # Effect of moisture diffusion on (moist) air mass
        end

        α_precip = CAP.α_vert_diff_tracer(params)
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
        ᶜdivᵥ_ρq = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        microphysics_tracers = (
            (@name(c.ρq_lcl), FT(1)),
            (@name(c.ρq_icl), FT(1)),
            (@name(c.ρq_rai), α_precip),
            (@name(c.ρq_sno), α_precip),
            (@name(c.ρn_lcl), FT(1)),
            (@name(c.ρn_rai), α_precip),
        )
        MatrixFields.unrolled_foreach(microphysics_tracers) do (ρχ_name, α)
            MatrixFields.has_field(Y, ρχ_name) || return
            ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
            ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq(-(ᶠρaK_h * α * ᶠgradᵥ(ᶜχ)))
            ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
            @. ᶜρχₜ -= ᶜρχₜ_diffusion
        end

        # Momentum diffusion
        ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
        @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Y.c.ρ)
    end

    return nothing
end
