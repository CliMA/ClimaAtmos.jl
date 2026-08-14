#####
##### Tendencies applied to the grid-mean atmospheric state due to subgrid-scale (SGS)
##### fluxes computed by the EDMFX scheme
#####

"""
    edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the divergence of the vertical SGS mass fluxes of the PROPHET scheme
(`EDMFX` in code) to the grid-mean prognostic variables.

For each subdomain `k` (updrafts `j` and environment `0`), the flux is the
difference-form vertical transport `ρᵏaᵏ (u³ᵏ - u³)(χᵏ - χ)` of a specific
scalar `χ` relative to the grid mean, reconstructed with the
`edmfx_sgsflux_upwinding` scheme; it vanishes identically when `χᵏ = χ`.
Tendencies are applied to `Yₜ.c.ρe_tot` (with `χʲ = mseʲ + Kʲ` for the
updrafts), `Yₜ.c.ρq_tot`, and every auto-discovered SGS tracer (microphysics
species and passive tracers). The `q_tot` flux also increments `Yₜ.c.ρ`,
since SGS moisture transport moves (moist) air mass, mirroring the
diffusive-flux treatment. Vertical SGS momentum fluxes are not yet included.

The `PrognosticEDMFX` method is gated on `p.atmos.edmfx_model.sgs_mass_flux`;
the generic method is a no-op. Mutates `Yₜ.c`; returns `nothing`. See the
"PROPHET Sub-Grid Scale Equations" page (`docs/src/edmf_equations.md`).
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
    (; edmfx_sgsflux_upwinding) = p.atmos.numerics
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
                @. Yₜ.c.ρ += vtt  # Effect of SGS water flux on (moist) air mass
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
            @. Yₜ.c.ρ += vtt  # Effect of SGS water flux on (moist) air mass
        end

        # Auto-discovered SGS tracer fluxes (microphysics species and any
        # user-defined passive tracers). Like the mse and q_tot fluxes above,
        # these are difference-form fluxes ρᵏaᵏ(u³ᵏ - u³)(χᵏ - χ), which
        # vanish identically for uniform χ, reconstructed with the same
        # upwinding as the mse and q_tot fluxes so that the water-species
        # fluxes stay consistent with the q_tot flux (the implied vapor flux
        # is their difference). The grid-mean advection -∇·(ρ u³ χ) of each
        # tracer is applied in explicit_vertical_advection_tendency!.
        # Draft fluxes
        for χ_name in sgs_tracer_names(Y)
            ρχ_name = get_ρχ_name(χ_name)
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            for j in 1:n
                ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
                @. ᶜa_scalar =
                    (ᶜχʲ - specific(ᶜρχ, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
                @. ᶜρχₜ += vtt
            end
        end
        # Environment fluxes
        @. ᶠu³_diff = ᶠu³⁰ - ᶠu³
        for χ_name in sgs_tracer_names(Y)
            ρχ_name = get_ρχ_name(χ_name)
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
            @. ᶜa_scalar =
                (ᶜχ⁰ - specific(ᶜρχ, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
            @. ᶜρχₜ += vtt
        end
    end
    # TODO - add vertical momentum fluxes
    return nothing
end

"""
    edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the divergence of the SGS diffusive (K-theory) fluxes of the PROPHET
scheme (`EDMFX` in code) to the grid-mean state.

All fluxes use the face-native eddy diffusivity/viscosity `ᶠK_h`/`ᶠK_u` plus
the interfacial entrainment diffusivity `ᶠK_entr` from
`set_face_diffusivities!`:

  - Total enthalpy: the single-gradient form
    `F_h = -K [∇s_d + (h_eff + Φ) ∇q_tot_eff]`, where
    `q_tot_eff = q_tot - q_rai - q_sno` is the water that diffuses and
    `h_eff = (h_v q_v + h_l q_lcl + h_i q_icl) / max(q_water_nonneg, ε)` is its
    mass-weighted enthalpy, applied to `Yₜ.c.ρe_tot`.
  - Total water: the `q_tot_eff` flux applied to `Yₜ.c.ρq_tot` and mirrored to
    `Yₜ.c.ρ` (moisture diffusion moves moist air mass). Cloud species take a
    share of it by tendency scaling rather than a flux of their own; rain and
    snow do not diffuse.
  - Other grid-scale tracers: passive tracers diffuse with the unscaled `K_h`;
    `K_entr` always enters at full weight.
  - Momentum: `-2 ρ K_u 𝔈` with the vertical strain rate, applied to `Yₜ.c.uₕ`.
  - TKE (when prognostic): turbulent transport plus dissipation
    (`tke_dissipation`), applied to `Yₜ.c.ρtke`; negative TKE is relaxed to
    zero within one time step.

When `p.atmos.edmfx_model.vertical_diffusion` is enabled for
`PrognosticEDMFX`, the same specific tendencies are additionally applied to
each updraft's `mse`, `q_tot`, and tracers (uniform vertical diffusion across
the grid box).

Methods: generic no-op, and `Union{EDOnlyEDMFX, PrognosticEDMFX}` gated on
`p.atmos.edmfx_model.sgs_diffusive_flux`. Mutates `Yₜ.c`; returns `nothing`.
"""
edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX},
)

    FT = Spaces.undertype(axes(Y.c))
    ϵ_FT = eps(FT)
    (; dt, params) = p
    turbconv_params = CAP.turbconv_params(params)
    (; ᶜu) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()
    n = n_mass_flux_subdomains(turbconv_model)
    # The SGS-updraft branches below apply the same specific tendency the grid
    # mean receives to each subdomain scalar (uniform vertical diffusion across
    # the grid box). Gated on `edmfx_model.vertical_diffusion` so configs can
    # opt in/out just like the old subdomain-native diffusion did.
    apply_sgs_updraft =
        turbconv_model isa PrognosticEDMFX &&
        p.atmos.edmfx_model.vertical_diffusion isa Val{true}

    if p.atmos.edmfx_model.sgs_diffusive_flux isa Val{true}

        # Face-native eddy diffusivity/viscosity and interfacial entrainment
        # diffusivity, evaluated at the faces where the fluxes live (see
        # `set_face_diffusivities!`): the stability closure collapses K_h at
        # an unresolved inversion at exactly (and only) the jump face, and
        # K_e restores the finite-velocity entrainment flux there. K_h and
        # K_e are held separately so they can be applied with different
        # structures: K_h is the "turbulent mixing" component, applied to
        # `q_tot_eff` (and distributed to cloud species) for water and to
        # `∇s_d + h_eff·∇q_tot_eff` for enthalpy; K_e is the "entrainment"
        # component, applied per-species with each species's own gradient
        # (bodily parcel transport). Passive tracers and dry static energy
        # transport at the combined (K_h + K_e); momentum uses (K_u + K_e).
        (; ᶠK_h, ᶠK_u, ᶠK_entr, ᶜl_mix) = p.precomputed
        ᶠρK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρK_h = ᶠinterp(Y.c.ρ) * ᶠK_h
        ᶠρK_e = p.scratch.ᶠtemp_scalar_3
        @. ᶠρK_e = ᶠinterp(Y.c.ρ) * ᶠK_entr
        ᶠρaK_u = p.scratch.ᶠtemp_scalar_2
        @. ᶠρaK_u = ᶠinterp(Y.c.ρ) * (ᶠK_u + ᶠK_entr)

        # Total enthalpy diffusion. K_h piece uses the spurious-transport-safe
        # decomposition (∇s_d + h_eff·∇q_tot_eff, moisture part added below
        # when non-dry). K_e piece uses bodily-parcel form (∇h_tot directly),
        # since interfacial entrainment transports every constituent —
        # including dry air — with the parcel.
        #   q_tot_eff = q_tot - q_rai - q_sno,
        #   h_eff = (h_v·q_v + h_l·q_lcl + h_i·q_icl) / max(q_water_nonneg, ε)
        # See `hyperdiffusion.jl` for the clipped-input protection.
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        thermo_params = CAP.thermodynamics_params(params)
        (; ᶜΦ) = p.core
        (; ᶜT) = p.precomputed
        (; ᶜh_tot) = p.precomputed
        ᶜρe_totₜ_diffusion = p.scratch.ᶜtemp_scalar_2
        @. ᶜρe_totₜ_diffusion =
            ᶜdivᵥ_ρe_tot(
                -(
                    ᶠρK_h * ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)) +
                    ᶠρK_e * ᶠgradᵥ(ᶜh_tot)
                ),
            )

        if use_prognostic_tke(turbconv_model)
            (; ρtke_flux) = p.precomputed
            ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
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
                        ᶜl_mix,
                    ),
                    Y.c.ρtke / dt,
                )
        end

        if !(p.atmos.microphysics_model isa DryModel)
            # Moisture contribution to the enthalpy K_h flux: adds
            # -ρK_h · (h_eff+Φ) · ∇q_tot_eff to the dry-part tendency
            # computed above.
            (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
            ᶜq_vap = @. lazy(TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
            ᶜq_lcl, ᶜq_icl =
                p.atmos.microphysics_model isa
                Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ?
                (
                    (@. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))),
                    (@. lazy(specific(Y.c.ρq_icl, Y.c.ρ))),
                ) :
                (ᶜq_liq, ᶜq_ice)
            ᶜh_eff_plus_Φ = p.scratch.ᶜtemp_scalar_3
            @. ᶜh_eff_plus_Φ =
                (
                    TD.enthalpy_vapor(thermo_params, ᶜT) * max(FT(0), ᶜq_vap) +
                    TD.enthalpy_liquid(thermo_params, ᶜT) * max(FT(0), ᶜq_lcl) +
                    TD.enthalpy_ice(thermo_params, ᶜT) * max(FT(0), ᶜq_icl)
                ) /
                max(max(FT(0), ᶜq_vap) + max(FT(0), ᶜq_lcl) + max(FT(0), ᶜq_icl), ϵ_FT) +
                ᶜΦ
            ᶜq_tot_eff =
                p.atmos.microphysics_model isa
                Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ?
                (@. lazy(specific(Y.c.ρq_tot - Y.c.ρq_rai - Y.c.ρq_sno, Y.c.ρ))) :
                (@. lazy(specific(Y.c.ρq_tot, Y.c.ρ)))
            @. ᶜρe_totₜ_diffusion +=
                ᶜdivᵥ_ρe_tot(-(ᶠρK_h * ᶠinterp(ᶜh_eff_plus_Φ) * ᶠgradᵥ(ᶜq_tot_eff)))

            # K_h water diffusion on q_tot_eff. Cloud species inherit via
            # clipped ratio; rain/snow/n_rai get no K_h transport. K_e
            # transport for all water species is handled in the unified
            # tracer loop below.
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq_tot(-(ᶠρK_h * ᶠgradᵥ(ᶜq_tot_eff)))
            @. Yₜ.c.ρq_tot -= ᶜρχₜ_diffusion
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion  # Effect of moisture diffusion on (moist) air mass
            if apply_sgs_updraft
                for j in 1:n
                    @. Yₜ.c.sgsʲs.:($$j).q_tot -= ᶜρχₜ_diffusion / Y.c.ρ
                end
                # The corresponding ρaⱼ dry-mass correction is deliberately neglected.
            end

            # Distribute K_h q_tot diffusion to cloud mass and number species.
            ᶜratio = p.scratch.ᶜtemp_scalar_4
            for (q_name, n_name) in (
                (@name(q_lcl), @name(n_lcl)),
                (@name(q_icl), @name(n_icl)),
            )
                ρq_name = get_ρχ_name(q_name)
                ρn_name = get_ρχ_name(n_name)
                MatrixFields.has_field(Y.c, ρq_name) || continue
                ᶜρq = MatrixFields.get_field(Y.c, ρq_name)
                ᶜρqₜ = MatrixFields.get_field(Yₜ.c, ρq_name)
                @. ᶜratio =
                    max(FT(0), min(FT(1), specific(ᶜρq, Y.c.ρ) / max(ᶜq_tot_eff, ϵ_FT)))
                @. ᶜρqₜ -= ᶜratio * ᶜρχₜ_diffusion
                if apply_sgs_updraft
                    for j in 1:n
                        if MatrixFields.has_field(Y.c.sgsʲs.:($j), q_name)
                            ᶜqⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), q_name)
                            @. ᶜqⱼₜ -= ᶜratio * ᶜρχₜ_diffusion / Y.c.ρ
                        end
                    end
                end
                if MatrixFields.has_field(Y.c, ρn_name)
                    ᶜρn = MatrixFields.get_field(Y.c, ρn_name)
                    ᶜρnₜ = MatrixFields.get_field(Yₜ.c, ρn_name)
                    @. ᶜρnₜ -= ᶜratio * max(FT(0), ᶜρn) / max(ᶜρq, ϵ_FT) * ᶜρχₜ_diffusion
                    if apply_sgs_updraft
                        for j in 1:n
                            ᶜnⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), n_name)
                            @. ᶜnⱼₜ -=
                                ᶜratio * max(FT(0), ᶜρn) / max(ᶜρq, ϵ_FT) *
                                ᶜρχₜ_diffusion / Y.c.ρ
                        end
                    end
                end
            end
        end

        # Apply the accumulated enthalpy tendency (dry parts + moisture
        # contribution added above in the moist branch). Every subdomain
        # sees the grid-mean specific-enthalpy tendency.
        @. Yₜ.c.ρe_tot -= ᶜρe_totₜ_diffusion
        if apply_sgs_updraft
            for j in 1:n
                @. Yₜ.c.sgsʲs.:($$j).mse -= ᶜρe_totₜ_diffusion / Y.c.ρ
            end
        end

        # Unified tracer diffusion loop covering both microphysics and
        # passive species. The `α` flag encodes the K_h contribution:
        # microphysics species (`α = 0`) receive only K_e transport (K_h
        # transport is applied above via q_tot_eff distribution to cloud
        # species; precip has no K_h transport), while passive tracers
        # (`α = 1`) receive the full ρ·(K_h + K_e) diffusion.
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
        ᶜdivᵥ_ρq = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
            α = ρχ_name in microphysics_tracer_names(Y) ? FT(0) : FT(1)
            ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq(-((α * ᶠρK_h + ᶠρK_e) * ᶠgradᵥ(ᶜχ)))
            @. ᶜρχₜ -= ᶜρχₜ_diffusion
            # K_e bodily transport of ρq_tot also moves moist-air mass.
            if ρχ_name == @name(ρq_tot)
                @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
            end
            # Uniform vertical diffusion: apply the same grid-mean specific
            # tendency to the matching subdomain field in each updraft.
            if apply_sgs_updraft
                χ_name = specific_tracer_name(ρχ_name)
                for j in 1:n
                    ᶜχⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                    @. ᶜχⱼₜ -= ᶜρχₜ_diffusion / Y.c.ρ
                end
            end
        end

        # Momentum diffusion
        ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
        @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Y.c.ρ)
    end

    return nothing
end
