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
"PROPHET: Overview and Equations" page (`docs/src/prophet.md`).
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

    if p.atmos.edmfx_model.sgs_mass_flux
        # Fluxes of enthalpy, total water, and the SGS tracers, as one fused
        # column loop per draft and one for the environment: the flux
        # velocity `ᶠu³_diff` and the flux variable `ᶜa_scalar` of each
        # quantity are loop temporaries (registers on GPUs) rather than scratch
        # fields, and all quantities of a subdomain share one kernel.
        # TODO: Isolate assembly of flux term pattern to a function and
        # reuse (both in prognostic and diagnostic EDMFX)
        # [best after removal of precomputed quantities]
        (; ᶜh_tot) = p.precomputed
        χ_names = sgs_tracer_names(Y)
        ρχ_names = unrolled_map(get_ρχ_name, χ_names)

        # Drafts
        for j in 1:n
            ᶜsgsʲ = Y.c.sgsʲs.:($j)
            ᶜχʲs = unrolled_map(
                χ_name -> MatrixFields.get_field(ᶜsgsʲ, χ_name), χ_names,
            )
            DataLayouts.foreach_column(
                Yₜ.c, Y.c, ᶜρʲs.:($j), ᶠu³, ᶠu³ʲs.:($j), ᶜKʲs.:($j), ᶜh_tot,
                ᶜsgsʲ, ᶜχʲs...,
            ) do Yₜc, Yc, ᶜρʲ, ᶠu³, ᶠu³ʲ, ᶜKʲ, ᶜh_tot, ᶜsgsʲ, ᶜχʲs...
                ᶜaʲ = @. draft_area(ᶜsgsʲ.ρa, ᶜρʲ)
                add_sgs_mass_flux_tendencies!(
                    Yₜc, Yc, ᶜρʲ, ᶠu³, ᶠu³ʲ, ᶜKʲ, ᶜh_tot, ᶜsgsʲ.mse, ᶜsgsʲ.q_tot,
                    ᶜaʲ, ᶜχʲs, ρχ_names, dt, edmfx_sgsflux_upwinding,
                )
                return nothing
            end
        end

        # Environment
        ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
        ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
        ᶜχ⁰s = unrolled_map(χ_name -> ᶜspecific_env_value(χ_name, Y, p), χ_names)
        DataLayouts.foreach_column(
            Yₜ.c, Y.c, ᶜρ⁰, ᶜρa⁰, ᶠu³, ᶠu³⁰, ᶜK⁰, ᶜh_tot, ᶜmse⁰, ᶜq_tot⁰, ᶜχ⁰s...,
        ) do Yₜc, Yc, ᶜρ⁰, ᶜρa⁰, ᶠu³, ᶠu³⁰, ᶜK⁰, ᶜh_tot, ᶜmse⁰, ᶜq_tot⁰, ᶜχ⁰s...
            ᶜa⁰ = @. draft_area(ᶜρa⁰, ᶜρ⁰)
            add_sgs_mass_flux_tendencies!(
                Yₜc, Yc, ᶜρ⁰, ᶠu³, ᶠu³⁰, ᶜK⁰, ᶜh_tot, ᶜmse⁰, ᶜq_tot⁰,
                ᶜa⁰, ᶜχ⁰s, ρχ_names, dt, edmfx_sgsflux_upwinding,
            )
            return nothing
        end
    end
    return nothing
end

"""
    add_sgs_mass_flux_tendencies!(Yₜc, Yc, ᶜρ, ᶠu³, ᶠu³_sub, ᶜK_sub, ᶜh_tot, ᶜmse_sub,
                                  ᶜq_tot_sub, ᶜa, ᶜχ_subs, ρχ_names, dt, upwinding)

Add the SGS mass-flux tendencies of one subdomain (a draft or the environment)
to the grid-mean `ρe_tot`, `ρq_tot`, `ρ`, and the SGS tracers `ρχ_names` of one
column. The subdomain is described by its density `ᶜρ`, velocity `ᶠu³_sub`,
kinetic energy `ᶜK_sub`, moist static energy `ᶜmse_sub`, total specific humidity
`ᶜq_tot_sub`, area fraction `ᶜa`, and tracer values `ᶜχ_subs` (one per name in
`ρχ_names`); `ᶠu³` and `ᶜh_tot` are the grid-mean velocity and total specific
enthalpy. Called inside the fused column loops of
[`edmfx_sgs_mass_flux_tendency!`](@ref), so its temporaries are loop
temporaries and it must stay inlined.
"""
@inline function add_sgs_mass_flux_tendencies!(
    Yₜc,
    Yc,
    ᶜρ,
    ᶠu³,
    ᶠu³_sub,
    ᶜK_sub,
    ᶜh_tot,
    ᶜmse_sub,
    ᶜq_tot_sub,
    ᶜa,
    ᶜχ_subs,
    ρχ_names,
    dt,
    upwinding,
)
    ᶠu³_diff = @. ᶠu³_sub - ᶠu³
    ᶜa_scalar = @. (ᶜmse_sub + ᶜK_sub - ᶜh_tot) * ᶜa
    vtt = vertical_transport(ᶜρ, ᶠu³_diff, ᶜa_scalar, dt, upwinding)
    @. Yₜc.ρe_tot += vtt
    if MatrixFields.has_field(Yc, @name(ρq_tot))
        ᶜa_scalar_q = @. (ᶜq_tot_sub - specific(Yc.ρq_tot, Yc.ρ)) * ᶜa
        vtt_q = vertical_transport(ᶜρ, ᶠu³_diff, ᶜa_scalar_q, dt, upwinding)
        @. Yₜc.ρq_tot += vtt_q
        @. Yₜc.ρ += vtt_q  # Effect of SGS water flux on (moist) air mass
    end
    ntuple(Val(length(ρχ_names))) do i
        ᶜρχ = MatrixFields.get_field(Yc, ρχ_names[i])
        ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_names[i])
        ᶜa_scalar_χ = @. (ᶜχ_subs[i] - specific(ᶜρχ, Yc.ρ)) * ᶜa
        vtt_χ = vertical_transport(ᶜρ, ᶠu³_diff, ᶜa_scalar_χ, dt, upwinding)
        @. ᶜρχₜ += vtt_χ
        return nothing
    end
    return nothing
end

"""
    edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the divergence of the SGS diffusive (K-theory) fluxes of the PROPHET
scheme (`EDMFX` in code) to the grid-mean state.

All fluxes use the face-native eddy diffusivity/viscosity `ᶠK_h`/`ᶠK_u`
from `set_face_diffusivities!`:

  - Total enthalpy: the single-gradient form
    `F_h = -K_h [∇s_d + (h_eff + Φ) ∇q_tot_eff]`, where
    `q_tot_eff = q_tot - q_rai - q_sno` is the water that diffuses and
    `h_eff = (h_v q_v + h_l q_lcl + h_i q_icl) / max(q_water_nonneg, ε)` is its
    mass-weighted enthalpy, applied to `Yₜ.c.ρe_tot`.
  - Total water: the `q_tot_eff` flux applied to `Yₜ.c.ρq_tot` and mirrored to
    `Yₜ.c.ρ` (moisture diffusion moves moist air mass). Cloud species take a
    share of it by tendency scaling rather than a flux of their own; rain and
    snow do not diffuse.
  - Other grid-scale tracers: passive tracers diffuse with `K_h`.
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
    n = n_mass_flux_subdomains(turbconv_model)
    # The SGS-updraft branches below apply the same specific tendency the grid
    # mean receives to each subdomain scalar (uniform vertical diffusion across
    # the grid box). Gated on `edmfx_model.vertical_diffusion` so configs can
    # opt in/out just like the old subdomain-native diffusion did.
    apply_sgs_updraft =
        turbconv_model isa PrognosticEDMFX &&
        p.atmos.edmfx_model.vertical_diffusion

    if p.atmos.edmfx_model.sgs_diffusive_flux

        # Face-native eddy diffusivity/viscosity, evaluated at the faces
        # where the fluxes live (see `set_face_diffusivities!`).
        (; ᶠK_h, ᶠK_u, ᶜl_mix) = p.precomputed
        ᶠρK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρK_h = ᶠinterp(Y.c.ρ) * ᶠK_h
        ᶠρaK_u = p.scratch.ᶠtemp_scalar_2
        @. ᶠρaK_u = ᶠinterp(Y.c.ρ) * ᶠK_u

        # Total enthalpy diffusion. Uses the spurious-transport-safe
        # decomposition (∇s_d + h_eff·∇q_tot_eff, moisture part added below
        # when non-dry).
        #   q_tot_eff = q_tot - q_rai - q_sno,
        #   h_eff = (h_v·q_v + h_l·q_lcl + h_i·q_icl) / max(q_water_nonneg, ε)
        # See `hyperdiffusion.jl` for the clipped-input protection.
        thermo_params = CAP.thermodynamics_params(params)
        (; ᶜΦ) = p.core
        (; ᶜT) = p.precomputed
        ᶜρe_totₜ_diffusion = p.scratch.ᶜtemp_scalar_2
        @. ᶜρe_totₜ_diffusion =
            ᶜdiffdivᵥ(
                -(ᶠρK_h * ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ))),
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
            ᶜq_vap, ᶜq_lcl, ᶜq_icl = ᶜsuspended_water(Y, p)
            ᶜh_eff_plus_Φ = ᶜh_eff_plus_Φ!(
                p.scratch.ᶜtemp_scalar_3,
                thermo_params,
                ᶜT,
                ᶜΦ,
                ᶜq_vap,
                ᶜq_lcl,
                ᶜq_icl,
            )
            ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
            @. ᶜρe_totₜ_diffusion +=
                ᶜdiffdivᵥ(-(ᶠρK_h * ᶠinterp(ᶜh_eff_plus_Φ) * ᶠgradᵥ(ᶜq_tot_eff)))

            # K_h water diffusion on q_tot_eff. Cloud species inherit via
            # clipped ratio; rain/snow/n_rai get no diffusion.
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜ∇ᵥρK∇q_tot = ᶜdiffusive_flux_divergenceᵥ(ᶠρK_h, ᶜq_tot_eff)
            @. ᶜρχₜ_diffusion = ᶜ∇ᵥρK∇q_tot
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
                (@name(q_icl), @name(n_ice)),
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

        # Passive (non-microphysics) grid-scale tracers diffuse with ρ·K_h.
        # Microphysics species (cloud mass/number) already picked up their
        # share above via the q_tot_eff distribution; precipitation species
        # get no vertical diffusion here.
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
        foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
            ρχ_name in microphysics_tracer_names(Y) && return
            ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
            ᶜ∇ᵥρK∇χ = ᶜdiffusive_flux_divergenceᵥ(ᶠρK_h, ᶜχ)
            @. ᶜρχₜ_diffusion = ᶜ∇ᵥρK∇χ
            @. ᶜρχₜ -= ᶜρχₜ_diffusion
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

"""
    edmfx_sgs_horizontal_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the tendency from the horizontal component of the EDMFX environment SGS
diffusive flux, using the TKE-based eddy diffusivity with the mixing length
limited by the horizontal node spacing rather than the resolvability filter
scale.
With prognostic updrafts and `edmfx_model.horizontal_diffusion` enabled, the
grid-mean specific tendencies are also applied to the subdomain scalars.

Always explicit; applied in the explicit remainder, independently of `diff_mode`.
"""
edmfx_sgs_horizontal_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model) =
    nothing

function edmfx_sgs_horizontal_diffusive_flux_tendency!(
    Yₜ, Y, p, t, turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX},
)
    p.atmos.edmfx_model.sgs_diffusive_flux_horizontal isa Val{true} ||
        return nothing
    iscolumn(axes(Y.c)) && return nothing
    FT = Spaces.undertype(axes(Y.c))
    ϵ_FT = eps(FT)
    (; params) = p
    (; ᶜK_u_h, ᶜK_h_h) = p.precomputed
    ᶜρ = Y.c.ρ
    ᶜtke = @. lazy(specific(Y.c.ρtke, ᶜρ))

    n = n_mass_flux_subdomains(turbconv_model)
    # With `edmfx_model.horizontal_diffusion`, each subdomain scalar receives
    # the same specific tendency as the grid mean (uniform horizontal diffusion
    # across the grid box).
    apply_sgs_updraft =
        turbconv_model isa PrognosticEDMFX &&
        p.atmos.edmfx_model.horizontal_diffusion isa Val{true}

    # Total enthalpy in the spurious-transport-safe form
    # `F_h = -ρ K_h [∇ₕs_d + (h_eff + Φ) ∇ₕq_tot_eff]`, matching the vertical
    # term in `edmfx_sgs_diffusive_flux_tendency!`. The dry static energy is
    # diffused in every configuration; the water contribution is added below
    # when `ρq_tot` is prognostic.
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜΦ) = p.core
    (; ᶜT) = p.precomputed
    ᶜρe_totₜ_diffusion = p.scratch.ᶜtemp_scalar_2
    @. ᶜρe_totₜ_diffusion =
        wdivₕ(ᶜρ * ᶜK_h_h * gradₕ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)))

    if !(p.atmos.microphysics_model isa DryModel)
        ᶜq_vap, ᶜq_lcl, ᶜq_icl = ᶜsuspended_water(Y, p)
        ᶜh_eff_plus_Φ = ᶜh_eff_plus_Φ!(
            p.scratch.ᶜtemp_scalar_6,
            thermo_params,
            ᶜT,
            ᶜΦ,
            ᶜq_vap,
            ᶜq_lcl,
            ᶜq_icl,
        )
        ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
        @. ᶜρe_totₜ_diffusion +=
            wdivₕ(ᶜρ * ᶜK_h_h * ᶜh_eff_plus_Φ * gradₕ(ᶜq_tot_eff))

        # Water diffusion on `q_tot_eff`, and its effect on the moist air mass.
        # Cloud species inherit their share below; rain and snow get no
        # horizontal transport.
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar_3
        @. ᶜρχₜ_diffusion = wdivₕ(ᶜρ * ᶜK_h_h * gradₕ(ᶜq_tot_eff))
        @. Yₜ.c.ρq_tot += ᶜρχₜ_diffusion
        @. Yₜ.c.ρ += ᶜρχₜ_diffusion
        if apply_sgs_updraft
            for j in 1:n
                @. Yₜ.c.sgsʲs.:($$j).q_tot += ᶜρχₜ_diffusion / ᶜρ
            end
            # The corresponding ρaⱼ dry-mass correction is omitted, matching
            # `edmfx_sgs_diffusive_flux_tendency!`.
        end

        # Distribute the q_tot_eff diffusion to cloud mass and number species.
        ᶜratio = p.scratch.ᶜtemp_scalar_4
        for (q_name, n_name, ρq_name, ρn_name) in (
            (@name(q_lcl), @name(n_lcl), @name(c.ρq_lcl), @name(c.ρn_lcl)),
            (@name(q_icl), @name(n_ice), @name(c.ρq_icl), @name(c.ρn_ice)),
        )
            MatrixFields.has_field(Y, ρq_name) || continue
            ᶜρq = MatrixFields.get_field(Y, ρq_name)
            ᶜρqₜ = MatrixFields.get_field(Yₜ, ρq_name)
            @. ᶜratio =
                max(FT(0), min(FT(1), specific(ᶜρq, ᶜρ) / max(ᶜq_tot_eff, ϵ_FT)))
            @. ᶜρqₜ += ᶜratio * ᶜρχₜ_diffusion
            if apply_sgs_updraft
                for j in 1:n
                    if MatrixFields.has_field(Y.c.sgsʲs.:($j), q_name)
                        ᶜqⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), q_name)
                        @. ᶜqⱼₜ += ᶜratio * ᶜρχₜ_diffusion / ᶜρ
                    end
                end
            end
            if MatrixFields.has_field(Y, ρn_name)
                ᶜρn = MatrixFields.get_field(Y, ρn_name)
                ᶜρnₜ = MatrixFields.get_field(Yₜ, ρn_name)
                @. ᶜρnₜ +=
                    ᶜratio * max(FT(0), ᶜρn) / max(ᶜρq, ϵ_FT) * ᶜρχₜ_diffusion
                if apply_sgs_updraft
                    for j in 1:n
                        ᶜnⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), n_name)
                        @. ᶜnⱼₜ +=
                            ᶜratio * max(FT(0), ᶜρn) / max(ᶜρq, ϵ_FT) *
                            ᶜρχₜ_diffusion / ᶜρ
                    end
                end
            end
        end
    end

    @. Yₜ.c.ρe_tot += ᶜρe_totₜ_diffusion
    if apply_sgs_updraft
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).mse += ᶜρe_totₜ_diffusion / ᶜρ
        end
    end

    # Grid-mean tracers. The water tracers are handled above: `ρq_tot` and the
    # cloud species through the aggregate flux, rain and snow not at all. The
    # vertical path gives them only the interfacial-entrainment `K_e`, which
    # has no horizontal analogue, so they receive no horizontal transport.
    # Passive tracers use the unscaled `K_h`.
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ρχ_name in microphysics_tracer_names(Y) && return
        ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar_3
        @. ᶜρχₜ_diffusion = wdivₕ(ᶜρ * ᶜK_h_h * gradₕ(ᶜχ))
        @. ᶜρχₜ += ᶜρχₜ_diffusion
        if apply_sgs_updraft
            χ_name = specific_tracer_name(ρχ_name)
            for j in 1:n
                if MatrixFields.has_field(Y.c.sgsʲs.:($j), χ_name)
                    ᶜχⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                    @. ᶜχⱼₜ += ᶜρχₜ_diffusion / ᶜρ
                end
            end
        end
    end

    (; ᶜu, ᶠu) = p.precomputed

    # Turbulent TKE transport, and shear production from horizontal gradients;
    # the production from vertical gradients is applied in the TKE tendency.
    # The strain-rate invariant of the nodal spectral-element gradient is too
    # large at element-boundary nodes and too small inside (the rectified
    # end-of-interval error of polynomial differentiation), so it is replaced
    # within each element by its lumped GLL{2} restriction (`lumpedₕ`), which
    # conserves the element's integral of the invariant.
    if use_prognostic_tke(turbconv_model)
        @. Yₜ.c.ρtke += wdivₕ(ᶜρ * ᶜK_u_h * gradₕ(ᶜtke))
        ᶜS_h = compute_strain_rate_center_horizontal(ᶜu)
        @. Yₜ.c.ρtke += 2 * ᶜρ * ᶜK_u_h * lumpedₕ(norm_sqr(ᶜS_h))
    end

    # Momentum: horizontal weak divergence of the SGS stress `τ = -2 K_u S`
    # with the full strain rate. The vertical stress divergence is handled by
    # the vertical diffusion pathway.
    ᶠρ = @. p.scratch.ᶠtemp_scalar = ᶠinterp(ᶜρ)
    ᶜτ_h = compute_strain_rate_center_full!(p.scratch.ᶜtemp_UVWxUVW, ᶜu, ᶠu)
    @. ᶜτ_h = -2 * ᶜK_u_h * ᶜτ_h
    @. Yₜ.c.uₕ -= C12(wdivₕ(ᶜρ * ᶜτ_h) / ᶜρ)
    ᶠτ_h = compute_strain_rate_face_full!(p.scratch.ᶠtemp_UVWxUVW, ᶜu, ᶠu)
    @. ᶠτ_h = -2 * ᶠinterp(ᶜK_u_h) * ᶠτ_h
    @. Yₜ.f.u₃ -= C3(wdivₕ(ᶠρ * ᶠτ_h) / ᶠρ)

    return nothing
end
