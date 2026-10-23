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
    ᶜenv_suspended_water(Y, p)

Return `(q_tot_eff⁰, q_lcl⁰, q_icl⁰)` for the EDMFX environment subdomain. With
non-equilibrium microphysics, `q_tot_eff⁰ = q_tot⁰ - q_rai⁰ - q_sno⁰` and the
suspended cloud species are the diagnosed environment specific humidities; with
an equilibrium scheme, `q_tot_eff⁰ = q_tot⁰` and the cloud species are the
cached diagnostic `ᶜq_liq⁰` and `ᶜq_ice⁰`.
"""
function ᶜenv_suspended_water(Y, p)
    if p.atmos.microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
        ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), Y, p)
        ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), Y, p)
        ᶜq_tot_eff⁰ = @. lazy(ᶜq_tot⁰ - ᶜq_rai⁰ - ᶜq_sno⁰)
        ᶜq_lcl⁰ = ᶜspecific_env_value(@name(q_lcl), Y, p)
        ᶜq_icl⁰ = ᶜspecific_env_value(@name(q_icl), Y, p)
        return (ᶜq_tot_eff⁰, ᶜq_lcl⁰, ᶜq_icl⁰)
    else
        (; ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
        return (ᶜspecific_env_value(@name(q_tot), Y, p), ᶜq_liq⁰, ᶜq_ice⁰)
    end
end

"""
    ᶜupdraft_suspended_water(Y, p, ::Val{j})

Return `(q_tot_effʲ, q_lclʲ, q_iclʲ)` for updraft `j`. With non-equilibrium
microphysics, `q_tot_effʲ = q_totʲ - q_raiʲ - q_snoʲ` and the suspended cloud
species are read from `Y.c.sgsʲs.:(j)`; with an equilibrium scheme,
`q_tot_effʲ = q_totʲ` and the cloud species are the cached diagnostic
`ᶜq_liqʲs.:(j)` and `ᶜq_iceʲs.:(j)`.
"""
function ᶜupdraft_suspended_water(Y, p, ::Val{j}) where {j}
    if p.atmos.microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        ᶜq_tot_effʲ = @. lazy(
            Y.c.sgsʲs.:($$j).q_tot - Y.c.sgsʲs.:($$j).q_rai -
            Y.c.sgsʲs.:($$j).q_sno,
        )
        return (ᶜq_tot_effʲ, Y.c.sgsʲs.:($j).q_lcl, Y.c.sgsʲs.:($j).q_icl)
    else
        (; ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
        return (Y.c.sgsʲs.:($j).q_tot, ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j))
    end
end

"""
    ᶜupdraft_diffusing_mse!(ᶜmse_eff, Y, p, ::Val{j})

Return the moist static energy of updraft `j` with the enthalpy of
non-diffusing precipitation removed (`mseʲ - (h_l(Tʲ) - h_d(Tʲ)) q_raiʲ - (h_i(Tʲ) - h_d(Tʲ)) q_snoʲ`), writing into `ᶜmse_eff` for non-equilibrium
microphysics and returning `Y.c.sgsʲs.:(j).mse` directly for an equilibrium
scheme.
"""
function ᶜupdraft_diffusing_mse!(ᶜmse_eff, Y, p, ::Val{j}) where {j}
    ᶜmseʲ = Y.c.sgsʲs.:($j).mse
    if p.atmos.microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        thermo_params = CAP.thermodynamics_params(p.params)
        ᶜTʲ = p.precomputed.ᶜTʲs.:($j)
        ᶜq_raiʲ = Y.c.sgsʲs.:($j).q_rai
        ᶜq_snoʲ = Y.c.sgsʲs.:($j).q_sno
        @. ᶜmse_eff =
            ᶜmseʲ -
            (
                TD.enthalpy_liquid(thermo_params, ᶜTʲ) -
                TD.enthalpy_dry(thermo_params, ᶜTʲ)
            ) * ᶜq_raiʲ -
            (
                TD.enthalpy_ice(thermo_params, ᶜTʲ) -
                TD.enthalpy_dry(thermo_params, ᶜTʲ)
            ) * ᶜq_snoʲ
        return ᶜmse_eff
    else
        return ᶜmseʲ
    end
end

"""
    add_subdomain_water_enthalpy_diffusionᵥ!(
        ᶜρe_totₜ_diffusion, ᶠw, ᶜT_sub, ᶜq_tot_eff_sub, ᶜq_lcl_sub, ᶜq_icl_sub,
        p, ᶜscratch, ᶜq_scratch,
    )

Add the vertical divergence of the Yatunin et al. (2026, Eq. 33) water-enthalpy
diffusive flux for a single subdomain (weighted by face coefficient `ᶠw`) to
`ᶜρe_totₜ_diffusion`, evaluating `h_v(T_sub) + Φ`, `h_l(T_sub) - h_v(T_sub)`,
and `h_i(T_sub) - h_v(T_sub)` at the subdomain temperature `ᶜT_sub`.
"""
function add_subdomain_water_enthalpy_diffusionᵥ!(
    ᶜρe_totₜ_diffusion,
    ᶠw,
    ᶜT_sub,
    ᶜq_tot_eff_sub,
    ᶜq_lcl_sub,
    ᶜq_icl_sub,
    p,
    ᶜscratch,
    ᶜq_scratch,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core
    @. ᶜscratch = TD.enthalpy_vapor(thermo_params, ᶜT_sub) + ᶜΦ
    @. ᶜq_scratch = ᶜq_tot_eff_sub
    @. ᶜρe_totₜ_diffusion +=
        ᶜdiffdivᵥ(-(ᶠw * ᶠinterp(ᶜscratch) * ᶠgradᵥ(ᶜq_scratch)))
    @. ᶜscratch =
        TD.enthalpy_liquid(thermo_params, ᶜT_sub) -
        TD.enthalpy_vapor(thermo_params, ᶜT_sub)
    @. ᶜq_scratch = ᶜq_lcl_sub
    @. ᶜρe_totₜ_diffusion +=
        ᶜdiffdivᵥ(-(ᶠw * ᶠinterp(ᶜscratch) * ᶠgradᵥ(ᶜq_scratch)))
    @. ᶜscratch =
        TD.enthalpy_ice(thermo_params, ᶜT_sub) -
        TD.enthalpy_vapor(thermo_params, ᶜT_sub)
    @. ᶜq_scratch = ᶜq_icl_sub
    @. ᶜρe_totₜ_diffusion +=
        ᶜdiffdivᵥ(-(ᶠw * ᶠinterp(ᶜscratch) * ᶠgradᵥ(ᶜq_scratch)))
    return nothing
end

"""
    add_subdomain_water_enthalpy_diffusionₕ!(
        ᶜρe_totₜ_diffusion, ᶜw, ᶜT_sub, ᶜq_tot_eff_sub, ᶜq_lcl_sub, ᶜq_icl_sub,
        p, ᶜscratch, ᶜq_scratch,
    )

Add the horizontal weak divergence of the Yatunin et al. (2026, Eq. 33)
water-enthalpy diffusive flux for a single subdomain (weighted by center
coefficient `ᶜw`) to `ᶜρe_totₜ_diffusion`, evaluating `h_v(T_sub) + Φ`,
`h_l(T_sub) - h_v(T_sub)`, and `h_i(T_sub) - h_v(T_sub)` at `ᶜT_sub`.
"""
function add_subdomain_water_enthalpy_diffusionₕ!(
    ᶜρe_totₜ_diffusion,
    ᶜw,
    ᶜT_sub,
    ᶜq_tot_eff_sub,
    ᶜq_lcl_sub,
    ᶜq_icl_sub,
    p,
    ᶜscratch,
    ᶜq_scratch,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core
    @. ᶜscratch = TD.enthalpy_vapor(thermo_params, ᶜT_sub) + ᶜΦ
    @. ᶜq_scratch = ᶜq_tot_eff_sub
    @. ᶜρe_totₜ_diffusion += wdivₕ(ᶜw * ᶜscratch * gradₕ(ᶜq_scratch))
    @. ᶜscratch =
        TD.enthalpy_liquid(thermo_params, ᶜT_sub) -
        TD.enthalpy_vapor(thermo_params, ᶜT_sub)
    @. ᶜq_scratch = ᶜq_lcl_sub
    @. ᶜρe_totₜ_diffusion += wdivₕ(ᶜw * ᶜscratch * gradₕ(ᶜq_scratch))
    @. ᶜscratch =
        TD.enthalpy_ice(thermo_params, ᶜT_sub) -
        TD.enthalpy_vapor(thermo_params, ᶜT_sub)
    @. ᶜq_scratch = ᶜq_icl_sub
    @. ᶜρe_totₜ_diffusion += wdivₕ(ᶜw * ᶜscratch * gradₕ(ᶜq_scratch))
    return nothing
end

"""
    edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the divergence of the SGS diffusive (K-theory) fluxes of the PROPHET
scheme (`EDMFX` in code) to the grid-mean and updraft states.

All fluxes use the face-native eddy diffusivity/viscosity `ᶠK_h`/`ᶠK_u`
from `set_face_diffusivities!`:

  - Total enthalpy: the flux of Eq. (33) of Yatunin et al. (2026),
    `F_h = -ρ K_h [∇s_d + (h_v + Φ) ∇q_tot_eff + (h_l - h_v) ∇q_lcl + (h_i - h_v) ∇q_icl]`,
    where `q_tot_eff = q_tot - q_rai - q_sno` is the water that diffuses,
    applied to `Yₜ.c.ρe_tot`.
  - Total water and suspended cloud species: `q_tot_eff` diffuses with `K_h`
    (mirrored to `Yₜ.c.ρ` for moist mass conservation), while suspended cloud
    species (`ρq_lcl`, `ρq_icl`, `ρn_lcl`, `ρn_ice`) diffuse down their own
    gradients and rain and snow do not diffuse.
  - Other grid-scale tracers: passive tracers diffuse down their own gradients
    with `K_h`.
  - Momentum: `-2 ρ K_u 𝔈` with the vertical strain rate, applied to `Yₜ.c.uₕ`.
  - TKE (when prognostic): turbulent transport plus dissipation
    (`tke_dissipation`), applied to `Yₜ.c.ρtke`; negative TKE is relaxed to
    zero within one time step.

For `PrognosticEDMFX`, the grid-mean diffusive flux is the subdomain sum
`a⁰ F⁰ + Σⱼ aʲ Fʲ` (with harmonic-mean face `ρaʲ`), and when
`p.atmos.edmfx_model.vertical_diffusion` is enabled each updraft scalar `χʲ`
(`mse`, `q_tot`, suspended cloud species, and passive tracers) diffuses down its
own vertical gradient (using `q_tot_effʲ = q_totʲ - q_raiʲ - q_snoʲ` for
`q_totʲ`, matching the updraft contribution to `Yₜ.c.ρq_tot`). When
`p.atmos.edmfx_model.vertical_diffusion` is `false`, `PrognosticEDMFX` reduces
to the environment diffusive flux `a⁰ F⁰`.

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
    apply_sgs_updraft =
        turbconv_model isa PrognosticEDMFX &&
        p.atmos.edmfx_model.vertical_diffusion
    updraft_kh_factor = CAP.edmfx_updraft_kh_diffusion_factor(turbconv_params)

    if p.atmos.edmfx_model.sgs_diffusive_flux

        (; ᶠK_h, ᶠK_u, ᶜl_mix) = p.precomputed
        thermo_params = CAP.thermodynamics_params(params)
        (; ᶜΦ) = p.core

        if turbconv_model isa PrognosticEDMFX
            (; ᶜT⁰, ᶜTʲs) = p.precomputed
            ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
            ᶠw⁰ = p.scratch.ᶠtemp_scalar
            @. ᶠw⁰ = ᶠinterp(ᶜρa⁰) * ᶠK_h
            ᶠwʲ = p.scratch.ᶠtemp_scalar_2

            # Environment dry-static-energy diffusion
            ᶜρe_totₜ_diffusion = p.scratch.ᶜtemp_scalar_2
            ᶜρq_totₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜq_scratch = p.scratch.ᶜtemp_scalar_4
            @. ᶜρe_totₜ_diffusion =
                ᶜdiffdivᵥ(
                    -(
                        ᶠw⁰ *
                        ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT⁰, ᶜΦ))
                    ),
                )

            if !(p.atmos.microphysics_model isa DryModel)
                ᶜq_tot_eff⁰, ᶜq_lcl⁰, ᶜq_icl⁰ = ᶜenv_suspended_water(Y, p)
                add_subdomain_water_enthalpy_diffusionᵥ!(
                    ᶜρe_totₜ_diffusion,
                    ᶠw⁰,
                    ᶜT⁰,
                    ᶜq_tot_eff⁰,
                    ᶜq_lcl⁰,
                    ᶜq_icl⁰,
                    p,
                    p.scratch.ᶜtemp_scalar_3,
                    ᶜq_scratch,
                )
                @. ᶜq_scratch = ᶜq_tot_eff⁰
                @. ᶜρq_totₜ_diffusion = ᶜdiffdivᵥ(-(ᶠw⁰ * ᶠgradᵥ(ᶜq_scratch)))

                MatrixFields.unrolled_foreach(
                    (
                    (@name(q_lcl), @name(c.ρq_lcl)),
                    (@name(q_icl), @name(c.ρq_icl)),
                    (@name(n_lcl), @name(c.ρn_lcl)),
                    (@name(n_ice), @name(c.ρn_ice)),
                ),
                ) do (χ_name, ρχ_name)
                    MatrixFields.has_field(Y, ρχ_name) || return
                    ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                    ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
                    @. ᶜq_scratch = ᶜχ⁰
                    @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠw⁰ * ᶠgradᵥ(ᶜq_scratch)))
                end
            end

            if apply_sgs_updraft
                for j in 1:n
                    ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
                    @. ᶠwʲ =
                        updraft_kh_factor * ᶠK_h / ᶠinterp(1 / max(ᶜρaʲ, ϵ_FT))
                    ᶜinv_ρaʲ = @. lazy(1 / max(ᶜρaʲ, ϵ_FT))
                    ᶜTʲ = ᶜTʲs.:($j)
                    @. ᶜρe_totₜ_diffusion +=
                        ᶜdiffdivᵥ(
                            -(
                                ᶠwʲ * ᶠgradᵥ(
                                    TD.dry_static_energy(
                                        thermo_params,
                                        ᶜTʲ,
                                        ᶜΦ,
                                    ),
                                )
                            ),
                        )
                    ᶜmse_effʲ = ᶜupdraft_diffusing_mse!(
                        p.scratch.ᶜtemp_scalar_3,
                        Y,
                        p,
                        Val(j),
                    )
                    @. Yₜ.c.sgsʲs.:($$j).mse -=
                        ᶜinv_ρaʲ * ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜmse_effʲ)))

                    if !(p.atmos.microphysics_model isa DryModel)
                        ᶜq_tot_effʲ, ᶜq_lclʲ, ᶜq_iclʲ =
                            ᶜupdraft_suspended_water(Y, p, Val(j))
                        add_subdomain_water_enthalpy_diffusionᵥ!(
                            ᶜρe_totₜ_diffusion,
                            ᶠwʲ,
                            ᶜTʲ,
                            ᶜq_tot_effʲ,
                            ᶜq_lclʲ,
                            ᶜq_iclʲ,
                            p,
                            p.scratch.ᶜtemp_scalar_3,
                            ᶜq_scratch,
                        )
                        @. ᶜq_scratch = ᶜq_tot_effʲ
                        ᶜρχₜ_up = p.scratch.ᶜtemp_scalar_3
                        @. ᶜρχₜ_up = ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜq_scratch)))
                        @. ᶜρq_totₜ_diffusion += ᶜρχₜ_up
                        @. Yₜ.c.sgsʲs.:($$j).q_tot -= ᶜinv_ρaʲ * ᶜρχₜ_up

                        MatrixFields.unrolled_foreach(
                            (
                            (@name(q_lcl), @name(c.ρq_lcl)),
                            (@name(q_icl), @name(c.ρq_icl)),
                            (@name(n_lcl), @name(c.ρn_lcl)),
                            (@name(n_ice), @name(c.ρn_ice)),
                        ),
                        ) do (χ_name, ρχ_name)
                            MatrixFields.has_field(Y.c.sgsʲs.:($j), χ_name) ||
                                return
                            ᶜχʲ = MatrixFields.get_field(
                                Y.c.sgsʲs.:($j),
                                χ_name,
                            )
                            ᶜχʲₜ = MatrixFields.get_field(
                                Yₜ.c.sgsʲs.:($j),
                                χ_name,
                            )
                            ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                            ᶜρχₜ_up = p.scratch.ᶜtemp_scalar_3
                            @. ᶜρχₜ_up = ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜχʲ)))
                            @. ᶜρχₜ -= ᶜρχₜ_up
                            @. ᶜχʲₜ -= ᶜinv_ρaʲ * ᶜρχₜ_up
                        end
                    end
                end
            end

            @. Yₜ.c.ρe_tot -= ᶜρe_totₜ_diffusion
            if !(p.atmos.microphysics_model isa DryModel)
                @. Yₜ.c.ρq_tot -= ᶜρq_totₜ_diffusion
                @. Yₜ.c.ρ -= ᶜρq_totₜ_diffusion
            end

            foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
                ρχ_name in microphysics_tracer_names(Y) && return
                χ_name = specific_tracer_name(ρχ_name)
                if MatrixFields.has_field(Y.c.sgsʲs.:(1), χ_name)
                    ᶜq_scratch = p.scratch.ᶜtemp_scalar_4
                    ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
                    @. ᶜq_scratch = ᶜχ⁰
                    @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠw⁰ * ᶠgradᵥ(ᶜq_scratch)))
                    if apply_sgs_updraft
                        for j in 1:n
                            ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
                            @. ᶠwʲ =
                                updraft_kh_factor * ᶠK_h /
                                ᶠinterp(1 / max(ᶜρaʲ, ϵ_FT))
                            ᶜinv_ρaʲ = @. lazy(1 / max(ᶜρaʲ, ϵ_FT))
                            ᶜχʲ =
                                MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                            ᶜχʲₜ =
                                MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                            ᶜρχₜ_up = p.scratch.ᶜtemp_scalar_3
                            @. ᶜρχₜ_up = ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜχʲ)))
                            @. ᶜρχₜ -= ᶜρχₜ_up
                            @. ᶜχʲₜ -= ᶜinv_ρaʲ * ᶜρχₜ_up
                        end
                    end
                else
                    ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
                    @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠinterp(Y.c.ρ) * ᶠK_h * ᶠgradᵥ(ᶜχ)))
                end
            end
        else
            ᶠρK_h = p.scratch.ᶠtemp_scalar
            @. ᶠρK_h = ᶠinterp(Y.c.ρ) * ᶠK_h
            (; ᶜT) = p.precomputed
            ᶜρe_totₜ_diffusion = p.scratch.ᶜtemp_scalar_2
            @. ᶜρe_totₜ_diffusion =
                ᶜdiffdivᵥ(
                    -(
                        ᶠρK_h *
                        ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ))
                    ),
                )

            if !(p.atmos.microphysics_model isa DryModel)
                add_water_enthalpy_diffusionᵥ!(
                    ᶜρe_totₜ_diffusion,
                    ᶠρK_h,
                    Y,
                    p,
                    p.scratch.ᶜtemp_scalar_3;
                    sign = +1,
                )
                ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
                ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
                @. ᶜρχₜ_diffusion = ᶜdiffdivᵥ(-(ᶠρK_h * ᶠgradᵥ(ᶜq_tot_eff)))
                @. Yₜ.c.ρq_tot -= ᶜρχₜ_diffusion
                @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
                diffuse_suspended_cloud_speciesᵥ!(Yₜ, Y, ᶠρK_h)
            end

            @. Yₜ.c.ρe_tot -= ᶜρe_totₜ_diffusion

            foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
                ρχ_name in microphysics_tracer_names(Y) && return
                ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
                @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠρK_h * ᶠgradᵥ(ᶜχ)))
            end
        end

        ᶠρaK_u = p.scratch.ᶠtemp_scalar_2
        @. ᶠρaK_u = ᶠinterp(Y.c.ρ) * ᶠK_u

        if use_prognostic_tke(turbconv_model)
            (; ρtke_flux) = p.precomputed
            ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
            ᶜdivᵥ_ρtke = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(ρtke_flux),
            )
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

        # Momentum diffusion
        ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
        @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Y.c.ρ)
    end

    return nothing
end

"""
    edmfx_sgs_horizontal_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the horizontal component of the PROPHET (`EDMFX`) SGS diffusive flux
tendency using the horizontal eddy viscosity `ᶜK_u_h` and diffusivity `ᶜK_h_h`
computed with the horizontal node spacing `Δx_h`:

  - **Momentum (`uₕ`, `u₃`)** and **TKE (`ρtke`)**: diffused on the grid mean
    with `ᶜK_u_h`.
  - **Thermodynamics, water, and tracers**: for `EDOnlyEDMFX`, diffused on the
    grid mean via `add_water_enthalpy_diffusionₕ!` and
    `diffuse_suspended_cloud_speciesₕ!`; for `PrognosticEDMFX`, assembled from
    the environment (`ρa⁰ K_h_h ∇ₕψ⁰`) plus, when
    `edmfx_model.horizontal_diffusion` is active, the updrafts
    (`ρaʲ f_j K_h_h ∇ₕψʲ` divided by `max(ρaʲ, ρ a_min, ε)` in the intensive
    updraft equations). Rain and snow do not diffuse.

Always explicit; applied in the explicit remainder independently of `diff_mode`.
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
    turbconv_params = CAP.turbconv_params(params)
    (; ᶜK_u_h, ᶜK_h_h) = p.precomputed
    ᶜρ = Y.c.ρ
    ᶜtke = @. lazy(specific(Y.c.ρtke, ᶜρ))

    n = n_mass_flux_subdomains(turbconv_model)
    apply_sgs_updraft =
        turbconv_model isa PrognosticEDMFX &&
        p.atmos.edmfx_model.horizontal_diffusion isa Val{true}
    updraft_kh_factor = CAP.edmfx_updraft_kh_diffusion_factor(turbconv_params)
    min_a = CAP.min_area(turbconv_params)

    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜΦ) = p.core
    ᶜρe_totₜ_diffusion = p.scratch.ᶜtemp_scalar_2

    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜTʲs) = p.precomputed
        ᶜρa⁰ = @. lazy(ρa⁰(ᶜρ, Y.c.sgsʲs, turbconv_model))
        ᶜw⁰ = @. lazy(ᶜρa⁰ * ᶜK_h_h)
        ᶜρq_totₜ_diffusion = p.scratch.ᶜtemp_scalar_3
        ᶜq_scratch = p.scratch.ᶜtemp_scalar_4
        @. ᶜρe_totₜ_diffusion =
            wdivₕ(ᶜw⁰ * gradₕ(TD.dry_static_energy(thermo_params, ᶜT⁰, ᶜΦ)))

        if !(p.atmos.microphysics_model isa DryModel)
            ᶜq_tot_eff⁰, ᶜq_lcl⁰, ᶜq_icl⁰ = ᶜenv_suspended_water(Y, p)
            add_subdomain_water_enthalpy_diffusionₕ!(
                ᶜρe_totₜ_diffusion,
                ᶜw⁰,
                ᶜT⁰,
                ᶜq_tot_eff⁰,
                ᶜq_lcl⁰,
                ᶜq_icl⁰,
                p,
                p.scratch.ᶜtemp_scalar_6,
                ᶜq_scratch,
            )
            @. ᶜq_scratch = ᶜq_tot_eff⁰
            @. ᶜρq_totₜ_diffusion = wdivₕ(ᶜw⁰ * gradₕ(ᶜq_scratch))

            MatrixFields.unrolled_foreach(
                (
                (@name(q_lcl), @name(c.ρq_lcl)),
                (@name(q_icl), @name(c.ρq_icl)),
                (@name(n_lcl), @name(c.ρn_lcl)),
                (@name(n_ice), @name(c.ρn_ice)),
            ),
            ) do (χ_name, ρχ_name)
                MatrixFields.has_field(Y, ρχ_name) || return
                ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
                @. ᶜq_scratch = ᶜχ⁰
                @. ᶜρχₜ += wdivₕ(ᶜw⁰ * gradₕ(ᶜq_scratch))
            end
        end

        if apply_sgs_updraft
            for j in 1:n
                ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
                ᶜwʲ = @. lazy(ᶜρaʲ * updraft_kh_factor * ᶜK_h_h)
                ᶜinv_ρaʲ = @. lazy(1 / max(ᶜρaʲ, ᶜρ * min_a, ϵ_FT))
                ᶜTʲ = ᶜTʲs.:($j)
                @. ᶜρe_totₜ_diffusion +=
                    wdivₕ(
                        ᶜwʲ *
                        gradₕ(TD.dry_static_energy(thermo_params, ᶜTʲ, ᶜΦ)),
                    )
                ᶜmse_effʲ = ᶜupdraft_diffusing_mse!(
                    p.scratch.ᶜtemp_scalar_6,
                    Y,
                    p,
                    Val(j),
                )
                @. Yₜ.c.sgsʲs.:($$j).mse +=
                    ᶜinv_ρaʲ * wdivₕ(ᶜwʲ * gradₕ(ᶜmse_effʲ))

                if !(p.atmos.microphysics_model isa DryModel)
                    ᶜq_tot_effʲ, ᶜq_lclʲ, ᶜq_iclʲ =
                        ᶜupdraft_suspended_water(Y, p, Val(j))
                    add_subdomain_water_enthalpy_diffusionₕ!(
                        ᶜρe_totₜ_diffusion,
                        ᶜwʲ,
                        ᶜTʲ,
                        ᶜq_tot_effʲ,
                        ᶜq_lclʲ,
                        ᶜq_iclʲ,
                        p,
                        p.scratch.ᶜtemp_scalar_6,
                        ᶜq_scratch,
                    )
                    @. ᶜq_scratch = ᶜq_tot_effʲ
                    ᶜρχₜ_up = p.scratch.ᶜtemp_scalar_6
                    @. ᶜρχₜ_up = wdivₕ(ᶜwʲ * gradₕ(ᶜq_scratch))
                    @. ᶜρq_totₜ_diffusion += ᶜρχₜ_up
                    @. Yₜ.c.sgsʲs.:($$j).q_tot += ᶜinv_ρaʲ * ᶜρχₜ_up

                    MatrixFields.unrolled_foreach(
                        (
                        (@name(q_lcl), @name(c.ρq_lcl)),
                        (@name(q_icl), @name(c.ρq_icl)),
                        (@name(n_lcl), @name(c.ρn_lcl)),
                        (@name(n_ice), @name(c.ρn_ice)),
                    ),
                    ) do (χ_name, ρχ_name)
                        MatrixFields.has_field(Y.c.sgsʲs.:($j), χ_name) ||
                            return
                        ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                        ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                        ᶜρχₜ = MatrixFields.get_field(Yₜ, ρχ_name)
                        ᶜρχₜ_up = p.scratch.ᶜtemp_scalar_6
                        @. ᶜρχₜ_up = wdivₕ(ᶜwʲ * gradₕ(ᶜχʲ))
                        @. ᶜρχₜ += ᶜρχₜ_up
                        @. ᶜχʲₜ += ᶜinv_ρaʲ * ᶜρχₜ_up
                    end
                end
            end
        end

        @. Yₜ.c.ρe_tot += ᶜρe_totₜ_diffusion
        if !(p.atmos.microphysics_model isa DryModel)
            @. Yₜ.c.ρq_tot += ᶜρq_totₜ_diffusion
            @. Yₜ.c.ρ += ᶜρq_totₜ_diffusion
        end

        foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
            ρχ_name in microphysics_tracer_names(Y) && return
            χ_name = specific_tracer_name(ρχ_name)
            if MatrixFields.has_field(Y.c.sgsʲs.:(1), χ_name)
                ᶜq_scratch = p.scratch.ᶜtemp_scalar_4
                ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
                @. ᶜq_scratch = ᶜχ⁰
                @. ᶜρχₜ += wdivₕ(ᶜw⁰ * gradₕ(ᶜq_scratch))
                if apply_sgs_updraft
                    for j in 1:n
                        ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
                        ᶜwʲ = @. lazy(ᶜρaʲ * updraft_kh_factor * ᶜK_h_h)
                        ᶜinv_ρaʲ = @. lazy(1 / max(ᶜρaʲ, ᶜρ * min_a, ϵ_FT))
                        ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                        ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                        ᶜρχₜ_up = p.scratch.ᶜtemp_scalar_6
                        @. ᶜρχₜ_up = wdivₕ(ᶜwʲ * gradₕ(ᶜχʲ))
                        @. ᶜρχₜ += ᶜρχₜ_up
                        @. ᶜχʲₜ += ᶜinv_ρaʲ * ᶜρχₜ_up
                    end
                end
            else
                ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
                @. ᶜρχₜ += wdivₕ(ᶜρ * ᶜK_h_h * gradₕ(ᶜχ))
            end
        end
    else
        (; ᶜT) = p.precomputed
        ᶜρK_h_h = @. lazy(ᶜρ * ᶜK_h_h)
        @. ᶜρe_totₜ_diffusion =
            wdivₕ(ᶜρK_h_h * gradₕ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)))

        if !(p.atmos.microphysics_model isa DryModel)
            add_water_enthalpy_diffusionₕ!(
                ᶜρe_totₜ_diffusion,
                ᶜρK_h_h,
                Y,
                p,
                p.scratch.ᶜtemp_scalar_6,
            )
            ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar_3
            @. ᶜρχₜ_diffusion = wdivₕ(ᶜρK_h_h * gradₕ(ᶜq_tot_eff))
            @. Yₜ.c.ρq_tot += ᶜρχₜ_diffusion
            @. Yₜ.c.ρ += ᶜρχₜ_diffusion
            diffuse_suspended_cloud_speciesₕ!(Yₜ, Y, ᶜρK_h_h)
        end

        @. Yₜ.c.ρe_tot += ᶜρe_totₜ_diffusion

        foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
            ρχ_name in microphysics_tracer_names(Y) && return
            ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
            @. ᶜρχₜ += wdivₕ(ᶜρK_h_h * gradₕ(ᶜχ))
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
