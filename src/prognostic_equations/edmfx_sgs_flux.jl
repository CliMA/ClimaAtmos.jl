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
`edmfx_sgs_mass_flux_tendency!`, so its temporaries are loop
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
    MatrixFields.unrolled_foreach(
        _unrolled_zip(ρχ_names, ᶜχ_subs),
    ) do (ρχ_name, ᶜχ_sub)
        ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
        ᶜa_scalar_χ = @. (ᶜχ_sub - specific(ᶜρχ, Yc.ρ)) * ᶜa
        vtt_χ = vertical_transport(ᶜρ, ᶠu³_diff, ᶜa_scalar_χ, dt, upwinding)
        @. ᶜρχₜ += vtt_χ
    end
    return nothing
end

"""
    ᶜenv_diffusing_water(Y, p)

Return `(q_tot_eff⁰, q_lcl⁰, q_icl⁰)`, the diffusing total, liquid, and ice water
of the EDMFX environment (the subdomain counterpart of
`ᶜdiffusing_water_phases`). With non-equilibrium microphysics,
`q_tot_eff⁰ = q_tot⁰ - q_rai⁰ - q_sno⁰` and the suspended cloud species are the
diagnosed environment specific humidities; with an equilibrium scheme,
`q_tot_eff⁰ = q_tot⁰` and the cloud species are the cached diagnostic `ᶜq_liq⁰`
and `ᶜq_ice⁰`.
"""
function ᶜenv_diffusing_water(Y, p)
    if p.atmos.microphysics_model isa NonEquilibriumMicrophysics
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

@inline function ᶜenv_diffusing_water(
    Yc,
    ᶜρa⁰,
    ᶜq_liq⁰,
    ᶜq_ice⁰,
    turbconv_model,
    microphysics_model,
)
    if microphysics_model isa NonEquilibriumMicrophysics
        ᶜq_tot_eff⁰ = @. specific_env_value(
            @name(q_tot), Yc.ρq_tot, Yc.ρ, Yc.sgsʲs, ᶜρa⁰, turbconv_model,
        ) - specific_env_value(
            @name(q_rai), Yc.ρq_rai, Yc.ρ, Yc.sgsʲs, ᶜρa⁰, turbconv_model,
        ) - specific_env_value(
            @name(q_sno), Yc.ρq_sno, Yc.ρ, Yc.sgsʲs, ᶜρa⁰, turbconv_model,
        )
        ᶜq_lcl⁰ = @. specific_env_value(
            @name(q_lcl), Yc.ρq_lcl, Yc.ρ, Yc.sgsʲs, ᶜρa⁰, turbconv_model,
        )
        ᶜq_icl⁰ = @. specific_env_value(
            @name(q_icl), Yc.ρq_icl, Yc.ρ, Yc.sgsʲs, ᶜρa⁰, turbconv_model,
        )
        return (ᶜq_tot_eff⁰, ᶜq_lcl⁰, ᶜq_icl⁰)
    else
        ᶜq_tot⁰ = @. specific_env_value(
            @name(q_tot), Yc.ρq_tot, Yc.ρ, Yc.sgsʲs, ᶜρa⁰, turbconv_model,
        )
        return (ᶜq_tot⁰, ᶜq_liq⁰, ᶜq_ice⁰)
    end
end

"""
    ᶜupdraft_diffusing_water(Y, p, ::Val{j})

Return `(q_tot_effʲ, q_lclʲ, q_iclʲ)`, the diffusing total, liquid, and ice water
of updraft `j`. With non-equilibrium microphysics,
`q_tot_effʲ = q_totʲ - q_raiʲ - q_snoʲ` and the suspended cloud species are read
from `Y.c.sgsʲs.:(j)`; with an equilibrium scheme, `q_tot_effʲ = q_totʲ` and the
cloud species are the cached diagnostic `ᶜq_liqʲs.:(j)` and `ᶜq_iceʲs.:(j)`.
"""
function ᶜupdraft_diffusing_water(Y, p, ::Val{j}) where {j}
    if p.atmos.microphysics_model isa NonEquilibriumMicrophysics
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

@inline function ᶜupdraft_diffusing_water(
    ᶜsgsʲ,
    ᶜq_liqʲ,
    ᶜq_iceʲ,
    microphysics_model,
)
    if microphysics_model isa NonEquilibriumMicrophysics
        ᶜq_tot_effʲ = @. ᶜsgsʲ.q_tot - ᶜsgsʲ.q_rai - ᶜsgsʲ.q_sno
        return (ᶜq_tot_effʲ, ᶜsgsʲ.q_lcl, ᶜsgsʲ.q_icl)
    else
        return (ᶜsgsʲ.q_tot, ᶜq_liqʲ, ᶜq_iceʲ)
    end
end

"""
    ᶜupdraft_diffusing_mse!(ᶜmse_eff, Y, p, ::Val{j})

Return the moist static energy of updraft `j` with the enthalpy of the
non-diffusing precipitation replaced by that of dry air,

    mse_effʲ = mseʲ - (h_l(Tʲ) - h_d(Tʲ)) q_raiʲ - (h_i(Tʲ) - h_d(Tʲ)) q_snoʲ,

written into `ᶜmse_eff` with non-equilibrium microphysics. With an equilibrium
scheme, return `Y.c.sgsʲs.:(j).mse` itself.
"""
function ᶜupdraft_diffusing_mse!(ᶜmse_eff, Y, p, ::Val{j}) where {j}
    ᶜmseʲ = Y.c.sgsʲs.:($j).mse
    if p.atmos.microphysics_model isa NonEquilibriumMicrophysics
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

@inline function ᶜupdraft_diffusing_mse!(
    ᶜmse_eff,
    ᶜsgsʲ,
    ᶜTʲ,
    thermo_params,
    microphysics_model,
)
    ᶜmseʲ = ᶜsgsʲ.mse
    if microphysics_model isa NonEquilibriumMicrophysics
        ᶜq_raiʲ = ᶜsgsʲ.q_rai
        ᶜq_snoʲ = ᶜsgsʲ.q_sno
        if ᶜmse_eff === nothing
            return @. ᶜmseʲ -
                      (
                TD.enthalpy_liquid(thermo_params, ᶜTʲ) -
                TD.enthalpy_dry(thermo_params, ᶜTʲ)
            ) * ᶜq_raiʲ -
                      (
                TD.enthalpy_ice(thermo_params, ᶜTʲ) -
                TD.enthalpy_dry(thermo_params, ᶜTʲ)
            ) * ᶜq_snoʲ
        else
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
        end
    else
        return ᶜmseʲ
    end
end

"""
    edmfx_diffused_scalars(Y)

Return the `(χ_name, ρχ_name)` pairs of the grid-scale scalars that the PROPHET
diffusive fluxes transport down their own gradients besides energy and total
water: the `suspended_cloud_species` present in `Y`, then the passive tracers.

A scalar that the updrafts carry diffuses down its subdomain gradients; one that
they do not (such as the P3 ice number and rime quantities) diffuses down its
grid-mean gradient.
"""
edmfx_diffused_scalars(Y) = (
    unrolled_filter(
        pair -> MatrixFields.has_field(Y.c, pair[2]),
        suspended_cloud_species,
    )...,
    unrolled_map(
        ρχ_name -> (specific_tracer_name(ρχ_name), ρχ_name),
        passive_gs_tracer_names(Y),
    )...,
)

"""
    ᶠupdraft_diffusion_weight!(ᶠwʲ, ᶜρaʲ, ᶠK_h, f_j)

Write into `ᶠwʲ` (or allocate a column register field when `ᶠwʲ` is `nothing`)
and return the face weight of the vertical updraft diffusive flux `-ᶠwʲ ∇χʲ`,

    ᶠwʲ = f_j K_h H(ρaʲ),    H(x, y) = 2 / (1/x + 1/y),

where `H` is the harmonic (series-conductance) face mean of the updraft mass
`ρaʲ` floored at `eps` and `f_j` is `edmfx_updraft_kh_diffusion_factor`.

The intensive updraft equation divides the flux divergence by the cell value
`ρaʲ_c`, which is exact because `ρaʲ` evolves only through advection,
entrainment, and detrainment. Since
`H(ρaʲ_c, ρaʲ_nb) / ρaʲ_c = 2 ρaʲ_nb / (ρaʲ_c + ρaʲ_nb) ≤ 2`, that division
cancels against the series conductance: the intensive tendency is a three-point
diffusion of `χʲ` with effective face diffusivities between `0` and `2 f_j K_h`,
so the updraft relaxes toward its vertical neighbors at most twice as fast as
grid-mean diffusion at the same `K_h` for every value of `ρaʲ_c`, and the
extensive flux through a face bounded by a draft-free cell vanishes. (An
arithmetic face mean amplifies the intensive tendency by `ρaʲ_nb / ρaʲ_c`.)
"""
@inline function ᶠupdraft_diffusion_weight!(ᶠwʲ, ᶜρaʲ, ᶠK_h, f_j)
    ϵ_FT = eps(eltype(ᶠwʲ))
    @. ᶠwʲ = f_j * ᶠK_h / ᶠinterp(1 / max(ᶜρaʲ, ϵ_FT))
    return ᶠwʲ
end
@inline function ᶠupdraft_diffusion_weight!(::Nothing, ᶜρaʲ, ᶠK_h, f_j)
    ϵ_FT = eps(eltype(ᶠK_h))
    return @. f_j * ᶠK_h / ᶠinterp(1 / max(ᶜρaʲ, ϵ_FT))
end

"""
    edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the divergence of the SGS diffusive (K-theory) fluxes of the PROPHET
scheme (`EDMFX` in code) to the grid-mean and updraft states.

All fluxes use the face-native eddy diffusivity/viscosity `ᶠK_h`/`ᶠK_u`
from `set_face_diffusivities!`:

  - Total enthalpy: the flux
    `F_h = -ρ K_h [c_p ∇T + ∇Φ + (h_v + Φ) ∇q_tot_eff + (h_l - h_v) ∇q_lcl + (h_i - h_v) ∇q_icl]`
    (`add_sensible_heat_diffusionᵥ!`, `add_water_enthalpy_diffusionᵥ!`), with
    `c_p` the heat capacity of the diffusing mixture and
    `q_tot_eff = q_tot - q_rai - q_sno` the water that diffuses, applied to
    `Yₜ.c.ρe_tot`.
  - Total water and suspended cloud species: `q_tot_eff` diffuses with `K_h`
    (mirrored to `Yₜ.c.ρ` for moist mass conservation), while the
    `suspended_cloud_species` diffuse down their own gradients and rain and
    snow do not diffuse.
  - Other grid-scale tracers: passive tracers diffuse down their own gradients
    with `K_h`.
  - Momentum: `-2 ρ K_u 𝔈` with the vertical strain rate, applied to `Yₜ.c.uₕ`.
  - TKE (when prognostic): turbulent transport plus dissipation
    (`tke_dissipation`), applied to `Yₜ.c.ρtke`; negative TKE is relaxed to
    zero within one time step.

For `PrognosticEDMFX`, the grid-mean diffusive flux is the subdomain sum
`a⁰ F⁰ + Σⱼ aʲ Fʲ`, and when `p.atmos.edmfx_model.vertical_diffusion` is
enabled each updraft scalar `χʲ` (`mse`, `q_tot`, suspended cloud species, and
passive tracers) diffuses down its own vertical gradient (using
`q_tot_effʲ = q_totʲ - q_raiʲ - q_snoʲ` for `q_totʲ`, matching the updraft
contribution to `Yₜ.c.ρq_tot`). The updraft mass `ρaʲ` evolves only through
advection, entrainment, and detrainment, so the intensive updraft tendency is
the extensive flux divergence divided by `ρaʲ`, with the harmonic face weight of
`ᶠupdraft_diffusion_weight!` bounding the intensive relaxation rate by
`2 f_j K_h / Δz²` for every value of `ρaʲ`. The grid mean receives the draft
flux with each subdomain's `c_p` and `(h_μ + Φ) ∇q_μʲ`, which exceeds `∇mse_effʲ`
by `(h_d + Φ) ∇q_tot_effʲ`: the moist mass that `q_tot` diffusion adds and the
enthalpy of the dry air it displaces are both absorbed by the environment
residual. When
`p.atmos.edmfx_model.vertical_diffusion` is `false`, `PrognosticEDMFX` reduces
to the environment diffusive flux `a⁰ F⁰`. Grid-scale scalars that the updrafts
do not carry diffuse down their grid-mean gradients with `ρ K_h`.

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
    moist = !(p.atmos.microphysics_model isa DryModel)

    if p.atmos.edmfx_model.sgs_diffusive_flux

        (; ᶠK_h, ᶠK_u, ᶜl_mix) = p.precomputed
        ᶜρ = Y.c.ρ
        ᶠρK_h = p.scratch.ᶠtemp_scalar_3
        @. ᶠρK_h = ᶠinterp(ᶜρ) * ᶠK_h

        if turbconv_model isa PrognosticEDMFX
            (; ᶜT⁰, ᶜTʲs) = p.precomputed
            ᶜρa⁰ = @. lazy(ρa⁰(ᶜρ, Y.c.sgsʲs, turbconv_model))
            ᶠw⁰ = p.scratch.ᶠtemp_scalar
            @. ᶠw⁰ = ᶠinterp(ᶜρa⁰) * ᶠK_h
            ᶠwʲ = p.scratch.ᶠtemp_scalar_2
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜscratch = p.scratch.ᶜtemp_scalar_3
            ᶜq_scratch = p.scratch.ᶜtemp_scalar_4

            # Environment: sensible heat, water enthalpy, and total water down
            # their environmental gradients. The blend flag and its face
            # fluxes are bound once here so that the closure over the
            # environment scalars captures them unboxed; the scratch fields
            # exist only with the 1M scheme.
            ᶜq_water⁰ = moist ? ᶜenv_diffusing_water(Y, p) : nothing
            blend = blended_cloud_species_flux(Y, p)
            ᶠF_qt⁰ = blend ? p.scratch.ᶠtemp_C3 : nothing
            ᶠF_sl⁰ = blend ? p.scratch.ᶠtemp_C3_2 : nothing
            add_sensible_heat_diffusionᵥ!(
                Yₜ.c.ρe_tot,
                ᶠw⁰,
                ᶜT⁰,
                ᶜq_water⁰,
                p,
                ᶜscratch,
            )
            if moist
                add_water_enthalpy_diffusionᵥ!(
                    Yₜ.c.ρe_tot,
                    ᶠw⁰,
                    ᶜT⁰,
                    ᶜq_water⁰,
                    p,
                    ᶜscratch;
                    ᶜq_scratch,
                )
                ᶜq_tot⁰ = maybe_materialize!(ᶜq_scratch, ᶜq_water⁰[1])
                @. ᶜρχₜ_diffusion = -(ᶜdiffdivᵥ(-(ᶠw⁰ * ᶠgradᵥ(ᶜq_tot⁰))))
                @. Yₜ.c.ρq_tot += ᶜρχₜ_diffusion
                @. Yₜ.c.ρ += ᶜρχₜ_diffusion
                # Face fluxes of environment total water and liquid–ice static
                # energy for the blended cloud-species flux
                if blend
                    @. ᶠF_qt⁰ = -(ᶠw⁰ * ᶠgradᵥ(ᶜq_tot⁰))
                    ᶠliquid_ice_static_energy_flux!(
                        ᶠF_sl⁰,
                        ᶠw⁰,
                        ᶜT⁰,
                        ᶜq_water⁰,
                        p,
                        ᶜscratch,
                        ᶜq_scratch,
                    )
                end
            end

            # Environment: the other scalars down their environmental
            # gradients, or their grid-mean gradients if the updrafts do not
            # carry them; cloud liquid and ice take the blended flux of
            # add_cloud_species_diffusionᵥ!, whose scratch fields exist only
            # with the 1M scheme
            ᶜblend_scratch =
                blend ?
                (
                    p.scratch.ᶜtemp_scalar_8,
                    p.scratch.ᶜtemp_scalar_9,
                    p.scratch.ᶜtemp_scalar_10,
                ) : nothing
            MatrixFields.unrolled_foreach(
                edmfx_diffused_scalars(Y),
            ) do (χ_name, ρχ_name)
                ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
                phase = cloud_species_phase(χ_name)
                if blend && !isnothing(phase)
                    ᶜχ⁰ = maybe_materialize!(
                        ᶜq_scratch,
                        ᶜspecific_env_value(χ_name, Y, p),
                    )
                    add_cloud_species_diffusionᵥ!(
                        ᶜρχₜ,
                        Y,
                        p,
                        ᶠw⁰,
                        ᶠF_qt⁰,
                        ᶠF_sl⁰,
                        ᶜT⁰,
                        ᶜq_water⁰,
                        p.precomputed.ᶜsgs_moments,
                        ᶜχ⁰,
                        phase,
                        ᶜblend_scratch,
                    )
                elseif MatrixFields.has_field(Y.c.sgsʲs.:(1), χ_name)
                    ᶜχ⁰ = maybe_materialize!(
                        ᶜq_scratch,
                        ᶜspecific_env_value(χ_name, Y, p),
                    )
                    @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠw⁰ * ᶠgradᵥ(ᶜχ⁰)))
                else
                    ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
                    @. ᶜρχₜ -=
                        ᶜdiffdivᵥ(-(ᶠρK_h * ᶠgradᵥ(specific(ᶜρχ, ᶜρ))))
                end
            end

            # Updrafts: each scalar down its own gradient, with the extensive
            # flux divergence added to the grid mean
            if apply_sgs_updraft
                for j in 1:n
                    ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
                    ᶠupdraft_diffusion_weight!(
                        ᶠwʲ,
                        ᶜρaʲ,
                        ᶠK_h,
                        updraft_kh_factor,
                    )
                    ᶜinv_ρaʲ = @. lazy(1 / max(ᶜρaʲ, ϵ_FT))
                    ᶜTʲ = ᶜTʲs.:($j)
                    ᶜq_waterʲ =
                        moist ? ᶜupdraft_diffusing_water(Y, p, Val(j)) : nothing
                    add_sensible_heat_diffusionᵥ!(
                        Yₜ.c.ρe_tot,
                        ᶠwʲ,
                        ᶜTʲ,
                        ᶜq_waterʲ,
                        p,
                        ᶜscratch,
                    )
                    ᶜmse_effʲ = ᶜupdraft_diffusing_mse!(ᶜscratch, Y, p, Val(j))
                    @. Yₜ.c.sgsʲs.:($$j).mse -=
                        ᶜinv_ρaʲ * ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜmse_effʲ)))

                    if moist
                        add_water_enthalpy_diffusionᵥ!(
                            Yₜ.c.ρe_tot,
                            ᶠwʲ,
                            ᶜTʲ,
                            ᶜq_waterʲ,
                            p,
                            ᶜscratch;
                            ᶜq_scratch,
                        )
                        ᶜq_totʲ = maybe_materialize!(ᶜq_scratch, ᶜq_waterʲ[1])
                        @. ᶜρχₜ_diffusion =
                            -(ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜq_totʲ))))
                        @. Yₜ.c.ρq_tot += ᶜρχₜ_diffusion
                        @. Yₜ.c.ρ += ᶜρχₜ_diffusion
                        @. Yₜ.c.sgsʲs.:($$j).q_tot += ᶜinv_ρaʲ * ᶜρχₜ_diffusion
                    end

                    MatrixFields.unrolled_foreach(
                        edmfx_diffused_scalars(Y),
                    ) do (χ_name, ρχ_name)
                        MatrixFields.has_field(Y.c.sgsʲs.:($j), χ_name) ||
                            return
                        ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                        ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                        ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
                        @. ᶜρχₜ_diffusion = -(ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜχʲ))))
                        @. ᶜρχₜ += ᶜρχₜ_diffusion
                        @. ᶜχʲₜ += ᶜinv_ρaʲ * ᶜρχₜ_diffusion
                    end
                end
            end
        else
            add_scalar_diffusionᵥ!(
                Yₜ,
                Y,
                p,
                ᶠρK_h,
                p.scratch.ᶜtemp_scalar_3,
                p.scratch.ᶜtemp_scalar,
            )
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
    ᶜupdraft_horizontal_diffusion_weight!(ᶜwʲ, ᶜρ, ᶜρaʲ, ᶜK_h_h, f_j)

Write into `ᶜwʲ` and return the center weight of the horizontal updraft
diffusive flux `-ᶜwʲ ∇ₕχʲ`,

    ᶜwʲ = ρ f_j K_h_h âʲ,    âʲ = ⟨max(aʲ, a_ε)⁻⁴⟩ₑ^(-1/4),

where `aʲ = ρaʲ / ρ` is the draft area fraction, `⟨·⟩ₑ` is the `WJ`-weighted
element mean (`element_meanₕ`), `a_ε = horizontal_draft_area_floor(FT)`, and
`f_j` is `edmfx_updraft_kh_diffusion_factor`.

The power mean `âʲ` is a smooth element-wise minimum of the draft area, the
spectral-element counterpart of the harmonic face mean of
`ᶠupdraft_diffusion_weight!`. The weak divergence at a node draws on the flux at
every node of its element, so with the nodal area `aʲ` in the weight, dividing
the flux divergence by the local `ρaʲ` would amplify the intensive tendency by
the area ratio between nodes. Since `⟨a⁻⁴⟩ₑ ≥ ωᵢ aᵢ⁻⁴` at every node `i`, with
`ωᵢ` the node's share of the element's `WJ`, `âʲ ≤ ωᵢ^(-1/4) aᵢʲ`: the updraft
relaxes toward its horizontal neighbors at most `ω_min^(-1/4)` times as fast as
grid-mean diffusion at the same `f_j K_h_h` (about 3.5 for `GLL{4}`) for every
value of `ρaʲ`, and an element containing a draft-free node carries almost no
updraft flux. The grid mean receives the same flux divergence as the updraft,
so the environment receives no residual.
"""
function ᶜupdraft_horizontal_diffusion_weight!(ᶜwʲ, ᶜρ, ᶜρaʲ, ᶜK_h_h, f_j)
    a_ε = horizontal_draft_area_floor(eltype(ᶜwʲ))
    @. ᶜwʲ =
        ᶜρ * f_j * ᶜK_h_h /
        sqrt(sqrt(element_meanₕ(1 / max(ᶜρaʲ / ᶜρ, a_ε)^4)))
    return ᶜwʲ
end

"""
    horizontal_draft_area_floor(FT)

Return the draft area fraction `a_ε = 10⁻⁸` below which a node counts as
draft-free in horizontal updraft diffusion, small enough to leave the updraft
fluxes unaffected and large enough that `a_ε⁻⁴` stays finite in single
precision.
"""
horizontal_draft_area_floor(FT) = FT(1e-8)

"""
    edmfx_sgs_horizontal_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Apply the horizontal component of the PROPHET (`EDMFX`) SGS diffusive flux
tendency using the horizontal eddy viscosity `ᶜK_u_h` and diffusivity `ᶜK_h_h`
computed with the horizontal node spacing `Δx_h`:

  - **Momentum (`uₕ`, `u₃`)** and **TKE (`ρtke`)**: diffused on the grid mean
    with `ᶜK_u_h`.
  - **Thermodynamics, water, and tracers**: for `EDOnlyEDMFX`, diffused on the
    grid mean via `add_scalar_diffusionₕ!`; for `PrognosticEDMFX`, assembled
    from the environment (`ρa⁰ K_h_h ∇ₕψ⁰`) plus, when
    `edmfx_model.horizontal_diffusion` is active, the updraft fluxes
    `ᶜwʲ ∇ₕψʲ` with the weight of `ᶜupdraft_horizontal_diffusion_weight!`. Each
    updraft equation receives its flux divergence divided by `ρaʲ` and the
    grid mean the flux divergence itself, so the environment's scalars receive
    no residual; as in `edmfx_sgs_diffusive_flux_tendency!`, the moist mass
    that `q_tot` diffusion adds and the enthalpy `(h_d + Φ) ∇ₕq_tot_effʲ` of
    the dry air it displaces go to the environment. Grid-scale scalars that
    the updrafts do not carry diffuse down
    their grid-mean gradients with `ρ K_h_h`. Rain and snow do not diffuse.

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
    moist = !(p.atmos.microphysics_model isa DryModel)

    ᶜρK_h_h = @. lazy(ᶜρ * ᶜK_h_h)

    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜTʲs) = p.precomputed
        ᶜρa⁰ = @. lazy(ρa⁰(ᶜρ, Y.c.sgsʲs, turbconv_model))
        ᶜw⁰ = @. lazy(ᶜρa⁰ * ᶜK_h_h)
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar_3
        ᶜq_scratch = p.scratch.ᶜtemp_scalar_4
        ᶜscratch = p.scratch.ᶜtemp_scalar_6

        # Environment: sensible heat, water enthalpy, and total water down
        # their environmental gradients
        ᶜq_water⁰ = moist ? ᶜenv_diffusing_water(Y, p) : nothing
        add_sensible_heat_diffusionₕ!(
            Yₜ.c.ρe_tot,
            ᶜw⁰,
            ᶜT⁰,
            ᶜq_water⁰,
            p,
            ᶜscratch,
        )
        if moist
            add_water_enthalpy_diffusionₕ!(
                Yₜ.c.ρe_tot,
                ᶜw⁰,
                ᶜT⁰,
                ᶜq_water⁰,
                p,
                ᶜscratch;
                ᶜq_scratch,
            )
            ᶜq_tot⁰ = maybe_materialize!(ᶜq_scratch, ᶜq_water⁰[1])
            @. ᶜρχₜ_diffusion = wdivₕ(ᶜw⁰ * gradₕ(ᶜq_tot⁰))
            @. Yₜ.c.ρq_tot += ᶜρχₜ_diffusion
            @. Yₜ.c.ρ += ᶜρχₜ_diffusion
        end

        # Environment: the other scalars down their environmental gradients, or
        # their grid-mean gradients if the updrafts do not carry them
        MatrixFields.unrolled_foreach(
            edmfx_diffused_scalars(Y),
        ) do (χ_name, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
            if MatrixFields.has_field(Y.c.sgsʲs.:(1), χ_name)
                ᶜχ⁰ = maybe_materialize!(
                    ᶜq_scratch,
                    ᶜspecific_env_value(χ_name, Y, p),
                )
                @. ᶜρχₜ += wdivₕ(ᶜw⁰ * gradₕ(ᶜχ⁰))
            else
                ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
                @. ᶜρχₜ += wdivₕ(ᶜρK_h_h * gradₕ(specific(ᶜρχ, ᶜρ)))
            end
        end

        # Updrafts: each scalar down its own gradient, with the flux divergence
        # added to the grid mean
        if apply_sgs_updraft
            a_ε = horizontal_draft_area_floor(eltype(Y))
            for j in 1:n
                ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
                ᶜwʲ = ᶜupdraft_horizontal_diffusion_weight!(
                    p.scratch.ᶜtemp_scalar_5,
                    ᶜρ,
                    ᶜρaʲ,
                    ᶜK_h_h,
                    updraft_kh_factor,
                )
                ᶜinv_ρaʲ = @. lazy(1 / max(ᶜρaʲ, a_ε * ᶜρ))
                ᶜTʲ = ᶜTʲs.:($j)
                ᶜq_waterʲ =
                    moist ? ᶜupdraft_diffusing_water(Y, p, Val(j)) : nothing
                add_sensible_heat_diffusionₕ!(
                    Yₜ.c.ρe_tot,
                    ᶜwʲ,
                    ᶜTʲ,
                    ᶜq_waterʲ,
                    p,
                    ᶜscratch,
                )
                ᶜmse_effʲ = ᶜupdraft_diffusing_mse!(ᶜscratch, Y, p, Val(j))
                @. Yₜ.c.sgsʲs.:($$j).mse +=
                    ᶜinv_ρaʲ * wdivₕ(ᶜwʲ * gradₕ(ᶜmse_effʲ))

                if moist
                    add_water_enthalpy_diffusionₕ!(
                        Yₜ.c.ρe_tot,
                        ᶜwʲ,
                        ᶜTʲ,
                        ᶜq_waterʲ,
                        p,
                        ᶜscratch;
                        ᶜq_scratch,
                    )
                    ᶜq_totʲ = maybe_materialize!(ᶜq_scratch, ᶜq_waterʲ[1])
                    @. ᶜρχₜ_diffusion = wdivₕ(ᶜwʲ * gradₕ(ᶜq_totʲ))
                    @. Yₜ.c.ρq_tot += ᶜρχₜ_diffusion
                    @. Yₜ.c.ρ += ᶜρχₜ_diffusion
                    @. Yₜ.c.sgsʲs.:($$j).q_tot += ᶜinv_ρaʲ * ᶜρχₜ_diffusion
                end

                MatrixFields.unrolled_foreach(
                    edmfx_diffused_scalars(Y),
                ) do (χ_name, ρχ_name)
                    MatrixFields.has_field(Y.c.sgsʲs.:($j), χ_name) || return
                    ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                    ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                    ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
                    @. ᶜρχₜ_diffusion = wdivₕ(ᶜwʲ * gradₕ(ᶜχʲ))
                    @. ᶜρχₜ += ᶜρχₜ_diffusion
                    @. ᶜχʲₜ += ᶜinv_ρaʲ * ᶜρχₜ_diffusion
                end
            end
        end
    else
        add_scalar_diffusionₕ!(
            Yₜ,
            Y,
            p,
            ᶜρK_h_h,
            p.scratch.ᶜtemp_scalar_6,
            p.scratch.ᶜtemp_scalar_3,
        )
    end

    (; ᶜu, ᶠu) = p.precomputed
    ᶜS = compute_strain_rate_center_full!(p.scratch.ᶜtemp_UVWxUVW, ᶜu, ᶠu)

    # Turbulent TKE transport, and the shear production of the horizontal SGS
    # stress `τ = -2 K_u S`: the resolved kinetic energy it removes,
    # `-τ : ∇ₕu = 2 K_u S : Sₕ`, with `Sₕ` the strain rate of the horizontal
    # gradients alone. Its cross terms `K_u ∂ₓw ∂_z u` are sign-indefinite; the
    # total production, including `K_u,v |∂_z uₕ|²` from the TKE tendency, is
    # non-negative where `K_u,v ≥ K_u,h / 4`, which holds up to interpolation
    # because the vertical mixing length is capped at `max(Δx_h, Δz)` and the
    # horizontal one at `Δx_h`.
    # The strain-rate invariant of the nodal spectral-element gradient is too
    # large at element-boundary nodes and too small inside (the rectified
    # end-of-interval error of polynomial differentiation), so it is replaced
    # within each element by its lumped GLL{2} restriction (`lumpedₕ`), which
    # conserves the element's integral of the invariant.
    if use_prognostic_tke(turbconv_model)
        @. Yₜ.c.ρtke += wdivₕ(ᶜρ * ᶜK_u_h * gradₕ(ᶜtke))
        ᶜS_h = compute_strain_rate_center_horizontal(ᶜu)
        @. Yₜ.c.ρtke += 2 * ᶜρ * ᶜK_u_h * lumpedₕ(strain_rate_inner(ᶜS, ᶜS_h))
    end

    # Momentum: horizontal weak divergence of the SGS stress `τ = -2 K_u S`
    # with the full strain rate. The vertical stress divergence is handled by
    # the vertical diffusion pathway.
    ᶠρ = @. p.scratch.ᶠtemp_scalar = ᶠinterp(ᶜρ)
    ᶜτ_h = ᶜS
    @. ᶜτ_h = -2 * ᶜK_u_h * ᶜτ_h
    @. Yₜ.c.uₕ -= C12(wdivₕ(ᶜρ * ᶜτ_h) / ᶜρ)
    ᶠτ_h = compute_strain_rate_face_full!(p.scratch.ᶠtemp_UVWxUVW, ᶜu, ᶠu)
    @. ᶠτ_h = -2 * ᶠinterp(ᶜK_u_h) * ᶠτ_h
    @. Yₜ.f.u₃ -= C3(wdivₕ(ᶠρ * ᶠτ_h) / ᶠρ)

    return nothing
end
