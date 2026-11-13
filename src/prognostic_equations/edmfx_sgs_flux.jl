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
    eps_FT = eps(eltype(thermo_params))
    ᶜρ⁰ = @. lazy(
        max(
            eps_FT,
            TD.air_density(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_nonneg⁰,
                ᶜq_liq⁰,
                ᶜq_ice⁰,
            ),
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
            DataLayouts.foreach_column(
                Yₜ.c, Y.c, ᶜρʲs.:($j), ᶠu³, ᶠu³ʲs.:($j), ᶜKʲs.:($j), ᶜh_tot,
                Y.c.sgsʲs.:($j),
            ) do Yₜc, Yc, ᶜρʲ, ᶠu³, ᶠu³ʲ, ᶜKʲ, ᶜh_tot, ᶜsgsʲ
                ᶜχʲs = unrolled_map(name -> MatrixFields.get_field(ᶜsgsʲ, name), χ_names)
                ᶜaʲ = @. draft_area(ᶜsgsʲ.ρa, ᶜρʲ)
                add_sgs_mass_flux_tendencies!(
                    Yₜc, Yc, ᶜρʲ, ᶠu³, ᶠu³ʲ, ᶜKʲ, ᶜh_tot, ᶜsgsʲ.mse, ᶜsgsʲ.q_tot,
                    ᶜaʲ, ᶜχʲs, ρχ_names, dt, edmfx_sgsflux_upwinding,
                )
            end
        end

        # Environment
        ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
        ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
        ᶜχ⁰s = unrolled_map(χ_name -> ᶜspecific_env_value(χ_name, Y, p), χ_names)
        DataLayouts.foreach_column(
            Yₜ.c, Y.c, ᶜρ⁰, ᶜρa⁰, ᶠu³, ᶠu³⁰, ᶜK⁰, ᶜh_tot, ᶜmse⁰, ᶜq_tot⁰, ᶜχ⁰s,
        ) do Yₜc, Yc, ᶜρ⁰, ᶜρa⁰, ᶠu³, ᶠu³⁰, ᶜK⁰, ᶜh_tot, ᶜmse⁰, ᶜq_tot⁰, ᶜχ⁰s
            ᶜa⁰ = @. draft_area(ᶜρa⁰, ᶜρ⁰)
            add_sgs_mass_flux_tendencies!(
                Yₜc, Yc, ᶜρ⁰, ᶠu³, ᶠu³⁰, ᶜK⁰, ᶜh_tot, ᶜmse⁰, ᶜq_tot⁰,
                ᶜa⁰, ᶜχ⁰s, ρχ_names, dt, edmfx_sgsflux_upwinding,
            )
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
`edmfx_sgs_mass_flux_tendency!`.
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
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    ᶠρ = @. ᶠinterp(ᶜρ * ᶜJ) / ᶠJ # shared by every transport
    ᶠu³_diff = @. ᶠu³_sub - ᶠu³
    ᶜa_scalar = @. (ᶜmse_sub + ᶜK_sub - ᶜh_tot) * ᶜa
    add_vertical_transport!(Yₜc.ρe_tot, ᶠρ, ᶠu³_diff, ᶜa_scalar, dt, upwinding)
    if MatrixFields.has_field(Yc, @name(ρq_tot))
        ᶜa_scalar_q = @. (ᶜq_tot_sub - specific(Yc.ρq_tot, Yc.ρ)) * ᶜa
        add_vertical_transport!(Yₜc.ρq_tot, ᶠρ, ᶠu³_diff, ᶜa_scalar_q, dt, upwinding)
        # Effect of SGS water flux on (moist) air mass
        add_vertical_transport!(Yₜc.ρ, ᶠρ, ᶠu³_diff, ᶜa_scalar_q, dt, upwinding)
    end
    MatrixFields.unrolled_foreach(map(tuple, ρχ_names, ᶜχ_subs)) do (ρχ_name, ᶜχ_sub)
        @inline
        ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
        ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
        ᶜa_scalar_χ = @. (ᶜχ_sub - specific(ᶜρχ, Yc.ρ)) * ᶜa
        add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³_diff, ᶜa_scalar_χ, dt, upwinding)
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
    (; turbconv_model, microphysics_model) = p.atmos
    (; ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    nonequil_val = Val(microphysics_model isa NonEquilibriumMicrophysics)
    return env_diffusing_water(
        Y.c, ᶜρa⁰, ᶜq_liq⁰, ᶜq_ice⁰, turbconv_model, nonequil_val,
    )
end

# `ᶜenv_diffusing_water` of the grid-mean state `ᶜY`, a whole field or one
# column of a fused loop, with the environment mass `ᶜρa⁰` and the cached
# diagnostic `ᶜq_liq⁰` and `ᶜq_ice⁰`.
@inline function env_diffusing_water(ᶜY, ᶜρa⁰, _, _, turbconv_model, ::Val{true})
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), ᶜY, ᶜρa⁰, turbconv_model)
    ᶜq_rai⁰ = ᶜspecific_env_value(@name(q_rai), ᶜY, ᶜρa⁰, turbconv_model)
    ᶜq_sno⁰ = ᶜspecific_env_value(@name(q_sno), ᶜY, ᶜρa⁰, turbconv_model)
    return (
        (@inline @. lazy(ᶜq_tot⁰ - ᶜq_rai⁰ - ᶜq_sno⁰)),
        ᶜspecific_env_value(@name(q_lcl), ᶜY, ᶜρa⁰, turbconv_model),
        ᶜspecific_env_value(@name(q_icl), ᶜY, ᶜρa⁰, turbconv_model),
    )
end
@inline env_diffusing_water(ᶜY, ᶜρa⁰, ᶜq_liq⁰, ᶜq_ice⁰, turbconv_model, ::Val{false}) =
    (ᶜspecific_env_value(@name(q_tot), ᶜY, ᶜρa⁰, turbconv_model), ᶜq_liq⁰, ᶜq_ice⁰)

"""
    ᶜupdraft_diffusing_water(Y, p, ::Val{j})

Return `(q_tot_effʲ, q_lclʲ, q_iclʲ)`, the diffusing total, liquid, and ice water
of updraft `j`. With non-equilibrium microphysics,
`q_tot_effʲ = q_totʲ - q_raiʲ - q_snoʲ` and the suspended cloud species are read
from `Y.c.sgsʲs.:(j)`; with an equilibrium scheme, `q_tot_effʲ = q_totʲ` and the
cloud species are the cached diagnostic `ᶜq_liqʲs.:(j)` and `ᶜq_iceʲs.:(j)`.
"""
function ᶜupdraft_diffusing_water(Y, p, ::Val{j}) where {j}
    (; ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
    nonequil_val = Val(p.atmos.microphysics_model isa NonEquilibriumMicrophysics)
    return updraft_diffusing_water(
        Y.c.sgsʲs.:($j), ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j), nonequil_val,
    )
end

# `ᶜupdraft_diffusing_water` of the updraft state `ᶜsgsʲ`, a whole field or one
# column of a fused loop, with its cached diagnostic `ᶜq_liqʲ` and `ᶜq_iceʲ`.
@inline updraft_diffusing_water(ᶜsgsʲ, ᶜq_liqʲ, ᶜq_iceʲ, ::Val{true}) = (
    (@inline @. lazy(ᶜsgsʲ.q_tot - ᶜsgsʲ.q_rai - ᶜsgsʲ.q_sno)),
    ᶜsgsʲ.q_lcl,
    ᶜsgsʲ.q_icl,
)
@inline updraft_diffusing_water(ᶜsgsʲ, ᶜq_liqʲ, ᶜq_iceʲ, ::Val{false}) =
    (ᶜsgsʲ.q_tot, ᶜq_liqʲ, ᶜq_iceʲ)

"""
    ᶜupdraft_diffusing_mse!(ᶜmse_eff, Y, p, ::Val{j})

Return the moist static energy of updraft `j` with the enthalpy of the
non-diffusing precipitation replaced by that of dry air,

    mse_effʲ = mseʲ - (h_l(Tʲ) - h_d(Tʲ)) q_raiʲ - (h_i(Tʲ) - h_d(Tʲ)) q_snoʲ,

written into `ᶜmse_eff` with non-equilibrium microphysics. With an equilibrium
scheme, return `Y.c.sgsʲs.:(j).mse` itself.
"""
ᶜupdraft_diffusing_mse!(ᶜmse_eff, Y, p, ::Val{j}) where {j} =
    updraft_diffusing_mse!(
        ᶜmse_eff,
        Y.c.sgsʲs.:($j),
        p.precomputed.ᶜTʲs.:($j),
        CAP.thermodynamics_params(p.params),
        Val(p.atmos.microphysics_model isa NonEquilibriumMicrophysics),
    )

# `ᶜupdraft_diffusing_mse!` of the updraft state `ᶜsgsʲ` at the temperature
# `ᶜTʲ`, a whole field or one column of a fused loop (see `materialize_into!`).
# In a fused loop (`ᶜmse_eff === nothing`), it is a register temporary with either
# microphysics model.
@inline updraft_diffusing_mse!(_, ᶜsgsʲ, _, _, ::Val{false}) = ᶜsgsʲ.mse
@inline updraft_diffusing_mse!(::Nothing, ᶜsgsʲ, _, _, ::Val{false}) =
    register_copy(ᶜsgsʲ.mse)
@inline updraft_diffusing_mse!(ᶜmse_eff, ᶜsgsʲ, ᶜTʲ, thermo_params, ::Val{true}) =
    materialize_into!(
        ᶜmse_eff,
        (@inline @. lazy(
            ᶜsgsʲ.mse -
            (
                TD.enthalpy_liquid(thermo_params, ᶜTʲ) -
                TD.enthalpy_dry(thermo_params, ᶜTʲ)
            ) * ᶜsgsʲ.q_rai -
            (
                TD.enthalpy_ice(thermo_params, ᶜTʲ) -
                TD.enthalpy_dry(thermo_params, ᶜTʲ)
            ) * ᶜsgsʲ.q_sno,
        )),
    )

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
    (; dt, params) = p
    turbconv_params = CAP.turbconv_params(params)
    (; ᶜu) = p.precomputed

    if p.atmos.edmfx_model.sgs_diffusive_flux

        (; ᶠK_h, ᶠK_u) = p.precomputed
        if turbconv_model isa PrognosticEDMFX
            # The vertical diffusive fluxes of energy, total water, and the
            # `edmfx_diffused_scalars`: the environment fluxes, then those of
            # each updraft when `apply_sgs_updraft`, with the updraft weight
            # factor `f_j`. Every subdomain shares one fused column loop. Each tendency is updated in the same order as in separate
            # broadcasts, so the result does not depend on the fusion.
            apply_sgs_updraft = p.atmos.edmfx_model.vertical_diffusion
            f_j = CAP.edmfx_updraft_kh_diffusion_factor(turbconv_params)
            microphysics_model = p.atmos.microphysics_model
            ϵ_FT = eps(FT)
            nonequil_val = Val(microphysics_model isa NonEquilibriumMicrophysics)
            thermo_params = CAP.thermodynamics_params(p.params)
            (; ᶜT⁰, ᶜTʲs, ᶜq_liq⁰, ᶜq_ice⁰, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
            # The blended cloud-species flux needs the closures of the 1M scheme.
            blend = Val(blended_cloud_species_flux(Y, p))
            blend_fields =
                blend isa Val{true} ?
                (p.precomputed.ᶜl_mix, p.precomputed.ᶜp, p.precomputed.ᶜsgs_moments) : ()
            blend_params = blend isa Val{true} ? cloud_species_blend_params(p) : nothing
            χ_pairs = edmfx_diffused_scalars(Y)
            DataLayouts.foreach_column(
                Yₜ.c, Y.c, ᶠK_h, p.core.ᶜΦ, ᶜT⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜTʲs, ᶜq_liqʲs,
                ᶜq_iceʲs, blend_fields,
            ) do Yₜc, Yc, ᶠK_h, ᶜΦ, ᶜT⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜTʲs, ᶜq_liqʲs, ᶜq_iceʲs,
            blend_fields
                # Environment: each scalar down its environmental gradient, or its
                # grid-mean gradient if the updrafts do not carry it
                ᶜρa⁰ = @. ρa⁰(Yc.ρ, Yc.sgsʲs, turbconv_model)
                ᶠw⁰ = @. ᶠinterp(ᶜρa⁰) * ᶠK_h
                ᶜT⁰ = register_copy(ᶜT⁰)
                ᶜq_water⁰ =
                    !(microphysics_model isa DryModel) ?
                    unrolled_map(
                        register_copy,
                        env_diffusing_water(
                            Yc, ᶜρa⁰, ᶜq_liq⁰, ᶜq_ice⁰, turbconv_model, nonequil_val,
                        ),
                    ) : nothing
                add_sensible_heat_diffusionᵥ!(
                    Yₜc.ρe_tot, ᶠw⁰, ᶜT⁰, ᶜq_water⁰, thermo_params, ᶜΦ,
                )
                !(microphysics_model isa DryModel) && add_water_diffusionᵥ!(
                    Yₜc, nothing, ᶠw⁰, ᶜT⁰, ᶜq_water⁰, thermo_params, ᶜΦ,
                )
                # Face fluxes of total water and liquid–ice static energy for the
                # blended cloud-species flux
                ᶜq_tot⁰ = isnothing(ᶜq_water⁰) ? nothing : ᶜq_water⁰[1]
                (ᶠF_qt⁰, ᶠF_sl⁰) =
                    blend isa Val{true} ?
                    (
                        (@. -(ᶠw⁰ * ᶠgradᵥ(ᶜq_tot⁰))),
                        ᶠliquid_ice_static_energy_flux!(
                            nothing, ᶠw⁰, ᶜT⁰, ᶜq_water⁰, thermo_params, ᶜΦ,
                        ),
                    ) : (nothing, nothing)
                MatrixFields.unrolled_foreach(χ_pairs) do (χ_name, ρχ_name)
                    @inline
                    ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
                    phase = cloud_species_phase(χ_name)
                    blended = blend isa Val{true} && !isnothing(phase)
                    if blended || MatrixFields.has_field(Yc.sgsʲs.:(1), χ_name)
                        # The blend implies the 1M scheme, whose environment cloud
                        # liquid and ice are the last two diffusing water values.
                        ᶜχ⁰ =
                            blended ? ᶜq_water⁰[phase isa TD.Liquid ? 2 : 3] :
                            Base.materialize(
                                ᶜspecific_env_value(χ_name, Yc, ᶜρa⁰, turbconv_model),
                            )
                        if blended
                            (ᶜl_mix, ᶜp, ᶜsgs_moments) = blend_fields
                            add_cloud_species_diffusionᵥ!(
                                ᶜρχₜ, ᶠw⁰, ᶠF_qt⁰, ᶠF_sl⁰, ᶜT⁰, ᶜq_water⁰, ᶜsgs_moments,
                                ᶜχ⁰, phase, nothing, blend_params, ᶜl_mix, ᶜp,
                                (@inline @. lazy(specific(Yc.ρtke, Yc.ρ))),
                            )
                        else
                            @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠw⁰ * ᶠgradᵥ(ᶜχ⁰)))
                        end
                    else
                        ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
                        @. ᶜρχₜ -= ᶜdiffdivᵥ(
                            -(ᶠinterp(Yc.ρ) * ᶠK_h * ᶠgradᵥ(specific(ᶜρχ, Yc.ρ))),
                        )
                    end
                end

                # Updrafts: each scalar down its own gradient, with the extensive flux
                # divergence added to the grid mean
                apply_sgs_updraft || return nothing
                for j in 1:n_mass_flux_subdomains(turbconv_model)
                    ᶜsgsʲ = Yc.sgsʲs.:($j)
                    ᶜsgsʲₜ = Yₜc.sgsʲs.:($j)
                    ᶜTʲ = register_copy(ᶜTʲs.:($j))
                    ᶠwʲ = ᶠupdraft_diffusion_weight!(nothing, ᶜsgsʲ.ρa, ᶠK_h, f_j)
                    ᶜinv_ρaʲ = @. 1 / max(ᶜsgsʲ.ρa, ϵ_FT)
                    ᶜq_waterʲ =
                        !(microphysics_model isa DryModel) ?
                        unrolled_map(
                            register_copy,
                            updraft_diffusing_water(
                                ᶜsgsʲ, ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j), nonequil_val,
                            ),
                        ) : nothing
                    add_sensible_heat_diffusionᵥ!(
                        Yₜc.ρe_tot, ᶠwʲ, ᶜTʲ, ᶜq_waterʲ, thermo_params, ᶜΦ,
                    )
                    ᶜmse_effʲ = updraft_diffusing_mse!(
                        nothing, ᶜsgsʲ, ᶜTʲ, thermo_params, nonequil_val,
                    )
                    @. ᶜsgsʲₜ.mse -= ᶜinv_ρaʲ * ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜmse_effʲ)))
                    !(microphysics_model isa DryModel) && add_water_diffusionᵥ!(
                        Yₜc, (ᶜsgsʲₜ.q_tot, ᶜinv_ρaʲ), ᶠwʲ, ᶜTʲ, ᶜq_waterʲ,
                        thermo_params, ᶜΦ,
                    )
                    MatrixFields.unrolled_foreach(χ_pairs) do (χ_name, ρχ_name)
                        @inline
                        MatrixFields.has_field(ᶜsgsʲ, χ_name) || return
                        ᶜχʲ = register_copy(MatrixFields.get_field(ᶜsgsʲ, χ_name))
                        ᶜχʲₜ = MatrixFields.get_field(ᶜsgsʲₜ, χ_name)
                        ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
                        ᶜρχₜ_diffusion = @. -(ᶜdiffdivᵥ(-(ᶠwʲ * ᶠgradᵥ(ᶜχʲ))))
                        @. ᶜρχₜ += ᶜρχₜ_diffusion
                        @. ᶜχʲₜ += ᶜinv_ρaʲ * ᶜρχₜ_diffusion
                    end
                end
            end
        else
            ᶠρK_h = p.scratch.ᶠtemp_scalar
            @. ᶠρK_h = ᶠinterp(Y.c.ρ) * ᶠK_h
            add_scalar_diffusionᵥ!(
                Yₜ,
                Y,
                p,
                ᶠρK_h,
                p.scratch.ᶜtemp_scalar_3,
                p.scratch.ᶜtemp_scalar,
            )
        end

        # TKE and momentum, in one fused column loop
        zero_FT = zero(FT)
        tke_fields =
            use_prognostic_tke(turbconv_model) ?
            (p.precomputed.ᶜl_mix, p.precomputed.ρtke_flux) : ()
        DataLayouts.foreach_column(
            Yₜ.c, Y.c, ᶠK_u, ᶜu, tke_fields,
        ) do Yₜc, Yc, ᶠK_u, ᶜu, tke_fields
            ᶠρaK_u = @. ᶠinterp(Yc.ρ) * ᶠK_u
            if !isempty(tke_fields)
                (ᶜl_mix, ρtke_flux) = tke_fields
                ᶜtke = @inline @. lazy(specific(Yc.ρtke, Yc.ρ))
                ᶜdivᵥ_ρtke = Operators.DivergenceF2C(
                    top = Operators.SetValue(C3(zero_FT)),
                    bottom = Operators.SetValue(ρtke_flux),
                )
                @. Yₜc.ρtke -=
                    ᶜdivᵥ_ρtke(-(ᶠρaK_u * ᶠgradᵥ(ᶜtke))) + ifelse(
                        ᶜtke >= zero_FT,
                        tke_dissipation(turbconv_params, Yc.ρtke, ᶜtke, ᶜl_mix),
                        Yc.ρtke / dt,
                    )
            end
            ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
            @. Yₜc.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Yc.ρ)
        end
    end

    return nothing
end

"""
    add_water_diffusionᵥ!(Yₜc, sgs_q_totₜ, ᶠw, ᶜT, ᶜq_water, thermo_params, ᶜΦ)

Add the water part of the vertical diffusive enthalpy flux of a subdomain
(`add_water_enthalpy_diffusionᵥ!`) to `Yₜc.ρe_tot`, and the divergence of its
diffusive flux of total water `-ᶠw ∇q_tot` to `Yₜc.ρq_tot` and `Yₜc.ρ`, in one
column of a fused loop. For an updraft, `sgs_q_totₜ` is the pair of its total
water tendency and inverse mass `1 / ρaʲ`, which receives the intensive
tendency; for the environment, it is `nothing`.
"""
@inline function add_water_diffusionᵥ!(
    Yₜc,
    sgs_q_totₜ,
    ᶠw,
    ᶜT,
    ᶜq_water,
    thermo_params,
    ᶜΦ,
)
    add_water_enthalpy_diffusionᵥ!(
        Yₜc.ρe_tot, ᶠw, ᶜT, ᶜq_water, thermo_params, ᶜΦ,
    )
    ᶜq_tot = ᶜq_water[1]
    ᶜρq_totₜ = @. -(ᶜdiffdivᵥ(-(ᶠw * ᶠgradᵥ(ᶜq_tot))))
    @. Yₜc.ρq_tot += ᶜρq_totₜ
    @. Yₜc.ρ += ᶜρq_totₜ
    if !isnothing(sgs_q_totₜ)
        (ᶜq_totʲₜ, ᶜinv_ρaʲ) = sgs_q_totₜ
        @. ᶜq_totʲₜ += ᶜinv_ρaʲ * ᶜρq_totₜ
    end
    return nothing
end

"""
    ᶜupdraft_horizontal_diffusion_weight!(ᶜwʲ, ᶜρ, ᶜρaʲ, ᶜK_h_h, f_j)

Write into `ᶜwʲ` (or, when `ᶜwʲ` is `nothing`, materialize as a register
temporary of a fused slab loop) and return the center weight of the horizontal
updraft diffusive flux `-ᶜwʲ ∇ₕχʲ`,

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
@inline function ᶜupdraft_horizontal_diffusion_weight!(ᶜwʲ, ᶜρ, ᶜρaʲ, ᶜK_h_h, f_j)
    a_ε = horizontal_draft_area_floor(eltype(ᶜK_h_h))
    return materialize_into!(
        ᶜwʲ,
        (@inline @. lazy(
            ᶜρ * f_j * ᶜK_h_h /
            sqrt(sqrt(element_meanₕ(1 / max(ᶜρaʲ / ᶜρ, a_ε)^4))),
        )),
    )
end

"""
    horizontal_draft_area_floor(FT)

Return the draft area fraction `a_ε = 10⁻⁸` below which a node counts as
draft-free in horizontal updraft diffusion, small enough to leave the updraft
fluxes unaffected and large enough that `a_ε⁻⁴` stays finite in single
precision.
"""
horizontal_draft_area_floor(FT) = FT(1e-8)

# Add the energy flux divergences of a subdomain with the weight `ᶜw` to
# `ᶜYₜ.ᶜρe_totₜ` in a fused slab loop, and its total-water flux
# divergence to `ᶜYₜ.ᶜρq_totₜ` and `ᶜYₜ.ᶜρₜ`; return that divergence (`nothing` in
# dry configurations).
@inline function add_subdomain_diffusionₕ!(ᶜYₜ, ᶜw, ᶜT, ᶜq_water, thermo_params, ᶜΦ)
    (; ᶜρe_totₜ, ᶜρq_totₜ, ᶜρₜ) = ᶜYₜ
    ᶜT = register_copy(ᶜT)
    ᶜq_water = isnothing(ᶜq_water) ? nothing : unrolled_map(register_copy, ᶜq_water)
    add_sensible_heat_diffusionₕ!(ᶜρe_totₜ, ᶜw, ᶜT, ᶜq_water, thermo_params, ᶜΦ)
    isnothing(ᶜq_water) && return nothing
    add_water_enthalpy_diffusionₕ!(ᶜρe_totₜ, ᶜw, ᶜT, ᶜq_water, thermo_params, ᶜΦ)
    ᶜρq_totₜ_diffusion = diffusion_divₕ(ᶜw, ᶜq_water[1])
    @. ᶜρq_totₜ += ᶜρq_totₜ_diffusion
    @. ᶜρₜ += ᶜρq_totₜ_diffusion
    return ᶜρq_totₜ_diffusion
end

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

The scalars, the TKE, and `uₕ` are updated in one center slab loop, and `u₃` in
one face slab loop and one column loop, with the vertical velocity differences
and the face `ρ K_u` stored by a first column loop in `p.scratch` (`ᶜtemp_UVW`,
`ᶠtemp_UVW`, and `ᶠtemp_scalar`).
"""
edmfx_sgs_horizontal_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model) =
    nothing

function edmfx_sgs_horizontal_diffusive_flux_tendency!(
    Yₜ, Y, p, t, turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX},
)
    p.atmos.edmfx_model.sgs_diffusive_flux_horizontal isa Val{true} ||
        return nothing
    iscolumn(axes(Y.c)) && return nothing
    (; microphysics_model) = p.atmos
    (; ᶜK_u_h, ᶜK_h_h, ᶜu, ᶠu) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

    # The full strain rate `S` is the sum of the strain rates of the vertical
    # and horizontal velocity gradients. Slab loops cannot apply vertical
    # operators, so one column loop stores, for the slab loops below, the
    # vertical velocity differences at centers and faces (3 values per point,
    # rather than the 9 of their strain rates) and the face values of `ρ K_u`.
    ᶜΔu = p.scratch.ᶜtemp_UVW
    ᶠΔu = p.scratch.ᶠtemp_UVW
    ᶠρK_u_h = p.scratch.ᶠtemp_scalar
    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), UVW(0, 0, 0))
    ᶠgradᵥ_uvw = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    DataLayouts.foreach_column(
        ᶜΔu, ᶠΔu, ᶠρK_u_h, Y.c.ρ, ᶜK_u_h, ᶜu, ᶠu,
    ) do ᶜΔu, ᶠΔu, ᶠρK_u_h, ᶜρ, ᶜK_u_h, ᶜu, ᶠu
        @. ᶜΔu = vertical_difference(ᶜgradᵥ(UVW(ᶠu)))
        @. ᶠΔu = vertical_difference(ᶠgradᵥ_uvw(UVW(ᶜu)))
        @. ᶠρK_u_h = ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u_h)
    end

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
    # Momentum: horizontal weak divergence of the SGS stress with the full
    # strain rate. The vertical stress divergence is handled by the vertical
    # diffusion pathway.
    # The center slab loop requests the shared memory of the `:medium` level,
    # which its lumped restrictions need, so that it is not first compiled
    # without it.
    turbconv_params = CAP.turbconv_params(p.params)
    apply_sgs_updraft = p.atmos.edmfx_model.horizontal_diffusion
    f_j = CAP.edmfx_updraft_kh_diffusion_factor(turbconv_params)
    χ_pairs = edmfx_diffused_scalars(Y)
    ρχ_names = self_diffused_tracer_names(Y, false)
    thermo_fields =
        turbconv_model isa PrognosticEDMFX ?
        (
            p.precomputed.ᶜT⁰, p.precomputed.ᶜq_liq⁰, p.precomputed.ᶜq_ice⁰,
            p.precomputed.ᶜTʲs, p.precomputed.ᶜq_liqʲs, p.precomputed.ᶜq_iceʲs,
        ) : (p.precomputed.ᶜT, p.precomputed.ᶜq_liq, p.precomputed.ᶜq_ice)
    DataLayouts.foreach_slab(
        Yₜ.c, Y.c, ᶜK_h_h, ᶜK_u_h, ᶜu, ᶜΔu, p.core.ᶜΦ, thermo_fields;
        shmem = Val(:medium),
    ) do Yₜc, Yc, ᶜK_h_h, ᶜK_u_h, ᶜu, ᶜΔu, ᶜΦ, thermo_fields
        if turbconv_model isa PrognosticEDMFX
            # The horizontal diffusive fluxes for energy, total water, and the
            # `edmfx_diffused_scalars`. The grid-mean tendencies accumulate the subdomain
            # fluxes in the order of separate broadcasts, and are stored after
            # all of them are computed.
            (ᶜT⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜTʲs, ᶜq_liqʲs, ᶜq_iceʲs) = thermo_fields
            a_ε = horizontal_draft_area_floor(eltype(ᶜK_h_h))
            nonequil_val = Val(microphysics_model isa NonEquilibriumMicrophysics)
            draft_indices = StaticOneTo(n_mass_flux_subdomains(turbconv_model))
            ᶜΦ = register_copy(ᶜΦ)
            ᶜρe_totₜ = register_copy(Yₜc.ρe_tot)
            ᶜρq_totₜ =
                !(microphysics_model isa DryModel) ? register_copy(Yₜc.ρq_tot) : nothing
            ᶜρₜ = !(microphysics_model isa DryModel) ? register_copy(Yₜc.ρ) : nothing
            ᶜρχₜs = unrolled_map(χ_pairs) do (_, ρχ_name)
                @inline
                register_copy(MatrixFields.get_field(Yₜc, ρχ_name))
            end
            ᶜYₜ = (; ᶜρe_totₜ, ᶜρq_totₜ, ᶜρₜ)

            # Environment: every scalar down its environmental gradient, or its
            # grid-mean gradient if the updrafts do not carry it
            ᶜρa⁰ = @. ρa⁰(Yc.ρ, Yc.sgsʲs, turbconv_model)
            ᶜw⁰ = @. ᶜρa⁰ * ᶜK_h_h
            ᶜq_water⁰ =
                !(microphysics_model isa DryModel) ?
                env_diffusing_water(
                    Yc,
                    ᶜρa⁰,
                    ᶜq_liq⁰,
                    ᶜq_ice⁰,
                    turbconv_model,
                    nonequil_val,
                ) :
                nothing
            add_subdomain_diffusionₕ!(ᶜYₜ, ᶜw⁰, ᶜT⁰, ᶜq_water⁰, thermo_params, ᶜΦ)
            unrolled_foreach(χ_pairs, ᶜρχₜs) do (χ_name, ρχ_name), ᶜρχₜ
                @inline
                if MatrixFields.has_field(Yc.sgsʲs.:(1), χ_name)
                    ᶜχ⁰ = Base.materialize(
                        ᶜspecific_env_value(χ_name, Yc, ᶜρa⁰, turbconv_model),
                    )
                    @. ᶜρχₜ += wdivₕ(ᶜw⁰ * gradₕ(ᶜχ⁰))
                else
                    ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
                    ᶜχ = @. specific(ᶜρχ, Yc.ρ)
                    @. ᶜρχₜ += wdivₕ(Yc.ρ * ᶜK_h_h * gradₕ(ᶜχ))
                end
            end

            # Updrafts: every scalar down its own gradient, with the flux divergence
            # added to the grid mean
            apply_sgs_updraft isa Val{true} && for j in draft_indices
                ᶜsgsʲ = Yc.sgsʲs.:($j)
                ᶜsgsʲₜ = Yₜc.sgsʲs.:($j)
                ᶜwʲ =
                    ᶜupdraft_horizontal_diffusion_weight!(
                        nothing,
                        Yc.ρ,
                        ᶜsgsʲ.ρa,
                        ᶜK_h_h,
                        f_j,
                    )
                ᶜinv_ρaʲ = @. 1 / max(ᶜsgsʲ.ρa, a_ε * Yc.ρ)
                ᶜTʲ = register_copy(ᶜTʲs.:($j))
                ᶜq_waterʲ =
                    !(microphysics_model isa DryModel) ?
                    updraft_diffusing_water(
                        ᶜsgsʲ,
                        ᶜq_liqʲs.:($j),
                        ᶜq_iceʲs.:($j),
                        nonequil_val,
                    ) :
                    nothing
                ᶜρq_totₜʲ =
                    add_subdomain_diffusionₕ!(ᶜYₜ, ᶜwʲ, ᶜTʲ, ᶜq_waterʲ, thermo_params, ᶜΦ)
                ᶜmse_effʲ =
                    updraft_diffusing_mse!(nothing, ᶜsgsʲ, ᶜTʲ, thermo_params, nonequil_val)
                @. ᶜsgsʲₜ.mse += ᶜinv_ρaʲ * wdivₕ(ᶜwʲ * gradₕ(ᶜmse_effʲ))
                isnothing(ᶜρq_totₜʲ) || @. ᶜsgsʲₜ.q_tot += ᶜinv_ρaʲ * ᶜρq_totₜʲ
                unrolled_foreach(χ_pairs, ᶜρχₜs) do (χ_name, _), ᶜρχₜ
                    @inline
                    MatrixFields.has_field(ᶜsgsʲ, χ_name) || return nothing
                    ᶜχʲ = register_copy(MatrixFields.get_field(ᶜsgsʲ, χ_name))
                    ᶜχʲₜ = MatrixFields.get_field(ᶜsgsʲₜ, χ_name)
                    ᶜρχₜʲ = diffusion_divₕ(ᶜwʲ, ᶜχʲ)
                    @. ᶜρχₜ += ᶜρχₜʲ
                    @. ᶜχʲₜ += ᶜinv_ρaʲ * ᶜρχₜʲ
                end
            end

            @. Yₜc.ρe_tot = ᶜρe_totₜ
            if !(microphysics_model isa DryModel)
                @. Yₜc.ρq_tot = ᶜρq_totₜ
                @. Yₜc.ρ = ᶜρₜ
            end
            unrolled_foreach(χ_pairs, ᶜρχₜs) do (_, ρχ_name), ᶜρχₜ
                @inline
                ᶜρχₜ_field = MatrixFields.get_field(Yₜc, ρχ_name)
                @. ᶜρχₜ_field = ᶜρχₜ
            end
        else
            (ᶜT, ᶜq_liq, ᶜq_ice) = thermo_fields
            ᶜq_water =
                diffusing_water_phases(Yc, ᶜq_liq, ᶜq_ice, microphysics_model, Val(false))
            ᶜρK_h_h = @. Yc.ρ * ᶜK_h_h
            add_scalar_diffusionₕ!(
                Yₜc, Yc, ᶜρK_h_h, ᶜT, ᶜΦ, ᶜq_water, thermo_params, ρχ_names,
            )
        end
        ᶜS_v = vertical_strain_rate(ᶜΔu)
        ᶜS_h = compute_strain_rate_horizontal(ᶜu)
        if use_prognostic_tke(turbconv_model)
            ᶜtke = @. specific(Yc.ρtke, Yc.ρ)
            ᶜρtkeₜ = register_copy(Yₜc.ρtke)
            @. ᶜρtkeₜ += wdivₕ(Yc.ρ * ᶜK_u_h * gradₕ(ᶜtke))
            # The integrand `S : Sₕ` of the shear production, with `S = S_v + Sₕ`, in one
            # pointwise call that reads `ᶜS_h` once
            shear_integrand = (S_v, S_h) -> strain_rate_inner(S_v + S_h, S_h)
            @. ᶜρtkeₜ += 2 * Yc.ρ * ᶜK_u_h * lumpedₕ(shear_integrand(ᶜS_v, ᶜS_h))
        end
        ᶜuₕₜ = @. C12(wdivₕ(Yc.ρ * (-2 * ᶜK_u_h * (ᶜS_v + ᶜS_h))) / Yc.ρ)
        use_prognostic_tke(turbconv_model) && @. Yₜc.ρtke = ᶜρtkeₜ
        @. Yₜc.uₕ -= ᶜuₕₜ
    end
    # The face slab loop overwrites `ρ K_u` with the stress divergence, which a
    # column loop divides by the face density (slab loops cannot interpolate it).
    DataLayouts.foreach_slab(ᶠu, ᶠΔu, ᶠρK_u_h) do ᶠu, ᶠΔu, ᶠρK_u_h
        ᶠS_v = vertical_strain_rate(ᶠΔu)
        ᶠS_h = compute_strain_rate_horizontal(ᶠu)
        ᶠρu₃ₜ = @. first(parent(C3(wdivₕ(-2 * ᶠρK_u_h * (ᶠS_v + ᶠS_h)))))
        @. ᶠρK_u_h = ᶠρu₃ₜ
    end
    DataLayouts.foreach_column(Yₜ.f.u₃, ᶠρK_u_h, Y.c.ρ) do ᶠu₃ₜ, ᶠρu₃ₜ, ᶜρ
        ᶠwₜ = ᶠu₃ₜ.components.data.:1
        @. ᶠwₜ -= ᶠρu₃ₜ / ᶠinterp(ᶜρ)
    end
    return nothing
end

# The components `Δu` of a vertical velocity gradient `C3 ⊗ UVW` (the velocity
# differences between the levels around a point), and the lazy strain rate of
# the gradient with the components `Δu`.
@inline vertical_difference(∇ᵥu) = UVW(parent(∇ᵥu)...)
@inline function vertical_strain_rate(Δu)
    axis_uvw = (Geometry.UVWAxis(),)
    return @inline @. lazy(
        symmetric_part(Geometry.project(axis_uvw, vertical_gradient(Δu))),
    )
end
@inline vertical_gradient(Δu) =
    Geometry.outer(Geometry.Covariant3Vector(one(eltype(Δu))), Δu)
