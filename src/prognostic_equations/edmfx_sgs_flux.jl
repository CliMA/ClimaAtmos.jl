#####
##### Tendencies applied to the grid-mean atmospheric state due to subgrid-scale (SGS)
##### fluxes computed by the EDMFX scheme
#####

# Diffusing total, liquid, and ice water `(q_tot_eff⁰, q_lcl⁰, q_icl⁰)` of the
# EDMFX environment, the subdomain counterpart of `ᶜdiffusing_water_phases`, for
# the grid-mean state `ᶜY` (a whole field or one column of a fused loop) with
# the environment mass `ᶜρa⁰`. With non-equilibrium microphysics (`Val(true)`),
# `q_tot_eff⁰ = q_tot⁰ - q_rai⁰ - q_sno⁰` and the cloud species are the
# diagnosed environment specific humidities; with an equilibrium scheme,
# `q_tot_eff⁰ = q_tot⁰` and the cloud species are the cached diagnostic
# `ᶜq_liq⁰` and `ᶜq_ice⁰`.
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

# Diffusing total, liquid, and ice water of the updraft state `ᶜsgsʲ` (a whole
# field or one column of a fused loop), the updraft counterpart of
# `env_diffusing_water`, with its cached diagnostic `ᶜq_liqʲ` and `ᶜq_iceʲ`.
@inline updraft_diffusing_water(ᶜsgsʲ, ᶜq_liqʲ, ᶜq_iceʲ, ::Val{true}) = (
    (@inline @. lazy(ᶜsgsʲ.q_tot - ᶜsgsʲ.q_rai - ᶜsgsʲ.q_sno)),
    ᶜsgsʲ.q_lcl,
    ᶜsgsʲ.q_icl,
)
@inline updraft_diffusing_water(ᶜsgsʲ, ᶜq_liqʲ, ᶜq_iceʲ, ::Val{false}) =
    (ᶜsgsʲ.q_tot, ᶜq_liqʲ, ᶜq_iceʲ)

# Moist static energy that the updraft diffuses, for the updraft state `ᶜsgsʲ`
# at the temperature `ᶜTʲ`, a whole field or one column of a fused loop (see
# `materialize_into!`). In a fused loop (`ᶜmse_eff === nothing`), it is a
# register temporary with either microphysics model.
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
spectral-element counterpart of the harmonic face mean `f_j K_h / ᶠinterp(1 / ρaʲ)` of the vertical updraft diffusion. The weak divergence at a node draws on the flux at
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
    no residual; as in the vertical diffusion of the EDMF column step, the moist mass
    that `q_tot` diffusion adds and the enthalpy `(h_d + Φ) ∇ₕq_tot_effʲ` of
    the dry air it displaces go to the environment. Grid-scale scalars that
    the updrafts do not carry diffuse down
    their grid-mean gradients with `ρ K_h_h`. Rain and snow do not diffuse.

Always explicit; applied in the explicit remainder tendency.

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
