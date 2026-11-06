#####
##### Applies tendencies to density, total energy, and total specific humidity
##### due to the sedimentation/precipitation of water and its associated enthalpy
#####


import ClimaCore: DataLayouts, Fields

"""
    vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

Add the grid-mean tendencies from sedimentation of the water species and the
energy they carry.

Increments `Yₜ.c.ρ`, `Yₜ.c.ρq_tot`, and `Yₜ.c.ρe_tot`. For each sedimenting
species present in `Y.c` (`ρq_lcl`, `ρq_icl`, `ρq_rai`, `ρq_sno`), the mass
tendency is the conservative flux convergence `-∇⋅F` of the downward flux
`ρ w q`, added to both density and total water, and the energy tendency carries
the specific energy `e_int(T) + Φ + Kin(w, u)`, with `e_int` the liquid or ice
internal energy, `w` the species terminal velocity, and `u` the air velocity.
Face fluxes are reconstructed with a first-order (right-biased) upwind scheme and
diverged with `ᶜprecipdivᵥ`. Species not carried by the state are skipped, so the
function is inactive for a `DryModel`.

For `PrognosticEDMFX`, the grid-mean energy flux is corrected so it equals the sum
of the subdomain (updraft and environment) fluxes: each correction is the
subdomain sedimentation mass flux times the difference between its specific energy
and the grid-mean one (`Φ` cancels, being identical across subdomains); see
`updraft_precip_energy_flux` and `environment_precip_energy_flux`, which the
surface-flux linearization shares. The environment mass flux is the residual
`ρqw - ρaʲqʲwʲ`, so the subdomain mass fluxes sum to the grid-mean flux by
construction. This path assumes one updraft, and approximates the environment
kinetic term with the grid-mean terminal velocity.

Reads the precomputed terminal velocities `ᶜwₗ`, `ᶜwᵢ`, `ᶜwᵣ`, `ᶜwₛ` (and their
updraft counterparts plus subdomain temperatures, densities, and velocities for
EDMFX), the temperature `ᶜT`, the geopotential `ᶜΦ` from `p.core`, and scratch
space; `t` is unused. Called from `implicit_tendency!`, so these terms are part of
the implicitly treated vertical transport. See the "Microphysics" page of the docs
(`docs/src/microphysics.md`). Returns `nothing`.
"""
function vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    (; params) = p
    (; ᶜΦ, ᶠ∂ξ³∂x³) = p.core
    (; ᶜu, ᶜT) = p.precomputed
    thp = CAP.thermodynamics_params(params)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J

    microphysics_tracers = unrolled_filter(
        ((ρq_name, _),) -> MatrixFields.has_field(Y.c, ρq_name),
        (
            (@name(ρq_lcl), @name(ᶜwₗ)),
            (@name(ρq_icl), @name(ᶜwᵢ)),
            (@name(ρq_rai), @name(ᶜwᵣ)),
            (@name(ρq_sno), @name(ᶜwₛ)),
        ),
    )
    isempty(microphysics_tracers) && return nothing
    ρq_names = unrolled_map(first, microphysics_tracers)
    ᶜws = unrolled_map(
        ((_, w_name),) -> MatrixFields.get_field(p.precomputed, w_name),
        microphysics_tracers,
    )

    # One fused column loop for all species: the face density and the per-species
    # specific humidity and energy density are loop temporaries (registers on
    # GPUs) rather than scratch fields, and the five broadcasts per species run in
    # the same kernel.
    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ
        ᶜuvw = @. Geometry.UVWVector(ᶜu) # converted once for every species
        MatrixFields.unrolled_foreach(map(tuple, ρq_names, ᶜws)) do (ρq_name, ᶜw)
            @inline
            ᶜρq = MatrixFields.get_field(Yc, ρq_name)
            e_int_func = internal_energy_func(ρq_name)
            ᶜq = @. specific_nonneg(ᶜρq, Yc.ρ)
            vtt = @. -1 * ᶜprecipdivᵥ(CT3(ᶠ∂ξ³∂x³ * (ᶠρ * ᶠtop_bias(-(ᶜw) * ᶜq))))
            @. Yₜc.ρ += vtt
            @. Yₜc.ρq_tot += vtt
            ᶜρwe = @. -(ᶜw) * ᶜq * (e_int_func(thp, ᶜT) + ᶜΦ + $(Kin(ᶜw, ᶜuvw)))
            @. Yₜc.ρe_tot -= ᶜprecipdivᵥ(CT3(ᶠ∂ξ³∂x³ * (ᶠρ * ᶠtop_bias(ᶜρwe))))
        end
        return nothing
    end

    # For prognostic EDMF, the grid-mean energy flux applied above is corrected
    # so that it equals the sum of the updraft and environment energy fluxes;
    # see `updraft_precip_energy_flux` and `environment_precip_energy_flux`.
    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜρʲs, ᶜTʲs, ᶜuʲs) = p.precomputed
        (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰) = p.precomputed

        # TODO the following code works for only one updraft
        sgs_microphysics_tracers = unrolled_filter(
            ((q_name, _, _),) -> MatrixFields.has_field(Y.c.sgsʲs.:(1), q_name),
            (
                (@name(q_lcl), @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
                (@name(q_icl), @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
                (@name(q_rai), @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
                (@name(q_sno), @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
            ),
        )
        isempty(sgs_microphysics_tracers) && return nothing
        q_names = unrolled_map(first, sgs_microphysics_tracers)
        # The updraft and grid-mean velocities of each species.
        ᶜw_pairs = unrolled_map(
            ((_, wʲ_name, w_name),) -> (
                MatrixFields.get_field(p.precomputed, wʲ_name),
                MatrixFields.get_field(p.precomputed, w_name),
            ),
            sgs_microphysics_tracers,
        )
        # One fused column loop for the updraft and environment corrections of
        # all species; the subdomain face densities and the converted
        # velocities are loop temporaries.
        DataLayouts.foreach_column(
            Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶜT, ᶜu, ᶜρʲs.:(1), ᶜTʲs.:(1), ᶜuʲs.:(1),
            ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰, ᶠ∂ξ³∂x³, ᶜw_pairs,
        ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶜT, ᶜu, ᶜρʲ, ᶜTʲ, ᶜuʲ,
        ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰, ᶠ∂ξ³∂x³, ᶜw_pairs
            ᶜρ⁰ = @. TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
            ᶠρʲ = @. ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ
            ᶠρ⁰ = @. ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ
            ᶜρaʲ = Yc.sgsʲs.:(1).ρa
            # The velocities are converted once for every species.
            ᶜuvw = @. Geometry.UVWVector(ᶜu)
            ᶜuvwʲ = @. Geometry.UVWVector(ᶜuʲ)
            ᶜuvw⁰ = @. Geometry.UVWVector(ᶜu⁰)
            MatrixFields.unrolled_foreach(
                map(tuple, q_names, ᶜw_pairs),
            ) do (q_name, (ᶜwʲ, ᶜw))
                @inline
                ᶜqʲ = MatrixFields.get_field(Yc.sgsʲs.:(1), q_name)
                ᶜρq = MatrixFields.get_field(Yc, get_ρχ_name(q_name))
                e_int_func = internal_energy_func(get_ρχ_name(q_name))
                @. Yₜc.ρe_tot -= ᶜprecipdivᵥ(
                    CT3(
                        ᶠ∂ξ³∂x³ * (
                            ᶠρʲ * ᶠtop_bias(
                                updraft_precip_energy_flux(
                                    ᶜqʲ, ᶜwʲ, ᶜTʲ, ᶜuvwʲ, ᶜw, ᶜT, ᶜuvw,
                                    ᶜρaʲ, ᶜρʲ, e_int_func, thp,
                                ),
                            ) +
                            ᶠρ⁰ * ᶠtop_bias(
                                environment_precip_energy_flux(
                                    ᶜqʲ, ᶜwʲ, ᶜρq, ᶜw, ᶜT, ᶜuvw, ᶜT⁰, ᶜuvw⁰,
                                    ᶜρ⁰, ᶜρaʲ, e_int_func, thp,
                                ),
                            )
                        ),
                    ),
                )
            end
            return nothing
        end
    end

    record_stage_precipitation_linearization!(Y, p)
    return nothing
end

"""
    sedimentation_kinetic_energy(w, uvw)

Return the specific kinetic energy `|uvw - w ẑ|² / 2` [J/kg] of a species that
falls at terminal velocity `w` [m/s] through air moving with the `UVWVector`
`uvw` [m/s]. The scalar counterpart of the lazy broadcast `Kin`.
"""
@inline sedimentation_kinetic_energy(w, uvw) =
    norm_sqr(Geometry.UVWVector(0, 0, -w) + uvw) / 2

"""
    updraft_precip_energy_flux(qʲ, wʲ, Tʲ, uvwʲ, w, T, uvw, ρaʲ, ρʲ, e_int_func, thp)

Return the updraft correction to the downward sedimentation energy flux per unit
updraft density, `-wʲ aʲ max(0, qʲ) Δeʲ` [m/s · J/kg], where
`aʲ = max(0, ρaʲ / ρʲ)` and

    Δeʲ = e_int(Tʲ) + Kin(wʲ, uʲ) - e_int(T) - Kin(w, u)

is the excess of the updraft specific energy over the grid-mean specific energy
that the grid-mean flux already carries; `Φ` is the same in every subdomain at
a level and cancels. `e_int_func(thp, T)` is the liquid or ice internal energy
of the species. Multiplying by the updraft density and applying `ᶠtop_bias`
gives the face flux; multiplying by `ᶜρʲ ᶜJ / ᶠJ` at level 1 gives its value
through the surface face.
"""
@inline function updraft_precip_energy_flux(
    qʲ, wʲ, Tʲ, uvwʲ, w, T, uvw, ρaʲ, ρʲ, e_int_func, thp,
)
    zero_FT = zero(w)
    aʲ_pos = max(zero_FT, draft_area(ρaʲ, ρʲ))
    Δeʲ =
        e_int_func(thp, Tʲ) + sedimentation_kinetic_energy(wʲ, uvwʲ) -
        e_int_func(thp, T) - sedimentation_kinetic_energy(w, uvw)
    return -wʲ * aʲ_pos * max(zero_FT, qʲ) * Δeʲ
end

"""
    environment_precip_energy_flux(qʲ, wʲ, ρq, w, T, uvw, T⁰, uvw⁰, ρ⁰, ρaʲ, e_int_func, thp)

Return the environment correction to the downward sedimentation energy flux per
unit environment density, `-(w a q)⁰ Δe⁰` [m/s · J/kg], where

    Δe⁰ = e_int(T⁰) + Kin(w, u⁰) - e_int(T) - Kin(w, u)

and the environment sedimentation mass flux is the residual of the grid mean
and the updraft, `(w a q)⁰ = max(0, (max(0, ρq) w - max(0, ρaʲ) max(0, qʲ) wʲ) / ρ⁰)`.
The environment falls at the grid-mean terminal velocity `w`, since no separate
environment velocity is stored. The clamp keeps the environment flux downward
where the updraft flux exceeds the grid-mean flux; there, the subdomain fluxes
sum to more than the grid-mean flux.
"""
@inline function environment_precip_energy_flux(
    qʲ, wʲ, ρq, w, T, uvw, T⁰, uvw⁰, ρ⁰, ρaʲ, e_int_func, thp,
)
    zero_FT = zero(w)
    Δe⁰ =
        e_int_func(thp, T⁰) + sedimentation_kinetic_energy(w, uvw⁰) -
        e_int_func(thp, T) - sedimentation_kinetic_energy(w, uvw)
    waq⁰ = max(
        zero_FT,
        (max(zero_FT, ρq) * w - max(zero_FT, ρaʲ) * max(zero_FT, qʲ) * wʲ) / ρ⁰,
    )
    return -waq⁰ * Δe⁰
end

"""
    set_sfc_sgs_precip_energy_flux!(dest, Y, p, turbconv_model, gm)

Write into the surface-level field `dest` the subdomain correction to the
bottom-face sedimentation energy flux at the Newton linearization state, summed
over the four condensate species [W/m²]. `gm` holds the surface-level grid-mean
inputs of `record_stage_precipitation_linearization!` (`sfc_lev`, `J_ratio`,
`int_T`, `int_uvw`, and the species' `int_ρq_*` and `sfc_w_*`). Each term is
`J_ratio (ρʲ fʲ + ρ⁰ f⁰)` with `fʲ` and `f⁰` from `updraft_precip_energy_flux`
and `environment_precip_energy_flux`, which is the value of the column
correction flux through the surface face. Zero for every turbulence-convection
model other than `PrognosticEDMFX`, and for a `PrognosticEDMFX` whose updrafts
carry no precipitation.

The environment density is first stored in `p.scratch.sfc_precip_env_density`,
and the species terms are added in separate broadcasts: a single fused
broadcast repeats the lazy density in every term and exceeds the 4 KiB kernel
parameter limit of some GPUs.
"""
set_sfc_sgs_precip_energy_flux!(dest, Y, p, _, gm) =
    fill!(dest, zero(eltype(p.params)))
function set_sfc_sgs_precip_energy_flux!(dest, Y, p, ::PrognosticEDMFX, gm)
    FT = eltype(p.params)
    fill!(dest, zero(FT))
    MatrixFields.has_field(Y.c.sgsʲs.:(1), @name(q_rai)) || return nothing
    (; ᶜρʲs, ᶜTʲs, ᶜuʲs, ᶜwₗʲs, ᶜwᵢʲs, ᶜwᵣʲs, ᶜwₛʲs) = p.precomputed
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰) = p.precomputed
    (; sfc_lev, J_ratio, int_T, int_uvw) = gm
    thp = CAP.thermodynamics_params(p.params)

    int_ρʲ = sfc_lev(ᶜρʲs.:(1))
    int_Tʲ = sfc_lev(ᶜTʲs.:(1))
    int_uvwʲ = @. lazy(Geometry.UVWVector($(sfc_lev(ᶜuʲs.:(1)))))
    int_T⁰ = sfc_lev(ᶜT⁰)
    int_uvw⁰ = @. lazy(Geometry.UVWVector($(sfc_lev(ᶜu⁰))))
    int_ρ⁰ = p.scratch.sfc_precip_env_density
    @. int_ρ⁰ = max(
        eps(FT),
        TD.air_density(
            thp,
            int_T⁰,
            $(sfc_lev(ᶜp)),
            $(sfc_lev(ᶜq_tot_nonneg⁰)),
            $(sfc_lev(ᶜq_liq⁰)),
            $(sfc_lev(ᶜq_ice⁰)),
        ),
    )
    int_ρaʲ = sfc_lev(Y.c.sgsʲs.:(1).ρa)

    species = (
        (
            sfc_lev(Y.c.sgsʲs.:(1).q_lcl), sfc_lev(ᶜwₗʲs.:(1)),
            gm.int_ρq_lcl, gm.sfc_wₗ, TD.internal_energy_liquid,
        ),
        (
            sfc_lev(Y.c.sgsʲs.:(1).q_icl), sfc_lev(ᶜwᵢʲs.:(1)),
            gm.int_ρq_icl, gm.sfc_wᵢ, TD.internal_energy_ice,
        ),
        (
            sfc_lev(Y.c.sgsʲs.:(1).q_rai), sfc_lev(ᶜwᵣʲs.:(1)),
            gm.int_ρq_rai, gm.sfc_wᵣ, TD.internal_energy_liquid,
        ),
        (
            sfc_lev(Y.c.sgsʲs.:(1).q_sno), sfc_lev(ᶜwₛʲs.:(1)),
            gm.int_ρq_sno, gm.sfc_wₛ, TD.internal_energy_ice,
        ),
    )
    MatrixFields.unrolled_foreach(species) do (int_qʲ, sfc_wʲ, int_ρq, sfc_w, e_int_func)
        @. dest +=
            J_ratio * (
                int_ρʲ * updraft_precip_energy_flux(
                    int_qʲ, sfc_wʲ, int_Tʲ, int_uvwʲ, sfc_w, int_T, int_uvw,
                    int_ρaʲ, int_ρʲ, e_int_func, thp,
                ) +
                int_ρ⁰ * environment_precip_energy_flux(
                    int_qʲ, sfc_wʲ, int_ρq, sfc_w, int_T, int_uvw, int_T⁰,
                    int_uvw⁰, int_ρ⁰, int_ρaʲ, e_int_func, thp,
                )
            )
    end
    return nothing
end

@inline _precip_lin_nt(
    w_lcl_base,
    w_icl_base,
    w_rai_jac,
    w_sno_jac,
    f_guess_lcl,
    f_guess_icl,
    f_guess_rai,
    f_guess_sno,
    ρq_lcl_guess,
    ρq_icl_guess,
    ρq_rai_guess,
    ρq_sno_guess,
    e_lcl,
    e_icl,
    e_rai,
    e_sno,
    f_e_sgs,
) = (;
    w_lcl_base,
    w_icl_base,
    w_rai_jac,
    w_sno_jac,
    f_guess_lcl,
    f_guess_icl,
    f_guess_rai,
    f_guess_sno,
    ρq_lcl_guess,
    ρq_icl_guess,
    ρq_rai_guess,
    ρq_sno_guess,
    e_lcl,
    e_icl,
    e_rai,
    e_sno,
    f_e_sgs,
)

"""
    record_stage_precipitation_linearization!(Y, p)

Cache the level-1 condensate mass densities, specific energies, and SGS bottom
energy-flux correction at the Newton linearization state `Y = U^g` into
`p.scratch.sfc_precip_lin_state`. Read after the Newton solve by
`accumulate_stage_precipitation_surface_fluxes!` to evaluate the stage
bottom-face sedimentation flux

    F = w_base * max(0, ρq^g) + w_jac * (ρq* - ρq^g),

where `w_base` is the level-1 terminal velocity (`ᶠJ_1 / ᶠJ_{1/2}`-scaled) and
`w_jac` is the Jacobian terminal velocity: `w_base` for cloud liquid and ice,
and the column maxima `ᶜwᵣ_max` / `ᶜwₛ_max` (scaled the same way) for rain and
snow. This linearization matches the discrete column sink applied by one direct
Newton iteration (`max_newton_iters_ode == 1`, no line search, non-Krylov linear
solve) with the Jacobian updated at every solve
(`update_jacobian_every = "solve"`). The SGS correction comes from
`set_sfc_sgs_precip_energy_flux!`, which dispatches on the
turbulence-convection model.
"""
function record_stage_precipitation_linearization!(Y, p)
    (
        eltype(Y) === eltype(p.params) &&
        has_precip_stage_accumulator(p) &&
        MatrixFields.has_field(Y.c, @name(ρq_lcl)) &&
        MatrixFields.has_field(Y.c, @name(ρq_icl)) &&
        MatrixFields.has_field(Y.c, @name(ρq_rai)) &&
        MatrixFields.has_field(Y.c, @name(ρq_sno))
    ) || return nothing

    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜT, ᶜu, ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ) = p.precomputed
    (; sfc_precip_lin_state) = p.scratch
    thp = CAP.thermodynamics_params(params)
    FT = eltype(params)
    zero_FT = zero(FT)
    sfc_space = axes(sfc_precip_lin_state)
    sfc_lev(x) =
        Fields.Field(Fields.field_values(Fields.level(x, 1)), sfc_space)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    int_J = sfc_lev(ᶜJ)
    sfc_J = Fields.level(ᶠJ, Fields.half)
    J_ratio = @. lazy(int_J / sfc_J)

    int_ρq_lcl = sfc_lev(Y.c.ρq_lcl)
    int_ρq_icl = sfc_lev(Y.c.ρq_icl)
    int_ρq_rai = sfc_lev(Y.c.ρq_rai)
    int_ρq_sno = sfc_lev(Y.c.ρq_sno)
    int_T = sfc_lev(ᶜT)
    int_Φ = sfc_lev(ᶜΦ)
    int_u = sfc_lev(ᶜu)
    sfc_wₗ = sfc_lev(ᶜwₗ)
    sfc_wᵢ = sfc_lev(ᶜwᵢ)
    sfc_wᵣ = sfc_lev(ᶜwᵣ)
    sfc_wₛ = sfc_lev(ᶜwₛ)

    fill_column_max!(p.scratch.ᶜtemp_scalar_3, ᶜwᵣ)
    sfc_wᵣ_jac = sfc_lev(p.scratch.ᶜtemp_scalar_3)
    fill_column_max!(p.scratch.ᶜtemp_scalar_4, ᶜwₛ)
    sfc_wₛ_jac = sfc_lev(p.scratch.ᶜtemp_scalar_4)

    w_lcl_base = @. lazy(-sfc_wₗ * J_ratio)
    w_icl_base = @. lazy(-sfc_wᵢ * J_ratio)
    w_rai_base = @. lazy(-sfc_wᵣ * J_ratio)
    w_sno_base = @. lazy(-sfc_wₛ * J_ratio)
    w_rai_jac = @. lazy(-sfc_wᵣ_jac * J_ratio)
    w_sno_jac = @. lazy(-sfc_wₛ_jac * J_ratio)
    f_guess_lcl = @. lazy(w_lcl_base * max(zero_FT, int_ρq_lcl))
    f_guess_icl = @. lazy(w_icl_base * max(zero_FT, int_ρq_icl))
    f_guess_rai = @. lazy(w_rai_base * max(zero_FT, int_ρq_rai))
    f_guess_sno = @. lazy(w_sno_base * max(zero_FT, int_ρq_sno))

    int_uvw = @. lazy(Geometry.UVWVector(int_u))
    e_int_liq = @. lazy(TD.internal_energy_liquid(thp, int_T))
    e_int_ice = @. lazy(TD.internal_energy_ice(thp, int_T))
    e_lcl = @. lazy(e_int_liq + int_Φ + $(Kin(sfc_wₗ, int_uvw)))
    e_icl = @. lazy(e_int_ice + int_Φ + $(Kin(sfc_wᵢ, int_uvw)))
    e_rai = @. lazy(e_int_liq + int_Φ + $(Kin(sfc_wᵣ, int_uvw)))
    e_sno = @. lazy(e_int_ice + int_Φ + $(Kin(sfc_wₛ, int_uvw)))

    gm = (;
        sfc_lev, J_ratio, int_T, int_uvw,
        int_ρq_lcl, int_ρq_icl, int_ρq_rai, int_ρq_sno,
        sfc_wₗ, sfc_wᵢ, sfc_wᵣ, sfc_wₛ,
    )
    @. sfc_precip_lin_state = _precip_lin_nt(
        w_lcl_base,
        w_icl_base,
        w_rai_jac,
        w_sno_jac,
        f_guess_lcl,
        f_guess_icl,
        f_guess_rai,
        f_guess_sno,
        int_ρq_lcl,
        int_ρq_icl,
        int_ρq_rai,
        int_ρq_sno,
        e_lcl,
        e_icl,
        e_rai,
        e_sno,
        zero_FT,
    )
    set_sfc_sgs_precip_energy_flux!(
        sfc_precip_lin_state.f_e_sgs,
        Y,
        p,
        p.atmos.turbconv_model,
        gm,
    )
    return nothing
end
