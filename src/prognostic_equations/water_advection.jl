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
surface-flux linearization shares. The environment mass flux is the nonnegative
residual `max(0, (max(0, ρq) w - max(0, ρaʲ) max(0, qʲ) wʲ) / ρ⁰)`, so the
subdomain mass fluxes sum to the grid-mean flux where that residual is
nonnegative and exceed it where the floor activates. This path assumes one
updraft, and approximates the environment kinetic term with the grid-mean
terminal velocity.

Reads the precomputed terminal velocities `ᶜwₗ`, `ᶜwᵢ`, `ᶜwᵣ`, `ᶜwₛ` (and their
updraft counterparts plus subdomain temperatures, densities, and velocities for
EDMFX), the temperature `ᶜT`, and the geopotential `ᶜΦ` from `p.core`, in one
column loop; `t` is unused. Called from `implicit_tendency!`, so these terms are
part of the implicitly treated vertical transport. See the "Microphysics" page of
the docs (`docs/src/microphysics.md`). Returns `nothing`.
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

    # For prognostic EDMF, the grid-mean energy flux is corrected so that it
    # equals the sum of the updraft and environment energy fluxes; see
    # `updraft_precip_energy_flux` and `environment_precip_energy_flux`.
    # TODO the corrections work for only one updraft
    ᶜsubdomain_inputs =
        p.atmos.turbconv_model isa PrognosticEDMFX ?
        let (; ᶜρʲs, ᶜTʲs, ᶜuʲs) = p.precomputed,
            (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰) = p.precomputed
            # The updraft velocity of each species, or an empty container for a
            # species that the updraft does not carry
            ᶜwʲs = unrolled_map(ρq_names) do ρq_name
                q_name = specific_tracer_name(ρq_name)
                MatrixFields.has_field(Y.c.sgsʲs.:(1), q_name) ?
                MatrixFields.get_field(
                    p.precomputed,
                    sgs_sedimentation_velocity_name(q_name),
                ) : ()
            end
            (
                ᶜρʲs.:(1), ᶜTʲs.:(1), ᶜuʲs.:(1), ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰,
                ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰, ᶜwʲs,
            )
        end : ()

    # One fused column loop for all species and subdomains: the face fluxes of
    # mass and energy are summed over the species (and, for the energy, over
    # the subdomain corrections), so that the density,
    # total water, and energy tendencies each take one flux divergence.
    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws, ᶜsubdomain_inputs,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws, ᶜsubdomain_inputs
        (ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws, ᶜsubdomain_inputs) =
            register_copy((ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws, ᶜsubdomain_inputs))
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ
        ᶜuvw = @. Geometry.UVWVector(ᶜu) # converted once for every species
        ᶠρwq = @. zero(ᶠρ)
        ᶠρwe = @. zero(ᶠρ)
        if ᶜsubdomain_inputs === ()
            MatrixFields.unrolled_foreach(map(tuple, ρq_names, ᶜws)) do (ρq_name, ᶜw)
                @inline
                ᶜρq = MatrixFields.get_field(Yc, ρq_name)
                e_int_func = internal_energy_func(ρq_name)
                ᶜq = @. specific_nonneg(ᶜρq, Yc.ρ)
                @. ᶠρwq += ᶠρ * ᶠtop_bias(-(ᶜw) * ᶜq)
                @. ᶠρwe +=
                    ᶠρ * ᶠtop_bias(
                        -(ᶜw) * ᶜq * (e_int_func(thp, ᶜT) + ᶜΦ + $(Kin(ᶜw, ᶜuvw))),
                    )
            end
        else
            (ᶜρʲ, ᶜTʲ, ᶜuʲ, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰, ᶜwʲs) =
                ᶜsubdomain_inputs
            eps_FT = eps(eltype(thp))
            ᶜρ⁰ = @. max(
                eps_FT,
                TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰),
            )
            ᶠρʲ = @. ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ
            ᶠρ⁰ = @. ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ
            ᶜρaʲ = Yc.sgsʲs.:(1).ρa
            # The velocities are converted once for every species.
            ᶜuvwʲ = @. Geometry.UVWVector(ᶜuʲ)
            ᶜuvw⁰ = @. Geometry.UVWVector(ᶜu⁰)
            MatrixFields.unrolled_foreach(
                map(tuple, ρq_names, ᶜws, ᶜwʲs),
            ) do (ρq_name, ᶜw, ᶜwʲ)
                @inline
                ᶜρq = MatrixFields.get_field(Yc, ρq_name)
                e_int_func = internal_energy_func(ρq_name)
                ᶜq = @. specific_nonneg(ᶜρq, Yc.ρ)
                @. ᶠρwq += ᶠρ * ᶠtop_bias(-(ᶜw) * ᶜq)
                if ᶜwʲ === ()
                    @. ᶠρwe +=
                        ᶠρ * ᶠtop_bias(
                            -(ᶜw) * ᶜq *
                            (e_int_func(thp, ᶜT) + ᶜΦ + $(Kin(ᶜw, ᶜuvw))),
                        )
                else
                    ᶜqʲ = MatrixFields.get_field(
                        Yc.sgsʲs.:(1),
                        specific_tracer_name(ρq_name),
                    )
                    # A function broadcasts through a heap `Ref` that this
                    # nested broadcast does not elide; the 1-tuple broadcasts
                    # as a scalar without one.
                    e_int_func_tuple = (e_int_func,)
                    @. ᶠρwe +=
                        ᶠρ * ᶠtop_bias(
                            -(ᶜw) * ᶜq *
                            (e_int_func(thp, ᶜT) + ᶜΦ + $(Kin(ᶜw, ᶜuvw))),
                        ) +
                        ᶠρʲ * ᶠtop_bias(
                            updraft_precip_energy_flux(
                                ᶜqʲ, ᶜwʲ, ᶜTʲ, ᶜuvwʲ, ᶜw, ᶜT, ᶜuvw,
                                ᶜρaʲ, ᶜρʲ, e_int_func_tuple, thp,
                            ),
                        ) +
                        ᶠρ⁰ * ᶠtop_bias(
                            environment_precip_energy_flux(
                                ᶜqʲ, ᶜwʲ, ᶜρq, ᶜw, ᶜT, ᶜuvw, ᶜT⁰, ᶜuvw⁰,
                                ᶜρ⁰, ᶜρaʲ, e_int_func_tuple, thp,
                            ),
                        )
                end
            end
        end
        ᶜvtt = @. -1 * ᶜprecipdivᵥ(CT3(ᶠ∂ξ³∂x³ * ᶠρwq))
        @. Yₜc.ρ += ᶜvtt
        @. Yₜc.ρq_tot += ᶜvtt
        @. Yₜc.ρe_tot -= ᶜprecipdivᵥ(CT3(ᶠ∂ξ³∂x³ * ᶠρwe))
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
    environment_precip_energy_flux(
        qʲ, wʲ, ρq, w, T, uvw, T⁰, uvw⁰, ρ⁰, ρaʲ, e_int_func, thp,
    )

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
inputs of `record_stage_precipitation_linearization!` (`sfc_lev`,
`int_T`, `int_u`, and the species' `int_ρq_*` and `sfc_w_*`). Each term is
`J_ratio (ρʲ fʲ + ρ⁰ f⁰)` with `fʲ` and `f⁰` from `updraft_precip_energy_flux`
and `environment_precip_energy_flux`, which is the value of the column
correction flux through the surface face. Zero for every turbulence-convection
model other than `PrognosticEDMFX`, and for a `PrognosticEDMFX` whose updrafts
carry no prognostic condensate species.

The environment density and the velocities of the subdomains and the grid mean
are temporaries of one point loop, which adds the term of every species.
"""
@inline function _sfc_sgs_species_energy_flux(
    J_ratio,
    int_qʲ,
    sfc_wʲ,
    int_ρq,
    sfc_w,
    int_T,
    int_uvw,
    int_Tʲ,
    int_uvwʲ,
    int_ρʲ,
    int_T⁰,
    int_uvw⁰,
    int_ρ⁰,
    int_ρaʲ,
    e_int_func,
    thp,
)
    return J_ratio * (
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

set_sfc_sgs_precip_energy_flux!(dest, Y, p, _, gm) =
    fill!(dest, zero(eltype(p.params)))
function set_sfc_sgs_precip_energy_flux!(dest, Y, p, ::PrognosticEDMFX, gm)
    FT = eltype(p.params)
    sgs_species = unrolled_filter(
        ((q_name, _, _, _, _),) ->
            MatrixFields.has_field(Y.c.sgsʲs.:(1), q_name),
        (
            (
                @name(q_lcl), @name(ᶜwₗʲs.:(1)),
                gm.int_ρq_lcl, gm.sfc_wₗ, TD.internal_energy_liquid,
            ),
            (
                @name(q_icl), @name(ᶜwᵢʲs.:(1)),
                gm.int_ρq_icl, gm.sfc_wᵢ, TD.internal_energy_ice,
            ),
            (
                @name(q_rai), @name(ᶜwᵣʲs.:(1)),
                gm.int_ρq_rai, gm.sfc_wᵣ, TD.internal_energy_liquid,
            ),
            (
                @name(q_sno), @name(ᶜwₛʲs.:(1)),
                gm.int_ρq_sno, gm.sfc_wₛ, TD.internal_energy_ice,
            ),
        ),
    )
    isempty(sgs_species) && return fill!(dest, zero(FT))
    (; ᶜρʲs, ᶜTʲs, ᶜuʲs) = p.precomputed
    (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰) = p.precomputed
    (; sfc_lev, int_T, int_u) = gm
    sfc_J = Fields.level(Fields.local_geometry_field(Y.f).J, Fields.half)
    thp = CAP.thermodynamics_params(p.params)
    eps_FT = eps(FT)
    # The loop takes the values of the fields: a point of a field would compare
    # its space with those of the other points in every statement.
    int_val(x) = Fields.field_values(sfc_lev(x))
    species_fields = unrolled_map(sgs_species) do (q_name, wʲ_name, int_ρq, sfc_w, _)
        int_qʲ = int_val(MatrixFields.get_field(Y.c.sgsʲs.:(1), q_name))
        sfc_wʲ = int_val(MatrixFields.get_field(p.precomputed, wʲ_name))
        (int_qʲ, sfc_wʲ, Fields.field_values(int_ρq), Fields.field_values(sfc_w))
    end
    # The 1-tuples broadcast the functions as scalars without a heap `Ref`.
    e_int_funcs = unrolled_map(species -> (last(species),), sgs_species)
    DataLayouts.foreach_point(
        Fields.field_values(dest), Fields.field_values(sfc_J),
        Fields.field_values(int_T), Fields.field_values(int_u),
        int_val(Fields.local_geometry_field(Y.c)), int_val(ᶜρʲs.:(1)),
        int_val(ᶜTʲs.:(1)), int_val(ᶜuʲs.:(1)), int_val(Y.c.sgsʲs.:(1).ρa), int_val(ᶜT⁰),
        int_val(ᶜu⁰), int_val(ᶜp), int_val(ᶜq_tot_nonneg⁰), int_val(ᶜq_liq⁰),
        int_val(ᶜq_ice⁰), species_fields,
    ) do dest, sfc_J, int_T, int_u, int_lg, int_ρʲ, int_Tʲ, int_uʲ, int_ρaʲ, int_T⁰,
    int_u⁰, int_p, int_q_tot⁰, int_q_liq⁰, int_q_ice⁰, species_fields
        # The velocities are converted with the local geometry of the first
        # cell, which the surface space does not hold.
        J_ratio = @. int_lg.J / sfc_J
        int_uvw = @. Geometry.UVWVector(int_u, int_lg)
        int_uvwʲ = @. Geometry.UVWVector(int_uʲ, int_lg)
        int_uvw⁰ = @. Geometry.UVWVector(int_u⁰, int_lg)
        int_ρ⁰ = @. max(
            eps_FT,
            TD.air_density(thp, int_T⁰, int_p, int_q_tot⁰, int_q_liq⁰, int_q_ice⁰),
        )
        @. dest = zero(int_T)
        MatrixFields.unrolled_foreach(
            species_fields, e_int_funcs,
        ) do (int_qʲ, sfc_wʲ, int_ρq, sfc_w), e_int_func
            @inline
            @. dest += _sfc_sgs_species_energy_flux(
                J_ratio, int_qʲ, sfc_wʲ, int_ρq, sfc_w, int_T, int_uvw, int_Tʲ,
                int_uvwʲ, int_ρʲ, int_T⁰, int_uvw⁰, int_ρ⁰, int_ρaʲ, e_int_func, thp,
            )
        end
    end
    return nothing
end

@inline function _precip_lin_nt(
    thp,
    int_lg,
    sfc_J,
    int_T,
    int_Φ,
    int_u,
    ρq_lcl_guess,
    ρq_icl_guess,
    ρq_rai_guess,
    ρq_sno_guess,
    sfc_wₗ,
    sfc_wᵢ,
    sfc_wᵣ,
    sfc_wₛ,
    sfc_wᵢ_jac,
    sfc_wᵣ_jac,
    sfc_wₛ_jac,
)
    zero_FT = zero(int_T)
    J_ratio = int_lg.J / sfc_J
    int_uvw = Geometry.UVWVector(int_u, int_lg)
    w_lcl_base = -sfc_wₗ * J_ratio
    w_icl_base = -sfc_wᵢ * J_ratio
    w_rai_base = -sfc_wᵣ * J_ratio
    w_sno_base = -sfc_wₛ * J_ratio
    w_lcl_jac = w_lcl_base
    w_icl_jac = -sfc_wᵢ_jac * J_ratio
    w_rai_jac = -sfc_wᵣ_jac * J_ratio
    w_sno_jac = -sfc_wₛ_jac * J_ratio
    f_guess_lcl = w_lcl_base * max(zero_FT, ρq_lcl_guess)
    f_guess_icl = w_icl_base * max(zero_FT, ρq_icl_guess)
    f_guess_rai = w_rai_base * max(zero_FT, ρq_rai_guess)
    f_guess_sno = w_sno_base * max(zero_FT, ρq_sno_guess)
    e_int_liq = TD.internal_energy_liquid(thp, int_T)
    e_int_ice = TD.internal_energy_ice(thp, int_T)
    e_lcl = e_int_liq + int_Φ + sedimentation_kinetic_energy(sfc_wₗ, int_uvw)
    e_icl = e_int_ice + int_Φ + sedimentation_kinetic_energy(sfc_wᵢ, int_uvw)
    e_rai = e_int_liq + int_Φ + sedimentation_kinetic_energy(sfc_wᵣ, int_uvw)
    e_sno = e_int_ice + int_Φ + sedimentation_kinetic_energy(sfc_wₛ, int_uvw)
    f_e_sgs = zero_FT
    return (;
        w_lcl_jac,
        w_icl_jac,
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
end

"""
    record_stage_precipitation_linearization!(Y, p)

Cache the level-1 condensate mass densities, specific energies, and SGS bottom
energy-flux correction at the Newton linearization state `Y = U^g` into
`p.scratch.sfc_precip_lin_state`. Read after the Newton solve by
`accumulate_stage_precipitation_surface_fluxes!` to evaluate the stage
bottom-face sedimentation flux

    F = w_base * max(0, ρq^g) + w_jac * (ρq* - ρq^g),

where `w_base` is the level-1 terminal velocity (`ᶜJ_1 / ᶠJ_{1/2}`-scaled) and
`w_jac` is the Jacobian terminal velocity: `w_base` for cloud liquid, and the
column maxima `ᶜwᵢ_max` / `ᶜwᵣ_max` / `ᶜwₛ_max` (scaled the same way) for cloud
ice, rain, and snow, matching `update_sedimentation_jacobian!`. This
linearization matches the discrete column sink applied by the stage under four
conditions: (1) direct Newton solve without Krylov or line search, with the
Jacobian refreshed at the final Newton iterate (`update_jacobian_every = "solve"`,
or `"stage"` when `max_newton_iters_ode == 1`); (2) `ManualSparseJacobian`
(autodiff uses the pointwise derivative and disables this accumulator);
(3) direct inversion of the block-lower-triangular scalar submatrix `A₁₁`
(`ApproximateBlockArrowheadIterativeSolve` wraps `SchurComplementReductionSolve`
with `alg₁ = BlockLowerTriangularSolve`, approximating only the velocity Schur
complement whose `ᶜadvdivᵥ` columns integrate to zero over each column, so
column sums hold at any `approximate_solve_iters ≥ 1`); and
(4) `constrain_state!` leaving column water unchanged (`ρq_tot ≥ 0`; setting
`update_constrain_state_every = "dss"` under condensate warm-starting preserves
conservation when `ρq_tot ≥ 0`, but re-clamping `U^g` before the solve can drive
the stage flux upward). A stage flux is upward (`F > 0`) whenever
`|w_jac| (ρq^g - ρq*) > |w_base| max(0, ρq^g)` — from (a) a negative bottom cell
with a net implicit sink and `w_jac > 0` (from condensate aloft, the 1M `10⁻⁸`
rain terminal-velocity floor, or `FixedTerminalVelocity`), or (b) a positive
bottom cell drained below `(1 - w_base / w_jac) ρq^g` within one stage (e.g.,
subcloud evaporation at linearized dimensionless rate
`dtγ λ > w_base / (w_jac - w_base)`) — and (c) negative IMEX `b_imp` stage
weights (`ARS343` has `b_imp[3] ≈ -0.644`) can make the step sum upward even
when every stage flux is downward; in all cases the accumulated sum equals the
discrete column sink (up to explicit-diagonal weight redistribution on `SSP` /
`IMKG` tableaus). The SGS correction comes from `set_sfc_sgs_precip_energy_flux!`,
which dispatches on the turbulence-convection model.
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
    sfc_space = axes(sfc_precip_lin_state)
    sfc_lev(x) =
        Fields.Field(Fields.field_values(Fields.level(x, 1)), sfc_space)

    int_lg = sfc_lev(Fields.local_geometry_field(Y.c))
    sfc_J = Fields.level(Fields.local_geometry_field(Y.f).J, Fields.half)

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

    # The column maxima of the rain, snow, and cloud ice velocities in one
    # column loop, stored in the first level of scratch fields.
    # TODO: Skip them in explicit-stage and diagnostic calls, and reuse
    # `sfc_precip_lin_state` in `surface_precipitation_tendency!` for
    # `SlabOceanTemperature`.
    sfc_wᵣ_jac = sfc_lev(p.scratch.ᶜtemp_scalar_3)
    sfc_wₛ_jac = sfc_lev(p.scratch.ᶜtemp_scalar_4)
    sfc_wᵢ_jac = sfc_lev(p.scratch.ᶜtemp_scalar_5)
    DataLayouts.foreach_column(
        unrolled_map(Fields.field_values, (sfc_wᵣ_jac, sfc_wₛ_jac, sfc_wᵢ_jac)),
        unrolled_map(Fields.field_values, (ᶜwᵣ, ᶜwₛ, ᶜwᵢ)),
    ) do sfc_w_jacs, ᶜws
        MatrixFields.unrolled_foreach(sfc_w_jacs, ᶜws) do sfc_w_jac, ᶜw
            @inline
            fill!(sfc_w_jac, reduce(max, ᶜw))
        end
    end

    # `_precip_lin_nt` computes the linearization fields from the pointwise
    # inputs so each input `Field` enters the broadcast kernel once (staying
    # within the 4 KiB CUDA kernel parameter limit).
    @. sfc_precip_lin_state = _precip_lin_nt(
        thp,
        int_lg,
        sfc_J,
        int_T,
        int_Φ,
        int_u,
        int_ρq_lcl,
        int_ρq_icl,
        int_ρq_rai,
        int_ρq_sno,
        sfc_wₗ,
        sfc_wᵢ,
        sfc_wᵣ,
        sfc_wₛ,
        sfc_wᵢ_jac,
        sfc_wᵣ_jac,
        sfc_wₛ_jac,
    )
    gm = (;
        sfc_lev, int_T, int_u,
        int_ρq_lcl, int_ρq_icl, int_ρq_rai, int_ρq_sno,
        sfc_wₗ, sfc_wᵢ, sfc_wᵣ, sfc_wₛ,
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
