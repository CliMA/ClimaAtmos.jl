#####
##### Applies tendencies to density, total energy, and total specific humidity
##### due to the sedimentation/precipitation of water and its associated enthalpy
#####


import ClimaCore: Fields

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
and the grid-mean one (`Φ` cancels, being identical across subdomains). The
environment mass flux is taken as the residual `ρqw - ρaʲqʲwʲ`, so the subdomain
mass fluxes sum to the grid-mean flux by construction. This path currently assumes
one updraft, and approximates the environment kinetic term with the grid-mean
terminal velocity.

Reads the precomputed terminal velocities `ᶜwₗ`, `ᶜwᵢ`, `ᶜwᵣ`, `ᶜwₛ` (and their
updraft counterparts plus subdomain temperatures, densities, and velocities for
EDMFX), the temperature `ᶜT`, the geopotential `ᶜΦ` from `p.core`, and scratch
space; `t` is unused. Called from `implicit_tendency!`, so these terms are part of
the implicitly treated vertical transport. See the "Microphysics" page of the docs
(`docs/src/microphysics.md`). Returns `nothing`.
"""
function vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜu, ᶜT) = p.precomputed
    thp = CAP.thermodynamics_params(params)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J

    microphysics_tracers = (
        (@name(ρq_lcl), @name(ᶜwₗ)),
        (@name(ρq_icl), @name(ᶜwᵢ)),
        (@name(ρq_rai), @name(ᶜwᵣ)),
        (@name(ρq_sno), @name(ᶜwₛ)),
    )
    internal_energy_func(name) =
        (name == @name(ρq_lcl) || name == @name(ρq_rai)) ? TD.internal_energy_liquid :
        (name == @name(ρq_icl) || name == @name(ρq_sno)) ? TD.internal_energy_ice :
        nothing

    ᶠρ = p.scratch.ᶠtemp_scalar
    ᶜq = p.scratch.ᶜtemp_scalar
    vtt = p.scratch.ᶜtemp_scalar_2
    @. ᶠρ = ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ
    MatrixFields.unrolled_foreach(microphysics_tracers) do (ρq_name, w_name)
        MatrixFields.has_field(Y.c, ρq_name) || return

        ᶜρq = MatrixFields.get_field(Y.c, ρq_name)
        ᶜw = MatrixFields.get_field(p.precomputed, w_name)
        @. ᶜq = specific(ᶜρq, Y.c.ρ)
        @. vtt =
            -1 * ᶜprecipdivᵥ(
                ᶠρ * ᶠright_bias(
                    Geometry.WVector(-(ᶜw)) * ᶜq,
                ),
            )
        @. Yₜ.c.ρ += vtt
        @. Yₜ.c.ρq_tot += vtt

        e_int_func = internal_energy_func(ρq_name)
        @. p.scratch.ᶜtemp_scalar_3 =
            -(ᶜw) * ᶜq * (e_int_func(thp, ᶜT) + ᶜΦ + $(Kin(ᶜw, ᶜu)))
        @. Yₜ.c.ρe_tot -= ᶜprecipdivᵥ(
            ᶠρ * ᶠright_bias(
                Geometry.WVector(p.scratch.ᶜtemp_scalar_3),
            ),
        )
    end

    # For prognostic edmf, augment the energy tendencies with the additional energy contributions
    # so that the total-grid energy flux remains consistent. Specifically, we enforce that the
    # grid-mean energy flux equals the sum of the subdomain (updraft/environment) energy fluxes
    # by accounting for the energy carried by sedimenting tracers.
    #
    # The grid-mean flux applied above carries the specific energy
    # e_int(T) + Φ + Kin(w, u) per unit sedimenting mass. Each subdomain
    # correction replaces this grid-mean specific energy with the subdomain
    # value, weighted by the subdomain sedimentation mass flux:
    #
    #   F_corr(k) = (mass flux)ᵏ * [(e_intᵏ + Kinᵏ) - (e_int + Kin)],
    #
    # where Φ cancels because it is identical in all subdomains at a given
    # level. The environment mass flux is defined as the residual
    # ρ⁰a⁰q⁰w⁰ = ρqw - ρaʲqʲwʲ, so the subdomain mass fluxes sum to the
    # grid-mean flux by construction, and both corrections enter with the
    # same sign convention as the grid-mean flux.
    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜρʲs, ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs, ᶜuʲs) = p.precomputed
        (; ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰) = p.precomputed

        ᶜρ⁰ = p.scratch.ᶜtemp_scalar
        @. ᶜρ⁰ =
            TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)

        # TODO the following code works for only one updraft
        sgs_microphysics_tracers = (
            (@name(q_lcl), @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
            (@name(q_icl), @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
            (@name(q_rai), @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
            (@name(q_sno), @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
        )
        MatrixFields.unrolled_foreach(sgs_microphysics_tracers) do (q_name, wʲ_name, w_name)
            MatrixFields.has_field(Y.c.sgsʲs.:(1), q_name) || return

            ᶜqʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), q_name)
            ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)
            ᶜρq = MatrixFields.get_field(Y.c, get_ρχ_name(q_name))
            ᶜw = MatrixFields.get_field(p.precomputed, w_name)
            ᶜuʲ = ᶜuʲs.:(1)

            e_int_func = internal_energy_func(get_ρχ_name(q_name))
            # Grid-mean specific energy carried by the sedimentation flux
            # applied above (Φ is identical in all subdomains at a given
            # level, so it cancels in the subdomain differences).
            @. p.scratch.ᶜtemp_scalar_2 = e_int_func(thp, ᶜT) + $(Kin(ᶜw, ᶜu))
            # Updraft correction: (e_intʲ + Kinʲ) - (e_int + Kin)
            @. p.scratch.ᶜtemp_scalar_3 =
                e_int_func(thp, ᶜTʲs.:(1)) + $(Kin(ᶜwʲ, ᶜuʲ)) -
                p.scratch.ᶜtemp_scalar_2
            @. Yₜ.c.ρe_tot -=
                ᶜprecipdivᵥ(
                    ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ * ᶠright_bias(
                        Geometry.WVector(-(ᶜwʲ)) *
                        draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)) * ᶜqʲ *
                        p.scratch.ᶜtemp_scalar_3,
                    ),
                )
            # Environment correction: (e_int⁰ + Kin⁰) - (e_int + Kin). The
            # environment sedimentation velocity is not stored separately
            # (the environment mass flux is the residual ρqw - ρaʲqʲwʲ), so
            # Kin⁰ is approximated with the grid-mean terminal velocity ᶜw.
            # TODO: Update for when updraft area fraction is not necessarily
            # small
            @. p.scratch.ᶜtemp_scalar_3 =
                e_int_func(thp, ᶜT⁰) + $(Kin(ᶜw, ᶜu⁰)) -
                p.scratch.ᶜtemp_scalar_2
            ᶜwaq⁰ = @. lazy((ᶜρq * ᶜw - Y.c.sgsʲs.:(1).ρa * ᶜqʲ * ᶜwʲ) / ᶜρ⁰)
            @. Yₜ.c.ρe_tot -=
                ᶜprecipdivᵥ(
                    ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ * ᶠright_bias(
                        Geometry.WVector(-(ᶜwaq⁰)) *
                        p.scratch.ᶜtemp_scalar_3,
                    ),
                )
        end
    end

    return nothing
end
