#####
##### Applies tendencies to density, total energy, and total specific humidity
##### due to the sedimentation/precipitation of water and its associated enthalpy
#####


import ClimaCore: Fields

"""
    vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

Computes and applies tendencies to grid-mean density (`ρ`), total energy (`ρe_tot`),
and total specific humidity (`ρq_tot`) due to sedimentation/precipitation of total water
and its associated energy.

This function is active only if the atmospheric model includes moisture (i.e.,
is not a `DryModel`). The tendencies are calculated in a conservative form,
`-∇ ⋅ F`, where `F` represents the vertical fluxes.

Fluxes at cell faces are reconstructed using a first-order upwind scheme.
Specifically, face-valued density (`ᶠρ`) is multiplied by a right-biased,
negated cell-centered specific flux term. The resulting face flux
is then diverged using the `ᶜprecipdivᵥ` operator.

Each sedimenting water species carries its specific internal, potential, and
kinetic energy, `e_int(T) + Φ + Kin(w, u)`, where `w` is the species terminal
velocity and `u` the air velocity. For `PrognosticEDMFX`, the grid-mean flux
computed with grid-mean quantities is corrected so the grid-mean energy flux
equals the sum of the subdomain (updraft and environment) fluxes: each
subdomain correction is its sedimentation mass flux times the difference
between its specific energy and the grid-mean one (`Φ` cancels). The
environment mass flux is taken as the residual `ρqw - ρaʲqʲwʲ`, so the
subdomain mass fluxes sum to the grid-mean flux by construction.

Arguments:

  - `Yₜ`: The tendency state vector, modified in place.
  - `Y`: The current state vector.
  - `p`: Cache containing parameters, precomputed fields (species terminal
    velocities `ᶜwₗ`/`ᶜwᵢ`/`ᶜwᵣ`/`ᶜwₛ` and, for EDMFX, their updraft
    counterparts and the subdomain temperatures and velocities),
    and atmospheric model configurations.
  - `t`: Current simulation time (not directly used in these calculations).

Modifies:

  - `Yₜ.c.ρ`, `Yₜ.c.ρe_tot`, and `Yₜ.c.ρq_tot` (always when moisture is present)
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
