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
and the grid-mean one (`Φ` cancels, being identical across subdomains). The
environment mass flux is taken as the residual `ρqw - ρaʲqʲwʲ`, so the subdomain
mass fluxes sum to the grid-mean flux by construction. This path currently assumes
one updraft, and approximates the environment kinetic term with the grid-mean
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

    # For prognostic edmf, the grid-mean energy flux is augmented with the
    # additional energy contributions, so that it equals the sum of the subdomain
    # (updraft/environment) energy fluxes, by accounting for the energy carried
    # by sedimenting tracers.
    #
    # The grid-mean flux carries the specific energy e_int(T) + Φ + Kin(w, u)
    # per unit sedimenting mass. Each subdomain correction replaces this
    # grid-mean specific energy with the subdomain value, weighted by the
    # subdomain sedimentation mass flux:
    #
    #   F_corr(k) = (mass flux)ᵏ * [(e_intᵏ + Kinᵏ) - (e_int + Kin)],
    #
    # where Φ cancels because it is identical in all subdomains at a given
    # level. The environment mass flux is defined as the residual
    # ρ⁰a⁰q⁰w⁰ = ρqw - ρaʲqʲwʲ, so the subdomain mass fluxes sum to the
    # grid-mean flux by construction, and both corrections enter with the
    # same sign convention as the grid-mean flux.
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
    # the subdomain corrections) in loop temporaries, so that the density,
    # total water, and energy tendencies each take one flux divergence.
    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws, ᶜsubdomain_inputs,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws, ᶜsubdomain_inputs
        (ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws, ᶜsubdomain_inputs) =
            as_register((ᶜJ, ᶠJ, ᶜT, ᶜΦ, ᶜu, ᶠ∂ξ³∂x³, ᶜws, ᶜsubdomain_inputs))
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ
        ᶜuvw = @. Geometry.UVWVector(ᶜu) # converted once for every species
        ᶠρwq = @. zero(ᶠρ)
        ᶠρwe = @. zero(ᶠρ)
        if ᶜsubdomain_inputs === ()
            MatrixFields.unrolled_foreach(map(tuple, ρq_names, ᶜws)) do (ρq_name, ᶜw)
                @inline
                ᶜρq = MatrixFields.get_field(Yc, ρq_name)
                e_int_func = internal_energy_func(ρq_name)
                ᶜq = @. specific(ᶜρq, Yc.ρ)
                @. ᶠρwq += ᶠρ * ᶠtop_bias(-(ᶜw) * ᶜq)
                @. ᶠρwe +=
                    ᶠρ * ᶠtop_bias(
                        -(ᶜw) * ᶜq * (e_int_func(thp, ᶜT) + ᶜΦ + $(Kin(ᶜw, ᶜuvw))),
                    )
            end
        else
            (ᶜρʲ, ᶜTʲ, ᶜuʲ, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰, ᶜu⁰, ᶜwʲs) =
                ᶜsubdomain_inputs
            ᶜρ⁰ = @. TD.air_density(thp, ᶜT⁰, ᶜp, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰)
            ᶠρʲ = @. ᶠinterp(ᶜρʲ * ᶜJ) / ᶠJ
            ᶠρ⁰ = @. ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ
            ᶜρaʲ = Yc.sgsʲs.:(1).ρa
            ᶜaʲ = @. draft_area(ᶜρaʲ, ᶜρʲ)
            # The velocities are converted once for every species.
            ᶜuvwʲ = @. Geometry.UVWVector(ᶜuʲ)
            ᶜuvw⁰ = @. Geometry.UVWVector(ᶜu⁰)
            MatrixFields.unrolled_foreach(
                map(tuple, ρq_names, ᶜws, ᶜwʲs),
            ) do (ρq_name, ᶜw, ᶜwʲ)
                @inline
                ᶜρq = MatrixFields.get_field(Yc, ρq_name)
                e_int_func = internal_energy_func(ρq_name)
                ᶜq = @. specific(ᶜρq, Yc.ρ)
                @. ᶠρwq += ᶠρ * ᶠtop_bias(-(ᶜw) * ᶜq)
                # Grid-mean specific energy carried by the sedimentation flux,
                # without Φ, which cancels in the subdomain differences
                ᶜe_gm = @. e_int_func(thp, ᶜT) + $(Kin(ᶜw, ᶜuvw))
                if ᶜwʲ === ()
                    @. ᶠρwe += ᶠρ * ᶠtop_bias(-(ᶜw) * ᶜq * (ᶜe_gm + ᶜΦ))
                else
                    ᶜqʲ = MatrixFields.get_field(
                        Yc.sgsʲs.:(1),
                        specific_tracer_name(ρq_name),
                    )
                    # Updraft correction: (e_intʲ + Kinʲ) - (e_int + Kin)
                    ᶜΔeʲ = @. e_int_func(thp, ᶜTʲ) + $(Kin(ᶜwʲ, ᶜuvwʲ)) - ᶜe_gm
                    # Environment correction: (e_int⁰ + Kin⁰) - (e_int + Kin). The
                    # environment sedimentation velocity is not stored separately
                    # (the environment mass flux is the residual ρqw - ρaʲqʲwʲ),
                    # so Kin⁰ is approximated with the grid-mean terminal
                    # velocity ᶜw.
                    # TODO: Update for when updraft area fraction is not
                    # necessarily small
                    ᶜΔe⁰ = @. e_int_func(thp, ᶜT⁰) + $(Kin(ᶜw, ᶜuvw⁰)) - ᶜe_gm
                    ᶜwaq⁰ = @. (ᶜρq * ᶜw - ᶜρaʲ * ᶜqʲ * ᶜwʲ) / ᶜρ⁰
                    @. ᶠρwe +=
                        ᶠρ * ᶠtop_bias(-(ᶜw) * ᶜq * (ᶜe_gm + ᶜΦ)) +
                        ᶠρʲ * ᶠtop_bias(-(ᶜwʲ) * ᶜaʲ * ᶜqʲ * ᶜΔeʲ) +
                        ᶠρ⁰ * ᶠtop_bias(-(ᶜwaq⁰) * ᶜΔe⁰)
                end
            end
        end
        ᶜvtt = @. -1 * ᶜprecipdivᵥ(CT3(ᶠ∂ξ³∂x³ * ᶠρwq))
        @. Yₜc.ρ += ᶜvtt
        @. Yₜc.ρq_tot += ᶜvtt
        @. Yₜc.ρe_tot -= ᶜprecipdivᵥ(CT3(ᶠ∂ξ³∂x³ * ᶠρwe))
    end

    return nothing
end
