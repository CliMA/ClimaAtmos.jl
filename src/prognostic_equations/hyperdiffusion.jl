#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

"""
    ν₄(hyperdiff, Y)

A `NamedTuple` of the hyperdiffusivity `ν₄_scalar` and the hyperviscosity
`ν₄_vorticity`. These quantities are assumed to scale with `h^3`, where `h` is
the mean nodal distance, following the empirical results of Lauritzen et al.
(2018, https://doi.org/10.1029/2017MS001257). When `h == 1`, these quantities
are equal to `hyperdiff.ν₄_scalar_coeff` and `hyperdiff.ν₄_vorticity_coeff`.
"""
function ν₄(hyperdiff, Y)
    h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c)))
    ν₄_scalar = hyperdiff.ν₄_scalar_coeff * h^3
    ν₄_vorticity = hyperdiff.ν₄_vorticity_coeff * h^3
    return (; ν₄_scalar, ν₄_vorticity)
end

hyperdiffusion_cache(Y, atmos) = hyperdiffusion_cache(
    Y,
    atmos.hyperdiff,
    atmos.turbconv_model,
    atmos.moisture_model,
    atmos.microphysics_model,
)

# No hyperdiffiusion
hyperdiffusion_cache(Y, hyperdiff::Nothing, _, _, _) = (;)

function hyperdiffusion_cache(
    Y,
    hyperdiff::ClimaHyperdiffusion,
    turbconv_model,
    moisture_model,
    microphysics_model,
)
    quadrature_style =
        Spaces.quadrature_style(Spaces.horizontal_space(axes(Y.c)))
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)

    # Grid scale quantities
    ᶜ∇²u = similar(Y.c, C123{FT})
    gs_quantities = (;
        ᶜ∇²u = similar(Y.c, C123{FT}),
        ᶜ∇²specific_energy = similar(Y.c, FT),
        ᶜ∇²q_tot = similar(Y.c, FT),
    )

    # Sub-grid scale quantities
    ᶜ∇²uʲs =
        turbconv_model isa PrognosticEDMFX ? similar(Y.c, NTuple{n, C123{FT}}) :
        (;)
    sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶜ∇²uₕʲs = similar(Y.c, NTuple{n, C12{FT}}),
            ᶜ∇²uᵥʲs = similar(Y.c, NTuple{n, C3{FT}}),
            ᶜ∇²mseʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_totʲs = similar(Y.c, NTuple{n, FT}),
        ) : (;)
    maybe_ᶜ∇²tke⁰ =
        use_prognostic_tke(turbconv_model) ? (; ᶜ∇²tke⁰ = similar(Y.c, FT)) :
        (;)
    sgs_quantities = (; sgs_quantities..., maybe_ᶜ∇²tke⁰...)
    quantities = (; gs_quantities..., sgs_quantities...)
    if do_dss(axes(Y.c))
        quantities = (;
            quantities...,
            hyperdiffusion_ghost_buffer = map(
                Spaces.create_dss_buffer,
                quantities,
            ),
        )
    end
    return (; quantities..., ᶜ∇²u, ᶜ∇²uʲs)
end

# This should prep variables that we will dss in
# dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    (; ᶜp) = p.precomputed
    (; ᶜh_ref) = p.core
    (; ᶜ∇²u, ᶜ∇²specific_energy, ᶜ∇²q_tot) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs, ᶜ∇²mseʲs, ᶜ∇²q_totʲs) = p.hyperdiff
    end

    # Grid scale hyperdiffusion
    @. ᶜ∇²u =
        C123(wgradₕ(divₕ(p.precomputed.ᶜu))) -
        C123(wcurlₕ(C123(curlₕ(p.precomputed.ᶜu))))

    @. ᶜ∇²specific_energy =
        wdivₕ(gradₕ(specific(Y.c.ρe_tot, Y.c.ρ) + ᶜp / Y.c.ρ - ᶜh_ref))

    @. ᶜ∇²q_tot = wdivₕ(gradₕ(specific(Y.c.ρq_tot, Y.c.ρ)))

    if diffuse_tke
        ᶜρa⁰ =
            turbconv_model isa PrognosticEDMFX ?
            (@. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))) : Y.c.ρ
        ᶜtke⁰ =
            @. lazy(specific_tke(Y.c.ρ, Y.c.sgs⁰.ρatke, ᶜρa⁰, turbconv_model))
        (; ᶜ∇²tke⁰) = p.hyperdiff
        @. ᶜ∇²tke⁰ = wdivₕ(gradₕ(ᶜtke⁰))
    end

    # Sub-grid scale hyperdiffusion
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. ᶜ∇²uʲs.:($$j) =
                C123(wgradₕ(divₕ(p.precomputed.ᶜuʲs.:($$j)))) -
                C123(wcurlₕ(C123(curlₕ(p.precomputed.ᶜuʲs.:($$j)))))
            @. ᶜ∇²mseʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).mse))
            @. ᶜ∇²q_totʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_tot))
            @. ᶜ∇²uₕʲs.:($$j) = C12(ᶜ∇²uʲs.:($$j))
            @. ᶜ∇²uᵥʲs.:($$j) = C3(ᶜ∇²uʲs.:($$j))
        end
    end
end

# This requires dss to have been called on
# variables in dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; divergence_damping_factor) = hyperdiff
    (; ν₄_scalar, ν₄_vorticity) = ν₄(hyperdiff, Y)
    # Rescale the hyperdiffusivity for precipitating species.
    ν₄_scalar_for_precip = CAP.α_hyperdiff_tracer(p.params) * ν₄_scalar

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))

    (; ᶜ∇²u, ᶜ∇²specific_energy, ᶜ∇²q_tot) = p.hyperdiff

    if turbconv_model isa PrognosticEDMFX
        ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs, ᶜ∇²mseʲs, ᶜ∇²q_totʲs) = p.hyperdiff
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke⁰) = p.hyperdiff
    end

    ###
    ### GS velocities
    ###
    # re-use to store the curl-curl part
    @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.c.uₕ -= ν₄_vorticity * C12(ᶜ∇²u)
    @. Yₜ.f.u₃ -= ν₄_vorticity * ᶠwinterp(ᶜJ * Y.c.ρ, C3(ᶜ∇²u))

    ###
    ### GS energy, density and total moisture
    ###
    @. Yₜ.c.ρe_tot -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²specific_energy))
    @. Yₜ.c.ρq_tot -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot))
    @. Yₜ.c.ρ      -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot))

    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?

    ###
    ### GS moisture tracers
    ###
    FT = eltype(Y.c.ρ)
    if p.atmos.moisture_model isa NonEquilMoistModel && (p.atmos.microphysics_model isa Microphysics1Moment || p.atmos.microphysics_model isa Microphysics2Moment)
        # cloud condensate mass
        @. Yₜ.c.ρq_liq -= ν₄_scalar * Y.c.ρq_liq / max(eps(FT), Y.c.ρq_tot) * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot))
        @. Yₜ.c.ρq_ice -= ν₄_scalar * Y.c.ρq_ice / max(eps(FT), Y.c.ρq_tot) * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot))
        # precipitation mass
        @. Yₜ.c.ρq_rai -= ν₄_scalar_for_precip * Y.c.ρq_rai / max(eps(FT), Y.c.ρq_tot) * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot))
        @. Yₜ.c.ρq_sno -= ν₄_scalar_for_precip * Y.c.ρq_sno / max(eps(FT), Y.c.ρq_tot) * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot))
    end
    if p.atmos.moisture_model isa NonEquilMoistModel && p.atmos.microphysics_model isa Microphysics2Moment
        # number concnetrations
        # TODO - should I multiply by som reference number concentration?
        @. Yₜ.c.ρn_liq -= ν₄_scalar            * Y.c.ρq_liq / max(eps(FT), Y.c.ρq_tot) * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot))
        @. Yₜ.c.ρn_rai -= ν₄_scalar_for_precip * Y.c.ρq_rai / max(eps(FT), Y.c.ρq_tot) * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot))
    end

    ###
    ### SGS
    ###
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. ᶜ∇²uʲs.:($$j) = C123(ᶜ∇²uₕʲs.:($$j)) + C123(ᶜ∇²uᵥʲs.:($$j))

            if point_type <: Geometry.Abstract3DPoint
                # only need curl-curl part
                @. ᶜ∇²uᵥʲs.:($$j) = C3(wcurlₕ(C123(curlₕ(ᶜ∇²uʲs.:($$j)))))
                @. Yₜ.f.sgsʲs.:($$j).u₃ +=
                    ν₄_vorticity * ᶠwinterp(ᶜJ * Y.c.ρ, ᶜ∇²uᵥʲs.:($$j))
            end
            # Note: It is more correct to have ρa inside and outside the divergence
            @. Yₜ.c.sgsʲs.:($$j).mse -=
                ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²mseʲs.:($$j)))

            @. Yₜ.c.sgsʲs.:($$j).ρa -=
                ν₄_scalar *
                wdivₕ(Y.c.sgsʲs.:($$j).ρa * gradₕ(ᶜ∇²q_totʲs.:($$j)))
            @. Yₜ.c.sgsʲs.:($$j).q_tot -=
                ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))

            if p.atmos.moisture_model isa NonEquilMoistModel &&
               (p.atmos.microphysics_model isa Microphysics1Moment || p.atmos.microphysics_model isa Microphysics2Moment)
                @. Yₜ.c.sgsʲs.:($$j).q_liq -=
                    ν₄_scalar * Y.c.sgsʲs.q_liq / max(FT(0), Y.c.sgsʲs.q_tot) *  wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_ice -=
                    ν₄_scalar * Y.c.sgsʲs.q_ice / max(FT(0), Y.c.sgsʲs.q_tot) * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_rai -=
                    ν₄_scalar_for_precip * Y.c.sgsʲs.q_rai / max(FT(0), Y.c.sgsʲs.q_tot) * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_sno -=
                    ν₄_scalar_for_precip * Y.c.sgsʲs.q_sno / max(FT(0), Y.c.sgsʲs.q_tot) * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
            end
            if p.atmos.moisture_model isa NonEquilMoistModel && p.atmos.microphysics_model isa Microphysics2Moment
                @. Yₜ.c.sgsʲs.:($$j).n_liq -=
                    ν₄_scalar * Y.c.sgsʲs.q_liq / max(FT(0), Y.c.sgsʲs.q_tot) * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).n_rai -=
                    ν₄_scalar_for_precip * Y.c.sgsʲs.q_rai / max(FT(0), Y.c.sgsʲs.q_tot) * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
            end
        end
    end

    ###
    ### SGS TKE
    ###
    if (turbconv_model isa PrognosticEDMFX) && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= ν₄_vorticity * wdivₕ(ᶜρa⁰ * gradₕ(ᶜ∇²tke⁰))
    end
    if turbconv_model isa DiagnosticEDMFX && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= ν₄_vorticity * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²tke⁰))
    end
end

function dss_hyperdiffusion_tendency_pairs(p)
    (; hyperdiff, turbconv_model) = p.atmos
    buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    (; ᶜ∇²u, ᶜ∇²specific_energy, ᶜ∇²q_tot) = p.hyperdiff
    diffuse_tke = use_prognostic_tke(turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²mseʲs, ᶜ∇²q_totʲs) = p.hyperdiff
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke⁰) = p.hyperdiff
    end

    core_dynamics_pairs = (
        ᶜ∇²u => buffer.ᶜ∇²u,
        ᶜ∇²specific_energy => buffer.ᶜ∇²specific_energy,
        ᶜ∇²q_tot => buffer.ᶜ∇²q_tot,
        (diffuse_tke ? (ᶜ∇²tke⁰ => buffer.ᶜ∇²tke⁰,) : ())...,
    )
    tc_dynamics_pairs =
        turbconv_model isa PrognosticEDMFX ?
        (
            ᶜ∇²uₕʲs => buffer.ᶜ∇²uₕʲs,
            ᶜ∇²uᵥʲs => buffer.ᶜ∇²uᵥʲs,
            ᶜ∇²mseʲs => buffer.ᶜ∇²mseʲs,
            ᶜ∇²q_totʲs => buffer.ᶜ∇²q_totʲs,
        ) : ()
    return (core_dynamics_pairs..., tc_dynamics_pairs...)
end
