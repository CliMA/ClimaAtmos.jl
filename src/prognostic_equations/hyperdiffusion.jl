#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

hyperdiffusion_cache(Y, atmos) =
    hyperdiffusion_cache(Y, atmos.hyperdiff, atmos.turbconv_model)

# No hyperdiffiusion
hyperdiffusion_cache(Y, hyperdiff::Nothing, _) = (;)

function hyperdiffusion_cache(Y, hyperdiff::ClimaHyperdiffusion, turbconv_model)
    do_dss =
        Spaces.horizontal_space(axes(Y.c)).quadrature_style isa
        Spaces.Quadratures.GLL
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)

    # Grid scale quantities
    ᶜ∇²u = similar(Y.c, C123{FT})
    gs_quantities = (;
        ᶜ∇²u = similar(Y.c, C123{FT}),
        ᶜ∇²specific_energy = similar(Y.c, FT),
        ᶜ∇²specific_tracers = remove_energy_var.(specific_gs.(Y.c)),
    )

    # Sub-grid scale quantities
    ᶜ∇²uʲs =
        turbconv_model isa PrognosticEDMFX ? similar(Y.c, NTuple{n, C123{FT}}) :
        (;)
    sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶜ∇²tke⁰ = similar(Y.c, FT),
            ᶜ∇²uₕʲs = similar(Y.c, NTuple{n, C12{FT}}),
            ᶜ∇²uᵥʲs = similar(Y.c, NTuple{n, C3{FT}}),
            ᶜ∇²mseʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_totʲs = similar(Y.c, NTuple{n, FT}),
        ) :
        turbconv_model isa DiagnosticEDMFX ? (; ᶜ∇²tke⁰ = similar(Y.c, FT)) :
        (;)
    quantities = (; gs_quantities..., sgs_quantities...)
    if do_dss
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

NVTX.@annotate function hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; κ₄_vorticity, κ₄_tracer, divergence_damping_factor) = hyperdiff
    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; do_dss) = p
    (; ᶜp, ᶜspecific) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜρa⁰, ᶜtke⁰) = p.precomputed
        (; ᶜ∇²tke⁰, ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs, ᶜ∇²mseʲs) = p.hyperdiff
    end
    if turbconv_model isa DiagnosticEDMFX
        (; ᶜtke⁰) = p.precomputed
        (; ᶜ∇²tke⁰) = p.hyperdiff
    end
    if do_dss
        buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    end

    # Grid scale hyperdiffusion
    @. ᶜ∇²u =
        C123(wgradₕ(divₕ(p.precomputed.ᶜu))) -
        C123(wcurlₕ(C123(curlₕ(p.precomputed.ᶜu))))

    @. ᶜ∇²specific_energy = wdivₕ(gradₕ(ᶜspecific.e_tot + ᶜp / Y.c.ρ))

    if diffuse_tke
        @. ᶜ∇²tke⁰ = wdivₕ(gradₕ(ᶜtke⁰))
    end

    # Sub-grid scale hyperdiffusion
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. ᶜ∇²uʲs.:($$j) =
                C123(wgradₕ(divₕ(p.precomputed.ᶜuʲs.:($$j)))) -
                C123(wcurlₕ(C123(curlₕ(p.precomputed.ᶜuʲs.:($$j)))))
            @. ᶜ∇²mseʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).mse))
        end
    end

    if do_dss
        NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
            # DSS on Grid scale quantities
            # Need to split the DSS computation here, because our DSS
            # operations do not accept Covariant123Vector types
            Spaces.weighted_dss!(
                ᶜ∇²u => buffer.ᶜ∇²u,
                ᶜ∇²specific_energy => buffer.ᶜ∇²specific_energy,
                (diffuse_tke ? (ᶜ∇²tke⁰ => buffer.ᶜ∇²tke⁰,) : ())...,
            )
            if turbconv_model isa PrognosticEDMFX
                # Need to split the DSS computation here, because our DSS
                # operations do not accept Covariant123Vector types
                for j in 1:n
                    @. ᶜ∇²uₕʲs.:($$j) = C12(ᶜ∇²uʲs.:($$j))
                    @. ᶜ∇²uᵥʲs.:($$j) = C3(ᶜ∇²uʲs.:($$j))
                end
                Spaces.weighted_dss!(
                    ᶜ∇²uₕʲs => buffer.ᶜ∇²uₕʲs,
                    ᶜ∇²uᵥʲs => buffer.ᶜ∇²uᵥʲs,
                    ᶜ∇²mseʲs => buffer.ᶜ∇²mseʲs,
                )
                for j in 1:n
                    @. ᶜ∇²uʲs.:($$j) =
                        C123(ᶜ∇²uₕʲs.:($$j)) + C123(ᶜ∇²uᵥʲs.:($$j))
                end
            end

        end
    end

    # re-use to store the curl-curl part
    @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.c.uₕ -= κ₄_vorticity * C12(ᶜ∇²u)
    @. Yₜ.f.u₃ -= κ₄_vorticity * ᶠwinterp(ᶜJ * Y.c.ρ, C3(ᶜ∇²u))

    @. Yₜ.c.ρe_tot -= κ₄_tracer * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²specific_energy))

    # Sub-grid scale hyperdiffusion continued
    if (turbconv_model isa PrognosticEDMFX) && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= κ₄_vorticity * wdivₕ(ᶜρa⁰ * gradₕ(ᶜ∇²tke⁰))
    end
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            if point_type <: Geometry.Abstract3DPoint
                # only need curl-curl part
                @. ᶜ∇²uᵥʲs.:($$j) = C3(wcurlₕ(C123(curlₕ(ᶜ∇²uʲs.:($$j)))))
                @. Yₜ.f.sgsʲs.:($$j).u₃ +=
                    κ₄_vorticity * ᶠwinterp(ᶜJ * Y.c.ρ, ᶜ∇²uᵥʲs.:($$j))
            end
            # Note: It is more correct to have ρa inside and outside the divergence
            @. Yₜ.c.sgsʲs.:($$j).mse -=
                κ₄_tracer * wdivₕ(gradₕ(ᶜ∇²mseʲs.:($$j)))
        end
    end

    if turbconv_model isa DiagnosticEDMFX && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= κ₄_vorticity * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²tke⁰))
    end
end

NVTX.@annotate function tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; κ₄_tracer) = hyperdiff
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜspecific) = p.precomputed
    (; do_dss) = p
    (; ᶜ∇²specific_tracers) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²q_totʲs) = p.hyperdiff
    end
    if do_dss
        buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    end

    for χ_name in propertynames(ᶜ∇²specific_tracers)
        @. ᶜ∇²specific_tracers.:($$χ_name) = wdivₕ(gradₕ(ᶜspecific.:($$χ_name)))
    end

    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            # Note: It is more correct to have ρa inside and outside the divergence
            @. ᶜ∇²q_totʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_tot))
        end
    end

    if do_dss
        NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
            if !isempty(propertynames(ᶜ∇²specific_tracers))
                Spaces.weighted_dss!(
                    ᶜ∇²specific_tracers => buffer.ᶜ∇²specific_tracers,
                )
            end
            if turbconv_model isa PrognosticEDMFX
                Spaces.weighted_dss!(ᶜ∇²q_totʲs => buffer.ᶜ∇²q_totʲs)
            end
        end
    end

    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    # TODO: Figure out why caching the duplicated tendencies in ᶜtemp_scalar
    # triggers allocations.
    for (ᶜρχₜ, ᶜ∇²χ, _) in matching_subfields(Yₜ.c, ᶜ∇²specific_tracers)
        @. ᶜρχₜ -= κ₄_tracer * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
        @. Yₜ.c.ρ -= κ₄_tracer * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
    end
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).ρa -=
                κ₄_tracer *
                wdivₕ(Y.c.sgsʲs.:($$j).ρa * gradₕ(ᶜ∇²q_totʲs.:($$j)))
            @. Yₜ.c.sgsʲs.:($$j).q_tot -=
                κ₄_tracer * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
        end
    end
    return nothing
end
