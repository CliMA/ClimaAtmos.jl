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
    if do_dss(Y)
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

    (; κ₄, divergence_damping_factor) = hyperdiff
    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))
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
    if do_dss(Y)
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

    if do_dss(Y)
        NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
            for dss_op! in (
                Spaces.weighted_dss_start!,
                Spaces.weighted_dss_internal!,
                Spaces.weighted_dss_ghost!,
            )
                # DSS on Grid scale quantities
                # Need to split the DSS computation here, because our DSS
                # operations do not accept Covariant123Vector types
                dss_op!(ᶜ∇²u, buffer.ᶜ∇²u)
                dss_op!(ᶜ∇²specific_energy, buffer.ᶜ∇²specific_energy)
                if diffuse_tke
                    dss_op!(ᶜ∇²tke⁰, buffer.ᶜ∇²tke⁰)
                end
                if turbconv_model isa PrognosticEDMFX
                    # Need to split the DSS computation here, because our DSS
                    # operations do not accept Covariant123Vector types
                    for j in 1:n
                        @. ᶜ∇²uₕʲs.:($$j) = C12(ᶜ∇²uʲs.:($$j))
                        @. ᶜ∇²uᵥʲs.:($$j) = C3(ᶜ∇²uʲs.:($$j))
                    end
                    dss_op!(ᶜ∇²uₕʲs, buffer.ᶜ∇²uₕʲs)
                    dss_op!(ᶜ∇²uᵥʲs, buffer.ᶜ∇²uᵥʲs)
                    for j in 1:n
                        @. ᶜ∇²uʲs.:($$j) =
                            C123(ᶜ∇²uₕʲs.:($$j)) + C123(ᶜ∇²uᵥʲs.:($$j))
                    end
                    dss_op!(ᶜ∇²mseʲs, buffer.ᶜ∇²mseʲs)
                end
            end
        end
    end

    # re-use to store the curl-curl part
    @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.c.uₕ -= κ₄ * C12(ᶜ∇²u)
    @. Yₜ.f.u₃ -= κ₄ * ᶠwinterp(ᶜJ * Y.c.ρ, C3(ᶜ∇²u))

    @. Yₜ.c.ρe_tot -= κ₄ * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²specific_energy))

    # Sub-grid scale hyperdiffusion continued
    if (turbconv_model isa PrognosticEDMFX) && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= κ₄ * wdivₕ(ᶜρa⁰ * gradₕ(ᶜ∇²tke⁰))
    end
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            if point_type <: Geometry.Abstract3DPoint
                # only need curl-curl part
                @. ᶜ∇²uᵥʲs.:($$j) = C3(wcurlₕ(C123(curlₕ(ᶜ∇²uʲs.:($$j)))))
                @. Yₜ.f.sgsʲs.:($$j).u₃ +=
                    κ₄ * ᶠwinterp(ᶜJ * Y.c.ρ, ᶜ∇²uᵥʲs.:($$j))
            end
            # Note: It is more correct to have ρa inside and outside the divergence
            @. Yₜ.c.sgsʲs.:($$j).mse -= κ₄ * wdivₕ(gradₕ(ᶜ∇²mseʲs.:($$j)))
        end
    end

    if turbconv_model isa DiagnosticEDMFX && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= κ₄ * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²tke⁰))
    end
end

NVTX.@annotate function tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; κ₄) = hyperdiff
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜspecific) = p.precomputed
    (; ᶜ∇²specific_tracers) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²q_totʲs) = p.hyperdiff
    end
    if do_dss(Y)
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

    if do_dss(Y)
        NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
            for dss_op! in (
                Spaces.weighted_dss_start!,
                Spaces.weighted_dss_internal!,
                Spaces.weighted_dss_ghost!,
            )
                dss_op!(ᶜ∇²specific_tracers, buffer.ᶜ∇²specific_tracers)
                if turbconv_model isa PrognosticEDMFX
                    dss_op!(ᶜ∇²q_totʲs, buffer.ᶜ∇²q_totʲs)
                end
            end
        end
    end

    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    # TODO: Figure out why caching the duplicated tendencies in ᶜtemp_scalar
    # triggers allocations.
    for (ᶜρχₜ, ᶜ∇²χ, _) in matching_subfields(Yₜ.c, ᶜ∇²specific_tracers)
        @. ᶜρχₜ -= κ₄ * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
    end
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).ρa -=
                κ₄ * wdivₕ(Y.c.sgsʲs.:($$j).ρa * gradₕ(ᶜ∇²q_totʲs.:($$j)))
            @. Yₜ.c.sgsʲs.:($$j).q_tot -= κ₄ * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
        end
    end
    return nothing
end
