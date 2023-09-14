#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

function hyperdiffusion_cache(Y, atmos, do_dss)
    isnothing(atmos.hyperdiff) && return (;)
    FT = eltype(Y)
    n = n_mass_flux_subdomains(atmos.turbconv_model)

    # Grid scale quantities
    ᶜ∇²u = similar(Y.c, C123{FT})
    gs_quantities = (;
        ᶜ∇²uₕ = similar(Y.c, C12{FT}),
        ᶜ∇²uᵥ = similar(Y.c, C3{FT}),
        ᶜ∇²specific_energy = similar(Y.c, FT),
        ᶜ∇²specific_tracers = remove_energy_var.(specific_gs.(Y.c)),
    )

    # Sub-grid scale quantities
    ᶜ∇²uʲs =
        atmos.turbconv_model isa EDMFX ? similar(Y.c, NTuple{n, C123{FT}}) : (;)
    sgs_quantities =
        atmos.turbconv_model isa EDMFX ?
        (;
            ᶜ∇²tke⁰ = similar(Y.c, FT),
            ᶜ∇²uₕʲs = similar(Y.c, NTuple{n, C12{FT}}),
            ᶜ∇²uᵥʲs = similar(Y.c, NTuple{n, C3{FT}}),
            ᶜ∇²specific_energyʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²specific_tracersʲs = remove_energy_var.(
                specific_sgsʲs.(Y.c, atmos.turbconv_model)
            ),
        ) :
        atmos.turbconv_model isa DiagnosticEDMFX ?
        (; ᶜ∇²tke⁰ = similar(Y.c, FT)) : (;)
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

    (; κ₄, divergence_damping_factor) = hyperdiff
    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; do_dss, ᶜp, ᶜspecific, ᶜ∇²uₕ, ᶜ∇²uᵥ, ᶜ∇²u, ᶜ∇²specific_energy) = p
    if turbconv_model isa EDMFX
        (;
            ᶜρa⁰,
            ᶜspecific⁰,
            ᶜρʲs,
            ᶜspecificʲs,
            ᶜ∇²tke⁰,
            ᶜ∇²uₕʲs,
            ᶜ∇²uᵥʲs,
            ᶜ∇²uʲs,
            ᶜ∇²specific_energyʲs,
        ) = p
    end
    if turbconv_model isa DiagnosticEDMFX
        (; ᶜtke⁰, ᶜ∇²tke⁰) = p
    end
    if do_dss
        buffer = p.hyperdiffusion_ghost_buffer
    end

    # Grid scale hyperdiffusion
    @. ᶜ∇²u = C123(wgradₕ(divₕ(p.ᶜu))) - C123(wcurlₕ(C123(curlₕ(p.ᶜu))))

    if :θ in propertynames(ᶜspecific)
        @. ᶜ∇²specific_energy = wdivₕ(gradₕ(ᶜspecific.θ))
    elseif :e_tot in propertynames(ᶜspecific)
        @. ᶜ∇²specific_energy = wdivₕ(gradₕ(ᶜspecific.e_tot + ᶜp / Y.c.ρ))
    end
    if turbconv_model isa EDMFX && diffuse_tke
        @. ᶜ∇²tke⁰ = wdivₕ(gradₕ(ᶜspecific⁰.tke))
    end
    if turbconv_model isa DiagnosticEDMFX && diffuse_tke
        @. ᶜ∇²tke⁰ = wdivₕ(gradₕ(ᶜtke⁰))
    end

    # Sub-grid scale hyperdiffusion
    if turbconv_model isa EDMFX
        for j in 1:n
            @. ᶜ∇²uʲs.:($$j) =
                C123(wgradₕ(divₕ(p.ᶜuʲs.:($$j)))) -
                C123(wcurlₕ(C123(curlₕ(p.ᶜuʲs.:($$j)))))

            if :θ in propertynames(ᶜspecificʲs.:($j))
                @. ᶜ∇²specific_energyʲs.:($$j) =
                    wdivₕ(gradₕ(ᶜspecificʲs.:($$j).θ))
            elseif :e_tot in propertynames(ᶜspecificʲs.:($j))
                @. ᶜ∇²specific_energyʲs.:($$j) =
                    wdivₕ(gradₕ(ᶜspecificʲs.:($$j).e_tot + ᶜp / ᶜρʲs.:($$j)))
            end
        end
    end

    if do_dss
        NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
            for dss_op! in (
                Spaces.weighted_dss_start!,
                Spaces.weighted_dss_internal!,
                Spaces.weighted_dss_ghost!,
            )
                # DSS on Grid scale quantities
                # Need to split the DSS computation here, because our DSS 
                # operations do not accept Covariant123Vector types
                @. ᶜ∇²uₕ = C12(ᶜ∇²u)
                @. ᶜ∇²uᵥ = C3(ᶜ∇²u)
                dss_op!(ᶜ∇²uₕ, buffer.ᶜ∇²uₕ)
                dss_op!(ᶜ∇²uᵥ, buffer.ᶜ∇²uᵥ)
                @. ᶜ∇²u = C123(ᶜ∇²uₕ) + C123(ᶜ∇²uᵥ)
                dss_op!(ᶜ∇²specific_energy, buffer.ᶜ∇²specific_energy)
                if turbconv_model isa EDMFX && diffuse_tke
                    dss_op!(ᶜ∇²tke⁰, buffer.ᶜ∇²tke⁰)
                end
                if turbconv_model isa EDMFX
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
                    dss_op!(ᶜ∇²specific_energyʲs, buffer.ᶜ∇²specific_energyʲs)
                end
                if turbconv_model isa DiagnosticEDMFX && diffuse_tke
                    dss_op!(ᶜ∇²tke⁰, buffer.ᶜ∇²tke⁰)
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

    ᶜρ_energyₜ = :θ in propertynames(ᶜspecific) ? Yₜ.c.ρθ : Yₜ.c.ρe_tot
    @. ᶜρ_energyₜ -= κ₄ * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²specific_energy))

    # Sub-grid scale hyperdiffusion continued
    if turbconv_model isa EDMFX && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= κ₄ * wdivₕ(ᶜρa⁰ * gradₕ(ᶜ∇²tke⁰))
    end
    if turbconv_model isa EDMFX
        for j in 1:n
            if point_type <: Geometry.Abstract3DPoint
                # only need curl-curl part
                @. ᶜ∇²uᵥʲs.:($$j) = C3(wcurlₕ(C123(curlₕ(ᶜ∇²uʲs.:($$j)))))
                @. Yₜ.f.sgsʲs.:($$j).u₃ +=
                    κ₄ * ᶠwinterp(ᶜJ * Y.c.ρ, ᶜ∇²uᵥʲs.:($$j))
            end
            ᶜρa_energyʲₜ =
                :θ in propertynames(ᶜspecificʲs.:($j)) ? Yₜ.c.sgsʲs.:($j).ρaθ :
                Yₜ.c.sgsʲs.:($j).ρae_tot
            @. ᶜρa_energyʲₜ -=
                κ₄ *
                wdivₕ(Y.c.sgsʲs.:($$j).ρa * gradₕ(ᶜ∇²specific_energyʲs.:($$j)))
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

    (; do_dss, ᶜspecific, ᶜ∇²specific_tracers) = p
    if turbconv_model isa EDMFX
        (; ᶜspecificʲs, ᶜ∇²specific_tracersʲs, ᶜp) = p
    end
    if do_dss
        buffer = p.hyperdiffusion_ghost_buffer
    end

    for χ_name in propertynames(ᶜ∇²specific_tracers)
        @. ᶜ∇²specific_tracers.:($$χ_name) = wdivₕ(gradₕ(ᶜspecific.:($$χ_name)))
    end
    if turbconv_model isa EDMFX
        for j in 1:n
            for χ_name in propertynames(ᶜ∇²specific_tracersʲs.:($j))
                @. ᶜ∇²specific_tracersʲs.:($$j).:($$χ_name) =
                    wdivₕ(gradₕ(ᶜspecificʲs.:($$j).:($$χ_name)))
            end
        end
    end

    if do_dss
        NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
            for dss_op! in (
                Spaces.weighted_dss_start!,
                Spaces.weighted_dss_internal!,
                Spaces.weighted_dss_ghost!,
            )
                dss_op!(ᶜ∇²specific_tracers, buffer.ᶜ∇²specific_tracers)
                if turbconv_model isa EDMFX
                    dss_op!(ᶜ∇²specific_tracersʲs, buffer.ᶜ∇²specific_tracersʲs)
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
    if turbconv_model isa EDMFX
        for j in 1:n
            for (ᶜρaχʲₜ, ᶜ∇²χʲ, _) in matching_subfields(
                Yₜ.c.sgsʲs.:($j),
                ᶜ∇²specific_tracersʲs.:($j),
            )
                @. ᶜρaχʲₜ -= κ₄ * wdivₕ(Y.c.sgsʲs.:($$j).ρa * gradₕ(ᶜ∇²χʲ))
                @. Yₜ.c.sgsʲs.:($$j).ρa -=
                    κ₄ * wdivₕ(Y.c.sgsʲs.:($$j).ρa * gradₕ(ᶜ∇²χʲ))
            end
        end
    end
    return nothing
end
