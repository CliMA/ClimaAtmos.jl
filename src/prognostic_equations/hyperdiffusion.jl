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
    gs_quantities = (;
        ᶜ∇²uₕ = similar(Y.c, C12{FT}),
        ᶜ∇²specific_energy = similar(Y.c, FT),
        ᶜ∇²specific_tracers = remove_energy_var.(specific_gs.(Y.c)),
    )
    sgs_quantities =
        atmos.turbconv_model isa EDMFX ?
        (;
            ᶜ∇²tke⁰ = similar(Y.c, FT),
            ᶠ∇²wʲs = similar(Y.f, NTuple{n, FT}),
            ᶜ∇²specific_energyʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²specific_tracersʲs = remove_energy_var.(
                specific_sgsʲs.(Y.c, atmos.turbconv_model)
            ),
        ) : (;)
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
    return quantities
end

function hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; κ₄, divergence_damping_factor) = hyperdiff
    n = n_mass_flux_subdomains(turbconv_model)
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; do_dss, ᶜp, ᶜspecific, ᶜ∇²uₕ, ᶜ∇²specific_energy) = p
    if turbconv_model isa EDMFX
        (;
            ᶜρa⁰,
            ᶜspecific⁰,
            ᶜρʲs,
            ᶜspecificʲs,
            ᶜ∇²tke⁰,
            ᶠ∇²wʲs,
            ᶜ∇²specific_energyʲs,
        ) = p
    end
    if do_dss
        buffer = p.hyperdiffusion_ghost_buffer
    end

    @. ᶜ∇²uₕ = C12(wgradₕ(divₕ(Y.c.uₕ)))
    # Without the C12(), the right-hand side would be a C1 or C2 in 2D space.
    if point_type <: Geometry.Abstract3DPoint
        # TODO: Are these vector conversions correct when there is topography?
        @. ᶜ∇²uₕ -= C12(wcurlₕ(C3(curlₕ(Y.c.uₕ))))
    end

    if :θ in propertynames(ᶜspecific)
        @. ᶜ∇²specific_energy = wdivₕ(gradₕ(ᶜspecific.θ))
    elseif :e_tot in propertynames(ᶜspecific)
        @. ᶜ∇²specific_energy = wdivₕ(gradₕ(ᶜspecific.e_tot + ᶜp / Y.c.ρ))
    end
    if turbconv_model isa EDMFX
        @. ᶜ∇²tke⁰ = wdivₕ(gradₕ(ᶜspecific⁰.tke))
    end
    if turbconv_model isa EDMFX
        for j in 1:n
            @. ᶠ∇²wʲs.:($$j) =
                wdivₕ(gradₕ(Y.f.sgsʲs.:($$j).u₃.components.data.:1))
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
                dss_op!(ᶜ∇²uₕ, buffer.ᶜ∇²uₕ)
                dss_op!(ᶜ∇²specific_energy, buffer.ᶜ∇²specific_energy)
                if turbconv_model isa EDMFX
                    dss_op!(ᶜ∇²tke⁰, buffer.ᶜ∇²tke⁰)
                    dss_op!(ᶠ∇²wʲs, buffer.ᶠ∇²wʲs)
                    dss_op!(ᶜ∇²specific_energyʲs, buffer.ᶜ∇²specific_energyʲs)
                end
            end
        end
    end

    @. Yₜ.c.uₕ -= κ₄ * divergence_damping_factor * C12(wgradₕ(divₕ(ᶜ∇²uₕ)))
    # Without the C12(), the right-hand side would be a C1 or C2 in 2D space.
    if point_type <: Geometry.Abstract3DPoint
        # TODO: Are these vector conversions correct when there is topography?
        @. Yₜ.c.uₕ += κ₄ * C12(wcurlₕ(C3(curlₕ(ᶜ∇²uₕ))))
    end

    ᶜρ_energyₜ = :θ in propertynames(ᶜspecific) ? Yₜ.c.ρθ : Yₜ.c.ρe_tot
    @. ᶜρ_energyₜ -= κ₄ * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²specific_energy))
    if turbconv_model isa EDMFX
        @. Yₜ.c.sgs⁰.ρatke -= κ₄ * wdivₕ(ᶜρa⁰ * gradₕ(ᶜ∇²tke⁰))
    end
    if turbconv_model isa EDMFX
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃.components.data.:1 -=
                κ₄ * wdivₕ(gradₕ(ᶠ∇²wʲs.:($$j)))
            ᶜρa_energyʲₜ =
                :θ in propertynames(ᶜspecificʲs.:($j)) ? Yₜ.c.sgsʲs.:($j).ρaθ :
                Yₜ.c.sgsʲs.:($j).ρae_tot
            @. ᶜρa_energyʲₜ -=
                κ₄ *
                wdivₕ(Y.c.sgsʲs.:($$j).ρa * gradₕ(ᶜ∇²specific_energyʲs.:($$j)))
        end
    end
end

function tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
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
end
