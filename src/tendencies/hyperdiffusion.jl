#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

function hyperdiffusion_cache(Y, atmos, do_dss)
    isnothing(atmos.hyperdiff) && return (;)
    FT = Spaces.undertype(axes(Y.c))
    moist_kwargs = if (:ρq_tot in propertynames(Y.c))
        (; ᶜχρq_tot = similar(Y.c, FT))
    else
        NamedTuple()
    end
    quantities =
        (; ᶜχ = similar(Y.c, FT), moist_kwargs..., ᶜχuₕ = similar(Y.c, C12{FT}))
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
    isnothing(p.atmos.hyperdiff) && return nothing
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ

    (; do_dss, ᶜp, ᶜχ, ᶜχuₕ) = p # assume ᶜp has been updated
    (; κ₄, divergence_damping_factor) = p.atmos.hyperdiff
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)
    is_2d_pt = point_type <: Geometry.Abstract2DPoint
    is_3d_pt = !is_2d_pt
    if do_dss
        buffer = p.hyperdiffusion_ghost_buffer
    end

    ᶜρs = :ρe_tot in propertynames(Y.c) ? Y.c.ρe_tot : Y.c.ρθ
    ᵗρs = :ρe_tot in propertynames(Y.c) ? Yₜ.c.ρe_tot : Yₜ.c.ρθ

    (:ρθ in propertynames(Y.c)) && (@. ᶜχ = wdivₕ(gradₕ(ᶜρs / ᶜρ)))
    !(:ρθ in propertynames(Y.c)) && (@. ᶜχ = wdivₕ(gradₕ((ᶜρs + ᶜp) / ᶜρ)))

    is_3d_pt && (@. ᶜχuₕ = wgradₕ(divₕ(ᶜuₕ)) - C12(wcurlₕ(C3(curlₕ(ᶜuₕ)))))
    is_2d_pt && (@. ᶜχuₕ = C12(wgradₕ(divₕ(ᶜuₕ))))

    if do_dss
        NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
            Spaces.weighted_dss_start2!(ᶜχ, buffer.ᶜχ)
            Spaces.weighted_dss_start2!(ᶜχuₕ, buffer.ᶜχuₕ)

            Spaces.weighted_dss_internal2!(ᶜχ, buffer.ᶜχ)
            Spaces.weighted_dss_internal2!(ᶜχuₕ, buffer.ᶜχuₕ)

            Spaces.weighted_dss_ghost2!(ᶜχ, buffer.ᶜχ)
            Spaces.weighted_dss_ghost2!(ᶜχuₕ, buffer.ᶜχuₕ)
        end
    end

    @. ᵗρs -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
    if is_3d_pt
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                C12(wcurlₕ(C3(curlₕ(ᶜχuₕ))))
            )
    elseif is_2d_pt
        @. Yₜ.c.uₕ -= κ₄ * divergence_damping_factor * C12(wgradₕ(divₕ(ᶜχuₕ)))
    end
    return nothing
end

function tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    isnothing(p.atmos.hyperdiff) && return nothing
    !(:ρq_tot in propertynames(Y.c)) && return nothing
    ᶜρ = Y.c.ρ
    (; κ₄) = p.atmos.hyperdiff
    if p.do_dss
        buffer = p.hyperdiffusion_ghost_buffer
    end

    (; ᶜχρq_tot) = p
    @. ᶜχρq_tot = wdivₕ(gradₕ(Y.c.ρq_tot / ᶜρ))

    if p.do_dss
        NVTX.@range "dss_hyperdiffusion_tendency" color = colorant"green" begin
            Spaces.weighted_dss_start2!(ᶜχρq_tot, buffer.ᶜχρq_tot)
            Spaces.weighted_dss_internal2!(ᶜχρq_tot, buffer.ᶜχρq_tot)
            Spaces.weighted_dss_ghost2!(ᶜχρq_tot, buffer.ᶜχρq_tot)
        end
    end

    # TODO: Since we are not applying the limiter to density, the mass
    # redistributed by hyperdiffusion will not be conserved by the limiter. Is
    # this a significant problem?

    @. Yₜ.c.ρq_tot -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
    @. Yₜ.c.ρ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
    return nothing
end
