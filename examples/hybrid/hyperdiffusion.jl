hyperdiffusion_cache(
    Y;
    κ₄ = FT(0),
    divergence_damping_factor = FT(1),
    use_tempest_mode = false,
) = merge(
    (;
        ᶜχ = similar(Y.c, FT),
        ᶜχuₕ = similar(Y.c, Geometry.Covariant12Vector{FT}),
        κ₄,
        divergence_damping_factor,
        use_tempest_mode,
    ),
    use_tempest_mode ? (; ᶠχw_data = similar(Y.F, FT)) : NamedTuple(),
)

function hyperdiffusion_tendency!(Yₜ, Y, p, t)
    NVTX.isactive() && (
        profile_hyperdiffusion_tendency = NVTX.range_start(;
            message = "hyperdiffusion tendency",
            color = colorant"yellow",
        )
    )
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume ᶜp has been updated
    (; ghost_buffer, κ₄, divergence_damping_factor, use_tempest_mode) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    if use_tempest_mode
        @. ᶜχ = wdivₕ(gradₕ(ᶜρ)) # ᶜχρ
        Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(gradₕ(ᶜχ))

        if :ρθ in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ)) # ᶜχρθ
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρθ -= κ₄ * wdivₕ(gradₕ(ᶜχ))
        else
            error("use_tempest_mode must be false when not using ρθ")
        end

        (; ᶠχw_data) = p
        @. ᶠχw_data = wdivₕ(gradₕ(Y.f.w.components.data.:1))
        Spaces.weighted_dss!(ᶠχw_data, ghost_buffer.χ)
        @. Yₜ.f.w.components.data.:1 -= κ₄ * wdivₕ(gradₕ(ᶠχw_data))
    else
        if :ρθ in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ / ᶜρ)) # ᶜχθ
            NVTX.isactive() && (
                hyperdiffusion_dss_rhotheta = NVTX.range_start(;
                    message = "hyperdiffusion_dss_rhotheta",
                    color = colorant"green",
                )
            )
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            NVTX.isactive() && NVTX.range_end(hyperdiffusion_dss_rhotheta)
            @. Yₜ.c.ρθ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        elseif :ρe_tot in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe_tot + ᶜp) / ᶜρ)) # ᶜχe
            NVTX.isactive() && (
                hyperdiffusion_dss_rhoe = NVTX.range_start(;
                    message = "hyperdiffusion_dss_rhoe",
                    color = colorant"green",
                )
            )
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            NVTX.isactive() && NVTX.range_end(hyperdiffusion_dss_rhoe)
            @. Yₜ.c.ρe_tot -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        elseif :ρe_int in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe_int + ᶜp) / ᶜρ)) # ᶜχe_int
            NVTX.isactive() && (
                hyperdiffusion_dss_rhoeint = NVTX.range_start(;
                    message = "hyperdiffusion_dss_rhoeint",
                    color = colorant"green",
                )
            )
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            NVTX.isactive() && NVTX.range_end(hyperdiffusion_dss_rhoeint)
            @. Yₜ.c.ρe_int -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        end
    end

    if :ρq_tot in propertynames(Y.c)
        @. ᶜχ = wdivₕ(gradₕ(Y.c.ρq_tot / ᶜρ))
        NVTX.isactive() && (
            hyperdiffusion_dss_rhoqtot = NVTX.range_start(;
                message = "hyperdiffusion_dss_rhoqtot",
                color = colorant"green",
            )
        )
        Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
        NVTX.isactive() && NVTX.range_end(hyperdiffusion_dss_rhoqtot)
        @. Yₜ.c.ρq_tot -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
    end

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜχuₕ =
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.Covariant12Vector(
                wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜuₕ))),
            )
        NVTX.isactive() && (
            hyperdiffusion_dss_3dpoint = NVTX.range_start(;
                message = "hyperdiffusion_dss_3dpoint",
                color = colorant"green",
            )
        )
        Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
        NVTX.isactive() && NVTX.range_end(hyperdiffusion_dss_3dpoint)
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                Geometry.Covariant12Vector(
                    wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜχuₕ))),
                )
            )
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜχuₕ = Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜuₕ)))
        NVTX.isactive() && (
            hyperdiffusion_dss_2dpoint = NVTX.range_start(;
                message = "hyperdiffusion_dss_2dpoint",
                color = colorant"green",
            )
        )
        Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
        NVTX.isactive() && NVTX.range_end(hyperdiffusion_dss_2dpoint)
        @. Yₜ.c.uₕ -=
            κ₄ *
            divergence_damping_factor *
            Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    end
    NVTX.isactive() && NVTX.range_end(profile_hyperdiffusion_tendency)
    return nothing
end
