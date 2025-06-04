#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

hyperdiffusion_cache(Y, atmos) = hyperdiffusion_cache(
    Y,
    atmos.hyperdiff,
    atmos.turbconv_model,
    atmos.moisture_model,
    atmos.precip_model,
)

# No hyperdiffiusion
hyperdiffusion_cache(Y, hyperdiff::Nothing, _, _, _) = (;)

function hyperdiffusion_cache(
    Y,
    hyperdiff::ClimaHyperdiffusion,
    turbconv_model,
    moisture_model,
    precip_model,
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
        ᶜ∇²specific_tracers = remove_energy_var.(specific_gs.(Y.c)),
    )

    # Sub-grid scale quantities
    ᶜ∇²uʲs =
        turbconv_model isa PrognosticEDMFX ? similar(Y.c, NTuple{n, C123{FT}}) :
        (;)
    moisture_sgs_quantities =
        turbconv_model isa PrognosticEDMFX &&
        moisture_model isa NonEquilMoistModel &&
        precip_model isa Microphysics1Moment ?
        (;
            ᶜ∇²q_liqʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_iceʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_raiʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_snoʲs = similar(Y.c, NTuple{n, FT}),
        ) : (;)
    sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶜ∇²uₕʲs = similar(Y.c, NTuple{n, C12{FT}}),
            ᶜ∇²uᵥʲs = similar(Y.c, NTuple{n, C3{FT}}),
            ᶜ∇²mseʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_totʲs = similar(Y.c, NTuple{n, FT}),
            moisture_sgs_quantities...,
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
    (; ᶜp, ᶜspecific) = p.precomputed
    (; ᶜh_ref) = p.core
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs, ᶜ∇²mseʲs) = p.hyperdiff
    end

    # Grid scale hyperdiffusion
    @. ᶜ∇²u =
        C123(wgradₕ(divₕ(p.precomputed.ᶜu))) -
        C123(wcurlₕ(C123(curlₕ(p.precomputed.ᶜu))))

    @. ᶜ∇²specific_energy = wdivₕ(gradₕ((ᶜspecific.e_tot + ᶜp / Y.c.ρ) - ᶜh_ref))

    if diffuse_tke
        (; ᶜtke⁰) = p.precomputed
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

    (; ν₄_vorticity_coeff, ν₄_scalar_coeff, divergence_damping_factor) =
        hyperdiff

    h_space = Spaces.horizontal_space(axes(Y.c))
    h_length_scale = Spaces.node_horizontal_length_scale(h_space) # mean nodal distance

    ν₄_scalar = ν₄_scalar_coeff * h_length_scale^3
    ν₄_vorticity = ν₄_vorticity_coeff * h_length_scale^3

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; ᶜp, ᶜspecific) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜρa⁰) = p.precomputed
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs, ᶜ∇²mseʲs) = p.hyperdiff
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜtke⁰) = p.precomputed
        (; ᶜ∇²tke⁰) = p.hyperdiff
    end

    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. ᶜ∇²uʲs.:($$j) = C123(ᶜ∇²uₕʲs.:($$j)) + C123(ᶜ∇²uᵥʲs.:($$j))
        end
    end

    # re-use to store the curl-curl part
    @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.c.uₕ -= ν₄_vorticity * C12(ᶜ∇²u)
    @. Yₜ.f.u₃ -= ν₄_vorticity * ᶠwinterp(ᶜJ * Y.c.ρ, C3(ᶜ∇²u))

    @. Yₜ.c.ρe_tot -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²specific_energy))

    # Sub-grid scale hyperdiffusion continued
    if (turbconv_model isa PrognosticEDMFX) && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= ν₄_vorticity * wdivₕ(ᶜρa⁰ * gradₕ(ᶜ∇²tke⁰))
    end
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            if point_type <: Geometry.Abstract3DPoint
                # only need curl-curl part
                @. ᶜ∇²uᵥʲs.:($$j) = C3(wcurlₕ(C123(curlₕ(ᶜ∇²uʲs.:($$j)))))
                @. Yₜ.f.sgsʲs.:($$j).u₃ +=
                    ν₄_vorticity * ᶠwinterp(ᶜJ * Y.c.ρ, ᶜ∇²uᵥʲs.:($$j))
            end
            # Note: It is more correct to have ρa inside and outside the divergence
            @. Yₜ.c.sgsʲs.:($$j).mse -=
                ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²mseʲs.:($$j)))
        end
    end

    if turbconv_model isa DiagnosticEDMFX && diffuse_tke
        @. Yₜ.c.sgs⁰.ρatke -= ν₄_vorticity * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²tke⁰))
    end
end

function dss_hyperdiffusion_tendency_pairs(p)
    (; hyperdiff, turbconv_model) = p.atmos
    buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    diffuse_tke = use_prognostic_tke(turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²mseʲs) = p.hyperdiff
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke⁰) = p.hyperdiff
    end

    core_dynamics_pairs = (
        ᶜ∇²u => buffer.ᶜ∇²u,
        ᶜ∇²specific_energy => buffer.ᶜ∇²specific_energy,
        (diffuse_tke ? (ᶜ∇²tke⁰ => buffer.ᶜ∇²tke⁰,) : ())...,
    )
    tc_dynamics_pairs =
        turbconv_model isa PrognosticEDMFX ?
        (
            ᶜ∇²uₕʲs => buffer.ᶜ∇²uₕʲs,
            ᶜ∇²uᵥʲs => buffer.ᶜ∇²uᵥʲs,
            ᶜ∇²mseʲs => buffer.ᶜ∇²mseʲs,
        ) : ()
    dynamics_pairs = (core_dynamics_pairs..., tc_dynamics_pairs...)

    (; ᶜ∇²specific_tracers) = p.hyperdiff
    core_tracer_pairs =
        !isempty(propertynames(ᶜ∇²specific_tracers)) ?
        (ᶜ∇²specific_tracers => buffer.ᶜ∇²specific_tracers,) : ()
    tc_tracer_pairs =
        turbconv_model isa PrognosticEDMFX ?
        (p.hyperdiff.ᶜ∇²q_totʲs => buffer.ᶜ∇²q_totʲs,) : ()
    tc_moisture_pairs =
        turbconv_model isa PrognosticEDMFX &&
        p.atmos.moisture_model isa NonEquilMoistModel &&
        p.atmos.precip_model isa Microphysics1Moment ?
        (
            p.hyperdiff.ᶜ∇²q_liqʲs => buffer.ᶜ∇²q_liqʲs,
            p.hyperdiff.ᶜ∇²q_iceʲs => buffer.ᶜ∇²q_iceʲs,
            p.hyperdiff.ᶜ∇²q_raiʲs => buffer.ᶜ∇²q_raiʲs,
            p.hyperdiff.ᶜ∇²q_snoʲs => buffer.ᶜ∇²q_snoʲs,
        ) : ()
    tracer_pairs =
        (core_tracer_pairs..., tc_tracer_pairs..., tc_moisture_pairs...)
    return (dynamics_pairs..., tracer_pairs...)
end

# This should prep variables that we will dss in
# dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function prep_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; ᶜspecific) = p.precomputed
    (; ᶜ∇²specific_tracers) = p.hyperdiff

    for χ_name in propertynames(ᶜ∇²specific_tracers)
        @. ᶜ∇²specific_tracers.:($$χ_name) = wdivₕ(gradₕ(ᶜspecific.:($$χ_name)))
    end

    if turbconv_model isa PrognosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        (; ᶜ∇²q_totʲs) = p.hyperdiff
        for j in 1:n
            # Note: It is more correct to have ρa inside and outside the divergence
            @. ᶜ∇²q_totʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_tot))
        end
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.precip_model isa Microphysics1Moment
            (; ᶜ∇²q_liqʲs, ᶜ∇²q_iceʲs, ᶜ∇²q_raiʲs, ᶜ∇²q_snoʲs) = p.hyperdiff
            for j in 1:n
                # Note: It is more correct to have ρa inside and outside the divergence
                @. ᶜ∇²q_liqʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_liq))
                @. ᶜ∇²q_iceʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_ice))
                @. ᶜ∇²q_raiʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_rai))
                @. ᶜ∇²q_snoʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_sno))
            end
        end
    end
    return nothing
end

# This requires dss to have been called on
# variables in dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function apply_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    α_hyperdiff_tracer = CAP.α_hyperdiff_tracer(p.params)
    (; ν₄_scalar_coeff) = hyperdiff
    h_space = Spaces.horizontal_space(axes(Y.c))
    h_length_scale = Spaces.node_horizontal_length_scale(h_space) # mean nodal distance
    ν₄_scalar = ν₄_scalar_coeff * h_length_scale^3
    n = n_mass_flux_subdomains(turbconv_model)

    (; ᶜ∇²specific_tracers) = p.hyperdiff

    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    # TODO: Figure out why caching the duplicated tendencies in ᶜtemp_scalar
    # triggers allocations.
    for (ᶜρχₜ, ᶜ∇²χ, χ_name) in matching_subfields(Yₜ.c, ᶜ∇²specific_tracers)
        ν₄_scalar = ifelse(
            χ_name in (:q_rai, :q_sno),
            α_hyperdiff_tracer * ν₄_scalar,
            ν₄_scalar,
        )
        @. ᶜρχₜ -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))

        # Exclude contributions from hyperdiffusion of condensate, 
        # precipitating species from mass tendency. 
        if χ_name == :q_tot
            @. Yₜ.c.ρ -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
        end
    end
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²q_totʲs) = p.hyperdiff
        if p.atmos.moisture_model isa NonEquilMoistModel &&
           p.atmos.precip_model isa Microphysics1Moment
            (; ᶜ∇²q_liqʲs, ᶜ∇²q_iceʲs, ᶜ∇²q_raiʲs, ᶜ∇²q_snoʲs) = p.hyperdiff
        end
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).ρa -=
                ν₄_scalar *
            @. Yₜ.c.sgsʲs.:($$j).q_tot -=
                ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
            if p.atmos.moisture_model isa NonEquilMoistModel &&
               p.atmos.precip_model isa Microphysics1Moment
                @. Yₜ.c.sgsʲs.:($$j).q_liq -=
                    ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_liqʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_ice -=
                    ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_iceʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_rai -=
                    α_hyperdiff_tracer *
                    ν₄_scalar *
                    wdivₕ(gradₕ(ᶜ∇²q_raiʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_sno -=
                    α_hyperdiff_tracer *
                    ν₄_scalar *
                    wdivₕ(gradₕ(ᶜ∇²q_snoʲs.:($$j)))
            end
        end
    end
    return nothing
end
