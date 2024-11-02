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
    quadrature_style =
        Spaces.quadrature_style(Spaces.horizontal_space(axes(Y.c)))
    do_dss = quadrature_style isa Quadratures.GLL
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

# This should prep variables that we will dss in
# dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    (; ghost_buffer) = p.scratch
    (; ᶜp, ᶜspecific) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs, ᶜ∇²mseʲs) = p.hyperdiff
    end

    # Grid scale hyperdiffusion
    ᶜdivₕ_u = @. p.scratch.ᶜtemp_scalar = wdivₕ(p.precomputed.ᶜu)
    Spaces.weighted_dss!(ᶜdivₕ_u => ghost_buffer.ᶜtemp_scalar)
    ᶜcurlₕ_u = @. p.scratch.ᶜtemp_C123 = C123(wcurlₕ(p.precomputed.ᶜu))
    Spaces.weighted_dss!(ᶜcurlₕ_u => ghost_buffer.ᶜtemp_C123)
    @. ᶜ∇²u = C123(wgradₕ(ᶜdivₕ_u)) - C123(wcurlₕ(ᶜcurlₕ_u))

    ᶜgradₕ_specific_energy =
        @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜspecific.e_tot + ᶜp / Y.c.ρ))
    Spaces.weighted_dss!(ᶜgradₕ_specific_energy => ghost_buffer.ᶜtemp_C12)
    @. ᶜ∇²specific_energy = wdivₕ(ᶜgradₕ_specific_energy)

    if diffuse_tke
        (; ᶜtke⁰) = p.precomputed
        (; ᶜ∇²tke⁰) = p.hyperdiff
        ᶜgradₕ_tke⁰ = @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜtke⁰))
        Spaces.weighted_dss!(ᶜgradₕ_tke⁰ => ghost_buffer.ᶜtemp_C12)
        @. ᶜ∇²tke⁰ = wdivₕ(ᶜgradₕ_tke⁰)
    end

    # Sub-grid scale hyperdiffusion
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            ᶜdivₕ_uʲ =
                @. p.scratch.ᶜtemp_scalar = wdivₕ(p.precomputed.ᶜuʲs.:($$j))
            Spaces.weighted_dss!(ᶜdivₕ_uʲ => ghost_buffer.ᶜtemp_scalar)
            ᶜcurlₕ_uʲ = @. p.scratch.ᶜtemp_C123 =
                C123(wcurlₕ(p.precomputed.ᶜuʲs.:($$j)))
            Spaces.weighted_dss!(ᶜcurlₕ_uʲ => ghost_buffer.ᶜtemp_C123)
            @. ᶜ∇²uʲs.:($$j) = C123(wgradₕ(ᶜdivₕ_uʲ)) - C123(wcurlₕ(ᶜcurlₕ_uʲ))

            ᶜgradₕ_mseʲ =
                @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(Y.c.sgsʲs.:($$j).mse))
            Spaces.weighted_dss!(ᶜgradₕ_mseʲ => ghost_buffer.ᶜtemp_C12)
            @. ᶜ∇²mseʲs.:($$j) = wdivₕ(ᶜgradₕ_mseʲ)

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
    (; ghost_buffer) = p.scratch
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

    ᶜdivₕ_∇²u = @. p.scratch.ᶜtemp_scalar = wdivₕ(ᶜ∇²u)
    Spaces.weighted_dss!(ᶜdivₕ_∇²u => ghost_buffer.ᶜtemp_scalar)
    ᶜcurlₕ_∇²u = @. p.scratch.ᶜtemp_C123 = C123(wcurlₕ(ᶜ∇²u))
    Spaces.weighted_dss!(ᶜcurlₕ_∇²u => ghost_buffer.ᶜtemp_C123)
    @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(ᶜdivₕ_∇²u)) -
        C123(wcurlₕ(ᶜcurlₕ_∇²u))
    @. Yₜ.c.uₕ -= ν₄_vorticity * C12(ᶜ∇²u)
    @. Yₜ.f.u₃ -= ν₄_vorticity * ᶠwinterp(ᶜJ * Y.c.ρ, C3(ᶜ∇²u))

    ᶜgradₕ_∇²specific_energy =
        @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜ∇²specific_energy))
    Spaces.weighted_dss!(ᶜgradₕ_∇²specific_energy => ghost_buffer.ᶜtemp_C12)
    @. Yₜ.c.ρe_tot -= ν₄_scalar * wdivₕ(Y.c.ρ * ᶜgradₕ_∇²specific_energy)

    # Sub-grid scale hyperdiffusion continued
    if (turbconv_model isa PrognosticEDMFX) && diffuse_tke
        ᶜgradₕ_∇²tke⁰ = @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜ∇²tke⁰))
        Spaces.weighted_dss!(ᶜgradₕ_∇²tke⁰ => ghost_buffer.ᶜtemp_C12)
        @. Yₜ.c.sgs⁰.ρatke -= ν₄_vorticity * wdivₕ(ᶜρa⁰ * ᶜgradₕ_∇²tke⁰)
    end
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            if point_type <: Geometry.Abstract3DPoint
                ᶜcurlₕ_∇²uʲ =
                    @. p.scratch.ᶜtemp_C123 = C123(wcurlₕ(ᶜ∇²uʲs.:($$j)))
                Spaces.weighted_dss!(ᶜcurlₕ_∇²uʲ => ghost_buffer.ᶜtemp_C123)
                # only need curl-curl part
                @. ᶜ∇²uᵥʲs.:($$j) = C3(wcurlₕ(ᶜcurlₕ_∇²uʲ))
                @. Yₜ.f.sgsʲs.:($$j).u₃ +=
                    ν₄_vorticity * ᶠwinterp(ᶜJ * Y.c.ρ, ᶜ∇²uᵥʲs.:($$j))
            end
            ᶜgradₕ_∇²mseʲ =
                @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜ∇²mseʲs.:($$j)))
            Spaces.weighted_dss!(ᶜgradₕ_∇²mseʲ => ghost_buffer.ᶜtemp_C12)
            # Note: It is more correct to have ρa inside and outside the divergence
            @. Yₜ.c.sgsʲs.:($$j).mse -= ν₄_scalar * wdivₕ(ᶜgradₕ_∇²mseʲ)
        end
    end

    if turbconv_model isa DiagnosticEDMFX && diffuse_tke
        ᶜgradₕ_∇²tke⁰ = @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜ∇²tke⁰))
        Spaces.weighted_dss!(ᶜgradₕ_∇²tke⁰ => ghost_buffer.ᶜtemp_C12)
        @. Yₜ.c.sgs⁰.ρatke -= ν₄_vorticity * wdivₕ(Y.c.ρ * ᶜgradₕ_∇²tke⁰)
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
    tracer_pairs = (core_tracer_pairs..., tc_tracer_pairs...)
    return (dynamics_pairs..., tracer_pairs...)
end

# This should prep variables that we will dss in
# dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function prep_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; ghost_buffer) = p.scratch
    (; ᶜspecific) = p.precomputed
    (; ᶜ∇²specific_tracers) = p.hyperdiff

    for χ_name in propertynames(ᶜ∇²specific_tracers)
        ᶜgradₕ_χ = @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜspecific.:($$χ_name)))
        Spaces.weighted_dss!(ᶜgradₕ_χ => ghost_buffer.ᶜtemp_C12)
        @. ᶜ∇²specific_tracers.:($$χ_name) = wdivₕ(ᶜgradₕ_χ)
    end

    if turbconv_model isa PrognosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        (; ᶜ∇²q_totʲs) = p.hyperdiff
        for j in 1:n
            ᶜgradₕ_q_totʲ =
                @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(Y.c.sgsʲs.:($$j).q_tot))
            Spaces.weighted_dss!(ᶜgradₕ_q_totʲ => ghost_buffer.ᶜtemp_C12)
            # Note: It is more correct to have ρa inside and outside the divergence
            @. ᶜ∇²q_totʲs.:($$j) = wdivₕ(ᶜgradₕ_q_totʲ)
        end
    end
    return nothing
end

# This requires dss to have been called on
# variables in dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function apply_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; ν₄_scalar_coeff) = hyperdiff
    h_space = Spaces.horizontal_space(axes(Y.c))
    h_length_scale = Spaces.node_horizontal_length_scale(h_space) # mean nodal distance
    ν₄_scalar = ν₄_scalar_coeff * h_length_scale^3
    n = n_mass_flux_subdomains(turbconv_model)

    (; ghost_buffer) = p.scratch
    (; ᶜ∇²specific_tracers) = p.hyperdiff

    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    # TODO: Figure out why caching the duplicated tendencies in ᶜtemp_scalar
    # triggers allocations.
    for (ᶜρχₜ, ᶜ∇²χ, χ_name) in matching_subfields(Yₜ.c, ᶜ∇²specific_tracers)
        ν₄_scalar = ifelse(χ_name in (:q_rai, :q_sno), 0 * ν₄_scalar, ν₄_scalar)
        ᶜgradₕ_∇²χ = @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜ∇²χ))
        Spaces.weighted_dss!(ᶜgradₕ_∇²χ => ghost_buffer.ᶜtemp_C12)
        @. ᶜρχₜ -= ν₄_scalar * wdivₕ(Y.c.ρ * ᶜgradₕ_∇²χ)
        if !(χ_name in (:q_rai, :q_sno))
            @. Yₜ.c.ρ -= ν₄_scalar * wdivₕ(Y.c.ρ * ᶜgradₕ_∇²χ)
        end
    end
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²q_totʲs) = p.hyperdiff
        for j in 1:n
            ᶜgradₕ_∇²q_totʲ =
                @. p.scratch.ᶜtemp_C12 = C12(wgradₕ(ᶜ∇²q_totʲs.:($$j)))
            Spaces.weighted_dss!(ᶜgradₕ_∇²q_totʲ => ghost_buffer.ᶜtemp_C12)
            @. Yₜ.c.sgsʲs.:($$j).ρa -=
                ν₄_scalar * wdivₕ(Y.c.sgsʲs.:($$j).ρa * ᶜgradₕ_∇²q_totʲ)
            @. Yₜ.c.sgsʲs.:($$j).q_tot -= ν₄_scalar * wdivₕ(ᶜgradₕ_∇²q_totʲ)
        end
    end
    return nothing
end
