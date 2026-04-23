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
(2018, https://doi.org/10.1029/2017MS001257). The scalar coefficient is computed
as `ν₄_scalar = ν₄_vorticity / prandtl_number`, where `ν₄_vorticity = ν₄_vorticity_coeff * h^3`.
"""
function ν₄(hyperdiff, Y)
    h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c)))
    # Vorticity coefficient unchanged
    ν₄_vorticity = hyperdiff.ν₄_vorticity_coeff * h^3
    # Scalar coefficient = vorticity coefficient / Prandtl number
    ν₄_scalar = ν₄_vorticity / hyperdiff.prandtl_number
    return (; ν₄_scalar, ν₄_vorticity)
end

function hyperdiffusion_cache(Y, atmos)
    (; hyperdiff, turbconv_model, microphysics_model) = atmos
    isnothing(hyperdiff) && return (;)  # No hyperdiffiusion
    hyperdiffusion_cache(Y, hyperdiff, turbconv_model, microphysics_model)
end

function hyperdiffusion_cache(
    Y, ::Hyperdiffusion, turbconv_model, microphysics_model,
)
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)

    # Grid scale quantities
    ᶜ∇²u = similar(Y.c, C123{FT})
    gs_quantities = (;
        ᶜ∇²u = similar(Y.c, C123{FT}),
        ᶜ∇²specific_energy = similar(Y.c, FT),
        ᶜ∇²specific_tracers = Base.materialize(ᶜspecific_gs_tracers(Y)),
    )

    # Sub-grid scale quantities

    moisture_sgs_quantities =
        microphysics_model isa NonEquilibriumMicrophysics1M ?
        (;
            ᶜ∇²q_lclʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_iclʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_raiʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_snoʲs = similar(Y.c, NTuple{n, FT}),
        ) :
        microphysics_model isa NonEquilibriumMicrophysics2M ?
        (;
            ᶜ∇²q_lclʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_iclʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_raiʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_snoʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²n_lclʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²n_raiʲs = similar(Y.c, NTuple{n, FT}),
        ) : (;)
    sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶠ∇²u₃ʲs = similar(Y.f, NTuple{n, FT}),
            ᶜ∇²aʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²mseʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_totʲs = similar(Y.c, NTuple{n, FT}),
            moisture_sgs_quantities...,
        ) : (;)
    maybe_ᶜ∇²tke =
        use_prognostic_tke(turbconv_model) ? (; ᶜ∇²tke = similar(Y.c, FT)) : (;)
    sgs_quantities = (; sgs_quantities..., maybe_ᶜ∇²tke...)
    quantities = (; gs_quantities..., sgs_quantities...)
    if do_dss(axes(Y.c))
        quantities = (;
            quantities...,
            hyperdiffusion_ghost_buffer = map(Spaces.create_dss_buffer, quantities),
        )
    end
    return (; quantities..., ᶜ∇²u)
end

# This should prep variables that we will dss in
# dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    (; params) = p
    (; ᶜΦ) = p.core
    thermo_params = CAP.thermodynamics_params(params)

    isnothing(hyperdiff) && return nothing

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    (; ᶜp, ᶜu) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶠ∇²u₃ʲs, ᶜ∇²aʲs, ᶜ∇²mseʲs) = p.hyperdiff
        (; ᶜρʲs) = p.precomputed
    end

    # Grid scale hyperdiffusion
    @. ᶜ∇²u = C123(wgradₕ(divₕ(ᶜu))) - C123(wcurlₕ(C123(curlₕ(ᶜu))))

    ᶜh_ref = @. lazy(h_dr(thermo_params, ᶜp, ᶜΦ))

    @. ᶜ∇²specific_energy = wdivₕ(gradₕ(specific(Y.c.ρe_tot, Y.c.ρ) + ᶜp / Y.c.ρ - ᶜh_ref))

    if diffuse_tke
        ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
        (; ᶜ∇²tke) = p.hyperdiff
        @. ᶜ∇²tke = wdivₕ(gradₕ(ᶜtke))
    end

    # Sub-grid scale hyperdiffusion
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. ᶠ∇²u₃ʲs.:($$j) = wdivₕ(gradₕ(Y.f.sgsʲs.:($$j).u₃.components.data.:1))
            @. ᶜ∇²aʲs.:($$j) = wdivₕ(gradₕ(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))))
            @. ᶜ∇²mseʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).mse))
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

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜρ = Y.c.ρ
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶠ∇²u₃ʲs, ᶜ∇²aʲs, ᶜ∇²mseʲs) = p.hyperdiff
        (; ᶜρʲs) = p.precomputed
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end



    # re-use to store the curl-curl part
    ᶜ∇⁴u = @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.c.uₕ -= ν₄_vorticity * C12(ᶜ∇⁴u)
    @. Yₜ.f.u₃ -= ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴u))

    @. Yₜ.c.ρe_tot -= ν₄_scalar * wdivₕ(ᶜρ * gradₕ(ᶜ∇²specific_energy))

    if (turbconv_model isa AbstractEDMF) && diffuse_tke
        @. Yₜ.c.ρtke -= ν₄_vorticity * wdivₕ(ᶜρ * gradₕ(ᶜ∇²tke))
    end
    # Sub-grid scale hyperdiffusion continued
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            # Scalar bi-Laplacian of u₃ computed directly on faces (no F→C→F interpolation)
            @. Yₜ.f.sgsʲs.:($$j).u₃.components.data.:1 -=
                CAP.α_hyperdiff_tracer(p.params) * ν₄_vorticity *
                wdivₕ(gradₕ(ᶠ∇²u₃ʲs.:($$j)))
            # @. Yₜ.c.sgsʲs.:($$j).ρa -= CAP.α_hyperdiff_tracer(p.params) * ν₄_scalar * wdivₕ(ᶜρʲs.:($$j) * gradₕ(ᶜ∇²aʲs.:($$j)))
            # Note: It is more correct to have ρa inside and outside the divergence
            @. Yₜ.c.sgsʲs.:($$j).mse -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²mseʲs.:($$j)))
        end
    end
end

function dss_hyperdiffusion_tendency_pairs(p)
    (; hyperdiff, turbconv_model) = p.atmos
    buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    diffuse_tke = use_prognostic_tke(turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        (; ᶠ∇²u₃ʲs, ᶜ∇²aʲs, ᶜ∇²mseʲs) = p.hyperdiff
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end

    core_dynamics_pairs = (
        ᶜ∇²u => buffer.ᶜ∇²u,
        ᶜ∇²specific_energy => buffer.ᶜ∇²specific_energy,
        (diffuse_tke ? (ᶜ∇²tke => buffer.ᶜ∇²tke,) : ())...,
    )
    tc_dynamics_pairs =
        turbconv_model isa PrognosticEDMFX ?
        (
            ᶠ∇²u₃ʲs => buffer.ᶠ∇²u₃ʲs,
            ᶜ∇²aʲs => buffer.ᶜ∇²aʲs,
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
        p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M ?
        (
            p.hyperdiff.ᶜ∇²q_lclʲs => buffer.ᶜ∇²q_lclʲs,
            p.hyperdiff.ᶜ∇²q_iclʲs => buffer.ᶜ∇²q_iclʲs,
            p.hyperdiff.ᶜ∇²q_raiʲs => buffer.ᶜ∇²q_raiʲs,
            p.hyperdiff.ᶜ∇²q_snoʲs => buffer.ᶜ∇²q_snoʲs,
        ) :
        turbconv_model isa PrognosticEDMFX &&
        p.atmos.microphysics_model isa NonEquilibriumMicrophysics2M ?
        (
            p.hyperdiff.ᶜ∇²q_lclʲs => buffer.ᶜ∇²q_lclʲs,
            p.hyperdiff.ᶜ∇²q_iclʲs => buffer.ᶜ∇²q_iclʲs,
            p.hyperdiff.ᶜ∇²q_raiʲs => buffer.ᶜ∇²q_raiʲs,
            p.hyperdiff.ᶜ∇²q_snoʲs => buffer.ᶜ∇²q_snoʲs,
            p.hyperdiff.ᶜ∇²n_lclʲs => buffer.ᶜ∇²n_lclʲs,
            p.hyperdiff.ᶜ∇²n_raiʲs => buffer.ᶜ∇²n_raiʲs,
        ) : ()
    tracer_pairs = (core_tracer_pairs..., tc_tracer_pairs..., tc_moisture_pairs...)
    return (dynamics_pairs..., tracer_pairs...)
end

# This should prep variables that we will dss in
# dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function prep_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model, microphysics_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; ᶜ∇²specific_tracers) = p.hyperdiff

    # TODO: Fix RecursiveApply bug in gradₕ to fuse this operation.
    # ᶜ∇²specific_tracers .= wdivₕ.(gradₕ.(ᶜspecific_gs_tracers(Y)))
    foreach_gs_tracer(Y, ᶜ∇²specific_tracers) do ᶜρχ, ᶜ∇²χ, _
        @. ᶜ∇²χ = wdivₕ(gradₕ(specific(ᶜρχ, Y.c.ρ)))
    end

    if turbconv_model isa PrognosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        (; ᶜ∇²q_totʲs) = p.hyperdiff
        for j in 1:n
            # Note: It is more correct to have ρa inside and outside the divergence
            @. ᶜ∇²q_totʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_tot))
        end
        if microphysics_model isa NonEquilibriumMicrophysics1M
            (; ᶜ∇²q_lclʲs, ᶜ∇²q_iclʲs, ᶜ∇²q_raiʲs, ᶜ∇²q_snoʲs) = p.hyperdiff
            for j in 1:n
                # Note: It is more correct to have ρa inside and outside the divergence
                @. ᶜ∇²q_lclʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_lcl))
                @. ᶜ∇²q_iclʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_icl))
                @. ᶜ∇²q_raiʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_rai))
                @. ᶜ∇²q_snoʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_sno))
            end
        elseif microphysics_model isa NonEquilibriumMicrophysics2M
            (; ᶜ∇²q_lclʲs, ᶜ∇²q_iclʲs, ᶜ∇²q_raiʲs, ᶜ∇²q_snoʲs, ᶜ∇²n_lclʲs, ᶜ∇²n_raiʲs) =
                p.hyperdiff
            for j in 1:n
                # Note: It is more correct to have ρa inside and outside the divergence
                @. ᶜ∇²q_lclʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_lcl))
                @. ᶜ∇²q_iclʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_icl))
                @. ᶜ∇²q_raiʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_rai))
                @. ᶜ∇²q_snoʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).q_sno))
                @. ᶜ∇²n_lclʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).n_lcl))
                @. ᶜ∇²n_raiʲs.:($$j) = wdivₕ(gradₕ(Y.c.sgsʲs.:($$j).n_rai))
            end
        end
    end
    return nothing
end

# This requires dss to have been called on
# variables in dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function apply_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model, microphysics_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    # Rescale the hyperdiffusivity for precipitating species.
    (; ν₄_scalar) = ν₄(hyperdiff, Y)
    ν₄_scalar_for_precip = CAP.α_hyperdiff_tracer(p.params) * ν₄_scalar

    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜ∇²specific_tracers) = p.hyperdiff

    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    foreach_gs_tracer(Yₜ, ᶜ∇²specific_tracers) do ᶜρχₜ, ᶜ∇²χ, ρχ_name
        ν₄_scalar_for_χ =
            ρχ_name in (@name(ρq_rai), @name(ρq_sno), @name(ρn_rai)) ?
            ν₄_scalar_for_precip : ν₄_scalar
        @. ᶜρχₜ -= ν₄_scalar_for_χ * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))

        # Take into account the effect of total water diffusion on density.
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
        end
    end
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²q_totʲs) = p.hyperdiff
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).q_tot -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
            @. Yₜ.c.sgsʲs.:($$j).ρa -=
                ν₄_scalar * Y.c.sgsʲs.:($$j).ρa / (1 - Y.c.sgsʲs.:($$j).q_tot) *
                wdivₕ(gradₕ(ᶜ∇²q_totʲs.:($$j)))
        end
        if microphysics_model isa NonEquilibriumMicrophysics1M
            (; ᶜ∇²q_lclʲs, ᶜ∇²q_iclʲs, ᶜ∇²q_raiʲs, ᶜ∇²q_snoʲs) = p.hyperdiff
            for j in 1:n
                @. Yₜ.c.sgsʲs.:($$j).q_lcl -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_lclʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_icl -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_iclʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_rai -=
                    ν₄_scalar_for_precip * wdivₕ(gradₕ(ᶜ∇²q_raiʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_sno -=
                    ν₄_scalar_for_precip * wdivₕ(gradₕ(ᶜ∇²q_snoʲs.:($$j)))
            end
        elseif microphysics_model isa NonEquilibriumMicrophysics2M
            (; ᶜ∇²q_lclʲs, ᶜ∇²q_iclʲs, ᶜ∇²q_raiʲs, ᶜ∇²q_snoʲs, ᶜ∇²n_lclʲs, ᶜ∇²n_raiʲs) =
                p.hyperdiff
            for j in 1:n
                @. Yₜ.c.sgsʲs.:($$j).q_lcl -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_lclʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_icl -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_iclʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).n_lcl -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²n_lclʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_rai -=
                    ν₄_scalar_for_precip * wdivₕ(gradₕ(ᶜ∇²q_raiʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).q_sno -=
                    ν₄_scalar_for_precip * wdivₕ(gradₕ(ᶜ∇²q_snoʲs.:($$j)))
                @. Yₜ.c.sgsʲs.:($$j).n_rai -=
                    ν₄_scalar_for_precip * wdivₕ(gradₕ(ᶜ∇²n_raiʲs.:($$j)))
            end
        end
    end
    return nothing
end
