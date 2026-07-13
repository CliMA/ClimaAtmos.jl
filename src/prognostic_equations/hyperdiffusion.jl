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
    # Energy hyperdiffusion is applied via the dry-static-energy +
    # water-enthalpy decomposition (mirrors `vertical_diffusion_boundary_layer_tendency!`),
    # so we DSS ∇²s_d and ∇²q_{v,l,i} separately instead of a single ∇²h_tot.
    gs_quantities = (;
        ᶜ∇²u = similar(Y.c, C123{FT}),
        ᶜ∇²s_d = similar(Y.c, FT),
        ᶜ∇²q_vap = similar(Y.c, FT),
        ᶜ∇²q_liq = similar(Y.c, FT),
        ᶜ∇²q_ice = similar(Y.c, FT),
        ᶜ∇²specific_tracers = Base.materialize(ᶜspecific_gs_tracers(Y)),
    )

    # Sub-grid scale quantities. All SGS hyperdiffusion (scalars AND updraft
    # u₃) is applied by inheriting the grid-mean tendency (uniform hyperdiff
    # across the resolved cell), so we do not compute any per-subdomain ∇².
    maybe_ᶜ∇²tke =
        use_prognostic_tke(turbconv_model) ? (; ᶜ∇²tke = similar(Y.c, FT)) : (;)
    quantities = (; gs_quantities..., maybe_ᶜ∇²tke...)
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

    diffuse_tke = use_prognostic_tke(turbconv_model)
    (; ᶜp, ᶜu, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_vap, ᶜ∇²q_liq, ᶜ∇²q_ice) = p.hyperdiff

    # Grid scale hyperdiffusion
    @. ᶜ∇²u = C123(wgradₕ(divₕ(ᶜu))) - C123(wcurlₕ(C123(curlₕ(ᶜu))))

    # Energy split: diffuse dry static energy and each water species
    # separately, so the ∇⁴ operator never sees a lumped h_tot that mixes
    # dry-air enthalpy with water enthalpy. `apply_hyperdiffusion_tendency!`
    # reassembles them into the total enthalpy flux, mirroring the vertical
    # BL flux F_h = -ρ K_h [∇s_d + Σ_μ h_μ ∇q_μ].
    ᶜq_vap = @. lazy(TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    @. ᶜ∇²s_d = wdivₕ(gradₕ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)))
    @. ᶜ∇²q_vap = wdivₕ(gradₕ(ᶜq_vap))
    @. ᶜ∇²q_liq = wdivₕ(gradₕ(ᶜq_liq))
    @. ᶜ∇²q_ice = wdivₕ(gradₕ(ᶜq_ice))

    if diffuse_tke
        ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
        (; ᶜ∇²tke) = p.hyperdiff
        @. ᶜ∇²tke = wdivₕ(gradₕ(ᶜtke))
    end

    # No per-subdomain wind ∇² prep: SGS updraft u₃ inherits the grid-mean
    # hyperdiff tendency in `apply_hyperdiffusion_tendency!`.
end

# This requires dss to have been called on
# variables in dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜΦ) = p.core
    (; divergence_damping_factor) = hyperdiff
    (; ν₄_scalar, ν₄_vorticity) = ν₄(hyperdiff, Y)

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜρ = Y.c.ρ
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜT) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_vap, ᶜ∇²q_liq, ᶜ∇²q_ice) = p.hyperdiff
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end

    # re-use to store the curl-curl part
    ᶜ∇⁴u = @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.c.uₕ -= ν₄_vorticity * C12(ᶜ∇⁴u)
    @. Yₜ.f.u₃ -= ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴u))
    # Uniform hyperdiff in the grid box: each subdomain u₃ feels the same
    # momentum hyperdiff tendency as the grid mean u₃.
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴u))
        end
    end

    # Total enthalpy hyperdiffusion flux divergence, using the dry-static-energy
    # + water-enthalpy decomposition, matching the vertical BL flux
    # F_h = -ρ K_h [∇s_d + Σ_μ h_tot,μ ∇q_μ]. Materialize once because we reuse
    # it for the grid-mean ρe_tot tendency and for each subdomain's mse (which
    # inherits the same specific enthalpy tendency dh_gm/dt / ρ).
    #
    # Split into four separate wdivₕ calls (accumulating into the scratch)
    # rather than one fused broadcast. Linearity of divergence makes this
    # mathematically identical, but each expression's Broadcasted type is much
    # shallower, avoiding a GPUCompiler `typekeyvalue_hash` segfault on the
    # deeply-nested fused version.
    ᶜh_flux_div = p.scratch.ᶜtemp_scalar
    @. ᶜh_flux_div = wdivₕ(ᶜρ * gradₕ(ᶜ∇²s_d))
    @. ᶜh_flux_div += wdivₕ(
        ᶜρ * (TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) * gradₕ(ᶜ∇²q_vap),
    )
    @. ᶜh_flux_div += wdivₕ(
        ᶜρ * (TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) * gradₕ(ᶜ∇²q_liq),
    )
    @. ᶜh_flux_div += wdivₕ(
        ᶜρ * (TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) * gradₕ(ᶜ∇²q_ice),
    )
    @. Yₜ.c.ρe_tot -= ν₄_scalar * ᶜh_flux_div

    if (turbconv_model isa AbstractEDMF) && diffuse_tke
        @. Yₜ.c.ρtke -= ν₄_vorticity * wdivₕ(ᶜρ * gradₕ(ᶜ∇²tke))
    end
    # Same specific enthalpy tendency as grid mean: uniform hyperdiff in the
    # grid box means each subdomain feels the grid-mean specific tendency
    # dh_gm/dt = -ν₄/ρ · wdivₕ(ρ · [∇s_d + Σ h_μ ∇q_μ]) (its ∇² operators are
    # all grid-mean quantities, not per-subdomain).
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. Yₜ.c.sgsʲs.:($$j).mse -= ν₄_scalar / ᶜρ * ᶜh_flux_div
        end
    end
end

function dss_hyperdiffusion_tendency_pairs(p)
    (; hyperdiff, turbconv_model) = p.atmos
    buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_vap, ᶜ∇²q_liq, ᶜ∇²q_ice) = p.hyperdiff
    diffuse_tke = use_prognostic_tke(turbconv_model)
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end

    dynamics_pairs = (
        ᶜ∇²u => buffer.ᶜ∇²u,
        ᶜ∇²s_d => buffer.ᶜ∇²s_d,
        ᶜ∇²q_vap => buffer.ᶜ∇²q_vap,
        ᶜ∇²q_liq => buffer.ᶜ∇²q_liq,
        ᶜ∇²q_ice => buffer.ᶜ∇²q_ice,
        (diffuse_tke ? (ᶜ∇²tke => buffer.ᶜ∇²tke,) : ())...,
    )

    (; ᶜ∇²specific_tracers) = p.hyperdiff
    core_tracer_pairs =
        !isempty(propertynames(ᶜ∇²specific_tracers)) ?
        (ᶜ∇²specific_tracers => buffer.ᶜ∇²specific_tracers,) : ()
    return (dynamics_pairs..., core_tracer_pairs...)
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

    # SGS scalar tracers (SGS q_tot, auto-discovered SGS microphysics) do not
    # compute their own ∇²; they inherit the grid-mean specific tendency in
    # `apply_tracer_hyperdiffusion_tendency!`.
    return nothing
end

# This requires dss to have been called on
# variables in dss_hyperdiffusion_tendency_pairs
NVTX.@annotate function apply_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model, microphysics_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    # Rescale the hyperdiffusivity for precipitating species.
    (; ν₄_scalar) = ν₄(hyperdiff, Y)
    ν₄_scalar_microphysics = CAP.α_hyperdiff_tracer(p.params) * ν₄_scalar

    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜ∇²specific_tracers) = p.hyperdiff

    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    foreach_gs_tracer(Yₜ, ᶜ∇²specific_tracers) do ᶜρχₜ, ᶜ∇²χ, ρχ_name
        ν₄_scalar_for_χ =
            ρχ_name in (
                @name(ρq_lcl), @name(ρq_icl), @name(ρq_rai),
                @name(ρq_sno), @name(ρn_lcl), @name(ρn_rai)
            ) ?
            ν₄_scalar_microphysics : ν₄_scalar
        # Grid mean tendency on ρχ.
        @. ᶜρχₜ -= ν₄_scalar_for_χ * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))

        # Take into account the effect of total water diffusion on density.
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
        end

        # Uniform hyperdiff in the grid box: every subdomain feels the same
        # specific tendency as the grid mean. dχⱼ/dt = dχ_gm/dt.
        if turbconv_model isa PrognosticEDMFX
            χ_name = specific_tracer_name(ρχ_name)
            for j in 1:n
                if MatrixFields.has_field(Y.c.sgsʲs.:($j), χ_name)
                    ᶜχⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                    @. ᶜχⱼₜ -=
                        ν₄_scalar_for_χ / Y.c.ρ *
                        wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
                end
            end
            # ρa correction from q_tot hyperdiff, mirroring the grid-mean
            # ρ correction. Preserves updraft dry mass ρa·(1-q_tot).
            if ρχ_name == @name(ρq_tot)
                for j in 1:n
                    @. Yₜ.c.sgsʲs.:($$j).ρa -=
                        ν₄_scalar / Y.c.ρ *
                        Y.c.sgsʲs.:($$j).ρa /
                        (1 - Y.c.sgsʲs.:($$j).q_tot) *
                        wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
                end
            end
        end
    end
    return nothing
end
