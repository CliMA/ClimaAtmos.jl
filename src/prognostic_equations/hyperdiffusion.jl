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

    # Grid scale quantities. Energy hyperdiffusion is applied via the
    # dry-static-energy + water-enthalpy decomposition (mirrors
    # `vertical_diffusion_boundary_layer_tendency!`), so we DSS ∇²s_d and
    # ∇²q_{v,l,i} separately instead of a single ∇²h_tot.
    ᶜ∇²u = similar(Y.c, C123{FT})
    gs_quantities = (;
        ᶜ∇²u = similar(Y.c, C123{FT}),
        ᶜ∇²s_d = similar(Y.c, FT),
        ᶜ∇²q_vap = similar(Y.c, FT),
        ᶜ∇²q_liq = similar(Y.c, FT),
        ᶜ∇²q_ice = similar(Y.c, FT),
        ᶜ∇²specific_tracers = Base.materialize(ᶜspecific_gs_tracers(Y)),
    )

    # Sub-grid scale quantities. SGS mse uses the same dry-static-energy +
    # water-enthalpy split as the grid mean, so we DSS ∇²s_dʲ and
    # ∇²q_{v,l,i}ʲ per subdomain instead of a single ∇²mseʲ.
    ᶜ∇²uʲs = turbconv_model isa PrognosticEDMFX ? similar(Y.c, NTuple{n, C123{FT}}) : (;)
    # Single reusable scratch field for auto-discovered SGS tracers
    sgs_tracer_hyperdiff =
        turbconv_model isa PrognosticEDMFX && !isempty(sgs_tracer_names(Y)) ?
        (; ᶜ∇²sgs_tracerʲs = similar(Y.c, NTuple{n, FT})) : (;)
    sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶜ∇²uₕʲs = similar(Y.c, NTuple{n, C12{FT}}),
            ᶜ∇²uᵥʲs = similar(Y.c, NTuple{n, C3{FT}}),
            ᶜ∇²s_dʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_vapʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_liqʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_iceʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_totʲs = similar(Y.c, NTuple{n, FT}),
            sgs_tracer_hyperdiff...,
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
    return (; quantities..., ᶜ∇²u, ᶜ∇²uʲs)
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
    (; ᶜp, ᶜu, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_vap, ᶜ∇²q_liq, ᶜ∇²q_ice) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs) = p.hyperdiff
        (; ᶜ∇²s_dʲs, ᶜ∇²q_vapʲs, ᶜ∇²q_liqʲs, ᶜ∇²q_iceʲs) = p.hyperdiff
        (; ᶜuʲs, ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
    end

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

    # Sub-grid scale hyperdiffusion. SGS mse uses the same dry-static-energy +
    # water-enthalpy split as the grid mean (reassembled with subdomain
    # thermodynamic quantities in `apply_hyperdiffusion_tendency!`).
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. ᶜ∇²uʲs.:($$j) =
                C123(wgradₕ(divₕ(ᶜuʲs.:($$j)))) - C123(wcurlₕ(C123(curlₕ(ᶜuʲs.:($$j)))))
            @. ᶜ∇²uₕʲs.:($$j) = C12(ᶜ∇²uʲs.:($$j))
            @. ᶜ∇²uᵥʲs.:($$j) = C3(ᶜ∇²uʲs.:($$j))
            ᶜq_vapʲ = @. lazy(
                TD.vapor_specific_humidity(
                    ᶜq_tot_nonnegʲs.:($$j),
                    ᶜq_liqʲs.:($$j),
                    ᶜq_iceʲs.:($$j),
                ),
            )
            @. ᶜ∇²s_dʲs.:($$j) =
                wdivₕ(gradₕ(TD.dry_static_energy(thermo_params, ᶜTʲs.:($$j), ᶜΦ)))
            @. ᶜ∇²q_vapʲs.:($$j) = wdivₕ(gradₕ(ᶜq_vapʲ))
            @. ᶜ∇²q_liqʲs.:($$j) = wdivₕ(gradₕ(ᶜq_liqʲs.:($$j)))
            @. ᶜ∇²q_iceʲs.:($$j) = wdivₕ(gradₕ(ᶜq_iceʲs.:($$j)))
        end
    end
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
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; ᶜT) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_vap, ᶜ∇²q_liq, ᶜ∇²q_ice) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        ᶜρa⁰ = @. lazy(ρa⁰(ᶜρ, Y.c.sgsʲs, turbconv_model))
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs) = p.hyperdiff
        (; ᶜ∇²s_dʲs, ᶜ∇²q_vapʲs, ᶜ∇²q_liqʲs, ᶜ∇²q_iceʲs) = p.hyperdiff
        (; ᶜTʲs) = p.precomputed
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end

    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            @. ᶜ∇²uʲs.:($$j) = C123(ᶜ∇²uₕʲs.:($$j)) + C123(ᶜ∇²uᵥʲs.:($$j))
        end
    end

    # re-use to store the curl-curl part
    ᶜ∇⁴u = @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.c.uₕ -= ν₄_vorticity * C12(ᶜ∇⁴u)
    @. Yₜ.f.u₃ -= ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴u))

    # Total enthalpy hyperdiffusion flux, using the dry-static-energy +
    # water-enthalpy decomposition, matching the vertical BL flux
    # F_h = -ρ K_h [∇s_d + Σ_μ h_tot,μ ∇q_μ]. This avoids applying ∇⁴ to a
    # lumped `h_tot` that would spuriously diffuse dry-air enthalpy along with
    # water enthalpy.
    #
    # Split into four separate wdivₕ calls (accumulating into a scratch field)
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
    # Sub-grid scale hyperdiffusion continued
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n
            if point_type <: Geometry.Abstract3DPoint
                # only need curl-curl part
                ᶜ∇⁴uᵥʲ = @. ᶜ∇²uᵥʲs.:($$j) = C3(wcurlₕ(C123(curlₕ(ᶜ∇²uʲs.:($$j)))))
                @. Yₜ.f.sgsʲs.:($$j).u₃ += ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, ᶜ∇⁴uᵥʲ)
            end
            # SGS mse hyperdiff, using the same dry-static-energy +
            # water-enthalpy split as the grid-mean energy flux above but
            # with subdomain thermodynamics (Tʲ, q_μʲ). No density weighting
            # — hyperdiff on the specific quantity mseʲ [J/kg/s], symmetric
            # with the unweighted SGS q_totʲ and tracer hyperdiff. Split
            # into four separate wdivₕ calls to avoid a GPUCompiler
            # `typekeyvalue_hash` segfault on the deeply-nested fused
            # broadcast.
            @. ᶜh_flux_div = wdivₕ(gradₕ(ᶜ∇²s_dʲs.:($$j)))
            @. ᶜh_flux_div += wdivₕ(
                (TD.enthalpy_vapor(thermo_params, ᶜTʲs.:($$j)) + ᶜΦ) *
                gradₕ(ᶜ∇²q_vapʲs.:($$j)),
            )
            @. ᶜh_flux_div += wdivₕ(
                (TD.enthalpy_liquid(thermo_params, ᶜTʲs.:($$j)) + ᶜΦ) *
                gradₕ(ᶜ∇²q_liqʲs.:($$j)),
            )
            @. ᶜh_flux_div += wdivₕ(
                (TD.enthalpy_ice(thermo_params, ᶜTʲs.:($$j)) + ᶜΦ) *
                gradₕ(ᶜ∇²q_iceʲs.:($$j)),
            )
            @. Yₜ.c.sgsʲs.:($$j).mse -= ν₄_scalar * ᶜh_flux_div
        end
    end
end

function dss_hyperdiffusion_tendency_pairs(p)
    (; hyperdiff, turbconv_model) = p.atmos
    buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_vap, ᶜ∇²q_liq, ᶜ∇²q_ice) = p.hyperdiff
    diffuse_tke = use_prognostic_tke(turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs) = p.hyperdiff
        (; ᶜ∇²s_dʲs, ᶜ∇²q_vapʲs, ᶜ∇²q_liqʲs, ᶜ∇²q_iceʲs) = p.hyperdiff
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end

    core_dynamics_pairs = (
        ᶜ∇²u => buffer.ᶜ∇²u,
        ᶜ∇²s_d => buffer.ᶜ∇²s_d,
        ᶜ∇²q_vap => buffer.ᶜ∇²q_vap,
        ᶜ∇²q_liq => buffer.ᶜ∇²q_liq,
        ᶜ∇²q_ice => buffer.ᶜ∇²q_ice,
        (diffuse_tke ? (ᶜ∇²tke => buffer.ᶜ∇²tke,) : ())...,
    )
    tc_dynamics_pairs =
        turbconv_model isa PrognosticEDMFX ?
        (
            ᶜ∇²uₕʲs => buffer.ᶜ∇²uₕʲs,
            ᶜ∇²uᵥʲs => buffer.ᶜ∇²uᵥʲs,
            ᶜ∇²s_dʲs => buffer.ᶜ∇²s_dʲs,
            ᶜ∇²q_vapʲs => buffer.ᶜ∇²q_vapʲs,
            ᶜ∇²q_liqʲs => buffer.ᶜ∇²q_liqʲs,
            ᶜ∇²q_iceʲs => buffer.ᶜ∇²q_iceʲs,
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
        # Auto-discovered SGS tracers: prep → DSS → apply per tracer,
        # reusing a single scratch field.
        if !isempty(sgs_tracer_names(Y))
            _microphysics_names = (
                @name(q_lcl), @name(q_icl), @name(q_rai),
                @name(q_sno), @name(n_lcl), @name(n_rai),
            )
            (; ᶜ∇²sgs_tracerʲs) = p.hyperdiff
            for χ_name in sgs_tracer_names(Y)
                for j in 1:n
                    # Prep: compute ∇²χ
                    ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                    # Note: It is more correct to have ρa inside and outside the divergence
                    @. ᶜ∇²sgs_tracerʲs.:($$j) = wdivₕ(gradₕ(ᶜχʲ))
                end
                # DSS
                if do_dss(axes(Y.c))
                    Spaces.weighted_dss!(
                        ᶜ∇²sgs_tracerʲs =>
                            p.hyperdiff.hyperdiffusion_ghost_buffer.ᶜ∇²sgs_tracerʲs,
                    )
                end
                # Apply: ∇⁴χ tendency
                ν = χ_name in _microphysics_names ?
                    ν₄_scalar_microphysics : ν₄_scalar
                for j in 1:n
                    ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                    @. ᶜχʲₜ -= ν * wdivₕ(gradₕ(ᶜ∇²sgs_tracerʲs.:($$j)))
                end
            end
        end
    end
    return nothing
end
