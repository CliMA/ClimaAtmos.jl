#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Quadratures as Quadratures

# Real-axis stability bound `C` of the integrator on the explicit biharmonic, in
# `dt * ρ(ν₄ ∇⁴) ≤ C`: forward Euler for the once-per-step coefficient reduction,
# the default ARS343 explicit tableau for the warning. See `docs/src/equations.md`.
const HYPERDIFFUSION_FORWARD_EULER_STABILITY = 2
const HYPERDIFFUSION_ARS343_STABILITY = 2.7853

"""
    hyperdiffusion_grid_scale_factor(degree)

Compute the grid-scale factor `β` of the horizontal biharmonic on a uniform,
degree-`degree` spectral element, defined by `ρ(∇⁴) = (β / h)⁴` with `h` the mean
nodal distance. Tabulated from the spectral radius of the assembled scalar
operator `(wdivₕ ∘ gradₕ)²`; the generating code is in
`test/prognostic_equations/hyperdiffusion_grid_factor.jl`.
"""
function hyperdiffusion_grid_scale_factor(degree)
    degree == 2 && return 3.4641
    degree == 3 && return 4.0637
    degree == 4 && return 4.7873
    degree == 5 && return 5.5997
    degree == 6 && return 6.4531
    degree == 7 && return 7.3250
    error("hyperdiffusion stability limit is not tabulated for polynomial \
        degree $degree.")
end

"""
    hyperdiffusion_grid_factor(space)

Compute the biharmonic grid factor `β` of the horizontal `space`, defined by
`ρ(∇⁴) = (β / h)⁴` with `h` the mean nodal distance. It is the uniform-grid factor
[`hyperdiffusion_grid_scale_factor`](@ref) for the quadrature degree, scaled by the
grid metric non-uniformity; see `docs/src/equations.md`.
"""
function hyperdiffusion_grid_factor(space)
    h = Spaces.node_horizontal_length_scale(space)
    degree = Quadratures.polynomial_degree(Spaces.quadrature_style(space))
    # Horizontal contravariant-metric components g¹¹, g¹², g²² (padded 3×3
    # column-major indices 1, 4, 5).
    g = Fields.local_geometry_field(space).gⁱʲ.components.data
    corner = g.:1 .+ g.:5 .+ 2 .* abs.(g.:4)
    uniform = 2 * (2 / (degree * h))^2
    metric_factor = sqrt(maximum(corner) / uniform)
    return oftype(h, hyperdiffusion_grid_scale_factor(degree) * metric_factor)
end

"""
    hyperdiffusion_dt_limit(hyperdiff, h, grid_factor, stability)

Compute the explicit stability limit, in seconds, of the hyperdiffusion at mean
nodal distance `h` and biharmonic `grid_factor`, using the largest of the divergent
and scalar coefficient factors `F = max(divergence_damping_factor, 1 / prandtl_number)`
and the integrator real-axis `stability` bound.
"""
function hyperdiffusion_dt_limit(hyperdiff, h, grid_factor, stability)
    F = max(hyperdiff.divergence_damping_factor, inv(hyperdiff.prandtl_number))
    return stability * h / (F * grid_factor^4 * hyperdiff.ν₄_vorticity_coeff)
end

"""
    ν₄(hyperdiff, Y, dt, grid_factor)

A `NamedTuple` of the hyperdiffusivity `ν₄_scalar` and the hyperviscosity
`ν₄_vorticity`. These quantities are assumed to scale with `h^3`, where `h` is
the mean nodal distance, following the empirical results of Lauritzen et al.
(2018, https://doi.org/10.1029/2017MS001257). The scalar coefficient is computed
as `ν₄_scalar = ν₄_vorticity / prandtl_number`, where `ν₄_vorticity = ν₄_vorticity_coeff * h^3`.

When `hyperdiff.dt_safety_factor > 0`, `ν₄_vorticity` is reduced so that the
hyperdiffusion is explicitly stable for `dt_safety_factor * dt`; see
[`hyperdiffusion_dt_limit`](@ref). `grid_factor` is
[`hyperdiffusion_grid_factor`](@ref) for the horizontal space.
"""
function ν₄(hyperdiff, Y, dt, grid_factor)
    h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c)))
    ν₄_vorticity = hyperdiff.ν₄_vorticity_coeff * h^3
    S = hyperdiff.dt_safety_factor
    if S > 0
        limit = hyperdiffusion_dt_limit(
            hyperdiff, h, grid_factor, HYPERDIFFUSION_FORWARD_EULER_STABILITY,
        )
        ν₄_vorticity = min(ν₄_vorticity, ν₄_vorticity * limit / (S * float(dt)))
    end
    ν₄_scalar = ν₄_vorticity / hyperdiff.prandtl_number
    return (; ν₄_scalar, ν₄_vorticity)
end

"""
    warn_if_hyperdiffusion_over_dt_limit(hyperdiff, Y, dt)

Warn when the hyperdiffusion tendency is integrated at a `dt` above its explicit
stability limit while no limit is applied (`dt_safety_factor == 0`). The limit uses
the ARS343 explicit-tableau stability bound, since the plain scheme integrates the
hyperdiffusion with the explicit IMEX tableau. See [`hyperdiffusion_dt_limit`](@ref).
"""
function warn_if_hyperdiffusion_over_dt_limit(hyperdiff, Y, dt)
    hyperdiff isa Hyperdiffusion || return nothing
    hyperdiff.dt_safety_factor > 0 && return nothing
    space = Spaces.horizontal_space(axes(Y.c))
    h = Spaces.node_horizontal_length_scale(space)
    grid_factor = hyperdiffusion_grid_factor(space)
    limit = hyperdiffusion_dt_limit(
        hyperdiff, h, grid_factor, HYPERDIFFUSION_ARS343_STABILITY,
    )
    float(dt) > limit && @warn "dt = $(float(dt)) s exceeds the explicit \
        stability limit ($limit s) of the hyperdiffusion coefficient. \
        Set hyperdiffusion_dt_safety_factor (recommended 2) or reduce \
        vorticity_hyperdiffusion_coefficient."
    return nothing
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
            ᶜ∇²mseʲs = similar(Y.c, NTuple{n, FT}),
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
    grid_factor = hyperdiffusion_grid_factor(Spaces.horizontal_space(axes(Y.c)))
    return (; quantities..., ᶜ∇²u, ᶜ∇²uʲs, grid_factor)
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
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs, ᶜ∇²mseʲs) = p.hyperdiff
        (; ᶜuʲs) = p.precomputed
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
            @. ᶜ∇²uʲs.:($$j) =
                C123(wgradₕ(divₕ(ᶜuʲs.:($$j)))) - C123(wcurlₕ(C123(curlₕ(ᶜuʲs.:($$j)))))
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

    (; divergence_damping_factor) = hyperdiff
    (; ν₄_scalar, ν₄_vorticity) =
        ν₄(hyperdiff, Y, p.dt, p.hyperdiff.grid_factor)

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜρ = Y.c.ρ
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; ᶜ∇²u, ᶜ∇²specific_energy) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        ᶜρa⁰ = @. lazy(ρa⁰(ᶜρ, Y.c.sgsʲs, turbconv_model))
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²uʲs, ᶜ∇²mseʲs) = p.hyperdiff
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

    @. Yₜ.c.ρe_tot -= ν₄_scalar * wdivₕ(ᶜρ * gradₕ(ᶜ∇²specific_energy))

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
        (; ᶜ∇²uₕʲs, ᶜ∇²uᵥʲs, ᶜ∇²mseʲs) = p.hyperdiff
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
    (; ν₄_scalar) = ν₄(hyperdiff, Y, p.dt, p.hyperdiff.grid_factor)
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
