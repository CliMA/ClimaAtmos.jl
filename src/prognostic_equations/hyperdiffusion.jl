#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

"""
    ν₄(hyperdiff, Y)

Return a `NamedTuple` with the scalar hyperdiffusivity `ν₄_scalar` and the
hyperviscosity `ν₄_vorticity` [m⁴/s].

Both coefficients scale with `h^3`, where `h` is the mean nodal distance of the
horizontal grid, following the empirical results of Lauritzen et al. (2018,
https://doi.org/10.1029/2017MS001257): `ν₄_vorticity = ν₄_vorticity_coeff * h^3`
and `ν₄_scalar = ν₄_vorticity / prandtl_number`, with both parameters taken from
`hyperdiff` (a `Hyperdiffusion` model).
"""
function ν₄(hyperdiff, Y)
    h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c)))
    # Vorticity coefficient unchanged
    ν₄_vorticity = hyperdiff.ν₄_vorticity_coeff * h^3
    # Scalar coefficient = vorticity coefficient / Prandtl number
    ν₄_scalar = ν₄_vorticity / hyperdiff.prandtl_number
    return (; ν₄_scalar, ν₄_vorticity)
end

"""
    hyperdiffusion_cache(Y, atmos)
    hyperdiffusion_cache(Y, hyperdiff::Hyperdiffusion, turbconv_model)

Allocate the cache fields that hold the DSSed Laplacians (`∇²`) used by the
hyperdiffusion tendencies.

Returns an empty `NamedTuple` when `atmos.hyperdiff` is `nothing`. Otherwise
allocates `ᶜ∇²u`, the energy-split fields `ᶜ∇²T`, `ᶜ∇²Φ`, and `ᶜ∇²q_tot_eff`
(energy hyperdiffusion acts on the sensible heat `c_p ∇∇²T`, the geopotential,
and an enthalpy-weighted effective total water, never on a lumped `h_tot`),
`ᶜ∇²specific_tracers`, the corresponding per-updraft fields for
`PrognosticEDMFX` (`ᶜ∇²uʲs`, `ᶜ∇²Tʲs`, and `ᶜ∇²q_tot_effʲs`, plus
`ᶜ∇²sgs_tracerʲs` when the state carries SGS tracers; the updrafts share
`ᶜ∇²Φ`),
`ᶜ∇²tke` when the turbulence-convection model carries prognostic TKE, and DSS
ghost buffers when the space requires DSS. For `PrognosticEDMFX`, also allocates
the per-updraft scratch field `ᶜscratchʲs`, which is not DSSed and holds the
updraft total-water tendencies `ᶜq_tot_hyperdiffʲs` of
`apply_tracer_hyperdiffusion_tendency!` between its two loops.
"""
function hyperdiffusion_cache(Y, atmos)
    (; hyperdiff, turbconv_model) = atmos
    isnothing(hyperdiff) && return (;)  # No hyperdiffiusion
    hyperdiffusion_cache(Y, hyperdiff, turbconv_model)
end

function hyperdiffusion_cache(Y, ::Hyperdiffusion, turbconv_model)
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)

    # Grid-scale hyperdiffusion inputs. Energy hyperdiffusion is applied as
    # the sensible heat `c_p ∇²T` and geopotential `∇²Φ` plus a total-water
    # enthalpy contribution weighted by the aggregate enthalpy
    # `h_eff = (h_v·q_v + h_l·q_liq + h_i·q_ice) / q_tot_eff`, so only `∇²T`,
    # `∇²Φ`, and `∇²q_tot_eff` are DSSed; the heat-capacity and enthalpy
    # coefficients multiply the DSSed Laplacian gradients. `q_tot_eff` is the
    # effective total-water field that undergoes hyperdiffusion.
    gs_quantities = (;
        ᶜ∇²u = similar(Y.c, C123{FT}),
        ᶜ∇²T = similar(Y.c, FT),
        ᶜ∇²Φ = similar(Y.c, FT),
        ᶜ∇²q_tot_eff = similar(Y.c, FT),
        ᶜ∇²specific_tracers = Base.materialize(ᶜspecific_gs_tracers(Y)),
    )

    # Sub-grid scale quantities. `ᶜ∇²uʲs` is DSSed as a full C123 vector
    # (matches the grid-mean pattern for `ᶜ∇²u`); only its C3 component is
    # used in `apply_hyperdiffusion_tendency!` for the u₃ⱼ tendency, but the
    # C12 components are still needed in the outer `wcurlₕ(C123(curlₕ(⋅)))`
    # under non-orthogonal metrics (topography).
    sgs_tracer_hyperdiff =
        turbconv_model isa PrognosticEDMFX && !isempty(sgs_tracer_names(Y)) ?
        (; ᶜ∇²sgs_tracerʲs = similar(Y.c, NTuple{n, FT})) : (;)
    sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶜ∇²uʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶜ∇²Tʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_tot_effʲs = similar(Y.c, NTuple{n, FT}),
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
    # Per-updraft scratch, not DSSed. `ᶜscratchʲs` holds the total-water
    # tendencies that the slab loop of `apply_tracer_hyperdiffusion_tendency!`
    # computes for its column loop.
    sgs_scratch =
        turbconv_model isa PrognosticEDMFX ?
        (; ᶜscratchʲs = similar(Y.c, NTuple{n, FT})) : (;)
    return (; quantities..., sgs_scratch...)
end

"""
    prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)

Compute the horizontal Laplacians that feed the dynamics/energy hyperdiffusion and
store them in `p.hyperdiff`, ready to be DSSed.

Fills `ᶜ∇²u` (grad-div minus curl-curl vector Laplacian), the energy-split fields
`ᶜ∇²T`, `ᶜ∇²Φ`, and `ᶜ∇²q_tot_eff`, `ᶜ∇²tke` when prognostic TKE is active, and
the analogous per-updraft fields for `PrognosticEDMFX`. Temperature, geopotential,
and effective total water are diffused separately (rather than a lumped `h_tot`)
so the `∇⁴` operator never mixes sensible heat with water enthalpy;
`apply_hyperdiffusion_tendency!` reassembles the pieces into a total enthalpy
flux with the heat-capacity and enthalpy coefficients outside the operator.

Does nothing when `p.atmos.hyperdiff` is `nothing`. `Yₜ` and `t` are unused. The
fields written here must be DSSed (see `dss_hyperdiffusion_tendency_pairs`) before
`apply_hyperdiffusion_tendency!` is called. Called from `hyperdiffusion_tendency!`.
Returns `nothing`.
"""
NVTX.@annotate function prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    (; params) = p
    (; ᶜΦ) = p.core
    thermo_params = CAP.thermodynamics_params(params)

    isnothing(hyperdiff) && return nothing

    (; ᶜu, ᶜT, ᶜp) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²T, ᶜ∇²Φ, ᶜ∇²q_tot_eff) = p.hyperdiff
    has_q_tot = Val(MatrixFields.has_field(Y, @name(c.ρq_tot)))
    nonequil = Val(p.atmos.microphysics_model isa NonEquilibriumMicrophysics)
    ᶜ∇²tke = use_prognostic_tke(turbconv_model) ? p.hyperdiff.ᶜ∇²tke : ()
    sgs_fields =
        turbconv_model isa PrognosticEDMFX ?
        (
            p.hyperdiff.ᶜ∇²uʲs,
            p.hyperdiff.ᶜ∇²Tʲs,
            p.hyperdiff.ᶜ∇²q_tot_effʲs,
            p.precomputed.ᶜuʲs,
            p.precomputed.ᶜTʲs,
        ) : ()

    # Grid scale hyperdiffusion. Scalars are hyperdiffused as perturbations
    # from a smooth hydrostatic reference profile: on tilted terrain-following
    # surfaces, `gradₕ` of a horizontally uniform `T(z)`, `Φ(z)`, or `q_tot(z)`
    # is nonzero purely from the coordinate tilt, so `∇⁴` on the raw field
    # would spuriously mix scalars across topography. The reference profiles
    # `T_r(p)`, `Φ_r(p)`, and `q_tot_r(p)` cancel this leading-order geometric
    # term, matching the split-form pressure-gradient treatment in
    # `advection.jl`. `Φ - Φ_r(p)` is recomputed every step because `Φ_r`
    # follows the evolving pressure. The vector Laplacians of the grid mean and
    # the updrafts are computed in one fused slab loop, and the scalar
    # Laplacians in another, where the reference profiles are register
    # temporaries shared by the grid mean and the updrafts. (A single loop needs
    # too many registers to keep its blocks resident.)
    ᶜu_sgs = turbconv_model isa PrognosticEDMFX ? (sgs_fields[1], sgs_fields[4]) : ()
    DataLayouts.foreach_slab(ᶜ∇²u, ᶜu, ᶜu_sgs) do ᶜ∇²u, ᶜu, ᶜu_sgs
        # Full vector Laplacian identity. Under non-orthogonal metrics
        # (topography), the C12 components of the updraft Laplacians matter
        # because they couple into the C3 of the outer
        # `wcurlₕ(C123(curlₕ(⋅)))` via g_{13}, g_{23}.
        vector_laplacian!(ᶜ∇²u, ᶜu)
        turbconv_model isa PrognosticEDMFX || return nothing
        (ᶜ∇²uʲs, ᶜuʲs) = ᶜu_sgs
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            vector_laplacian!(ᶜ∇²uʲs.:($j), ᶜuʲs.:($j))
        end
    end
    DataLayouts.foreach_slab(
        (ᶜ∇²T, ᶜ∇²Φ, ᶜ∇²q_tot_eff, ᶜ∇²tke), Y.c, ᶜT, ᶜp, ᶜΦ, sgs_fields,
    ) do (ᶜ∇²T, ᶜ∇²Φ, ᶜ∇²q_tot_eff, ᶜ∇²tke), Yc, ᶜT, ᶜp, ᶜΦ, sgs_fields
        ᶜT_r = @. air_temperature_reference(thermo_params, ᶜp)
        scalar_laplacian!(ᶜ∇²T, (@. ᶜT - ᶜT_r))
        scalar_laplacian!(ᶜ∇²Φ, (@. ᶜΦ - phi_r(thermo_params, ᶜp)))
        ᶜq_tot_r =
            has_q_tot isa Val{true} || turbconv_model isa PrognosticEDMFX ?
            (@. q_tot_r(thermo_params, ᶜp)) : nothing
        if has_q_tot isa Val{true}
            ᶜq_tot_eff = gs_diffusing_q_tot(Yc, nonequil)
            scalar_laplacian!(ᶜ∇²q_tot_eff, (@. ᶜq_tot_eff - ᶜq_tot_r))
        end
        if ᶜ∇²tke isa Fields.Field
            scalar_laplacian!(ᶜ∇²tke, (@. specific(Yc.ρtke, Yc.ρ)))
        end

        # Sub-grid scale hyperdiffusion. SGS mseⱼ uses the same temperature +
        # geopotential + q_tot_eff split as the grid mean (assembled with
        # subdomain thermodynamics in `apply_hyperdiffusion_tendency!`); the
        # geopotential Laplacian `ᶜ∇²Φ` is shared. Updrafts share the grid-mean
        # pressure, so `T_r(ᶜp)` and `q_tot_r(ᶜp)` reuse the same profiles.
        turbconv_model isa PrognosticEDMFX || return nothing
        (_, ᶜ∇²Tʲs, ᶜ∇²q_tot_effʲs, _, ᶜTʲs) = sgs_fields
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            scalar_laplacian!(ᶜ∇²Tʲs.:($j), (@. ᶜTʲs.:($$j) - ᶜT_r))
            ᶜq_tot_effʲ = sgs_diffusing_q_tot(Yc.sgsʲs.:($j), nonequil)
            scalar_laplacian!(ᶜ∇²q_tot_effʲs.:($j), (@. ᶜq_tot_effʲ - ᶜq_tot_r))
        end
    end
    return nothing
end

# The Laplacians of a fused slab loop. Every operator statement reads and writes
# register temporaries of one type (the scalar argument, and the two terms of
# the vector Laplacian), so that the Laplacians of a loop share their inferred
# methods, whatever their inputs and destinations. A vector argument stays a
# slice of its Field: a vector register temporary read by spectral operators is
# heap-allocated on CPUs.
@inline function scalar_laplacian!(ᶜ∇²χ, ᶜχ)
    ᶜ∇²χ_register = @. wdivₕ(gradₕ(ᶜχ))
    @. ᶜ∇²χ = ᶜ∇²χ_register
    return nothing
end
@inline function vector_laplacian!(ᶜ∇²u, ᶜu)
    ᶜgrad_div = @. C123(wgradₕ(divₕ(ᶜu)))
    ᶜcurl_curl = @. C123(wcurlₕ(C123(curlₕ(ᶜu))))
    @. ᶜ∇²u = ᶜgrad_div - ᶜcurl_curl
    return nothing
end

# The specific total water that diffuses in the grid mean `ᶜY` (a whole field or a slice
# of a fused loop): with non-equilibrium microphysics, precipitation does not diffuse.
@inline gs_diffusing_q_tot(ᶜY, ::Val{true}) =
    @. lazy(specific(ᶜY.ρq_tot - ᶜY.ρq_rai - ᶜY.ρq_sno, ᶜY.ρ))
@inline gs_diffusing_q_tot(ᶜY, ::Val{false}) = @. lazy(specific(ᶜY.ρq_tot, ᶜY.ρ))

# The specific total water that diffuses in an updraft `ᶜsgsʲ`
@inline sgs_diffusing_q_tot(ᶜsgsʲ, ::Val{true}) =
    @. lazy(ᶜsgsʲ.q_tot - ᶜsgsʲ.q_rai - ᶜsgsʲ.q_sno)
@inline sgs_diffusing_q_tot(ᶜsgsʲ, ::Val{false}) = ᶜsgsʲ.q_tot

"""
    apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)

Add the fourth-order (`∇⁴`) hyperdiffusion tendencies for momentum, energy, and
TKE, built from the DSSed Laplacians in `p.hyperdiff`.

Increments (each term enters with a minus sign, so hyperdiffusion damps
grid-scale noise):

  - `Yₜ.c.uₕ` and `Yₜ.f.u₃`: `ν₄_vorticity` times the vector `∇⁴u`, with the grad-div
    part scaled by `hyperdiff.divergence_damping_factor`.
  - `Yₜ.c.ρe_tot`: `ν₄_scalar` times the divergence of the total enthalpy
    hyperdiffusion flux,
    `∇⋅(ρ c_p ∇∇²T) + ∇⋅(ρ ∇∇²Φ) + ∇⋅(ρ (h_eff + Φ) ∇∇²q_tot_eff)`, with `c_p`
    the heat capacity of the suspended mixture (`ᶜcp_diffusing!`; `c_pd` in dry
    configurations). The water term is applied only when `ρq_tot` is prognostic.
    Here `q_tot_eff` is the water that hyperdiffuses, vapor plus cloud
    liquid plus cloud ice with rain and snow excluded, and
    `h_eff = (h_v q_v + h_l q_lcl + h_i q_icl) / q_tot_eff` is its aggregate
    specific enthalpy; because the phase shares sum to one, the geopotential
    collapses into the single `+ Φ`.
  - `Yₜ.c.ρtke`: `ν₄_vorticity` times `∇⋅(ρ ∇∇²tke)`, when prognostic TKE is active.
  - For `PrognosticEDMFX`, `Yₜ.f.sgsʲs.:(j).u₃` (curl-curl part only) and
    `Yₜ.c.sgsʲs.:(j).mse`: `ν₄_scalar` times
    `∇⋅(c_pʲ ∇∇²Tʲ) + ∇⋅∇∇²Φ + ∇⋅((h_effʲ - h_d(Tʲ)) ∇∇²q_tot_effʲ)`, the
    decomposition of `∇mseʲ = c_pʲ ∇Tʲ + ∇Φ + Σ_μ (h_μ - h_d) ∇q_μʲ` with the
    intensive water-enthalpy difference `h_effʲ - h_d(Tʲ)` (diffusing water
    replaces dry air at constant specific mass in the intensive `mseʲ`
    equation).

The two coefficients come from `ν₄`: `ν₄_vorticity` for momentum and TKE, and
`ν₄_scalar = ν₄_vorticity / prandtl_number` for scalars. Requires DSS to have been
applied to the pairs from `dss_hyperdiffusion_tendency_pairs`; does nothing when
`p.atmos.hyperdiff` is `nothing`. Called from `hyperdiffusion_tendency!`. Returns
`nothing`.
"""
NVTX.@annotate function apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model, microphysics_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; divergence_damping_factor) = hyperdiff
    (; ν₄_scalar, ν₄_vorticity) = ν₄(hyperdiff, Y)
    (; ᶜΦ) = p.core
    (; ᶜT) = p.precomputed

    n = n_mass_flux_subdomains(turbconv_model)
    draft_indices = StaticOneTo(n)
    is_edmf = turbconv_model isa PrognosticEDMFX
    nonequilibrium_val = Val(microphysics_model isa NonEquilibriumMicrophysics)
    # Whether the updraft momentum `Yₜ.f.sgsʲs.:(j).u₃` is hyperdiffused (on 3D
    # points only).
    updraft_u₃ = Val(eltype(Fields.coordinate_field(Y.c)) <: Geometry.Abstract3DPoint)
    (; ᶜ∇²u, ᶜ∇²T, ᶜ∇²Φ, ᶜ∇²q_tot_eff) = p.hyperdiff
    ᶜ∇²tke =
        turbconv_model isa AbstractEDMF && use_prognostic_tke(turbconv_model) ?
        p.hyperdiff.ᶜ∇²tke : ()
    ᶜq_liq_ice =
        microphysics_model isa DryModel ? () :
        (p.precomputed.ᶜq_liq, p.precomputed.ᶜq_ice)
    drafts =
        is_edmf ?
        (
            p.hyperdiff.ᶜ∇²uʲs,
            p.hyperdiff.ᶜ∇²Tʲs,
            p.hyperdiff.ᶜ∇²q_tot_effʲs,
            p.precomputed.ᶜTʲs,
            p.precomputed.ᶜq_liqʲs,
            p.precomputed.ᶜq_iceʲs,
        ) : ()

    # Every horizontal derivative of the grid mean and of the updrafts, in one slab
    # loop. The density and `∇²Φ` are register temporaries shared by the
    # divergences that read them, and the heat-capacity and enthalpy coefficients
    # of the energy fluxes (see the docstring) are register temporaries computed
    # from the state where the divergences read them.
    DataLayouts.foreach_slab(
        Yₜ.c, Y.c, ᶜ∇²u, (ᶜ∇²T, ᶜ∇²Φ, ᶜ∇²q_tot_eff, ᶜ∇²tke), ᶜT, ᶜΦ,
        ᶜq_liq_ice, drafts,
    ) do Yₜc, Yc, ᶜ∇²u, (ᶜ∇²T, ᶜ∇²Φ, ᶜ∇²q_tot_eff, ᶜ∇²tke), ᶜT, ᶜΦ,
    ᶜq_liq_ice, drafts
        # ∇⁴u overwrites ∇²u, for the column loop below.
        @. ᶜ∇²u =
            divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
            C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
        ᶜρ = as_register(Yc.ρ)
        ᶜ∇²Φ_register = as_register(ᶜ∇²Φ)
        ᶜρe_totₜ = Yₜc.ρe_tot
        if isempty(ᶜq_liq_ice)
            ᶜcp = ᶜcp_diffusing!(nothing, thermo_params, nothing)
            @. ᶜρe_totₜ -=
                ν₄_scalar * (
                    wdivₕ(ᶜρ * ᶜcp * gradₕ(ᶜ∇²T)) +
                    wdivₕ(ᶜρ * gradₕ(ᶜ∇²Φ_register))
                )
        else
            ᶜq_water = diffusing_water_phases(
                Yc, ᶜq_liq_ice..., microphysics_model, Val(false),
            )
            (ᶜq_tot, ᶜq_liq, ᶜq_ice) = ᶜq_water
            ᶜcp = ᶜcp_diffusing!(nothing, thermo_params, ᶜq_water)
            # `ᶜh_eff_plus_Φ!` in registers
            ᶜh_eff_plus_Φ =
                @. h_eff(thermo_params, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice) + ᶜΦ
            @. ᶜρe_totₜ -=
                ν₄_scalar * (
                    wdivₕ(ᶜρ * ᶜcp * gradₕ(ᶜ∇²T)) +
                    wdivₕ(ᶜρ * gradₕ(ᶜ∇²Φ_register)) +
                    wdivₕ(ᶜρ * ᶜh_eff_plus_Φ * gradₕ(ᶜ∇²q_tot_eff))
                )
        end
        if ᶜ∇²tke isa Fields.Field
            ᶜρtkeₜ = Yₜc.ρtke
            @. ᶜρtkeₜ -= ν₄_vorticity * wdivₕ(ᶜρ * gradₕ(ᶜ∇²tke))
        end

        isempty(drafts) && return nothing
        (ᶜ∇²uʲs, ᶜ∇²Tʲs, ᶜ∇²q_tot_effʲs, ᶜTʲs, ᶜq_liqʲs, ᶜq_iceʲs) = drafts
        unrolled_foreach(draft_indices) do j
            @inline
            if updraft_u₃ isa Val{true}
                # Only the C3 component of ∇⁴uⱼ contributes to Yₜ.f.u₃ⱼ, so drop
                # the grad-div term (its C3 is zero).
                ᶜ∇²uʲ = ᶜ∇²uʲs.:($j)
                @. ᶜ∇²uʲ = -C123(wcurlₕ(C123(curlₕ(ᶜ∇²uʲ))))
            end
            ᶜTʲ = ᶜTʲs.:($j)
            ᶜq_waterʲ = updraft_diffusing_water(
                Yc.sgsʲs.:($j), ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j),
                nonequilibrium_val,
            )
            (ᶜq_totʲ, ᶜq_liqʲ, ᶜq_iceʲ) = ᶜq_waterʲ
            ᶜcpʲ = ᶜcp_diffusing!(nothing, thermo_params, ᶜq_waterʲ)
            # `ᶜh_eff_minus_h_d!` in registers
            ᶜh_eff_minus_h_dʲ =
                @. h_eff(thermo_params, ᶜTʲ, ᶜq_totʲ, ᶜq_liqʲ, ᶜq_iceʲ) -
                   TD.enthalpy_dry(thermo_params, ᶜTʲ)
            ᶜ∇²Tʲ = ᶜ∇²Tʲs.:($j)
            ᶜ∇²q_tot_effʲ = ᶜ∇²q_tot_effʲs.:($j)
            ᶜmseʲₜ = Yₜc.sgsʲs.:($j).mse
            @. ᶜmseʲₜ -=
                ν₄_scalar * (
                    wdivₕ(ᶜcpʲ * gradₕ(ᶜ∇²Tʲ)) +
                    wdivₕ(gradₕ(ᶜ∇²Φ_register)) +
                    wdivₕ(ᶜh_eff_minus_h_dʲ * gradₕ(ᶜ∇²q_tot_effʲ))
                )
            return nothing
        end
        return nothing
    end

    # The horizontal momentum tendency and the face tendencies, which interpolate
    # ∇⁴u vertically, in one column loop.
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶜ∇⁴uʲs = is_edmf && updraft_u₃ isa Val{true} ? p.hyperdiff.ᶜ∇²uʲs : ()
    DataLayouts.foreach_column(
        Yₜ.c.uₕ, Yₜ.f, ᶜJ, Y.c.ρ, ᶜ∇²u, ᶜ∇⁴uʲs,
    ) do ᶜuₕₜ, Yₜf, ᶜJ, ᶜρ, ᶜ∇⁴u, ᶜ∇⁴uʲs
        @. ᶜuₕₜ -= ν₄_vorticity * C12(ᶜ∇⁴u)
        ᶠu₃ₜ = Yₜf.u₃
        @. ᶠu₃ₜ -= ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴u))
        ᶜ∇⁴uʲs isa Fields.Field || return nothing
        unrolled_foreach(draft_indices) do j
            @inline
            ᶠu₃ʲₜ = Yₜf.sgsʲs.:($j).u₃
            ᶜ∇⁴uʲ = ᶜ∇⁴uʲs.:($j)
            @. ᶠu₃ʲₜ -= ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴uʲ))
        end
    end
    return nothing
end

"""
    dss_hyperdiffusion_tendency_pairs(p)

Return the tuple of `field => ghost_buffer` pairs that must be DSSed between the
`prep_*` and `apply_*` stages of the hyperdiffusion tendencies.

Covers the dynamics fields (`ᶜ∇²u`, the energy-split `ᶜ∇²T`, `ᶜ∇²Φ`, and
`ᶜ∇²q_tot_eff`, `ᶜ∇²tke` when active, and the per-updraft counterparts `ᶜ∇²uʲs`,
`ᶜ∇²Tʲs`, and `ᶜ∇²q_tot_effʲs` for `PrognosticEDMFX`) and the grid-scale tracer
field `ᶜ∇²specific_tracers`.
Called from `hyperdiffusion_tendency!`, which passes the pairs to
`ClimaCore.Spaces.weighted_dss!`.
"""
function dss_hyperdiffusion_tendency_pairs(p)
    (; turbconv_model) = p.atmos
    buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    (; ᶜ∇²u, ᶜ∇²T, ᶜ∇²Φ, ᶜ∇²q_tot_eff) = p.hyperdiff
    diffuse_tke = use_prognostic_tke(turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uʲs, ᶜ∇²Tʲs, ᶜ∇²q_tot_effʲs) = p.hyperdiff
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end

    core_dynamics_pairs = (
        ᶜ∇²u => buffer.ᶜ∇²u,
        ᶜ∇²T => buffer.ᶜ∇²T,
        ᶜ∇²Φ => buffer.ᶜ∇²Φ,
        ᶜ∇²q_tot_eff => buffer.ᶜ∇²q_tot_eff,
        (diffuse_tke ? (ᶜ∇²tke => buffer.ᶜ∇²tke,) : ())...,
    )
    tc_dynamics_pairs =
        turbconv_model isa PrognosticEDMFX ?
        (
            ᶜ∇²uʲs => buffer.ᶜ∇²uʲs,
            ᶜ∇²Tʲs => buffer.ᶜ∇²Tʲs,
            ᶜ∇²q_tot_effʲs => buffer.ᶜ∇²q_tot_effʲs,
        ) : ()
    dynamics_pairs = (core_dynamics_pairs..., tc_dynamics_pairs...)

    (; ᶜ∇²specific_tracers) = p.hyperdiff
    has_passive_tracers = any(propertynames(ᶜ∇²specific_tracers)) do name
        is_hyperdiffused_passive_tracer(MatrixFields.FieldName((name,)))
    end
    core_tracer_pairs =
        has_passive_tracers ?
        (ᶜ∇²specific_tracers => buffer.ᶜ∇²specific_tracers,) : ()
    return (dynamics_pairs..., core_tracer_pairs...)
end

"""
    prep_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)

Compute the horizontal Laplacians of the specific grid-scale tracers and store
them in `p.hyperdiff`, ready to be DSSed.

Fills `ᶜ∇²specific_tracers` with `∇²(ρχ/ρ)` for every passive grid-scale tracer.
Updraft tracers are handled separately, through the shared `ᶜ∇²sgs_tracerʲs`
scratch field. Does nothing when `p.atmos.hyperdiff` is `nothing`. `Yₜ` and `t`
are unused. The fields written here must be DSSed (see
`dss_hyperdiffusion_tendency_pairs`) before `apply_tracer_hyperdiffusion_tendency!`
is called. Called from `hyperdiffusion_tendency!`. Returns `nothing`.
"""
NVTX.@annotate function prep_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; ᶜ∇²specific_tracers) = p.hyperdiff

    foreach_gs_tracer(Y, ᶜ∇²specific_tracers) do ᶜρχ, ᶜ∇²χ, ρχ_name
        is_hyperdiffused_passive_tracer(ρχ_name) || return
        @. ᶜ∇²χ = wdivₕ(gradₕ(specific(ᶜρχ, Y.c.ρ)))
    end
    return nothing
end

# The species the generic ∇⁴ tracer tendency skips. They equal the sedimenting
# tracers today, and are named separately because the two decisions are
# distinct: one is which tracers fall, the other is which take their share of
# the total-water tendency (or none at all) in place of a ∇⁴ tendency.
const hyperdiffusion_excluded_gs_names = gs_sedimenting_tracer_candidates
const hyperdiffusion_excluded_sgs_names = sgs_sedimenting_tracer_candidates

# Whether the grid-mean tracer `ρχ_name` takes the generic ∇⁴ tracer tendency.
is_hyperdiffused_passive_tracer(ρχ_name) =
    ρχ_name != @name(ρq_tot) && !(ρχ_name in hyperdiffusion_excluded_gs_names)

"""
    apply_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)

Add the fourth-order (`∇⁴`) hyperdiffusion tendencies for tracers, built from the
DSSed Laplacians in `p.hyperdiff`.

Increments (each with a minus sign):

  - `ρq_tot` is hyperdiffused on `q_tot_eff = q_tot - q_rai - q_sno` and the
    resulting mass tendency is applied to `Yₜ.c.ρq_tot` and `Yₜ.c.ρ` (so water
    hyperdiffusion moves mass consistently). Cloud mass species (`ρq_lcl`,
    `ρq_icl`) receive their share by pure scaling of that tendency with the
    clipped ratio `min(q_μ/q_tot_eff, 1)`; their number densities scale
    proportionally. Rain, snow, and rain number density (`ρq_rai`, `ρq_sno`,
    `ρn_rai`) receive no hyperdiffusion.
  - Every passive (non-microphysics) grid-mean tracer `Yₜ.c.ρχ`:
    `ν₄_scalar * ∇⋅(ρ ∇∇²χ)`.
  - For `PrognosticEDMFX`, `Yₜ.c.sgsʲs.:(j).q_tot` and the compensating
    `Yₜ.c.sgsʲs.:(j).ρa` term, plus every auto-discovered SGS tracer (with its own
    prep → DSS → apply cycle through the shared scratch field `ᶜ∇²sgs_tracerʲs`).

Requires DSS to have been applied to the pairs from
`dss_hyperdiffusion_tendency_pairs`; does nothing when `p.atmos.hyperdiff` is
`nothing`. Called from `hyperdiffusion_tendency!` with the limited tendency vector
`Yₜ_lim`. Returns `nothing`.
"""
NVTX.@annotate function apply_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model, microphysics_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; ν₄_scalar) = ν₄(hyperdiff, Y)
    FT = eltype(p.params)
    (zero_FT, one_FT, ϵ_FT) = (zero(FT), one(FT), eps(FT))
    n = n_mass_flux_subdomains(turbconv_model)
    draft_indices = StaticOneTo(n)
    is_edmf = turbconv_model isa PrognosticEDMFX
    has_q_tot = MatrixFields.has_field(Y, @name(c.ρq_tot))
    nonequilibrium_val = Val(microphysics_model isa NonEquilibriumMicrophysics)
    (; ᶜ∇²specific_tracers) = p.hyperdiff

    # Total-water hyperdiff via the single ∇²q_tot_eff field of the grid mean and
    # of each updraft. The grid-mean `ρq_tot_hyperdiff` is applied to ρq_tot and ρ
    # (in divergence form for total-water mass conservation), the updraft one to
    # q_totʲ and, compensating, ρaʲ. Both are then *distributed* to each cloud mass
    # prognostic by pure scaling with the clipped ratio `min(q_μ/q_tot_eff, 1)`.
    # This intentionally drops the per-species divergence form — the alternative
    # flux distribution implies advecting each species at ν₀·∇³q_t/q_t, which
    # explodes in dry regions. Rain, snow and n_rai do not hyperdiffuse.
    #
    # The quantities attached to a cloud mass species follow it in proportion,
    # dρχ = (ρχ/ρq) · dρq: for the number densities ρn_lcl and ρn_ice this
    # preserves the mean particle mass, and for the P3 rime mass ρq_rim and rime
    # volume ρb_rim it preserves the rime fraction and rime density of the ice.
    #
    # The divergences of the grid mean and of every updraft are computed in one
    # slab loop, and distributed in a column loop.
    ᶜρq_tot_hyperdiff = p.scratch.ᶜtemp_scalar
    ᶜq_tot_hyperdiffʲs = is_edmf ? p.hyperdiff.ᶜscratchʲs : ()
    gs_hd = has_q_tot ? (ᶜρq_tot_hyperdiff, p.hyperdiff.ᶜ∇²q_tot_eff) : ()
    sgs_hd = is_edmf ? (ᶜq_tot_hyperdiffʲs, p.hyperdiff.ᶜ∇²q_tot_effʲs) : ()
    if has_q_tot || is_edmf
        DataLayouts.foreach_slab(Y.c.ρ, gs_hd, sgs_hd) do ᶜρ, gs_hd, sgs_hd
            if !isempty(gs_hd)
                (ᶜρq_tot_hd, ᶜ∇²q_tot_eff) = gs_hd
                @. ᶜρq_tot_hd = ν₄_scalar * wdivₕ(ᶜρ * gradₕ(ᶜ∇²q_tot_eff))
            end
            isempty(sgs_hd) && return nothing
            (ᶜq_tot_hdʲs, ᶜ∇²q_tot_effʲs) = sgs_hd
            unrolled_foreach(draft_indices) do j
                @inline
                ᶜq_tot_hdʲ = ᶜq_tot_hdʲs.:($j)
                ᶜ∇²q_tot_effʲ = ᶜ∇²q_tot_effʲs.:($j)
                @. ᶜq_tot_hdʲ = ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_tot_effʲ))
            end
        end
    end
    gs_species = present_species(
        Y.c,
        (
            (@name(ρq_lcl), (@name(ρn_lcl),)),
            (@name(ρq_icl), (@name(ρn_ice), @name(ρq_rim), @name(ρb_rim))),
        ),
    )
    sgs_species =
        is_edmf ?
        present_species(
            Y.c.sgsʲs.:(1),
            ((@name(q_lcl), (@name(n_lcl),)), (@name(q_icl), (@name(n_ice),))),
        ) : ()

    if has_q_tot || n > 0
        DataLayouts.foreach_column(
            Yₜ.c, Y.c, ᶜρq_tot_hyperdiff, ᶜq_tot_hyperdiffʲs,
        ) do Yₜc, Yc, ᶜρq_tot_hd, ᶜq_tot_hdʲs
            if MatrixFields.has_field(Yc, @name(ρq_tot))
                @. Yₜc.ρq_tot -= ᶜρq_tot_hd
                @. Yₜc.ρ -= ᶜρq_tot_hd
                ᶜρq_tot_eff =
                    nonequilibrium_val isa Val{true} ?
                    (@. max(Yc.ρq_tot - Yc.ρq_rai - Yc.ρq_sno, ϵ_FT)) :
                    (@. max(Yc.ρq_tot, ϵ_FT))
                distribute_water_hyperdiffusion!(
                    Yₜc, Yc, gs_species, ᶜρq_tot_eff, ᶜρq_tot_hd,
                    zero_FT, one_FT, ϵ_FT,
                )
            end
            unrolled_foreach(draft_indices) do j
                @inline
                ᶜsgsʲ = Yc.sgsʲs.:($j)
                ᶜsgsʲₜ = Yₜc.sgsʲs.:($j)
                ᶜq_totʲ_hd = ᶜq_tot_hdʲs.:($j)
                @. ᶜsgsʲₜ.q_tot -= ᶜq_totʲ_hd
                @. ᶜsgsʲₜ.ρa -= ᶜsgsʲ.ρa / (1 - ᶜsgsʲ.q_tot) * ᶜq_totʲ_hd
                ᶜq_tot_diffʲ = first(
                    updraft_diffusing_water(
                        ᶜsgsʲ, nothing, nothing, nonequilibrium_val,
                    ),
                )
                ᶜq_tot_effʲ = @. max(ᶜq_tot_diffʲ, ϵ_FT)
                distribute_water_hyperdiffusion!(
                    ᶜsgsʲₜ, ᶜsgsʲ, sgs_species, ᶜq_tot_effʲ, ᶜq_totʲ_hd,
                    zero_FT, one_FT, ϵ_FT,
                )
                return nothing
            end
            return nothing
        end
    end
    passive_tracer_hyperdiffusion_tendency!(
        Yₜ, Y, p, ᶜ∇²specific_tracers, ν₄_scalar, n,
    )
    return nothing
end

# The `(mass name, attached names)` pairs of `species` whose mass is a field of
# `ᶜstate`, each with only the attached names that are fields of `ᶜstate`.
present_species(ᶜstate, species) = unrolled_map(
    unrolled_filter(((q_name, _),) -> MatrixFields.has_field(ᶜstate, q_name), species),
) do (q_name, attached_names)
    (q_name, unrolled_filter(n -> MatrixFields.has_field(ᶜstate, n), attached_names))
end

"""
    distribute_water_hyperdiffusion!(
        ᶜstateₜ, ᶜstate, species, ᶜq_tot_eff, ᶜq_tot_hd, zero_FT, one_FT, ϵ_FT,
    )

Column body of the cloud species share of the total-water hyperdiffusion
tendency `ᶜq_tot_hd`: each cloud mass of `species` (pairs of a mass name and its
attached names, relative to `ᶜstate`) is scaled by the clipped ratio
`min(q_μ/q_tot_eff, 1)`, with `ᶜq_tot_eff` floored at `ϵ_FT`, and each attached
quantity follows its mass in proportion.
"""
@inline function distribute_water_hyperdiffusion!(
    ᶜstateₜ, ᶜstate, species, ᶜq_tot_eff, ᶜq_tot_hd, zero_FT, one_FT, ϵ_FT,
)
    MatrixFields.unrolled_foreach(species) do (q_name, attached_names)
        @inline
        ᶜq = MatrixFields.get_field(ᶜstate, q_name)
        ᶜqₜ = MatrixFields.get_field(ᶜstateₜ, q_name)
        ᶜratio = @. max(zero_FT, min(one_FT, ᶜq / ᶜq_tot_eff))
        @. ᶜqₜ -= ᶜratio * ᶜq_tot_hd
        MatrixFields.unrolled_foreach(attached_names) do χ_name
            @inline
            ᶜχ = MatrixFields.get_field(ᶜstate, χ_name)
            ᶜχₜ = MatrixFields.get_field(ᶜstateₜ, χ_name)
            @. ᶜχₜ -= ᶜratio * max(zero_FT, ᶜχ) / max(ᶜq, ϵ_FT) * ᶜq_tot_hd
        end
    end
    return nothing
end

"""
    passive_tracer_hyperdiffusion_tendency!(
        Yₜ, Y, p, ᶜ∇²specific_tracers, ν₄_scalar, n,
    )

Add the standard ∇⁴ hyperdiffusion tendency of the passive grid-scale and SGS
tracers. `ρq_tot` and the microphysics species are excluded: they take their share
of the total-water flux in `apply_tracer_hyperdiffusion_tendency!`, except rain,
snow, and rain number, which receive no hyperdiffusion.
"""
function passive_tracer_hyperdiffusion_tendency!(
    Yₜ, Y, p, ᶜ∇²specific_tracers, ν₄_scalar, n,
)
    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    foreach_gs_tracer(Yₜ, ᶜ∇²specific_tracers) do ᶜρχₜ, ᶜ∇²χ, ρχ_name
        is_hyperdiffused_passive_tracer(ρχ_name) || return
        @. ᶜρχₜ -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
    end
    isempty(sgs_tracer_names(Y)) && return nothing
    (; ᶜ∇²sgs_tracerʲs) = p.hyperdiff
    for χ_name in sgs_tracer_names(Y)
        χ_name in hyperdiffusion_excluded_sgs_names && continue
        for j in 1:n
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
            @. ᶜ∇²sgs_tracerʲs.:($$j) = wdivₕ(gradₕ(ᶜχʲ))
        end
        if do_dss(axes(Y.c))
            Spaces.weighted_dss!(
                ᶜ∇²sgs_tracerʲs =>
                    p.hyperdiff.hyperdiffusion_ghost_buffer.ᶜ∇²sgs_tracerʲs,
            )
        end
        for j in 1:n
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
            @. ᶜχʲₜ -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²sgs_tracerʲs.:($$j)))
        end
    end
    return nothing
end
