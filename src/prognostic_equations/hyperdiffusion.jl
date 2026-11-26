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
allocates one struct-valued field `ᶜ∇²` that holds every Laplacian, so that a
single DSS covers them, and a DSS buffer for it when the space requires DSS. The
cache also holds views of the components of `ᶜ∇²`: `ᶜ∇²u`, the energy-split
fields `ᶜ∇²T`, `ᶜ∇²Φ`, and, when `ρq_tot` is prognostic, `ᶜ∇²q_tot_eff`
(energy hyperdiffusion acts on the sensible heat `c_p ∇∇²T`, the geopotential,
and an enthalpy-weighted effective total water, never on a lumped `h_tot`),
`ᶜ∇²specific_tracers` when the state has passive grid-scale tracers, the
corresponding per-updraft fields for `PrognosticEDMFX` (`ᶜ∇²uʲs`, `ᶜ∇²Tʲs`, and
`ᶜ∇²q_tot_effʲs`, plus `ᶜ∇²sgs_tracerʲs` when the state has passive SGS tracers;
the updrafts share `ᶜ∇²Φ`), and `ᶜ∇²tke` when the turbulence-convection model
carries prognostic TKE. For `PrognosticEDMFX`, also allocates
the per-updraft scratch fields `ᶜscratchʲs` and `ᶜscratch_2ʲs`, which are not
DSSed. Both apply functions use `ᶜscratchʲs` (`apply_hyperdiffusion_tendency!`
for `ᶜcpʲs`, `apply_tracer_hyperdiffusion_tendency!` for `ᶜq_tot_hyperdiffʲs`),
each filling it before reading it, so its contents never carry from one apply
function to the other, and the two must not be interleaved.
"""
function hyperdiffusion_cache(Y, atmos)
    (; hyperdiff, turbconv_model) = atmos
    isnothing(hyperdiff) && return (;)  # No hyperdiffiusion
    hyperdiffusion_cache(Y, hyperdiff, turbconv_model)
end

function hyperdiffusion_cache(Y, ::Hyperdiffusion, turbconv_model)
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)
    is_edmf = turbconv_model isa PrognosticEDMFX

    # Grid-scale hyperdiffusion inputs. Energy hyperdiffusion is applied as
    # the sensible heat `c_p ∇²T` and geopotential `∇²Φ` plus a total-water
    # enthalpy contribution weighted by the aggregate enthalpy
    # `h_eff = (h_v·q_v + h_l·q_liq + h_i·q_ice) / q_tot_eff`, so only `∇²T`,
    # `∇²Φ`, and `∇²q_tot_eff` are DSSed; the heat-capacity and enthalpy
    # coefficients multiply the DSSed Laplacian gradients. `q_tot_eff` is the
    # effective total-water field that undergoes hyperdiffusion.
    gs_syms = unrolled_map(hyperdiffused_gs_tracer_names(Y)) do ρχ_name
        MatrixFields.extract_first(specific_tracer_name(ρχ_name))
    end
    gs_types = (;
        u = C123{FT},
        T = FT,
        Φ = FT,
        (MatrixFields.has_field(Y, @name(c.ρq_tot)) ? (; q_tot_eff = FT) : (;))...,
        (
            isempty(gs_syms) ? (;) :
            (; specific_tracers = NamedTuple{gs_syms, NTuple{length(gs_syms), FT}})
        )...,
        (use_prognostic_tke(turbconv_model) ? (; tke = FT) : (;))...,
    )

    # Sub-grid scale quantities. `∇²uʲs` is DSSed as a full C123 vector
    # (matches the grid-mean pattern for `∇²u`); only its C3 component is
    # used in `apply_hyperdiffusion_tendency!` for the u₃ⱼ tendency, but the
    # C12 components are still needed in the outer `wcurlₕ(C123(curlₕ(⋅)))`
    # under non-orthogonal metrics (topography).
    sgs_syms = unrolled_map(
        MatrixFields.extract_first,
        is_edmf ? hyperdiffused_sgs_tracer_names(Y) : (),
    )
    sgs_types =
        is_edmf ?
        (;
            uʲs = NTuple{n, C123{FT}},
            Tʲs = NTuple{n, FT},
            q_tot_effʲs = NTuple{n, FT},
            (
                isempty(sgs_syms) ? (;) :
                (;
                    sgs_tracerʲs = NamedTuple{
                        sgs_syms,
                        NTuple{length(sgs_syms), NTuple{n, FT}},
                    },
                )
            )...,
        ) : (;)

    # All Laplacians live in one struct-valued field, so a single DSS (one
    # halo exchange and one set of DSS kernels) covers them. The `ᶜ∇²*` entries
    # of the cache are views of its components.
    types = (; gs_types..., sgs_types...)
    ᶜ∇² = similar(Y.c, NamedTuple{keys(types), Tuple{values(types)...}})
    views = NamedTuple{unrolled_map(name -> Symbol(:ᶜ∇², name), keys(types))}(
        unrolled_map(name -> getproperty(ᶜ∇², name), keys(types)),
    )
    quantities = (; ᶜ∇², views...)
    if do_dss(axes(Y.c))
        quantities = (;
            quantities...,
            hyperdiffusion_ghost_buffer = Spaces.create_dss_buffer(ᶜ∇²),
        )
    end
    # Per-updraft scratch, not DSSed: the coefficients and tendencies that the
    # fused column loops of the apply functions compute for every updraft at once.
    # `ᶜscratchʲs` is shared by the two apply functions, so it holds valid values
    # only within one of them.
    sgs_scratch =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶜscratchʲs = similar(Y.c, NTuple{n, FT}),
            ᶜscratch_2ʲs = similar(Y.c, NTuple{n, FT}),
        ) : (;)
    return (; quantities..., sgs_scratch...)
end

"""
    prep_hyperdiffusion_tendency!(Y, p)

Compute the horizontal Laplacians that feed the hyperdiffusion tendencies and
store them in the packed field `p.hyperdiff.ᶜ∇²`, ready to be DSSed.

Fills `∇²u` (grad-div minus curl-curl vector Laplacian), the energy-split fields
`∇²T`, `∇²Φ`, and `∇²q_tot_eff`, `∇²tke` when prognostic TKE is active,
`∇²(ρχ/ρ)` for every passive grid-scale tracer, and, for `PrognosticEDMFX`, the
analogous per-updraft fields and `∇²χʲ` for every passive SGS tracer, all in one
horizontal slab loop. Temperature, geopotential, and effective total water are
diffused separately so the `∇⁴` operator never mixes sensible heat with water
enthalpy; `apply_hyperdiffusion_tendency!` reassembles the pieces into a total
enthalpy flux with the heat-capacity and enthalpy coefficients outside the
operator.

Does nothing when `p.atmos.hyperdiff` is `nothing`. `p.hyperdiff.ᶜ∇²` must be
DSSed before the apply functions read it. Called from
`start_hyperdiffusion_tendency!`. Returns `nothing`.
"""
NVTX.@annotate function prep_hyperdiffusion_tendency!(Y, p)
    (; hyperdiff, turbconv_model, microphysics_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    thermo_params = CAP.thermodynamics_params(p.params)
    is_edmf = turbconv_model isa PrognosticEDMFX
    draft_indices = StaticOneTo(n_mass_flux_subdomains(turbconv_model))
    nonequilibrium_val = Val(microphysics_model isa NonEquilibriumMicrophysics)
    gs_names = hyperdiffused_gs_tracer_names(Y)
    sgs_names = is_edmf ? hyperdiffused_sgs_tracer_names(Y) : ()
    (; ᶜu, ᶜT, ᶜp) = p.precomputed
    ᶜdraft_inputs = is_edmf ? (p.precomputed.ᶜuʲs, p.precomputed.ᶜTʲs) : ()

    # Scalars are hyperdiffused as perturbations from a smooth hydrostatic
    # reference profile: on tilted terrain-following surfaces, `gradₕ` of a
    # horizontally uniform `T(z)`, `Φ(z)`, or `q_tot(z)` is nonzero purely from
    # the coordinate tilt, so `∇⁴` on the raw field would spuriously mix scalars
    # across topography. The reference profiles `T_r(p)`, `Φ_r(p)`, and
    # `q_tot_r(p)` cancel this leading-order geometric term, matching the
    # split-form pressure-gradient treatment in `advection.jl`. `Φ - Φ_r(p)` is
    # recomputed every step because `Φ_r` follows the evolving pressure. Updrafts
    # share the grid-mean pressure, so they reuse the same reference profiles,
    # and they share `∇²Φ`.
    DataLayouts.foreach_slab(
        p.hyperdiff.ᶜ∇², Y.c, ᶜu, ᶜT, ᶜp, p.core.ᶜΦ, ᶜdraft_inputs,
    ) do ᶜ∇², Yc, ᶜu, ᶜT, ᶜp, ᶜΦ, ᶜdraft_inputs
        ᶜ∇²u = ᶜ∇².u
        ᶜ∇²T = ᶜ∇².T
        ᶜ∇²Φ = ᶜ∇².Φ
        @. ᶜ∇²u = C123(wgradₕ(divₕ(ᶜu))) - C123(wcurlₕ(C123(curlₕ(ᶜu))))
        @. ᶜ∇²T = wdivₕ(gradₕ(ᶜT - air_temperature_reference(thermo_params, ᶜp)))
        @. ᶜ∇²Φ = wdivₕ(gradₕ(ᶜΦ - phi_r(thermo_params, ᶜp)))
        if MatrixFields.has_field(Yc, @name(ρq_tot))
            ᶜ∇²q_tot_eff = ᶜ∇².q_tot_eff
            if nonequilibrium_val isa Val{true}
                @. ᶜ∇²q_tot_eff = wdivₕ(
                    gradₕ(
                        specific(Yc.ρq_tot - Yc.ρq_rai - Yc.ρq_sno, Yc.ρ) -
                        q_tot_r(thermo_params, ᶜp),
                    ),
                )
            else
                @. ᶜ∇²q_tot_eff = wdivₕ(
                    gradₕ(specific(Yc.ρq_tot, Yc.ρ) - q_tot_r(thermo_params, ᶜp)),
                )
            end
        end
        # Branch on the packed field's components, which are known at compile
        # time, so the GPU kernel never compiles an access to a missing one.
        if MatrixFields.has_field(ᶜ∇², @name(tke))
            ᶜ∇²tke = ᶜ∇².tke
            @. ᶜ∇²tke = wdivₕ(gradₕ(specific(Yc.ρtke, Yc.ρ)))
        end
        MatrixFields.unrolled_foreach(gs_names) do ρχ_name
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜ∇²χ = MatrixFields.get_field(
                ᶜ∇².specific_tracers,
                specific_tracer_name(ρχ_name),
            )
            @. ᶜ∇²χ = wdivₕ(gradₕ(specific(ᶜρχ, Yc.ρ)))
        end
        if ᶜdraft_inputs !== ()
            (ᶜuʲs, ᶜTʲs) = ᶜdraft_inputs
            MatrixFields.unrolled_foreach(draft_indices) do j
                @inline
                ᶜsgsʲ = Yc.sgsʲs.:($j)
                ᶜuʲ = ᶜuʲs.:($j)
                ᶜTʲ = ᶜTʲs.:($j)
                ᶜ∇²uʲ = ᶜ∇².uʲs.:($j)
                ᶜ∇²Tʲ = ᶜ∇².Tʲs.:($j)
                ᶜ∇²q_tot_effʲ = ᶜ∇².q_tot_effʲs.:($j)
                # Full vector Laplacian identity, matching the grid-mean
                # pattern for `∇²u`. Under non-orthogonal metrics (topography),
                # the C12 components matter because they couple into the C3 of
                # the outer `wcurlₕ(C123(curlₕ(⋅)))` via g_{13}, g_{23}.
                @. ᶜ∇²uʲ =
                    C123(wgradₕ(divₕ(ᶜuʲ))) - C123(wcurlₕ(C123(curlₕ(ᶜuʲ))))
                @. ᶜ∇²Tʲ = wdivₕ(
                    gradₕ(ᶜTʲ - air_temperature_reference(thermo_params, ᶜp)),
                )
                if nonequilibrium_val isa Val{true}
                    @. ᶜ∇²q_tot_effʲ = wdivₕ(
                        gradₕ(
                            ᶜsgsʲ.q_tot - ᶜsgsʲ.q_rai - ᶜsgsʲ.q_sno -
                            q_tot_r(thermo_params, ᶜp),
                        ),
                    )
                else
                    @. ᶜ∇²q_tot_effʲ =
                        wdivₕ(gradₕ(ᶜsgsʲ.q_tot - q_tot_r(thermo_params, ᶜp)))
                end
                MatrixFields.unrolled_foreach(sgs_names) do χ_name
                    @inline
                    ᶜχʲ = MatrixFields.get_field(ᶜsgsʲ, χ_name)
                    ᶜ∇²χʲ =
                        MatrixFields.get_field(ᶜ∇².sgs_tracerʲs, χ_name).:($j)
                    @. ᶜ∇²χʲ = wdivₕ(gradₕ(ᶜχʲ))
                end
            end
        end
    end
    return nothing
end

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
`ν₄_scalar = ν₄_vorticity / prandtl_number` for scalars. Requires
`p.hyperdiff.ᶜ∇²` to have been DSSed; does nothing when `p.atmos.hyperdiff` is
`nothing`. Called from `finish_hyperdiffusion_tendency!`. Returns `nothing`.
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
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜρ = Y.c.ρ
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; ᶜ∇²u, ᶜ∇²T, ᶜ∇²Φ) = p.hyperdiff
    if is_edmf
        (; ᶜ∇²uʲs, ᶜ∇²Tʲs, ᶜ∇²q_tot_effʲs) = p.hyperdiff
        # Shares `ᶜscratchʲs` with `ᶜq_tot_hyperdiffʲs` in
        # `apply_tracer_hyperdiffusion_tendency!`.
        ᶜcpʲs = p.hyperdiff.ᶜscratchʲs
        ᶜh_eff_minus_h_dʲs = p.hyperdiff.ᶜscratch_2ʲs
    end
    if diffuse_tke
        (; ᶜ∇²tke) = p.hyperdiff
    end

    # re-use to store the curl-curl part
    ᶜ∇⁴u = @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.f.u₃ -= ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴u))

    # The horizontal momentum tendency and the heat-capacity and enthalpy
    # coefficients of the energy fluxes (see the docstring) of the grid mean and of
    # every updraft, in one column loop. The coefficients go to scratch fields,
    # since they multiply DSSed Laplacian gradients in the horizontal divergences
    # below.
    ᶜcp = p.scratch.ᶜtemp_scalar
    ᶜh_eff_plus_Φ = p.scratch.ᶜtemp_scalar_2
    ᶜq_water = ᶜdiffusing_water_phases(Y, p)
    drafts =
        is_edmf ?
        (;
            ᶜcpʲs,
            ᶜh_eff_minus_h_dʲs,
            p.precomputed.ᶜTʲs,
            p.precomputed.ᶜq_liqʲs,
            p.precomputed.ᶜq_iceʲs,
        ) : (;)
    DataLayouts.foreach_column(
        Yₜ.c.uₕ, ᶜcp, ᶜh_eff_plus_Φ, ᶜ∇⁴u, Y.c, ᶜT, ᶜΦ, drafts,
        something(ᶜq_water, ()),
    ) do ᶜuₕₜ, ᶜcp, ᶜh_eff_plus_Φ, ᶜ∇⁴u, Yc, ᶜT, ᶜΦ, drafts, ᶜq_water
        @. ᶜuₕₜ -= ν₄_vorticity * C12(ᶜ∇⁴u)
        ᶜcp_diffusing!(ᶜcp, thermo_params, isempty(ᶜq_water) ? nothing : ᶜq_water)
        isempty(ᶜq_water) ||
            ᶜh_eff_plus_Φ!(ᶜh_eff_plus_Φ, thermo_params, ᶜT, ᶜΦ, ᶜq_water)
        unrolled_foreach(draft_indices) do j
            @inline
            ᶜq_waterʲ = updraft_diffusing_water(
                Yc.sgsʲs.:($j), drafts.ᶜq_liqʲs.:($j), drafts.ᶜq_iceʲs.:($j),
                nonequilibrium_val,
            )
            ᶜcp_diffusing!(drafts.ᶜcpʲs.:($j), thermo_params, ᶜq_waterʲ)
            ᶜh_eff_minus_h_d!(
                drafts.ᶜh_eff_minus_h_dʲs.:($j), thermo_params, drafts.ᶜTʲs.:($j),
                ᶜq_waterʲ,
            )
            return nothing
        end
        return nothing
    end

    if isnothing(ᶜq_water)
        @. Yₜ.c.ρe_tot -=
            ν₄_scalar *
            (wdivₕ(ᶜρ * ᶜcp * gradₕ(ᶜ∇²T)) + wdivₕ(ᶜρ * gradₕ(ᶜ∇²Φ)))
    else
        @. Yₜ.c.ρe_tot -=
            ν₄_scalar * (
                wdivₕ(ᶜρ * ᶜcp * gradₕ(ᶜ∇²T)) +
                wdivₕ(ᶜρ * gradₕ(ᶜ∇²Φ)) +
                wdivₕ(ᶜρ * ᶜh_eff_plus_Φ * gradₕ(p.hyperdiff.ᶜ∇²q_tot_eff))
            )
    end
    if (turbconv_model isa AbstractEDMF) && diffuse_tke
        @. Yₜ.c.ρtke -= ν₄_vorticity * wdivₕ(ᶜρ * gradₕ(ᶜ∇²tke))
    end
    for j in 1:n
        if point_type <: Geometry.Abstract3DPoint
            # Only the C3 component of ∇⁴uⱼ contributes to Yₜ.f.u₃ⱼ, so drop the
            # grad-div term (its C3 is zero).
            ᶜ∇⁴uⱼ = @. ᶜ∇²uʲs.:($$j) =
                -C123(wcurlₕ(C123(curlₕ(ᶜ∇²uʲs.:($$j)))))
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴uⱼ))
        end
        @. Yₜ.c.sgsʲs.:($$j).mse -=
            ν₄_scalar * (
                wdivₕ(ᶜcpʲs.:($$j) * gradₕ(ᶜ∇²Tʲs.:($$j))) +
                wdivₕ(gradₕ(ᶜ∇²Φ)) +
                wdivₕ(ᶜh_eff_minus_h_dʲs.:($$j) * gradₕ(ᶜ∇²q_tot_effʲs.:($$j)))
            )
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

# Whether the SGS tracer `χ_name` takes the generic ∇⁴ tracer tendency.
is_hyperdiffused_sgs_tracer(χ_name) =
    !(χ_name in hyperdiffusion_excluded_sgs_names)

# The grid-mean and SGS tracers that take the generic ∇⁴ tracer tendency.
hyperdiffused_gs_tracer_names(Y) =
    unrolled_filter(is_hyperdiffused_passive_tracer, gs_tracer_names(Y))
hyperdiffused_sgs_tracer_names(Y) =
    unrolled_filter(is_hyperdiffused_sgs_tracer, sgs_tracer_names(Y))

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
    `Yₜ.c.sgsʲs.:(j).ρa` term, plus every passive SGS tracer from
    `ᶜ∇²sgs_tracerʲs`.

Requires `p.hyperdiff.ᶜ∇²` to have been DSSed; does nothing when
`p.atmos.hyperdiff` is `nothing`. Called from `finish_hyperdiffusion_tendency!`
with the limited tendency vector `Yₜ_lim`. Returns `nothing`.
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
    ᶜρq_tot_hyperdiff = p.scratch.ᶜtemp_scalar
    if has_q_tot
        (; ᶜ∇²q_tot_eff) = p.hyperdiff
        @. ᶜρq_tot_hyperdiff = ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²q_tot_eff))
    end
    if is_edmf
        (; ᶜ∇²q_tot_effʲs) = p.hyperdiff
        # Shares `ᶜscratchʲs` with `ᶜcpʲs` in `apply_hyperdiffusion_tendency!`.
        ᶜq_tot_hyperdiffʲs = p.hyperdiff.ᶜscratchʲs
        for j in 1:n
            @. ᶜq_tot_hyperdiffʲs.:($$j) =
                ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_tot_effʲs.:($$j)))
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
            Yₜ.c, Y.c, ᶜρq_tot_hyperdiff, is_edmf ? ᶜq_tot_hyperdiffʲs : (),
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
    passive_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, ν₄_scalar, n)
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
    passive_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, ν₄_scalar, n)

Add the standard ∇⁴ hyperdiffusion tendency of the passive grid-scale and SGS
tracers, from the DSSed `∇²(ρχ/ρ)` and `∇²χʲ` in `p.hyperdiff`. `ρq_tot` and the
microphysics species are excluded: they take their share of the total-water flux
in `apply_tracer_hyperdiffusion_tendency!`, except rain, snow, and rain number,
which receive no hyperdiffusion.
"""
function passive_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, ν₄_scalar, n)
    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    gs_names = hyperdiffused_gs_tracer_names(Y)
    if !isempty(gs_names)
        (; ᶜ∇²specific_tracers) = p.hyperdiff
        unrolled_foreach(gs_names) do ρχ_name
            ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
            ᶜ∇²χ = MatrixFields.get_field(
                ᶜ∇²specific_tracers,
                specific_tracer_name(ρχ_name),
            )
            @. ᶜρχₜ -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
        end
    end
    sgs_names =
        p.atmos.turbconv_model isa PrognosticEDMFX ?
        hyperdiffused_sgs_tracer_names(Y) : ()
    isempty(sgs_names) && return nothing
    (; ᶜ∇²sgs_tracerʲs) = p.hyperdiff
    unrolled_foreach(sgs_names) do χ_name
        ᶜ∇²χʲs = MatrixFields.get_field(ᶜ∇²sgs_tracerʲs, χ_name)
        for j in 1:n
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
            @. ᶜχʲₜ -= ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²χʲs.:($$j)))
        end
    end
    return nothing
end
