"""
Vertical transport diagnostics for the microphysics tracers.

ClimaAtmos registers the 1-moment microphysics process rates but not the transport terms of the same
budgets, so a tendency budget assembled from its diagnostics alone is missing sedimentation, vertical
advection and vertical diffusion. These register the missing terms through the same public extension
point (`add_diagnostic_variable!`) the process rates use.

Each is computed with the model's own operators on the model's own state, so it is the term the model
applied, not a reconstruction: sedimentation reuses the `ᶜprecipdivᵥ ∘ ᶠright_bias` flux divergence
from `implicit_vertical_advection_tendency!`, advection reuses `vertical_transport` from
`explicit_vertical_advection_tendency!` with the configured upwinding, and diffusion reuses the face
diffusivities and `α_vert_diff_tracer` weighting from `edmfx_sgs_diffusive_flux_tendency!`. Fall
speeds come from `p.precomputed`, so they follow `FixedTerminalVelocity` or
`DiagnosticTerminalVelocity` automatically.

ClimaAtmos resolves a diagnostic's short name through `ALL_DIAGNOSTICS` and offers no per-simulation
variable table, so `add_diagnostic_variable!` is the only way to make a new name requestable.
[`socrates_diagnostics`](@ref) calls the registration, so the entry a run uses is the one defined by
the code that run loaded.
"""

using ClimaAtmos: ClimaAtmos as CA

"""Tracer suffix → (`ρq` field name, precomputed fall-speed name)."""
const TRANSPORT_SPECIES = (
    ("q_lcl", :ρq_lcl, :ᶜwₗ),
    ("q_icl", :ρq_icl, :ᶜwᵢ),
    ("q_rai", :ρq_rai, :ᶜwᵣ),
    ("q_sno", :ρq_sno, :ᶜwₛ),
)

"""
    sedimentation_tendency(state, cache, ρq_name, w_name)

`∂q/∂t` from sedimentation [kg kg^-1 s^-1], as the model applies it.

`implicit_vertical_advection_tendency!` subtracts this flux divergence from `ρq`, so the sign here is
negative and the result is divided by `ρ` to give a specific-content tendency.
"""
function sedimentation_tendency(state, cache, ρq_name::Symbol, w_name::Symbol)
    ᶜρ = state.c.ρ
    ᶜJ = CA.CC.Fields.local_geometry_field(axes(state.c)).J
    ᶠJ = CA.CC.Fields.local_geometry_field(axes(state.f)).J
    ᶜw = getproperty(cache.precomputed, w_name)
    ᶜρq = getproperty(state.c, ρq_name)
    return @. CA.lazy(
        -CA.ᶜprecipdivᵥ(
            CA.ᶠinterp(ᶜρ * ᶜJ) / ᶠJ *
            CA.ᶠright_bias(CA.CC.Geometry.WVector(-(ᶜw)) * CA.specific(ᶜρq, ᶜρ)),
        ) / ᶜρ,
    )
end

"""
    vertical_advection_tendency(state, cache, ρq_name)

`∂q/∂t` from vertical advection by the resolved flow [kg kg^-1 s^-1], using the same
`vertical_transport` and `tracer_upwinding` the explicit tendency uses.
"""
function vertical_advection_tendency(state, cache, ρq_name::Symbol)
    ᶜρ = state.c.ρ
    ᶠu³ = cache.precomputed.ᶠu³
    ᶜρq = getproperty(state.c, ρq_name)
    ᶜχ = @. CA.lazy(CA.specific(ᶜρq, ᶜρ))
    vtt = CA.vertical_transport(
        ᶜρ,
        ᶠu³,
        ᶜχ,
        cache.dt,
        cache.atmos.numerics.tracer_upwinding,
    )
    return @. CA.lazy(vtt / ᶜρ)
end

"""
    vertical_diffusion_tendency(state, cache, ρq_name)

`∂q/∂t` from SGS vertical diffusion [kg kg^-1 s^-1], using the same face diffusivities and
`α_vert_diff_tracer` scaling `edmfx_sgs_diffusive_flux_tendency!` applies to sedimenting tracers.

Errors rather than returning zero when the configuration routes tracer diffusion through a path this
does not reproduce, so a missing term can never read as an absent one.
"""
function vertical_diffusion_tendency(state, cache, ρq_name::Symbol)
    cache.atmos.edmfx_model.sgs_diffusive_flux isa Val{true} || error(
        "sgs_diffusive_flux is off, so `edmfx_sgs_diffusive_flux_tendency!` applies no tracer \
         diffusion and this diagnostic would not be the term the model used.",
    )
    isnothing(cache.atmos.vertical_diffusion) || error(
        "vertical_diffusion = $(cache.atmos.vertical_diffusion) also diffuses tracers through \
         `vertical_diffusion_boundary_layer_tendency!`, which this diagnostic does not include.",
    )
    ᶜρ = state.c.ρ
    FT = eltype(ᶜρ)
    (; ᶠK_h, ᶠK_entr) = cache.precomputed
    α = CA.CAP.α_vert_diff_tracer(cache.params)
    ᶜρq = getproperty(state.c, ρq_name)
    ᶜχ = @. CA.lazy(CA.specific(ᶜρq, ᶜρ))
    ᶠgradᵥ = CA.CC.Operators.GradientC2F()
    ᶜdivᵥ_ρq = CA.CC.Operators.DivergenceF2C(
        top = CA.CC.Operators.SetValue(CA.C3(zero(FT))),
        bottom = CA.CC.Operators.SetValue(CA.C3(zero(FT))),
    )
    # α scales only the turbulent part; interfacial entrainment crosses at the same velocity for
    # every scalar, so `α ρ (K_h + K_e) + (1 - α) ρ K_e` collapses to `ρ (α K_h + K_e)`.
    ᶠρK = @. CA.lazy(CA.ᶠinterp(ᶜρ) * (α * ᶠK_h + ᶠK_entr))
    return @. CA.lazy(ᶜdivᵥ_ρq(ᶠρK * ᶠgradᵥ(ᶜχ)) / ᶜρ)
end

"""
    register_transport_diagnostics!()

Register `sed_<species>`, `adv_<species>` and `dif_<species>` for every microphysics tracer.

Any previous entry is dropped first, so a re-included file replaces the closure registered earlier in
the session instead of leaving the stale one in place.
"""
function register_transport_diagnostics!()
    for (species, ρq_name, w_name) in TRANSPORT_SPECIES
        for (prefix, description, compute) in (
            (
                "sed",
                "sedimentation",
                (state, cache, _) -> sedimentation_tendency(state, cache, ρq_name, w_name),
            ),
            (
                "adv",
                "resolved vertical advection",
                (state, cache, _) -> vertical_advection_tendency(state, cache, ρq_name),
            ),
            (
                "dif",
                "SGS vertical diffusion",
                (state, cache, _) -> vertical_diffusion_tendency(state, cache, ρq_name),
            ),
        )
            name = "$(prefix)_$(species)"
            delete!(CA.Diagnostics.ALL_DIAGNOSTICS, name)
            CA.Diagnostics.add_diagnostic_variable!(;
                short_name = name,
                long_name = "Tendency of $species from $description",
                units = "kg kg^-1 s^-1",
                comments = "Computed with the model's own vertical operators on the model state.",
                compute,
            )
        end
    end
    return nothing
end
