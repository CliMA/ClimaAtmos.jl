#####
##### TKE Tendency for Eddy Diffusion Closure Used in EDMFX
#####

"""
    edmfx_tke_tendency!(Yₜ, Y, p, t, turbconv_model)

Add the PROPHET (`EDMFX` in code) shear and buoyancy TKE production terms to
`Yₜ.c.ρtke`.

The generic method is a no-op. The method for
`turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX}` forwards to
`edmfx_tke_sources!` when `use_prognostic_tke(turbconv_model)` holds.
Turbulent TKE transport and dissipation are applied separately, in
`edmfx_sgs_diffusive_flux_tendency!`.

Mutates `Yₜ.c.ρtke`; returns `nothing`. See the "PROPHET Sub-Grid Scale
Equations" page (`docs/src/edmf_equations.md`) for the TKE budget.
"""
edmfx_tke_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_tke_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX},
)
    use_prognostic_tke(turbconv_model) || return nothing
    edmfx_tke_sources!(Yₜ, Y, p)
    return nothing
end

"""
    edmfx_tke_sources!(Yₜ, Y, p)

Add the shear and buoyancy sources of the isotropic (intra-subdomain) TKE to
`Yₜ.c.ρtke`.

Both terms use the same face diffusivities and face buoyancy gradient as the
diffusive fluxes they parameterize (`set_face_diffusivities!`):

  - Buoyancy production/destruction `−ρ interp((ᶠK_h + ᶠK_entr) ᶠbuoygrad)`
    is stencil-exact: the product is formed at the faces from the same
    factors as the scalar fluxes and only then interpolated, so it is
    exactly the (interpolated) buoyancy content of those fluxes. In
    unstable layers it is the usual convective production; at stable
    unresolved jumps the `ᶠK_entr ᶠbuoygrad` part carries the interfacial-
    entrainment sink `−γ w_e Δb` per face automatically (bounded by
    `A κ^{3/2}/ℓ_e`, a fixed multiple of the dissipation).
  - Shear production `+2 ρ interp(ᶠK_u + ᶠK_entr) ‖S‖²` corresponds to the
    momentum flux `−2 ρ (ᶠK_u + ᶠK_entr) 𝔈` at the adjacent faces, but only
    approximately at the stencil level: the viscosity is interpolated
    separately and multiplied by the *center* strain-rate norm (the face
    norm is not precomputed), rather than interpolating the face-local
    product. The two agree to second order in smooth flow and differ by an
    O(1) factor only where `K` or `‖S‖²` jumps between adjacent faces.

Only the diffusive (intra-subdomain) piece of the Favre-averaged buoyancy
flux enters this budget. The coherent (mass-flux) piece
`Σ_m ρa^m (w^m - w) b^m` powers the inter-subdomain (coherent) kinetic energy
through the buoyancy term of the subdomain momentum equations, which the
prognostic subdomain velocities already carry; adding it here would
double-count buoyancy production and spuriously inflate K near cloud tops
with active drafts.

Reads `ᶜstrain_rate_norm`, `ᶠbuoygrad`, `ᶠK_h`, `ᶠK_u`, and `ᶠK_entr` from
`p.precomputed`; mutates `Yₜ.c.ρtke` and returns `nothing`.
"""
function edmfx_tke_sources!(Yₜ, Y, p)
    (; ᶜstrain_rate_norm) = p.precomputed
    (; ᶠbuoygrad, ᶠK_h, ᶠK_u, ᶠK_entr) = p.precomputed

    # shear production (face viscosities brought to centers)
    @. Yₜ.c.ρtke +=
        2 * Y.c.ρ * ᶜinterp(ᶠK_u + ᶠK_entr) * ᶜstrain_rate_norm
    # buoyancy production/destruction (face-flux consistent; includes the
    # interfacial-entrainment sink through ᶠK_entr)
    @. Yₜ.c.ρtke -= Y.c.ρ * ᶜinterp((ᶠK_h + ᶠK_entr) * ᶠbuoygrad)
    return nothing
end

"""
    tke_dissipation(turbconv_params, ρtke, tke, mixing_length)

Return the TKE dissipation rate per unit volume, `ρ ε_d` [kg m⁻¹ s⁻³]:

    ρ ε_d = c_d * ρtke * sqrt(abs(tke)) / mixing_length,

where `c_d` is the TKE dissipation coefficient
(`tke_dissipation_coefficient`).

# Arguments

  - `turbconv_params`: Turbulence and convection model parameters.
  - `ρtke`: TKE density `ρ * tke` [kg m⁻¹ s⁻²].
  - `tke`: Specific turbulent kinetic energy [m² s⁻²].
  - `mixing_length`: Turbulent mixing length [m].
"""
function tke_dissipation(turbconv_params, ρtke, tke, mixing_length)
    FT = typeof(tke)
    c_d = tke_dissipation_coefficient(turbconv_params)
    dissipation_rate_vol = c_d * ρtke * sqrt(abs(tke)) / mixing_length
    return dissipation_rate_vol
end
