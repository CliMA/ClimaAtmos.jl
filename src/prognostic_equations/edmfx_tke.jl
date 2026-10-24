#####
##### TKE Tendency for Eddy Diffusion Closure Used in EDMFX
#####

"""
    edmfx_tke_tendency!(Yₜ, Y, p, t, turbconv_model)

Add the PROPHET (`EDMFX` in code) TKE sources to `Yₜ.c.ρtke`: shear and
buoyancy production, and, for `PrognosticEDMFX`, the pressure-drag
return-to-isotropy and detrainment shear-mixing sources.

The generic method is a no-op. The method for
`turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX}` forwards to
`edmfx_tke_sources!` when `use_prognostic_tke(turbconv_model)` holds.
Turbulent TKE transport and dissipation are applied separately, in
`edmfx_sgs_diffusive_flux_tendency!`.

Mutates `Yₜ.c.ρtke`; returns `nothing`. See the "PROPHET: Closures" page
(`docs/src/prophet_closures.md`) for the TKE budget.
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

Add the sources of the isotropic (intra-subdomain) TKE to `Yₜ.c.ρtke`: shear and
buoyancy production for every EDMF model, plus the pressure-drag
return-to-isotropy source (when `edmfx_nh_pressure` is on) and the detrainment
shear-mixing source for `PrognosticEDMFX`.

Both terms are formed at the faces from the same factors as the diffusive
fluxes they parameterize (`set_face_diffusivities!`) and only then
interpolated to the centers, so each is the interpolated energy content of
its flux:

  - Buoyancy production/destruction `−ρ interp(ᶠK_h ᶠbuoygrad)`, with the
    face buoyancy gradient of the scalar fluxes.
  - Shear production `+2 ρ interp(ᶠK_u ‖𝔈‖²)`, with the face strain rate `𝔈`
    of the momentum flux `−2 ρ K_u 𝔈` (`compute_strain_rate_face_vertical`
    of the grid-mean `ᶜu`, as in `edmfx_sgs_vertical_diffusive_flux_tendency!`).
    That flux acts on the horizontal wind only, so `‖𝔈‖² = |∂_z uₕ|² / 2`
    (`vertical_shear_norm_sqr`), and `2 K_u ‖𝔈‖² Δz` at a face is the
    resolved kinetic energy the flux removes between the two adjacent
    centers; summed over a column, the production equals the kinetic energy
    loss up to the interpolation of `ρ` between centers and faces. The
    center norm `ᶜstrain_rate_norm` is the shear in `Ri` and the mixing
    lengths, with a wider stencil.

Only the diffusive (intra-subdomain) piece of the Favre-averaged buoyancy
flux enters this budget. The coherent (mass-flux) piece
`Σ_m ρa^m (w^m - w) b^m` powers the inter-subdomain (coherent) kinetic energy
through the buoyancy term of the subdomain momentum equations, which the
prognostic subdomain velocities already carry; adding it here would
double-count buoyancy production and spuriously inflate K near cloud tops
with active drafts.

Reads `ᶜu`, `ᶠbuoygrad`, `ᶠK_h`, and `ᶠK_u` from `p.precomputed` and uses
`p.scratch.ᶠtemp_scalar`; mutates `Yₜ.c.ρtke` and returns `nothing`.
"""
function edmfx_tke_sources!(Yₜ, Y, p)
    (; ᶜu, ᶠbuoygrad, ᶠK_h, ᶠK_u) = p.precomputed

    # shear production: the face-local work of the momentum flux on the
    # horizontal wind, brought to centers
    ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
    ᶠshear_production = p.scratch.ᶠtemp_scalar
    @. ᶠshear_production = 2 * ᶠK_u * vertical_shear_norm_sqr(ᶠstrain_rate)
    @. Yₜ.c.ρtke += Y.c.ρ * ᶜinterp(ᶠshear_production)
    # buoyancy production/destruction (face-flux consistent)
    @. Yₜ.c.ρtke -= Y.c.ρ * ᶜinterp(ᶠK_h * ᶠbuoygrad)

    # Pressure-drag return-to-isotropy
    edmfx_pressure_drag_tke_source!(Yₜ, Y, p, p.atmos.turbconv_model)
    # Entr/detr shear-mixing source: 0.5 δ · ρa · |Δw|² per updraft.
    edmfx_entr_detr_tke_source!(Yₜ, Y, p, p.atmos.turbconv_model)
    return nothing
end

edmfx_pressure_drag_tke_source!(Yₜ, Y, p, turbconv_model) = nothing
function edmfx_pressure_drag_tke_source!(
    Yₜ, Y, p, turbconv_model::PrognosticEDMFX,
)
    p.atmos.edmfx_model.nh_pressure || return nothing
    n = n_mass_flux_subdomains(turbconv_model)
    n == 0 && return nothing

    turbconv_params = CAP.turbconv_params(p.params)
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    a_min = CAP.min_area(turbconv_params)
    a_max = CAP.max_area(turbconv_params)
    scale_height = CAP.R_d(p.params) * CAP.T_surf_ref(p.params) / CAP.grav(p.params)
    (; ᶜρʲs, ᶜuʲs, ᶜu⁰) = p.precomputed
    # Environment area, shared across all updrafts.
    ᶜa⁰ = @. lazy(a⁰(Y.c.sgsʲs, ᶜρʲs, turbconv_model))
    ᶜdrag_coeff = p.scratch.ᶜtemp_scalar
    ᶜlg = Fields.local_geometry_field(Y.c)
    for j in 1:n
        ᶜaʲ = @. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
        # The coefficient of the momentum equation's drag sink, so the energy
        # removed there is the energy received here.
        @. ᶜdrag_coeff =
            pressure_drag_coefficient(α_d, scale_height, ᶜaʲ, ᶜa⁰, a_min, a_max)
        @. Yₜ.c.ρtke +=
            Y.c.sgsʲs.:($$j).ρa * ᶜa⁰ * ᶜdrag_coeff *
            abs(get_physical_w(ᶜuʲs.:($$j) - ᶜu⁰, ᶜlg))^3
    end
    return nothing
end

edmfx_entr_detr_tke_source!(Yₜ, Y, p, turbconv_model) = nothing
function edmfx_entr_detr_tke_source!(
    Yₜ, Y, p, turbconv_model::PrognosticEDMFX,
)
    n = n_mass_flux_subdomains(turbconv_model)
    n == 0 && return nothing

    (;
        ᶜρʲs,
        ᶜρ_diffʲs,
        ᶜarea_bounding_entr_detrʲs,
        ᶜuʲs,
        ᶠu³ʲs,
        ᶜu⁰,
    ) = p.precomputed
    (; ᶠgradᵥ_ᶜΦ) = p.core

    turbconv_params = CAP.turbconv_params(p.params)
    entr_detr_buoy_inv_tau_max =
        CAP.entr_detr_buoy_inv_tau_max(turbconv_params)
    detr_model = p.atmos.edmfx_model.detr_model
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠbottom_bias_zero_bot = Operators.BottomBiasedC2F(bottom = Operators.SetValue(0))
    FT = Spaces.undertype(axes(Y.c))

    for j in 1:n
        ᶜaʲ = @. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
        ᶜbuoy_inv_time_scale = @. lazy(
            ᶜinterp(
                detr_buoy_inv_time_scale(
                    get_physical_w(ᶠu³ʲs.:($$j), ᶠlg),
                    vertical_buoyancy_acceleration(
                        ᶠinterp(ᶜρ_diffʲs.:($$j)),
                        ᶠgradᵥ_ᶜΦ,
                        ᶠlg,
                    ),
                    entr_detr_buoy_inv_tau_max,
                ),
            ),
        )
        ᶜdetr = @. lazy(
            compute_detrainment(
                turbconv_params,
                ᶜaʲ,
                Y.c.sgsʲs.:($$j).ρa,
                ᶜbuoy_inv_time_scale,
                ᶜdivᵥ(ᶠbottom_bias_zero_bot(Y.c.sgsʲs.:($$j).ρa) * ᶠu³ʲs.:($$j)),
                ᶜarea_bounding_entr_detrʲs.:($$j),
                detr_model,
            ),
        )

        @. Yₜ.c.ρtke +=
            FT(0.5) * ᶜdetr * Y.c.sgsʲs.:($$j).ρa *
            get_physical_w(ᶜuʲs.:($$j) - ᶜu⁰, ᶜlg)^2
    end
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
    c_d = tke_dissipation_coefficient(turbconv_params)
    FT = eltype(mixing_length)
    # `max(mixing_length, 1)` is a numerical divide-by-zero guard applied at
    # the point of division (dissipation only), not a physical floor on
    # `mixing_length`.
    dissipation_rate_vol =
        c_d * ρtke * sqrt(abs(tke)) / max(mixing_length, FT(1))
    return dissipation_rate_vol
end
