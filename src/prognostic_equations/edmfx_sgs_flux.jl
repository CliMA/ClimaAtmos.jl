#####
##### Tendencies applied to the grid-mean atmospheric state due to subgrid-scale (SGS)
##### fluxes computed by the EDMFX scheme
#####

"""
    edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Computes and applies tendencies to the grid-mean prognostic variables due to the
divergence of subgrid-scale (SGS) mass fluxes from EDMFX updrafts and the environment.

This involves terms of the form `- ∂(ρₖ aₖ w′ₖ ϕ′ₖ)/∂z`, where `k` denotes
an SGS component (updraft `j` or environment `0`), `aₖ` is the area fraction,
`w′ₖ` is the vertical velocity deviation from the grid mean, and `ϕ′ₖ` is the
deviation of a conserved variable `ϕ` (such as total enthalpy or specific humidity)
from its grid-mean value. These terms represent the redistribution of energy and tracers
by the resolved SGS circulations relative to the grid mean flow.

The SGS flux of `q_tot` redistributes water mass, so `Yₜ.c.ρ` receives the
same tendency as `Yₜ.c.ρq_tot` (mirroring the diffusive-flux treatment of
moist air mass).

The specific implementation depends on the `turbconv_model` (e.g., `PrognosticEDMFX`).
A generic fallback doing nothing is also provided.
The function modifies `Yₜ.c` (grid-mean tendencies) in place.

Arguments:

  - `Yₜ`: The tendency state vector for grid-mean variables.
  - `Y`: The current state vector (used for grid-mean and SGS properties).
  - `p`: Cache containing parameters, precomputed fields, atmospheric model settings,
    and scratch space.
  - `t`: Current simulation time.
  - `turbconv_model`: The turbulence convection model instance.
"""
edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_mass_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; edmfx_sgsflux_upwinding) = p.atmos.numerics
    (; ᶜp, ᶠu³) = p.precomputed
    (; ᶠu³ʲs, ᶜKʲs, ᶜρʲs) = p.precomputed
    (; ᶠu³⁰, ᶜK⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) = p.precomputed
    (; dt) = p

    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜρ⁰ = @. lazy(
        TD.air_density(
            thermo_params,
            ᶜT⁰,
            ᶜp,
            ᶜq_tot_nonneg⁰,
            ᶜq_liq⁰,
            ᶜq_ice⁰,
        ),
    )
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))

    if p.atmos.edmfx_model.sgs_mass_flux isa Val{true}

        # Enthalpy fluxes. First sum up the draft fluxes
        # TODO: Isolate assembly of flux term pattern to a function and
        # reuse (both in prognostic and diagnostic EDMFX)
        # [best after removal of precomputed quantities]
        ᶠu³_diff = p.scratch.ᶠtemp_CT3
        ᶜa_scalar = p.scratch.ᶜtemp_scalar
        (; ᶜh_tot) = p.precomputed
        for j in 1:n
            @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
            @. ᶜa_scalar =
                (Y.c.sgsʲs.:($$j).mse + ᶜKʲs.:($$j) - ᶜh_tot) *
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
            vtt = vertical_transport(
                ᶜρʲs.:($j),
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρe_tot += vtt
        end
        # Add the environment fluxes
        @. ᶠu³_diff = ᶠu³⁰ - ᶠu³
        ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
        @. ᶜa_scalar = (ᶜmse⁰ + ᶜK⁰ - ᶜh_tot) * draft_area(ᶜρa⁰, ᶜρ⁰)
        vtt = vertical_transport(
            ᶜρ⁰,
            ᶠu³_diff,
            ᶜa_scalar,
            dt,
            edmfx_sgsflux_upwinding,
        )
        @. Yₜ.c.ρe_tot += vtt

        if !(p.atmos.microphysics_model isa DryModel)
            # Specific humidity fluxes: First sum up the draft fluxes
            for j in 1:n
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
                @. ᶜa_scalar =
                    (Y.c.sgsʲs.:($$j).q_tot - specific(Y.c.ρq_tot, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                @. Yₜ.c.ρq_tot += vtt
                @. Yₜ.c.ρ += vtt  # Effect of SGS water flux on (moist) air mass
            end
            # Add the environment fluxes
            ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
            @. ᶠu³_diff = ᶠu³⁰ - ᶠu³
            @. ᶜa_scalar =
                (ᶜq_tot⁰ - specific(Y.c.ρq_tot, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            @. Yₜ.c.ρq_tot += vtt
            @. Yₜ.c.ρ += vtt  # Effect of SGS water flux on (moist) air mass
        end

        # Auto-discovered SGS tracer fluxes (microphysics species and any
        # user-defined passive tracers). Like the mse and q_tot fluxes above,
        # these are difference-form fluxes ρᵏaᵏ(u³ᵏ - u³)(χᵏ - χ), which
        # vanish identically for uniform χ, reconstructed with the same
        # upwinding as the mse and q_tot fluxes so that the water-species
        # fluxes stay consistent with the q_tot flux (the implied vapor flux
        # is their difference). The grid-mean advection -∇·(ρ u³ χ) of each
        # tracer is applied in explicit_vertical_advection_tendency!.
        # Draft fluxes
        for χ_name in sgs_tracer_names(Y)
            ρχ_name = get_ρχ_name(χ_name)
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            for j in 1:n
                ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                @. ᶠu³_diff = ᶠu³ʲs.:($$j) - ᶠu³
                @. ᶜa_scalar =
                    (ᶜχʲ - specific(ᶜρχ, Y.c.ρ)) *
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
                vtt = vertical_transport(
                    ᶜρʲs.:($j),
                    ᶠu³_diff,
                    ᶜa_scalar,
                    dt,
                    edmfx_sgsflux_upwinding,
                )
                ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
                @. ᶜρχₜ += vtt
            end
        end
        # Environment fluxes
        @. ᶠu³_diff = ᶠu³⁰ - ᶠu³
        for χ_name in sgs_tracer_names(Y)
            ρχ_name = get_ρχ_name(χ_name)
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
            @. ᶜa_scalar =
                (ᶜχ⁰ - specific(ᶜρχ, Y.c.ρ)) * draft_area(ᶜρa⁰, ᶜρ⁰)
            vtt = vertical_transport(
                ᶜρ⁰,
                ᶠu³_diff,
                ᶜa_scalar,
                dt,
                edmfx_sgsflux_upwinding,
            )
            ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
            @. ᶜρχₜ += vtt
        end
    end
    # TODO - add vertical momentum fluxes
    return nothing
end

"""
    edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model)

Computes and applies the tendency to the grid-mean state `Y` due to SGS
diffusive fluxes from the EDMFX environment. This involves calculating the
divergence of turbulent fluxes, which are parameterized using eddy diffusivity
and viscosity closures.

This function parameterizes these fluxes using an eddy-diffusivity/viscosity
approach (K-theory) for the grid-mean. Tendencies are calculated for
total energy, moisture species, momentum, and optionally TKE.
The form is typically `- ∂/∂z(-D ∂ϕ/∂z)`, where `D` is an effective SGS eddy
diffusivity for the quantity `ϕ`.

The specific implementation depends on the `turbconv_model`. A generic fallback
doing nothing is also provided. The function modifies `Yₜ.c` (grid-mean tendencies)
in place.

Arguments:

  - `Yₜ`: The tendency state vector for grid-mean variables.
  - `Y`: The current state vector (used for grid-mean and SGS properties).
  - `p`: Cache containing parameters, precomputed fields, atmospheric model settings,
    and scratch space.
  - `t`: Current simulation time.
  - `turbconv_model`: The turbulence convection model instance.
"""
edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_sgs_diffusive_flux_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::Union{EDOnlyEDMFX, PrognosticEDMFX},
)

    FT = Spaces.undertype(axes(Y.c))
    (; dt, params) = p
    turbconv_params = CAP.turbconv_params(params)
    (; ᶜu) = p.precomputed
    (; ρtke_flux) = p.precomputed
    ᶠgradᵥ = Operators.GradientC2F()
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))

    if p.atmos.edmfx_model.sgs_diffusive_flux isa Val{true}

        (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
        # scratch to prevent GPU Kernel parameter memory error
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_2
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜK_u = @. lazy(
            eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field),
        )
        ᶜprandtl_nvec = @. lazy(
            turbulent_prandtl_number(
                params,
                ᶜlinear_buoygrad,
                ᶜstrain_rate_norm,
            ),
        )
        ᶜK_h = @. lazy(eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec))

        ᶠρaK_h = p.scratch.ᶠtemp_scalar
        @. ᶠρaK_h = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_h)
        ᶠρaK_u = p.scratch.ᶠtemp_scalar_2
        @. ᶠρaK_u = ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜK_u)

        # Total enthalpy diffusion, using the dry-static-energy + water-
        # enthalpy decomposition
        #   F_h = -K_h ∇s_d + Σ_μ h_tot,μ F_qμ,   F_qμ = -K_h ∇q_μ,
        # with s_d = h_d + Φ, h_tot,μ = h_μ(T) + Φ for μ ∈ {vap, liq, ice},
        # and unit turbulent Lewis number (K_qμ = K_h). Diffusing h_tot
        # directly would imply a spurious enthalpy flux carried by dry-air
        # diffusion (h_tot depends on 1 - q_t through the dry-air mass
        # fraction), systematically warming entrained air at inversions where
        # h_tot jumps up while q_t jumps down. The thermal piece diffuses dry
        # static energy (constant under dry-adiabatic displacement), and each
        # water constituent carries its own enthalpy with its diffusive mass
        # flux, consistent with the ρq_tot and ρ updates below.
        #
        # Note: F_qμ for liquid and ice uses unscaled K_h (omitting the tracer
        # vertical diffusion factor α_vert_diff_tracer). While microphysics
        # tracers are diffused with α * K_h to prevent unphysical upward
        # transport of precipitation, omitting α here maintains exact energetic
        # consistency with the unscaled ρq_tot diffusion equation, preserves
        # total water invariance under moist-adiabatic processes, and aligns
        # with the implicit solver's Jacobian formulation.
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        thermo_params = CAP.thermodynamics_params(params)
        (; ᶜΦ) = p.core
        (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
        ᶜq_vap = @. lazy(
            TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
        )
        @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(
            -(
                ᶠρaK_h * (
                    ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)) +
                    ᶠinterp(TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ) *
                    ᶠgradᵥ(ᶜq_vap) +
                    ᶠinterp(TD.enthalpy_liquid(thermo_params, ᶜT) + ᶜΦ) *
                    ᶠgradᵥ(ᶜq_liq) +
                    ᶠinterp(TD.enthalpy_ice(thermo_params, ᶜT) + ᶜΦ) *
                    ᶠgradᵥ(ᶜq_ice)
                )
            ),
        )

        if use_prognostic_tke(turbconv_model)
            # Turbulent TKE transport (diffusion)
            ᶜdivᵥ_ρtke = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(ρtke_flux),
            )
            # Add flux divergence and dissipation term, relaxing TKE to zero
            # in one time step if tke < 0
            @. Yₜ.c.ρtke -=
                ᶜdivᵥ_ρtke(-(ᶠρaK_u * ᶠgradᵥ(ᶜtke))) + ifelse(
                    ᶜtke >= FT(0),
                    tke_dissipation(
                        turbconv_params,
                        Y.c.ρtke,
                        ᶜtke,
                        ᶜmixing_length_field,
                    ),
                    Y.c.ρtke / dt,
                )
        end

        if !(p.atmos.microphysics_model isa DryModel)
            # Specific humidity diffusion
            ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
            ᶜdivᵥ_ρq_tot = Operators.DivergenceF2C(
                top = Operators.SetValue(C3(FT(0))),
                bottom = Operators.SetValue(C3(FT(0))),
            )
            @. ᶜρχₜ_diffusion =
                ᶜdivᵥ_ρq_tot(-(ᶠρaK_h * ᶠgradᵥ(specific(Y.c.ρq_tot, Y.c.ρ))))
            @. Yₜ.c.ρq_tot -= ᶜρχₜ_diffusion
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion  # Effect of moisture diffusion on (moist) air mass
        end

        α_vert_diff_microphysics = CAP.α_vert_diff_tracer(params)
        ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
        ᶜdivᵥ_ρq = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(C3(FT(0))),
        )
        # Auto-discovered grid-scale tracers (microphysics species and any
        # user-defined passive tracers)
        for χ_name in sgs_tracer_names(Y)
            ρχ_name = get_ρχ_name(χ_name)
            MatrixFields.has_field(Y.c, ρχ_name) || continue
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
            ᶜχ = (@. lazy(specific(ᶜρχ, Y.c.ρ)))
            @. ᶜρχₜ_diffusion = ᶜdivᵥ_ρq(-(ᶠρaK_h * α_vert_diff_microphysics * ᶠgradᵥ(ᶜχ)))
            @. ᶜρχₜ -= ᶜρχₜ_diffusion
        end

        # Momentum diffusion
        ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
        @. Yₜ.c.uₕ -= C12(ᶜdivᵥ(-(2 * ᶠρaK_u * ᶠstrain_rate)) / Y.c.ρ)
    end

    return nothing
end
