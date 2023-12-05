#####
##### Precomputed quantities
#####
import Thermodynamics as TD
import ClimaCore: Spaces, Fields

"""
    precomputed_quantities(Y, atmos)

Allocates and returns the precomputed quantities:
    - `ᶜspecific`: the specific quantities on cell centers (for every prognostic
        quantity `ρχ`, there is a corresponding specific quantity `χ`)
    - `ᶜu`: the covariant velocity on cell centers
    - `ᶠu³`: the third component of contravariant velocity on cell faces
    - `ᶜK`: the kinetic energy on cell centers
    - `ᶜts`: the thermodynamic state on cell centers
    - `ᶜp`: the air pressure on cell centers
    - `sfc_conditions`: the conditions at the surface (at the bottom cell faces)
    - `ᶜh_tot`: the total enthalpy on cell centers

If the `turbconv_model` is EDMFX, there also two SGS versions of every quantity
except for `ᶜp` (we assume that the pressure is the same across all subdomains):
    - `_⁰`: the value for the environment
    - `_ʲs`: a tuple of the values for the mass-flux subdomains
In addition, there are several other SGS quantities for the EDMFX model:
    - `ᶜρa⁰`: the area-weighted air density of the environment on cell centers
    - `ᶠu₃⁰`: the vertical component of the covariant velocity of the environment
        on cell faces
    - `ᶜρ⁰`: the air density of the environment on cell centers
    - `ᶜρʲs`: a tuple of the air densities of the mass-flux subdomains on cell
        centers

TODO: Rename `ᶜK` to `ᶜκ`.
"""
function precomputed_quantities(Y, atmos)
    FT = eltype(Y)
    @assert !(atmos.moisture_model isa DryModel) ||
            !(atmos.turbconv_model isa DiagnosticEDMFX)
    @assert !(atmos.moisture_model isa DryModel) ||
            !(atmos.turbconv_model isa PrognosticEDMFX)
    @assert isnothing(atmos.turbconv_model) || isnothing(atmos.vert_diff)
    TST = thermo_state_type(atmos.moisture_model, FT)
    SCT = SurfaceConditions.surface_conditions_type(atmos, FT)
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    gs_quantities = (;
        ᶜspecific = specific_gs.(Y.c),
        ᶜu = similar(Y.c, C123{FT}),
        ᶠu³ = similar(Y.f, CT3{FT}),
        ᶜK = similar(Y.c, FT),
        ᶜts = similar(Y.c, TST),
        ᶜp = similar(Y.c, FT),
        ᶜh_tot = similar(Y.c, FT),
        sfc_conditions = Fields.Field(SCT, Spaces.level(axes(Y.f), half)),
    )
    advective_sgs_quantities =
        atmos.turbconv_model isa PrognosticEDMFX ?
        (;
            ᶜtke⁰ = similar(Y.c, FT),
            ᶜρa⁰ = similar(Y.c, FT),
            ᶠu₃⁰ = similar(Y.f, C3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜK⁰ = similar(Y.c, FT),
            ᶜmse⁰ = similar(Y.c, FT),
            ᶜq_tot⁰ = similar(Y.c, FT),
            ᶜts⁰ = similar(Y.c, TST),
            ᶜρ⁰ = similar(Y.c, FT),
            ᶜlinear_buoygrad = similar(Y.c, FT),
            ᶜstrain_rate_norm = similar(Y.c, FT),
            ᶜmixing_length = similar(Y.c, FT),
            ᶜK_u = similar(Y.c, FT),
            ᶜK_h = similar(Y.c, FT),
            ρatke_flux = similar(Fields.level(Y.f, half), C3{FT}),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶜtsʲs = similar(Y.c, NTuple{n, TST}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
            ᶜentrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜdetrʲs = similar(Y.c, NTuple{n, FT}),
        ) : (;)
    diagnostic_sgs_quantities =
        atmos.turbconv_model isa DiagnosticEDMFX ?
        (;
            ᶜρaʲs = similar(Y.c, NTuple{n, FT}),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶜtsʲs = similar(Y.c, NTuple{n, TST}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
            ᶜh_totʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_totʲs = similar(Y.c, NTuple{n, FT}),
            ᶜentrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜdetrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜnh_pressureʲs = similar(Y.c, NTuple{n, CT3{FT}}),
            ᶜS_q_totʲs = similar(Y.c, NTuple{n, FT}),
            ᶜS_q_tot⁰ = similar(Y.c, FT),
            ᶜS_e_totʲs_helper = similar(Y.c, NTuple{n, FT}),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶜK⁰ = similar(Y.c, FT),
            ᶜh_tot⁰ = similar(Y.c, FT),
            ᶜtke⁰ = similar(Y.c, FT),
            ᶜlinear_buoygrad = similar(Y.c, FT),
            ᶜstrain_rate_norm = similar(Y.c, FT),
            ᶜmixing_length = similar(Y.c, FT),
            ᶜK_u = similar(Y.c, FT),
            ᶜK_h = similar(Y.c, FT),
            ρatke_flux = similar(Fields.level(Y.f, half), C3{FT}),
        ) : (;)
    vert_diff_quantities = if atmos.vert_diff isa VerticalDiffusion
        ᶜK_h = similar(Y.c, FT)
        (; ᶜK_u = ᶜK_h, ᶜK_h) # ᶜK_u aliases ᶜK_h because they are always equal.
    else
        (;)
    end
    precipitation_quantities =
        atmos.precip_model isa Microphysics1Moment ?
        (; ᶜwᵣ = similar(Y.c, FT), ᶜwₛ = similar(Y.c, FT)) : (;)
    return (;
        gs_quantities...,
        advective_sgs_quantities...,
        diagnostic_sgs_quantities...,
        vert_diff_quantities...,
        precipitation_quantities...,
    )
end

# Interpolates the third contravariant component of Y.c.uₕ to cell faces.
function set_ᶠuₕ³!(ᶠuₕ³, Y)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    @. ᶠuₕ³ = ᶠwinterp(Y.c.ρ * ᶜJ, CT3(Y.c.uₕ))
    return nothing
end

"""
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)

Modifies `Y.f.u₃` so that `ᶠu³` is 0 at the surface. Specifically, since
`u³ = uₕ³ + u³ = uₕ³ + u₃ * g³³`, setting `u³` to 0 gives `u₃ = -uₕ³ / g³³`. If
the `turbconv_model` is EDMFX, the `Y.f.sgsʲs` are also modified so that each
`u₃ʲ` is equal to `u₃` at the surface.
"""
function set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
    sfc_u₃ = Fields.level(Y.f.u₃.components.data.:1, half)
    sfc_uₕ³ = Fields.level(ᶠuₕ³.components.data.:1, half)
    sfc_g³³ = g³³_field(sfc_u₃)
    @. sfc_u₃ = -sfc_uₕ³ / sfc_g³³ # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            sfc_u₃ʲ = Fields.level(Y.f.sgsʲs.:($j).u₃.components.data.:1, half)
            @. sfc_u₃ʲ = sfc_u₃
        end
    end
    return nothing
end

"""
    set_velocity_at_top!(Y, turbconv_model)

Modifies `Y.f.u₃` so that `u₃` is 0 at the model top.
"""
function set_velocity_at_top!(Y, turbconv_model)
    top_u₃ = Fields.level(
        Y.f.u₃.components.data.:1,
        Spaces.nlevels(axes(Y.c)) + half,
    )
    @. top_u₃ = 0
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            top_u₃ʲ = Fields.level(
                Y.f.sgsʲs.:($j).u₃.components.data.:1,
                Spaces.nlevels(axes(Y.c)) + half,
            )
            @. top_u₃ʲ = top_u₃
        end
    end
    return nothing
end

# This is used to set the grid-scale velocity quantities ᶜu, ᶠu³, ᶜK based on
# ᶠu₃, and it is also used to set the SGS quantities based on ᶠu₃⁰ and ᶠu₃ʲ.
function set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠu₃, ᶜuₕ, ᶠuₕ³)
    @. ᶜu = C123(ᶜuₕ) + ᶜinterp(C123(ᶠu₃))
    @. ᶠu³ = ᶠuₕ³ + CT3(ᶠu₃)
    compute_kinetic!(ᶜK, ᶜuₕ, ᶠu₃)
    return nothing
end

function set_sgs_ᶠu₃!(w_function, ᶠu₃, Y, turbconv_model)
    ρaʲs(sgsʲs) = map(sgsʲ -> sgsʲ.ρa, sgsʲs)
    u₃ʲs(sgsʲs) = map(sgsʲ -> sgsʲ.u₃, sgsʲs)
    @. ᶠu₃ = w_function(
        ᶠinterp(ρaʲs(Y.c.sgsʲs)),
        u₃ʲs(Y.f.sgsʲs),
        ᶠinterp(Y.c.ρ),
        Y.f.u₃,
        turbconv_model,
    )
    return nothing
end

function add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠu₃⁰, turbconv_model)
    @. ᶜK += ᶜρa⁰ * ᶜinterp(dot(ᶠu₃⁰ - Yf.u₃, CT3(ᶠu₃⁰ - Yf.u₃))) / 2 / Yc.ρ
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        ᶜρaʲ = Y.c.sgsʲs.:($j).ρa
        ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
        @. ᶜK += ᶜρaʲ * ᶜinterp(dot(ᶠu₃ʲ - Yf.u₃, CT3(ᶠu₃ʲ - Yf.u₃))) / 2 / Yc.ρ
    end
    return nothing
end

function thermo_state(
    thermo_params;
    ρ = nothing,
    p = nothing,
    θ = nothing,
    e_int = nothing,
    q_tot = nothing,
    q_pt = nothing,
)
    get_ts(ρ::Real, ::Nothing, θ::Real, ::Nothing, ::Nothing, ::Nothing) =
        TD.PhaseDry_ρθ(thermo_params, ρ, θ)
    get_ts(ρ::Real, ::Nothing, θ::Real, ::Nothing, q_tot::Real, ::Nothing) =
        TD.PhaseEquil_ρθq(thermo_params, ρ, θ, q_tot)
    get_ts(ρ::Real, ::Nothing, θ::Real, ::Nothing, ::Nothing, q_pt) =
        TD.PhaseNonEquil_ρθq(thermo_params, ρ, θ, q_pt)
    get_ts(ρ::Real, ::Nothing, ::Nothing, e_int::Real, ::Nothing, ::Nothing) =
        TD.PhaseDry_ρe(thermo_params, ρ, e_int)
    get_ts(ρ::Real, ::Nothing, ::Nothing, e_int::Real, q_tot::Real, ::Nothing) =
        TD.PhaseEquil_ρeq(
            thermo_params,
            ρ,
            e_int,
            q_tot,
            3,
            eltype(thermo_params)(0.003),
        )
    get_ts(ρ::Real, ::Nothing, ::Nothing, e_int::Real, ::Nothing, q_pt) =
        TD.PhaseNonEquil(thermo_params, e_int, ρ, q_pt)
    get_ts(::Nothing, p::Real, θ::Real, ::Nothing, ::Nothing, ::Nothing) =
        TD.PhaseDry_pθ(thermo_params, p, θ)
    get_ts(::Nothing, p::Real, θ::Real, ::Nothing, q_tot::Real, ::Nothing) =
        TD.PhaseEquil_pθq(thermo_params, p, θ, q_tot)
    get_ts(::Nothing, p::Real, θ::Real, ::Nothing, ::Nothing, q_pt) =
        TD.PhaseNonEquil_pθq(thermo_params, p, θ, q_pt)
    get_ts(::Nothing, p::Real, ::Nothing, e_int::Real, ::Nothing, ::Nothing) =
        TD.PhaseDry_pe(thermo_params, p, e_int)
    get_ts(::Nothing, p::Real, ::Nothing, e_int::Real, q_tot::Real, ::Nothing) =
        TD.PhaseEquil_peq(thermo_params, p, e_int, q_tot)
    get_ts(::Nothing, p::Real, ::Nothing, e_int::Real, ::Nothing, q_pt) =
        TD.PhaseNonEquil_peq(thermo_params, p, e_int, q_pt)
    return get_ts(ρ, p, θ, e_int, q_tot, q_pt)
end

function thermo_vars(moisture_model, specific, K, Φ)
    energy_var = (; e_int = specific.e_tot - K - Φ)
    moisture_var = if moisture_model isa DryModel
        (;)
    elseif moisture_model isa EquilMoistModel
        (; specific.q_tot)
    elseif moisture_model isa NonEquilMoistModel
        q_pt_args = (specific.q_tot, specific.q_liq, specific.q_ice)
        (; q_pt = TD.PhasePartition(q_pt_args...))
    end
    return (; energy_var..., moisture_var...)
end

ts_gs(thermo_params, moisture_model, specific, K, Φ, ρ) = thermo_state(
    thermo_params;
    thermo_vars(moisture_model, specific, K, Φ)...,
    ρ,
)

ts_sgs(thermo_params, moisture_model, specific, K, Φ, p) = thermo_state(
    thermo_params;
    thermo_vars(moisture_model, specific, K, Φ)...,
    p,
)

function eddy_diffusivity_coefficient(C_E, norm_v_a, z_a, p)
    p_pbl = 85000
    p_strato = 10000
    K_E = C_E * norm_v_a * z_a
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end

"""
    set_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities stored in `p` based on the current state `Y`.

This function also applies a "filter" to `Y` in order to ensure that `ᶠu³` is 0
at the surface (i.e., to enforce the impenetrable boundary condition). If the
`turbconv_model` is EDMFX, the filter also ensures that `ᶠu³⁰` and `ᶠu³ʲs` are 0
at the surface. In the future, we will probably want to move this filtering
elsewhere, but doing it here ensures that it occurs whenever the precomputed
quantities are updated.

Note: If you need to use any of the precomputed quantities, please call this
function instead of recomputing the value yourself. Otherwise, it will be
difficult to ensure that the duplicated computations are consistent.
"""
NVTX.@annotate function set_precomputed_quantities!(Y, p, t)
    (; moisture_model, turbconv_model, vert_diff, precip_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_args = (thermo_params, moisture_model)
    (; ᶜΦ) = p.core
    (; ᶜspecific, ᶜu, ᶠu³, ᶜK, ᶜts, ᶜp) = p.precomputed
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3

    @. ᶜspecific = specific_gs(Y.c)
    set_ᶠuₕ³!(ᶠuₕ³, Y)

    # TODO: We might want to move this to dss! (and rename dss! to something
    # like enforce_constraints!).
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
    set_velocity_at_top!(Y, turbconv_model)

    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y.f.u₃, Y.c.uₕ, ᶠuₕ³)
    if n > 0
        # TODO: In the following increments to ᶜK, we actually need to add
        # quantities of the form ᶜρaχ⁰ / ᶜρ⁰ and ᶜρaχʲ / ᶜρʲ to ᶜK, rather than
        # quantities of the form ᶜρaχ⁰ / ᶜρ and ᶜρaχʲ / ᶜρ. However, we cannot
        # compute ᶜρ⁰ and ᶜρʲ without first computing ᶜts⁰ and ᶜtsʲ, both of
        # which depend on the value of ᶜp, which in turn depends on ᶜts. Since
        # ᶜts depends on ᶜK, this
        # means that the amount by which ᶜK needs to be incremented is a
        # function of ᶜK itself. So, unless we run a nonlinear solver here, this
        # circular dependency will prevent us from computing the exact value of
        # ᶜK. For now, we will make the anelastic approximation ᶜρ⁰ ≈ ᶜρʲ ≈ ᶜρ.
        # add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠu₃⁰, turbconv_model)
        # @. ᶜK += Y.c.sgs⁰.ρatke / Y.c.ρ
        # TODO: We should think more about these increments before we use them.
    end
    @. ᶜts = ts_gs(thermo_args..., ᶜspecific, ᶜK, ᶜΦ, Y.c.ρ)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)

    (; ᶜh_tot) = p.precomputed
    @. ᶜh_tot = TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜspecific.e_tot)

    if !isnothing(p.sfc_setup)
        SurfaceConditions.update_surface_conditions!(Y, p, t)
    end

    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_draft_and_bc!(Y, p, ᶠuₕ³, t)
        set_prognostic_edmf_precomputed_quantities_environment!(Y, p, ᶠuₕ³, t)
        set_prognostic_edmf_precomputed_quantities_closures!(Y, p, t)
    end

    if turbconv_model isa DiagnosticEDMFX
        set_diagnostic_edmf_precomputed_quantities_bottom_bc!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_do_integral!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_top_bc!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_env_closures!(Y, p, t)
    end

    if vert_diff isa VerticalDiffusion
        (; ᶜK_h) = p.precomputed
        interior_uₕ = Fields.level(Y.c.uₕ, 1)
        ᶜΔz_surface = Fields.Δz_field(interior_uₕ)
        @. ᶜK_h = eddy_diffusivity_coefficient(
            p.atmos.vert_diff.C_E,
            norm(interior_uₕ),
            ᶜΔz_surface / 2,
            ᶜp,
        )
    end

    if precip_model isa Microphysics1Moment
        set_precipitation_precomputed_quantities!(Y, p, t)
    end

    return nothing
end

"""
    output_prognostic_sgs_quantities(Y, p, t)

Sets `ᶜu⁺`, `ᶠu³⁺`, `ᶜts⁺` and `ᶜa⁺` to be the same as the
values of the first updraft.
"""
function output_prognostic_sgs_quantities(Y, p, t)
    (; turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜρa⁰, ᶜρ⁰, ᶜtsʲs) = p.precomputed
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    set_ᶠuₕ³!(ᶠuₕ³, Y)
    (ᶠu₃⁺, ᶜu⁺, ᶠu³⁺, ᶜK⁺) =
        similar.((
            p.precomputed.ᶠu₃⁰,
            p.precomputed.ᶜu⁰,
            p.precomputed.ᶠu³⁰,
            p.precomputed.ᶜK⁰,
        ))
    set_sgs_ᶠu₃!(u₃⁺, ᶠu₃⁺, Y, turbconv_model)
    set_velocity_quantities!(ᶜu⁺, ᶠu³⁺, ᶜK⁺, ᶠu₃⁺, Y.c.uₕ, ᶠuₕ³)
    ᶜts⁺ = ᶜtsʲs.:1
    ᶜa⁺ = @. draft_area(ρa⁺(Y.c), TD.air_density(thermo_params, ᶜts⁺))
    ᶜa⁰ = @. draft_area(ᶜρa⁰, ᶜρ⁰)
    return (; ᶠu₃⁺, ᶜu⁺, ᶠu³⁺, ᶜK⁺, ᶜts⁺, ᶜa⁺, ᶜa⁰)
end

"""
    output_diagnostic_sgs_quantities(Y, p, t)

Sets `ᶜu⁺`, `ᶠu³⁺`, `ᶜts⁺` and `ᶜa⁺` to be the same as the
values of the first updraft.
"""
function output_diagnostic_sgs_quantities(Y, p, t)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜρaʲs, ᶜtsʲs) = p.precomputed
    ᶠu³⁺ = p.precomputed.ᶠu³ʲs.:1
    ᶜu⁺ = @. (C123(Y.c.uₕ) + C123(ᶜinterp(ᶠu³⁺)))
    ᶜts⁺ = @. ᶜtsʲs.:1
    ᶜa⁺ = @. draft_area(ᶜρaʲs.:1, TD.air_density(thermo_params, ᶜts⁺))
    return (; ᶜu⁺, ᶠu³⁺, ᶜts⁺, ᶜa⁺)
end
