#####
##### Precomputed quantities
#####
import Thermodynamics as TD
import ClimaCore: Spaces, Fields

"""
    implicit_precomputed_quantities(Y, atmos)

Allocates precomputed quantities that are treated implicitly (i.e., updated
on each iteration of the implicit solver). This includes all quantities related
to velocity and thermodynamics that are used in the implicit tendency.

The following grid-scale quantities are treated implicitly:
    - `ᶜspecific`: specific quantities on cell centers (for every prognostic
        quantity `ρχ`, there is a corresponding specific quantity `χ`)
    - `ᶜu`: covariant velocity on cell centers
    - `ᶠu`: contravariant velocity on cell faces
    - `ᶜK`: kinetic energy on cell centers
    - `ᶜts`: thermodynamic state on cell centers
    - `ᶜp`: air pressure on cell centers
    - `ᶜh_tot`: total enthalpy on cell centers
If the `turbconv_model` is `PrognosticEDMFX`, there also two SGS versions of
every quantity except for `ᶜp` (which is shared across all subdomains):
    - `_⁰`: value for the environment
    - `_ʲs`: a tuple of values for the mass-flux subdomains
In addition, there are several other SGS quantities for `PrognosticEDMFX`:
    - `ᶜtke⁰`: turbulent kinetic energy of the environment on cell centers
    - `ᶜρʲs`: a tuple of the air densities of the mass-flux subdomains on cell
        centers
For every other `AbstractEDMF`, only `ᶜtke⁰` is added as a precomputed quantity.

TODO: Rename `ᶜK` to `ᶜκ`.
"""
function implicit_precomputed_quantities(Y, atmos)
    (; moisture_model, turbconv_model, precip_model) = atmos
    FT = eltype(Y)
    TST = thermo_state_type(moisture_model, FT)
    n = n_mass_flux_subdomains(turbconv_model)
    gs_quantities = (;
        ᶜspecific = specific_gs.(Y.c),
        ᶜu = similar(Y.c, C123{FT}),
        ᶠu³ = similar(Y.f, CT3{FT}),
        ᶠu = similar(Y.f, CT123{FT}),
        ᶜK = similar(Y.c, FT),
        ᶜts = similar(Y.c, TST),
        ᶜp = similar(Y.c, FT),
        ᶜh_tot = similar(Y.c, FT),
    )
    sgs_quantities =
        turbconv_model isa AbstractEDMF ? (; ᶜtke⁰ = similar(Y.c, FT)) : (;)
    moisture_sgs_quantities =
        (
            turbconv_model isa PrognosticEDMFX &&
            moisture_model isa NonEquilMoistModel &&
            precip_model isa Microphysics1Moment
        ) ?
        (;
            ᶜq_liq⁰ = similar(Y.c, FT),
            ᶜq_ice⁰ = similar(Y.c, FT),
            ᶜq_rai⁰ = similar(Y.c, FT),
            ᶜq_sno⁰ = similar(Y.c, FT),
        ) : (;)
    prognostic_sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶠu₃⁰ = similar(Y.f, C3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜK⁰ = similar(Y.c, FT),
            ᶜts⁰ = similar(Y.c, TST),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶠKᵥʲs = similar(Y.f, NTuple{n, FT}),
            ᶜtsʲs = similar(Y.c, NTuple{n, TST}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
            ᶠnh_pressure₃_dragʲs = similar(Y.f, NTuple{n, C3{FT}}),
            moisture_sgs_quantities...,
        ) : (;)
    return (; gs_quantities..., sgs_quantities..., prognostic_sgs_quantities...)
end

"""
    precomputed_quantities(Y, atmos)

Allocates all precomputed quantities. This includes the quantities treated
implicitly (updated before each tendency evaluation), and also the quantities
treated explicitly (updated only before explicit tendency evaluations).

TODO: Reduce the number of cached values by computing them on the fly.
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
    cspace = axes(Y.c)
    fspace = axes(Y.f)
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    gs_quantities = (;
        ᶜwₜqₜ = similar(Y.c, Geometry.WVector{FT}),
        ᶜwₕhₜ = similar(Y.c, Geometry.WVector{FT}),
        ᶜmixing_length = similar(Y.c, FT),
        ᶜlinear_buoygrad = similar(Y.c, FT),
        ᶜstrain_rate_norm = similar(Y.c, FT),
        sfc_conditions = similar(Spaces.level(Y.f, half), SCT),
    )
    cloud_diagnostics_tuple =
        similar(Y.c, @NamedTuple{cf::FT, q_liq::FT, q_ice::FT})
    surface_precip_fluxes = (;
        surface_rain_flux = zeros(axes(Fields.level(Y.f, half))),
        surface_snow_flux = zeros(axes(Fields.level(Y.f, half))),
    )
    sedimentation_quantities =
        atmos.moisture_model isa NonEquilMoistModel ?
        (; ᶜwₗ = similar(Y.c, FT), ᶜwᵢ = similar(Y.c, FT)) : (;)
    if atmos.precip_model isa Microphysics0Moment
        precipitation_quantities =
            (; ᶜS_ρq_tot = similar(Y.c, FT), ᶜS_ρe_tot = similar(Y.c, FT))
    elseif atmos.precip_model isa Microphysics1Moment
        precipitation_quantities = (;
            ᶜwᵣ = similar(Y.c, FT),
            ᶜwₛ = similar(Y.c, FT),
            ᶜSqₗᵖ = similar(Y.c, FT),
            ᶜSqᵢᵖ = similar(Y.c, FT),
            ᶜSqᵣᵖ = similar(Y.c, FT),
            ᶜSqₛᵖ = similar(Y.c, FT),
        )
    elseif atmos.precip_model isa Microphysics2Moment
        precipitation_quantities = (;
            ᶜwᵣ = similar(Y.c, FT),
            ᶜwₛ = similar(Y.c, FT),
            ᶜSqₗᵖ = similar(Y.c, FT),
            ᶜSqᵢᵖ = similar(Y.c, FT),
            ᶜSqᵣᵖ = similar(Y.c, FT),
            ᶜSqₛᵖ = similar(Y.c, FT),
            ᶜwnₗ = similar(Y.c, FT),
            ᶜwnᵣ = similar(Y.c, FT),
            ᶜSnₗᵖ = similar(Y.c, FT),
            ᶜSnᵣᵖ = similar(Y.c, FT),
        )
    else
        precipitation_quantities = (;)
    end
    precipitation_sgs_quantities =
        atmos.precip_model isa Microphysics0Moment ?
        (; ᶜSqₜᵖʲs = similar(Y.c, NTuple{n, FT}), ᶜSqₜᵖ⁰ = similar(Y.c, FT)) :
        atmos.precip_model isa Microphysics1Moment ?
        (;
            ᶜSqₗᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqᵢᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqᵣᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqₛᵖʲs = similar(Y.c, NTuple{n, FT}),
            ᶜSqₗᵖ⁰ = similar(Y.c, FT),
            ᶜSqᵢᵖ⁰ = similar(Y.c, FT),
            ᶜSqᵣᵖ⁰ = similar(Y.c, FT),
            ᶜSqₛᵖ⁰ = similar(Y.c, FT),
        ) : (;)
    advective_sgs_quantities =
        atmos.turbconv_model isa PrognosticEDMFX ?
        (;
            # ᶜmixing_length_tuple = similar(Y.c, MixingLength{FT}),
            ρatke_flux = similar(Fields.level(Y.f, half), C3{FT}),
            bdmr_l = similar(Y.c, BidiagonalMatrixRow{FT}),
            bdmr_r = similar(Y.c, BidiagonalMatrixRow{FT}),
            bdmr = similar(Y.c, BidiagonalMatrixRow{FT}),
            ᶜentrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜdetrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜturb_entrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜgradᵥ_θ_virt⁰ = Fields.Field(C3{FT}, cspace),
            ᶜgradᵥ_q_tot⁰ = Fields.Field(C3{FT}, cspace),
            ᶜgradᵥ_θ_liq_ice⁰ = Fields.Field(C3{FT}, cspace),
            ᶠnh_pressure₃_buoyʲs = similar(Y.f, NTuple{n, C3{FT}}),
            precipitation_sgs_quantities...,
        ) : (;)

    edonly_quantities =
        atmos.turbconv_model isa EDOnlyEDMFX ?
        (;
            ᶜmixing_length_tuple = similar(Y.c, MixingLength{FT}),
            ᶜtke⁰ = similar(Y.c, FT),
            ρatke_flux = similar(Fields.level(Y.f, half), C3{FT}),
            ᶜK_h = similar(Y.c, FT),
        ) : (;)

    sgs_quantities = (;
        ᶜgradᵥ_θ_virt = Fields.Field(C3{FT}, cspace),
        ᶜgradᵥ_q_tot = Fields.Field(C3{FT}, cspace),
        ᶜgradᵥ_θ_liq_ice = Fields.Field(C3{FT}, cspace),
    )

    diagnostic_sgs_quantities =
        atmos.turbconv_model isa DiagnosticEDMFX ?
        (;
            ᶜρaʲs = similar(Y.c, NTuple{n, FT}),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶜtsʲs = similar(Y.c, NTuple{n, TST}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
            ᶜmseʲs = similar(Y.c, NTuple{n, FT}),
            ᶜq_totʲs = similar(Y.c, NTuple{n, FT}),
            ᶜentrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜdetrʲs = similar(Y.c, NTuple{n, FT}),
            ᶜturb_entrʲs = similar(Y.c, NTuple{n, FT}),
            ᶠnh_pressure³_buoyʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶠnh_pressure³_dragʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶜK⁰ = similar(Y.c, FT),
            ᶜmixing_length_tuple = similar(Y.c, MixingLength{FT}),
            ᶜK_u = similar(Y.c, FT),
            ᶜK_h = similar(Y.c, FT),
            ρatke_flux = similar(Fields.level(Y.f, half), C3{FT}),
            precipitation_sgs_quantities...,
        ) : (;)
    vert_diff_quantities =
        if atmos.vert_diff isa
           Union{VerticalDiffusion, DecayWithHeightDiffusion}
            ᶜK_h = similar(Y.c, FT)
            (; ᶜK_u = ᶜK_h, ᶜK_h) # ᶜK_u aliases ᶜK_h because they are always equal.
        else
            (;)
        end
    smagorinsky_lilly_quantities =
        if atmos.smagorinsky_lilly isa SmagorinskyLilly
            uvw_vec = UVW(FT(0), FT(0), FT(0))
            (;
                ᶜτ_smag = similar(Y.c, typeof(uvw_vec * uvw_vec')),
                ᶠτ_smag = similar(Y.f, typeof(uvw_vec * uvw_vec')),
                ᶜD_smag = similar(Y.c, FT),
                ᶠD_smag = similar(Y.f, FT),
            )
        else
            (;)
        end

    return (;
        implicit_precomputed_quantities(Y, atmos)...,
        gs_quantities...,
        sgs_quantities...,
        advective_sgs_quantities...,
        edonly_quantities...,
        diagnostic_sgs_quantities...,
        vert_diff_quantities...,
        sedimentation_quantities...,
        precipitation_quantities...,
        surface_precip_fluxes...,
        cloud_diagnostics_tuple,
        smagorinsky_lilly_quantities...,
    )
end

# Interpolates the third contravariant component of Y.c.uₕ to cell faces.
function compute_ᶠuₕ³(ᶜuₕ, ᶜρ)
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    return @. lazy(ᶠwinterp(ᶜρ * ᶜJ, CT3(ᶜuₕ)))
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
    sfc_u₃ .= surface_velocity(Y.f.u₃, ᶠuₕ³)
    if turbconv_model isa PrognosticEDMFX
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            sfc_u₃ʲ = Fields.level(Y.f.sgsʲs.:($j).u₃.components.data.:1, half)
            @. sfc_u₃ʲ = sfc_u₃
        end
    end
    return nothing
end

function surface_velocity(ᶠu₃, ᶠuₕ³)
    sfc_u₃ = Fields.level(ᶠu₃.components.data.:1, half)
    sfc_uₕ³ = Fields.level(ᶠuₕ³.components.data.:1, half)
    sfc_g³³ = g³³_field(axes(sfc_u₃))
    return @. lazy(-sfc_uₕ³ / sfc_g³³) # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
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
    ᶜK .= compute_kinetic(ᶜuₕ, ᶠu₃)
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

function thermo_vars(moisture_model, precip_model, ᶜY, K, Φ)
    energy_var = (; e_int = specific(ᶜY.ρe_tot, ᶜY.ρ) - K - Φ)
    moisture_var = if moisture_model isa DryModel
        (;)
    elseif moisture_model isa EquilMoistModel
        (; q_tot = specific(ᶜY.ρq_tot, ᶜY.ρ))
    elseif moisture_model isa NonEquilMoistModel
        q_pt_args = (
            q_tot = specific(ᶜY.ρq_tot, ᶜY.ρ),
            q_liq = specific(ᶜY.ρq_liq, ᶜY.ρ),
            q_ice = specific(ᶜY.ρq_ice, ᶜY.ρ),
        )
        (; q_pt = TD.PhasePartition(q_pt_args...))
    end
    return (; energy_var..., moisture_var...)
end

ts_gs(thermo_params, moisture_model, precip_model, ᶜY, K, Φ, ρ) =
    thermo_state(
        thermo_params;
        thermo_vars(moisture_model, precip_model, ᶜY, K, Φ)...,
        ρ,
    )

ts_sgs(thermo_params, moisture_model, precip_model, ᶜY, K, Φ, p) =
    thermo_state(
        thermo_params;
        thermo_vars(moisture_model, precip_model, ᶜY, K, Φ)...,
        p,
    )

function eddy_diffusivity_coefficient_H(D₀, H, z_sfc, z)
    return D₀ * exp(-(z - z_sfc) / H)
end
function eddy_diffusivity_coefficient(C_E, norm_v_a, z_a, p)
    p_pbl = 85000
    p_strato = 10000
    K_E = C_E * norm_v_a * z_a
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end

"""
    set_implicit_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities that are handled implicitly based on the
current state `Y`. This is called before each evaluation of either
`implicit_tendency!` or `remaining_tendency!`, and it includes quantities used
in both tedencies.

This function also applies a "filter" to `Y` in order to ensure that `ᶠu³` is 0
at the surface (i.e., to enforce the impenetrable boundary condition). If the
`turbconv_model` is EDMFX, the filter also ensures that `ᶠu³⁰` and `ᶠu³ʲs` are 0
at the surface. In the future, we will probably want to move this filtering
elsewhere, but doing it here ensures that it occurs whenever the precomputed
quantities are updated.
"""
NVTX.@annotate function set_implicit_precomputed_quantities!(Y, p, t)
    (; turbconv_model, moisture_model, precip_model) = p.atmos
    (; ᶜΦ) = p.core
    (; ᶜspecific, ᶜu, ᶠu³, ᶠu, ᶜK, ᶜts, ᶜp, ᶜh_tot) = p.precomputed
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_params = CAP.thermodynamics_params(p.params)
    thermo_args = (thermo_params, moisture_model, precip_model)

    @. ᶜspecific = specific_gs(Y.c)
    @. ᶠuₕ³ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)

    # TODO: We might want to move this to dss! (and rename dss! to something
    # like enforce_constraints!).
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
    set_velocity_at_top!(Y, turbconv_model)

    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y.f.u₃, Y.c.uₕ, ᶠuₕ³)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    @. ᶠu = CT123(ᶠwinterp(Y.c.ρ * ᶜJ, CT12(ᶜu))) + CT123(ᶠu³)
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
    @. ᶜts = ts_gs(thermo_args..., Y.c, ᶜK, ᶜΦ, Y.c.ρ)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)
    @. ᶜh_tot = TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜspecific.e_tot)

    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_draft!(Y, p, ᶠuₕ³, t)
        set_prognostic_edmf_precomputed_quantities_environment!(Y, p, ᶠuₕ³, t)
        set_prognostic_edmf_precomputed_quantities_implicit_closures!(Y, p, t)
    elseif turbconv_model isa AbstractEDMF
        (; ᶜtke⁰) = p.precomputed
        @. ᶜtke⁰ = Y.c.sgs⁰.ρatke / Y.c.ρ
    end
end

"""
    set_explicit_precomputed_quantities!(Y, p, t)

Updates the precomputed quantities that are handled explicitly based on the
current state `Y`. This is only called before each evaluation of
`remaining_tendency!`, though it includes quantities used in both
`implicit_tendency!` and `remaining_tendency!`.
"""
NVTX.@annotate function set_explicit_precomputed_quantities!(Y, p, t)
    (; turbconv_model, moisture_model, precip_model, cloud_model) = p.atmos
    (; vert_diff, call_cloud_diagnostics_per_stage) = p.atmos
    (; ᶜΦ) = p.core
    (; ᶜu, ᶜts, ᶜp) = p.precomputed
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3 # updated in set_implicit_precomputed_quantities!
    thermo_params = CAP.thermodynamics_params(p.params)

    if !isnothing(p.sfc_setup)
        SurfaceConditions.update_surface_conditions!(Y, p, float(t))
    end

    if turbconv_model isa AbstractEDMF
        @. p.precomputed.ᶜgradᵥ_θ_virt =
            ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ᶜts)))
        @. p.precomputed.ᶜgradᵥ_q_tot =
            ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ᶜts)))
        @. p.precomputed.ᶜgradᵥ_θ_liq_ice =
            ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts)))
    end

    # TODO: It is too slow to calculate mixing length at every timestep
    # if isnothing(turbconv_model)
    #     (; ᶜmixing_length) = p.precomputed
    #     compute_gm_mixing_length!(ᶜmixing_length, Y, p)
    # end

    if turbconv_model isa PrognosticEDMFX
        set_prognostic_edmf_precomputed_quantities_bottom_bc!(Y, p, t)
        set_prognostic_edmf_precomputed_quantities_explicit_closures!(Y, p, t)
        set_prognostic_edmf_precomputed_quantities_precipitation!(
            Y,
            p,
            p.atmos.precip_model,
        )
    end
    if turbconv_model isa DiagnosticEDMFX
        set_diagnostic_edmf_precomputed_quantities_bottom_bc!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_do_integral!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_top_bc!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_env_closures!(Y, p, t)
        set_diagnostic_edmf_precomputed_quantities_env_precipitation!(
            Y,
            p,
            t,
            p.atmos.precip_model,
        )
    end
    if turbconv_model isa EDOnlyEDMFX
        set_diagnostic_edmf_precomputed_quantities_env_closures!(Y, p, t)
        # TODO do I need env precipitation/cloud formation here?
    end

    set_precipitation_velocities!(
        Y,
        p,
        p.atmos.moisture_model,
        p.atmos.precip_model,
    )
    # Needs to be done after edmf precipitation is computed in sub-domains
    set_precipitation_cache!(Y, p, p.atmos.precip_model, p.atmos.turbconv_model)
    set_precipitation_surface_fluxes!(Y, p, p.atmos.precip_model)

    if vert_diff isa DecayWithHeightDiffusion
        (; ᶜK_h) = p.precomputed
        @. ᶜK_h = $compute_eddy_diffusivity_coefficient(Y.c.ρ, vert_diff)
    elseif vert_diff isa VerticalDiffusion
        (; ᶜK_h) = p.precomputed
        @. ᶜK_h = $compute_eddy_diffusivity_coefficient(Y.c.uₕ, ᶜp, vert_diff)
    end

    # TODO
    if call_cloud_diagnostics_per_stage isa CallCloudDiagnosticsPerStage
        set_cloud_fraction!(Y, p, moisture_model, cloud_model)
    end

    if p.atmos.smagorinsky_lilly isa SmagorinskyLilly
        set_smagorinsky_lilly_precomputed_quantities!(Y, p)
    end

    return nothing
end

"""
    set_precomputed_quantities!(Y, p, t)

Updates all precomputed quantities based on the current state `Y`.
"""
function set_precomputed_quantities!(Y, p, t)
    set_implicit_precomputed_quantities!(Y, p, t)
    set_explicit_precomputed_quantities!(Y, p, t)
end
