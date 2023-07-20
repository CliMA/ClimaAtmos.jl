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

If the `energy_form` is TotalEnergy, there is an additional quantity:
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
    @assert (
        !(atmos.moisture_model isa DryModel) &&
        atmos.energy_form isa TotalEnergy
    ) || !(atmos.turbconv_model isa DiagnosticEDMFX)
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
        (
            atmos.energy_form isa TotalEnergy ?
            (; ᶜh_tot = similar(Y.c, FT)) : (;)
        )...,
        sfc_conditions = Fields.Field(SCT, Spaces.level(axes(Y.f), half)),
    )
    sgs_quantities =
        atmos.turbconv_model isa EDMFX ?
        (;
            ᶜspecific⁰ = specific_full_sgs⁰.(Y.c, atmos.turbconv_model),
            ᶜρa⁰ = similar(Y.c, FT),
            ᶠu₃⁰ = similar(Y.f, C3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜK⁰ = similar(Y.c, FT),
            ᶜts⁰ = similar(Y.c, TST),
            ᶜρ⁰ = similar(Y.c, FT),
            ᶜlinear_buoygrad = similar(Y.c, FT),
            ᶜshear² = similar(Y.c, FT),
            ᶜmixing_length = similar(Y.c, FT),
            ᶜspecificʲs = specific_sgsʲs.(Y.c, atmos.turbconv_model),
            ᶜuʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶠu³ʲs = similar(Y.f, NTuple{n, CT3{FT}}),
            ᶜKʲs = similar(Y.c, NTuple{n, FT}),
            ᶜtsʲs = similar(Y.c, NTuple{n, TST}),
            ᶜρʲs = similar(Y.c, NTuple{n, FT}),
            ᶜentr_detrʲs = similar(
                Y.c,
                NTuple{n, NamedTuple{(:entr, :detr), NTuple{2, FT}}},
            ),
            (
                atmos.energy_form isa TotalEnergy ?
                (;
                    ᶜh_totʲs = similar(Y.c, NTuple{n, FT}),
                    ᶜh_tot⁰ = similar(Y.c, FT),
                ) : (;)
            )...,
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
            ᶜentr_detrʲs = similar(
                Y.c,
                NTuple{n, NamedTuple{(:entr, :detr), NTuple{2, FT}}},
            ),
            ᶜρa⁰ = similar(Y.c, FT),
            ᶠu³⁰ = similar(Y.f, CT3{FT}),
            ᶜu⁰ = similar(Y.c, C123{FT}),
            ᶜtke⁰ = similar(Y.c, FT),
            ᶜlinear_buoygrad = similar(Y.c, FT),
            ᶜshear² = similar(Y.c, FT),
            ᶜmixing_length = similar(Y.c, FT),
        ) : (;)
    return (; gs_quantities..., sgs_quantities..., diagnostic_sgs_quantities...)
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
    if turbconv_model isa EDMFX
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            sfc_u₃ʲ = Fields.level(Y.f.sgsʲs.:($j).u₃.components.data.:1, half)
            @. sfc_u₃ʲ = sfc_u₃
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
        TD.PhaseNonEquil_ρeq(thermo_params, ρ, e_int, q_pt)
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

function thermo_vars(energy_form, moisture_model, specific, K, Φ)
    energy_var = if energy_form isa PotentialTemperature
        (; specific.θ)
    elseif energy_form isa TotalEnergy
        (; e_int = specific.e_tot - K - Φ)
    end
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

ts_gs(thermo_params, energy_form, moisture_model, specific, K, Φ, ρ) =
    thermo_state(
        thermo_params;
        thermo_vars(energy_form, moisture_model, specific, K, Φ)...,
        ρ,
    )

ts_sgs(thermo_params, energy_form, moisture_model, specific, K, Φ, p) =
    thermo_state(
        thermo_params;
        thermo_vars(energy_form, moisture_model, specific, K, Φ)...,
        p,
    )

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
function set_precomputed_quantities!(Y, p, t)
    (; energy_form, moisture_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    n = n_mass_flux_subdomains(turbconv_model)
    thermo_args = (thermo_params, energy_form, moisture_model)
    (; ᶜspecific, ᶜu, ᶠu³, ᶜK, ᶜts, ᶜp, ᶜΦ) = p
    ᶠuₕ³ = p.ᶠtemp_CT3

    @. ᶜspecific = specific_gs(Y.c)
    set_ᶠuₕ³!(ᶠuₕ³, Y)

    # TODO: We might want to move this to dss! (and rename dss! to something
    # like enforce_constraints!).
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)

    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y.f.u₃, Y.c.uₕ, ᶠuₕ³)
    if n > 0
        # TODO: In the following increments to ᶜK, we actually need to add
        # quantities of the form ᶜρaχ⁰ / ᶜρ⁰ and ᶜρaχʲ / ᶜρʲ to ᶜK, rather than
        # quantities of the form ᶜρaχ⁰ / ᶜρ and ᶜρaχʲ / ᶜρ. However, we cannot
        # compute ᶜρ⁰ and ᶜρʲ without first computing ᶜts⁰ and ᶜtsʲ, both of
        # which depend on the value of ᶜp, which in turn depends on ᶜts. Since
        # ᶜts depends on ᶜK (at least when the energy_form is TotalEnergy), this
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

    if energy_form isa TotalEnergy
        (; ᶜh_tot) = p
        @. ᶜh_tot =
            TD.total_specific_enthalpy(thermo_params, ᶜts, ᶜspecific.e_tot)
    end

    SurfaceConditions.update_surface_conditions!(Y, p, t)

    if turbconv_model isa EDMFX
        set_edmf_precomputed_quantities!(Y, p, ᶠuₕ³, t)
    end

    if turbconv_model isa DiagnosticEDMFX
        set_diagnostic_edmf_precomputed_quantities!(Y, p, t)
    end

    return nothing
end

"""
    output_sgs_quantities(Y, p, t)

Allocates, sets, and returns `ᶜspecific⁺`, `ᶠu₃⁺`, `ᶜu⁺`, `ᶠu³⁺`, `ᶜK⁺`, `ᶜts⁺`,
`ᶜa⁺`, and `ᶜa⁰` in a way that is consistent with `set_precomputed_quantities!`.
This function assumes that `set_precomputed_quantities!` has already been
called.
"""
function output_sgs_quantities(Y, p, t)
    (; energy_form, moisture_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    thermo_args = (thermo_params, energy_form, moisture_model)
    (; ᶜp, ᶜρa⁰, ᶜρ⁰, ᶜΦ) = p
    ᶠuₕ³ = p.ᶠtemp_CT3
    set_ᶠuₕ³!(ᶠuₕ³, Y)
    ᶜspecific⁺ = @. specific_sgs⁺(Y.c, turbconv_model)
    (ᶠu₃⁺, ᶜu⁺, ᶠu³⁺, ᶜK⁺) = similar.((p.ᶠu₃⁰, p.ᶜu⁰, p.ᶠu³⁰, p.ᶜK⁰))
    set_sgs_ᶠu₃!(u₃⁺, ᶠu₃⁺, Y, turbconv_model)
    set_velocity_quantities!(ᶜu⁺, ᶠu³⁺, ᶜK⁺, ᶠu₃⁺, Y.c.uₕ, ᶠuₕ³)
    ᶜts⁺ = @. ts_sgs(thermo_args..., ᶜspecific⁺, ᶜK⁺, ᶜΦ, ᶜp)
    ᶜa⁺ = @. ρa⁺(Y.c) / TD.air_density(thermo_params, ᶜts⁺)
    ᶜa⁰ = @. ᶜρa⁰ / ᶜρ⁰
    return (; ᶜspecific⁺, ᶠu₃⁺, ᶜu⁺, ᶠu³⁺, ᶜK⁺, ᶜts⁺, ᶜa⁺, ᶜa⁰)
end

"""
    output_diagnostic_sgs_quantities(Y, p, t)

Sets `ᶜu⁺`, `ᶠu³⁺`, `ᶜts⁺` and `ᶜa⁺` to be the same as the
values of the first updraft.
"""
function output_diagnostic_sgs_quantities(Y, p, t)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜρaʲs, ᶜtsʲs) = p
    ᶠu³⁺ = p.ᶠu³ʲs[1]
    ᶜu⁺ = @. (C123(Y.c.uₕ) + C123(ᶜinterp(ᶠu³⁺)))
    ᶜts⁺ = @. ᶜtsʲs[1]
    ᶜa⁺ = @. ᶜρaʲs[1] / TD.air_density(thermo_params, ᶜts⁺)
    return (; ᶜu⁺, ᶠu³⁺, ᶜts⁺, ᶜa⁺)
end
