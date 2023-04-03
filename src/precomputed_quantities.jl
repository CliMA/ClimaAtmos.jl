#####
##### Precomputed quantities
#####

import Thermodynamics as TD
import ClimaCore: Geometry, Spaces, Fields

"""
    precomputed_quantities(atmos, center_space, face_space)

Allocates and returns the precomputed quantities:
    - `ᶜu`: the covariant velocity on cell centers
    - `ᶠu³`: the third component of contravariant velocity on cell faces
    - `ᶜK`: the kinetic energy on cell centers
    - `ᶜts`: the thermodynamic state on cell centers
    - `ᶜp`: the air pressure on cell centers

If the `turbconv_model` is EDMFX, there also two SGS versions of every quantity
except for `ᶜp` (we assume that `ᶜp⁰ = ᶜpʲ = ᶜp`):
    - `_⁰`: the value for the environment
    - `_ʲs`: a tuple of the values for the mass-flux subdomains
In addition, there are several other SGS quantities for the EDMFX model:
    - `ᶜρ⁰`: the air density of the environment on cell centers
    - `ᶜρa⁰`: the area-weighted air density of the environment on cell centers
    - `ᶠw⁰`: the vertical component of the covariant velocity of the environment
        on cell faces
    - `ᶜρʲs`: a tuple of the air densities of the mass-flux subdomains on cell
        centers

TODO: Rename `ᶜK` to `ᶜκ`, and rename `ᶠw` to `ᶠuᵥ`.
"""
function precomputed_quantities(atmos, center_space, face_space)
    C3 = Geometry.Covariant3Vector
    C123 = Geometry.Covariant123Vector
    CT3 = Geometry.Contravariant3Vector
    FT = Spaces.undertype(center_space)
    TST = thermo_state_type(atmos.moisture_model, FT)
    n = n_mass_flux_subdomains(atmos.turbconv_model)
    T_or_Nothing(::Type{T}) where {T} = n > 0 ? T : Nothing
    return (;
        ᶜu = Fields.Field(C123{FT}, center_space),
        ᶠu³ = Fields.Field(CT3{FT}, face_space),
        ᶜK = Fields.Field(FT, center_space),
        ᶜts = Fields.Field(TST, center_space),
        ᶜp = Fields.Field(FT, center_space),
        ᶜρa⁰ = Fields.Field(T_or_Nothing(FT), center_space),
        ᶠw⁰ = Fields.Field(T_or_Nothing(C3{FT}), face_space),
        ᶜu⁰ = Fields.Field(T_or_Nothing(C123{FT}), center_space),
        ᶠu³⁰ = Fields.Field(T_or_Nothing(CT3{FT}), face_space),
        ᶜK⁰ = Fields.Field(T_or_Nothing(FT), center_space),
        ᶜts⁰ = Fields.Field(T_or_Nothing(TST), center_space),
        ᶜρ⁰ = Fields.Field(T_or_Nothing(FT), center_space),
        ᶜuʲs = Fields.Field(T_or_Nothing(NTuple{n, C123{FT}}), center_space),
        ᶠu³ʲs = Fields.Field(T_or_Nothing(NTuple{n, CT3{FT}}), face_space),
        ᶜKʲs = Fields.Field(T_or_Nothing(NTuple{n, FT}), center_space),
        ᶜtsʲs = Fields.Field(T_or_Nothing(NTuple{n, TST}), center_space),
        ᶜρʲs = Fields.Field(T_or_Nothing(NTuple{n, FT}), center_space),
    )
end

"""
    divide_by_ρa(ρaχ, ρa, ρχ, ρ, a_min)

Computes `ρaχ / ρa`, regularizing the result to avoid issues when `a` is small.
This is done by performing a linear interpolation from `ρaχ / ρa` to `ρχ / ρ`,
interpolating closer to `ρχ / ρ` when `a` is small. When `a == 0`, the result is
exactly `ρχ / ρ` (a tiny value of `ε` is added to the denominator of `ρaχ / ρa`
in order to avoid returning `NaN` when `a == 0`, since `0 * Inf` is `NaN`).
"""
function divide_by_ρa(ρaχ, ρa, ρχ, ρ, a_min)
    ε = eps(zero(ρa)) # This is the smallest positive number we can add to ρa.
    sgs_weight = 1 / (1 + a_min^2 * ρ^2 / ρa^2) # TODO: Make this exponential.
    return sgs_weight * ρaχ / (ρa + ε) + (1 - sgs_weight) * ρχ / ρ
end

# Interpolates the third contravariant component of Y.c.uₕ to cell faces.
function set_ᶠuₕ³!(ᶠuₕ³, Y)
    ᶠwinterp = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    Fields.bycolumn(axes(Y.c)) do colidx
        CT3 = Geometry.Contravariant3Vector
        ᶜJ = Fields.local_geometry_field(Y.c).J
        @. ᶠuₕ³[colidx] =
            ᶠwinterp(Y.c.ρ[colidx] * ᶜJ[colidx], CT3(Y.c.uₕ[colidx]))
    end
end

"""
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)

Modifies `Y.f.w` so that `ᶠu³` is 0 at the surface. If `turbconv_model` is
EDMFX, also modifies `Y.f.sgsʲs` so that each `wʲ` is equal to `w` at the
surface.
"""
function set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)
    sfc_w₃ = Fields.level(Y.f.w.components.data.:1, half)
    sfc_uₕ³ = Fields.level(ᶠuₕ³.components.data.:1, half)
    sfc_g = Fields.local_geometry_field(sfc_w₃).gⁱʲ.components.data
    end_index = fieldcount(eltype(sfc_g)) # This will be 4 in 2D and 9 in 3D.
    sfc_g³³ = sfc_g.:($end_index) # For both 2D and 3D spaces, g³³ = g[end].
    @. sfc_w₃ = -sfc_uₕ³ / sfc_g³³ # u³ = uₕ³ + w³ = uₕ³ + w₃ * g³³
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        sfc_w₃ʲ = Fields.level(Y.f.sgsʲs.:($j).w.components.data.:1, half)
        @. sfc_w₃ʲ = sfc_w₃
    end
end

# This is used to set the grid-scale velocity quantities ᶜu, ᶠu³, ᶜK based on
# ᶠw, and it is also used to set the SGS quantities based on ᶠw⁰ and ᶠwʲ.
function set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠw, ᶜuₕ, ᶠuₕ³)
    ᶜinterp = Operators.InterpolateF2C()
    Fields.bycolumn(axes(ᶜu)) do colidx
        C123 = Geometry.Covariant123Vector
        CT3 = Geometry.Contravariant3Vector
        @. ᶜu[colidx] = C123(ᶜuₕ[colidx]) + ᶜinterp(C123(ᶠw[colidx]))
        @. ᶠu³[colidx] = ᶠuₕ³[colidx] + CT3(ᶠw[colidx])
        compute_kinetic!(ᶜK[colidx], ᶜuₕ[colidx], ᶠw[colidx])
    end
end

# Putting the anonymous function sgsʲ -> sgsʲ.<symbol> directly into broadcast
# expressions causes allocations, so this function should be used instead.
get_ʲs(sgsʲs, ::Val{symbol}) where {symbol} =
    map(sgsʲ -> sgsʲ.:($symbol), sgsʲs)

function set_ᶠw⁰!(ᶠw⁰, Y, ᶜρa⁰, a_min)
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    w⁰(ρa⁰, ρaʲs, wʲs, ρ, w, a_min) =
        divide_by_ρa(ρ * w - sum(ρaʲs .* wʲs), ρa⁰, ρ * w, ρ, a_min)
    Fields.bycolumn(axes(Y.c)) do colidx
        @. ᶠw⁰[colidx] = w⁰(
            ᶠinterp(ᶜρa⁰[colidx]),
            ᶠinterp(get_ʲs(Y.c.sgsʲs[colidx], Val{:ρa}())),
            get_ʲs(Y.f.sgsʲs[colidx], Val{:w}()),
            ᶠinterp(Y.c.ρ[colidx]),
            Y.f.w[colidx],
            a_min,
        )
    end
end

# TODO: Ask Tapio whether this function is correct.
function add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠw⁰, turbconv_model)
    ᶜinterp = Operators.InterpolateF2C()
    function do_col!(ᶜK, Yc, Yf, ᶜρa⁰, ᶠw⁰)
        CT3 = Geometry.Contravariant3Vector
        @. ᶜK += ᶜρa⁰ * ᶜinterp(dot(ᶠw⁰ - Yf.w, CT3(ᶠw⁰ - Yf.w))) / 2 / Yc.ρ
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            ᶜρaʲ = Yc.sgsʲs[j].ρa
            ᶠwʲ = Yf.sgsʲs[j].w
            @. ᶜK += ᶜρaʲ * ᶜinterp(dot(ᶠwʲ - Yf.w, CT3(ᶠwʲ - Yf.w))) / 2 / Yc.ρ
        end
    end
    Fields.bycolumn(axes(Y.c)) do colidx
        do_col!(ᶜK[colidx], Y.c[colidx], Y.f[colidx], ᶜρa⁰[colidx], ᶠw⁰[colidx])
    end
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
        TD.PhaseEquil_ρeq(thermo_params, ρ, e_int, q_tot)
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

function ts(Yc, K, Φ, thermo_params, energy_form, moisture_model)
    energy_var = if energy_form isa PotentialTemperature
        (; θ = Yc.ρθ / Yc.ρ)
    elseif energy_form isa TotalEnergy
        e_tot = Yc.ρe_tot / Yc.ρ
        (; e_int = e_tot - K - Φ)
    end
    moisture_var = if moisture_model isa DryModel
        (;)
    else
        q_tot = Yc.ρq_tot / Yc.ρ
        if moisture_model isa EquilMoistModel
            (; q_tot)
        elseif moisture_model isa NonEquilMoistModel
            q_liq = Yc.ρq_liq / Yc.ρ
            q_ice = Yc.ρq_ice / Yc.ρ
            (; q_pt = TD.PhasePartition(q_tot, q_liq, q_ice))
        end
    end
    return thermo_state(thermo_params; ρ = Yc.ρ, energy_var..., moisture_var...)
end

function tsʲ(
    sgsʲ,
    Yc,
    Kʲ,
    p,
    Φ,
    a_min,
    thermo_params,
    energy_form,
    moisture_model,
)
    energy_var = if energy_form isa PotentialTemperature
        (; θ = divide_by_ρa(sgsʲ.ρaθ, sgsʲ.ρa, Yc.ρθ, Yc.ρ, a_min))
    elseif energy_form isa TotalEnergy
        e_totʲ = divide_by_ρa(sgsʲ.ρae_tot, sgsʲ.ρa, Yc.ρe_tot, Yc.ρ, a_min)
        (; e_int = e_totʲ - Kʲ - Φ)
    end
    moisture_var = if moisture_model isa DryModel
        (;)
    else
        q_totʲ = divide_by_ρa(sgsʲ.ρaq_tot, sgsʲ.ρa, Yc.ρq_tot, Yc.ρ, a_min)
        if moisture_model isa EquilMoistModel
            (; q_tot = q_totʲ)
        elseif moisture_model isa NonEquilMoistModel
            q_liqʲ = divide_by_ρa(sgsʲ.ρaq_liq, sgsʲ.ρa, Yc.ρq_liq, Yc.ρ, a_min)
            q_iceʲ =
                divide_by_ρa(sgsʲ.ρaq_ice, sgsʲ.ρa, Yc.ρq_ice, Yc.ρ, a_min)
            (; q_pt = TD.PhasePartition(q_totʲ, q_liqʲ, q_iceʲ))
        end
    end
    return thermo_state(thermo_params; p, energy_var..., moisture_var...)
end

function ts⁰(
    Yc,
    ρa⁰,
    K⁰,
    p,
    Φ,
    a_min,
    thermo_params,
    energy_form,
    moisture_model,
)
    energy_var = if energy_form isa PotentialTemperature
        ρθ⁰ = Yc.ρθ - sum(get_ʲs(Yc.sgsʲs, Val{:ρaθ}()))
        (; θ = divide_by_ρa(ρθ⁰, ρa⁰, Yc.ρθ, Yc.ρ, a_min))
    elseif energy_form isa TotalEnergy
        ρae_tot⁰ = Yc.ρe_tot - sum(get_ʲs(Yc.sgsʲs, Val{:ρae_tot}()))
        e_tot⁰ = divide_by_ρa(ρae_tot⁰, ρa⁰, Yc.ρe_tot, Yc.ρ, a_min)
        (; e_int = e_tot⁰ - K⁰ - Φ)
    end
    moisture_var = if moisture_model isa DryModel
        (;)
    else
        ρaq_tot⁰ = Yc.ρq_tot - sum(get_ʲs(Yc.sgsʲs, Val{:ρaq_tot}()))
        q_tot⁰ = divide_by_ρa(ρaq_tot⁰, ρa⁰, Yc.ρq_tot, Yc.ρ, a_min)
        if moisture_model isa EquilMoistModel
            (; q_tot = q_tot⁰)
        elseif moisture_model isa NonEquilMoistModel
            ρaq_liq⁰ = Yc.ρq_liq - sum(get_ʲs(Yc.sgsʲs, Val{:ρaq_liq}()))
            ρaq_ice⁰ = Yc.ρq_ice - sum(get_ʲs(Yc.sgsʲs, Val{:ρaq_ice}()))
            q_liq⁰ = divide_by_ρa(ρaq_liq⁰, ρa⁰, Yc.ρq_liq, Yc.ρ, a_min)
            q_ice⁰ = divide_by_ρa(ρaq_ice⁰, ρa⁰, Yc.ρq_ice, Yc.ρ, a_min)
            (; q_pt = TD.PhasePartition(q_tot⁰, q_liq⁰, q_ice⁰))
        end
    end
    return thermo_state(thermo_params; p, energy_var..., moisture_var...)
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
function set_precomputed_quantities!(Y, p, t)
    (; energy_form, moisture_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    thermo_args = (thermo_params, energy_form, moisture_model)
    (; ᶜu, ᶠu³, ᶜK, ᶜts, ᶜp, ᶜΦ) = p
    (; ᶜρa⁰, ᶠw⁰, ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜts⁰, ᶜρ⁰, ᶜuʲs, ᶠu³ʲs, ᶜKʲs, ᶜtsʲs, ᶜρʲs) = p
    ᶠuₕ³ = p.ᶠtemp_CT3

    set_ᶠuₕ³!(ᶠuₕ³, Y)

    # TODO: We might want to move this to dss! (and rename dss! to something
    # like enforce_constraints!).
    set_velocity_at_surface!(Y, ᶠuₕ³, turbconv_model)

    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y.f.w, Y.c.uₕ, ᶠuₕ³)
    if turbconv_model isa EDMFX
        (; a_min) = turbconv_model
        @. ᶜρa⁰ = Y.c.ρ - sum(get_ʲs(Y.c.sgsʲs, Val{:ρa}()))
        set_ᶠw⁰!(ᶠw⁰, Y, ᶜρa⁰, a_min)

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
        add_sgs_ᶜK!(ᶜK, Y, ᶜρa⁰, ᶠw⁰, turbconv_model)
        @. ᶜK += Y.c.sgs⁰.ρatke / Y.c.ρ
    end
    @. ᶜts = ts(Y.c, ᶜK, ᶜΦ, thermo_args...)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)

    if turbconv_model isa EDMFX
        set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠw⁰, Y.c.uₕ, ᶠuₕ³)
        @. ᶜK⁰ += divide_by_ρa(Y.c.sgs⁰.ρatke, ᶜρa⁰, 0, Y.c.ρ, a_min)
        @. ᶜts⁰ = ts⁰(Y.c, ᶜρa⁰, ᶜK⁰, ᶜp, ᶜΦ, a_min, thermo_args...)
        @. ᶜρ⁰ = TD.air_density(thermo_params, ᶜts⁰)
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            ᶜuʲ = ᶜuʲs.:($j)
            ᶠu³ʲ = ᶠu³ʲs.:($j)
            ᶜKʲ = ᶜKʲs.:($j)
            ᶠwʲ = Y.f.sgsʲs.:($j).w
            set_velocity_quantities!(ᶜuʲ, ᶠu³ʲ, ᶜKʲ, ᶠwʲ, Y.c.uₕ, ᶠuₕ³)
            @. ᶜtsʲs[j] =
                tsʲ(Y.c.sgsʲs[j], Y.c, ᶜKʲ, ᶜp, ᶜΦ, a_min, thermo_args...)
            @. ᶜρʲs[j] = TD.air_density(thermo_params, ᶜtsʲs[j])
        end
    end

    return nothing
end

"""
    diagnostic_edmfx_quantities(Y, p, t)

Allocates, sets, and returns `ᶜsgs⁰`, `ᶜsgs⁺`, `ᶜu⁺`, and `ᶜts⁺` in a manner
that is consistent with `set_precomputed_quantities!`. This function should only
be used to generate diagnostics for EDMFX, and it should only be called after
`set_precomputed_quantities!` has been called.
"""
function diagnostic_edmfx_quantities(Y, p, t)
    ᶠinterp = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    (; energy_form, moisture_model, turbconv_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    thermo_args = (thermo_params, energy_form, moisture_model)
    (; ᶜp, ᶜΦ) = p
    ᶠuₕ³ = p.ᶠtemp_CT3
    (ᶠw⁺, ᶜu⁺, ᶠu³⁺, ᶜK⁺) = similar.((p.ᶠw⁰, p.ᶜu⁰, p.ᶠu³⁰, p.ᶜK⁰))

    @assert turbconv_model isa EDMFX
    (; a_min) = turbconv_model

    set_ᶠuₕ³!(ᶠuₕ³, Y)
    w⁺(ρaʲs, wʲs, ρ, w) =
        divide_by_ρa(sum(ρaʲs .* wʲs), sum(ρaʲs), ρ * w, ρ, a_min)
    Fields.bycolumn(axes(Y.c)) do colidx
        @. ᶠw⁺[colidx] = w⁺(
            ᶠinterp(get_ʲs(Y.c.sgsʲs[colidx], Val{:ρa}())),
            get_ʲs(Y.f.sgsʲs[colidx], Val{:w}()),
            ᶠinterp(Y.c.ρ[colidx]),
            Y.f.w[colidx],
        )
    end
    set_velocity_quantities!(ᶜu⁺, ᶠu³⁺, ᶜK⁺, ᶠw⁺, Y.c.uₕ, ᶠuₕ³)

    sgs⁺(Yc) = map(+, Yc.sgsʲs...)
    ᶜsgs⁺ = @. sgs⁺(Y.c)
    sgs⁰(Yc, sgs⁺) = merge(map(-, times_a(Yc, 1), sgs⁺), Yc.sgs⁰)
    ᶜsgs⁰ = @. sgs⁰(Y.c, ᶜsgs⁺)
    ᶜts⁺ = @. tsʲ(ᶜsgs⁺, Y.c, ᶜK⁺, ᶜp, ᶜΦ, a_min, thermo_args...)

    return (; ᶜsgs⁰, ᶜsgs⁺, ᶜu⁺, ᶜts⁺)
end
