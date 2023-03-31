#####
##### Precomputed quantities
#####

import Thermodynamics as TD
import ClimaCore: Geometry, Spaces, Fields

"""
    set_boundary_velocity!(Y, turbconv_model)

Modifies `Y.f.w` so that `ᶠu³` is 0 at the surface. If `turbconv_model` is
EDMFX, also modifies `Y.f.sgsʲs` so that all of the subdomain's values of `w`
are the same as `Y.f.w` at the surface.
"""
function set_boundary_velocity!(Y, turbconv_model)
    uₕ_surface_data = Fields.level(Fields.field_values(Y.c.uₕ), 1)
    uₕ_surface_geom = Fields.level(Spaces.local_geometry_data(axes(Y.c.uₕ)), 1)
    uᵥ_surface_data = Fields.level(Fields.field_values(Y.f.w), 1)
    uᵥ_surface_geom = Fields.level(Spaces.local_geometry_data(axes(Y.f.w)), 1)
    @. uᵥ_surface_data = Geometry.Covariant3Vector(
        -Geometry.contravariant3(uₕ_surface_data, uₕ_surface_geom) /
        Geometry.contravariant3(one(uᵥ_surface_data), uᵥ_surface_geom),
    )
    if turbconv_model isa EDMFX
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            Fields.level(Y.f.sgsʲs.:($j).w, half) .= Fields.level(Y.f.w, half)
        end
    end
end

# TODO: Ask Tapio whether we should use this version instead. We need to account
# for the fact that Fields.level(ᶠuₕ, half) != Fields.level(Y.c.uₕ, 1). If we do
# use this version, we will probably want to make ᶠuₕ a precomputed quantity.
function set_boundary_velocity!(Y, ᶠuₕ, turbconv_model)
    w_sfc = Fields.level(Y.f.w, half)
    uₕ_sfc = Fields.level(ᶠuₕ, half)
    lg_sfc = Fields.local_geometry_field(w_sfc)
    @. w_sfc = Geometry.Covariant3Vector(
        -Geometry.contravariant3(uₕ_sfc, lg_sfc) /
        Geometry.contravariant3(one(w_sfc), lg_sfc),
    )
    for j in 1:n_mass_flux_subdomains(turbconv_model)
        wʲ_sfc = Fields.level(Y.f.sgsʲs.:($j).w, half)
        @. wʲ_sfc = w_sfc
    end
end

"""
    precomputed_quantities(atmos, center_space, face_space)

Allocates and returns the precomputed quantities:
    - `ᶜu`: the covariant velocity on cell centers
    - `ᶠu³`: the third component of contravariant velocity on cell faces
    - `ᶜK`: the kinetic energy on cell centers
    - `ᶜts`: the thermodynamic state on cell centers
    - `ᶜp`: the air pressure on cell centers

If the `turbconv_model` is EDMFX, there also two SGS versions of each quantity:
    - `_⁰`: the value for the environment
    - `_ʲs`: a tuple of the values for the mass-flux subdomains
In addition, there are several other SGS quantities:
    - `ᶜρ⁰`: the air density of the environment on cell centers
    - `ᶜρa⁰`: the area-weighted air density of the environment on cell centers
    - `ᶠw⁰`: the vertical component of the covariant velocity of the environment
        on cell faces
    - `ᶜρʲs`: a tuple of the air densities of the mass-flux subdomains on cell
        centers

TODO: Rename `ᶜK` to `ᶜκ`, and rename `ᶠw` to `ᶠuᵥ`.
"""
function precomputed_quantities(atmos, center_space, face_space)
    C123 = Geometry.Covariant123Vector
    CT3 = Geometry.Contravariant3Vector
    FT = Spaces.undertype(center_space)
    TST = thermo_state_type(atmos.moisture_model, FT)
    precomputed_sgs_quantities = if atmos.turbconv_model isa EDMFX
        n = n_mass_flux_subdomains(atmos.turbconv_model)
        (;
            ᶜρa⁰ = Fields.Field(FT, center_space),
            ᶠw⁰ = Fields.Field(Geometry.Covariant3Vector{FT}, face_space),
            ᶜu⁰ = Fields.Field(C123{FT}, center_space),
            ᶠu³⁰ = Fields.Field(CT3{FT}, face_space),
            ᶜK⁰ = Fields.Field(FT, center_space),
            ᶜts⁰ = Fields.Field(TST, center_space),
            ᶜρ⁰ = Fields.Field(FT, center_space),
            ᶜuʲs = Fields.Field(NTuple{n, C123{FT}}, center_space),
            ᶠu³ʲs = Fields.Field(NTuple{n, CT3{FT}}, face_space),
            ᶜKʲs = Fields.Field(NTuple{n, FT}, center_space),
            ᶜtsʲs = Fields.Field(NTuple{n, TST}, center_space),
            ᶜρʲs = Fields.Field(NTuple{n, FT}, center_space),
        )
    else
        (;)
    end
    return (;
        ᶜu = Fields.Field(C123{FT}, center_space),
        ᶠu³ = Fields.Field(CT3{FT}, face_space),
        ᶜK = Fields.Field(FT, center_space),
        ᶜts = Fields.Field(TST, center_space),
        ᶜp = Fields.Field(FT, center_space),
        precomputed_sgs_quantities...,
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
    ε = eps(zero(ρa)) # the smallest number bigger than 0 of the same type as ρa
    sgs_weight = 1 / (1 + a_min^2 * ρ^2 / ρa^2) # TODO: Make this exponential.
    return sgs_weight * ρaχ / (ρa + ε) + (1 - sgs_weight) * ρχ / ρ
end

# The arguments ᶜu, ᶠu³, ᶜK, and ᶠw can be replaced with their SGS versions when
# calling this function.
function set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠw, ᶜYc)
    ᶜinterp = Operators.InterpolateF2C()
    ᶠwinterp = Operators.WeightedInterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    ᶜJ = Fields.local_geometry_field(axes(ᶜYc)).J
    Fields.bycolumn(axes(ᶜYc)) do colidx
        C123 = Geometry.Covariant123Vector
        CT123 = Geometry.Contravariant123Vector
        @. ᶜu[colidx] = C123(ᶜYc.uₕ[colidx]) + ᶜinterp(C123(ᶠw[colidx]))
        @. ᶠu³[colidx] = Geometry.project(
            Geometry.Contravariant3Axis(),
            ᶠwinterp(ᶜYc.ρ[colidx] * ᶜJ[colidx], CT123(ᶜYc.uₕ[colidx])) +
            CT123(ᶠw[colidx]),
        )
        compute_kinetic!(ᶜK[colidx], ᶜYc.uₕ[colidx], ᶠw[colidx])
    end
end

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

    # TODO: This should probably be moved to dss! (i.e., enforce_constraints!).
    set_boundary_velocity!(Y, turbconv_model)

    (; ᶜu, ᶠu³, ᶜK, ᶜts, ᶜp, ᶜΦ) = p
    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y.f.w, Y.c)
    if turbconv_model isa EDMFX
        (; ᶜρa⁰, ᶠw⁰) = p
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
        (; ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶜts⁰, ᶜρ⁰, ᶜuʲs, ᶠu³ʲs, ᶜKʲs, ᶜtsʲs, ᶜρʲs) = p
        set_velocity_quantities!(ᶜu⁰, ᶠu³⁰, ᶜK⁰, ᶠw⁰, Y.c)
        @. ᶜK⁰ += divide_by_ρa(Y.c.sgs⁰.ρatke, ᶜρa⁰, 0, Y.c.ρ, a_min)
        @. ᶜts⁰ = ts⁰(Y.c, ᶜρa⁰, ᶜK⁰, ᶜp, ᶜΦ, a_min, thermo_args...)
        @. ᶜρ⁰ = TD.air_density(thermo_params, ᶜts⁰)
        for j in 1:n_mass_flux_subdomains(turbconv_model)
            set_velocity_quantities!(
                ᶜuʲs.:($j),
                ᶠu³ʲs.:($j),
                ᶜKʲs.:($j),
                Y.f.sgsʲs.:($j).w,
                Y.c,
            )
            @. ᶜtsʲs[j] =
                tsʲ(Y.c.sgsʲs[j], Y.c, ᶜKʲs[j], ᶜp, ᶜΦ, a_min, thermo_args...)
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

    @assert turbconv_model isa EDMFX
    (; a_min) = turbconv_model
    (ᶠw⁺, ᶜu⁺, ᶠu³⁺, ᶜK⁺) = similar.((p.ᶠw⁰, p.ᶜu⁰, p.ᶠu³⁰, p.ᶜK⁰))

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
    set_velocity_quantities!(ᶜu⁺, ᶠu³⁺, ᶜK⁺, ᶠw⁺, Y.c)

    sgs⁺(Yc) = map(+, Yc.sgsʲs...)
    ᶜsgs⁺ = @. sgs⁺(Y.c)
    sgs⁰(Yc, sgs⁺) = merge(map(-, times_a(Yc, 1), sgs⁺), Yc.sgs⁰)
    ᶜsgs⁰ = @. sgs⁰(Y.c, ᶜsgs⁺)
    ᶜts⁺ = @. tsʲ(ᶜsgs⁺, Y.c, ᶜK⁺, ᶜp, ᶜΦ, a_min, thermo_args...)

    return (; ᶜsgs⁰, ᶜsgs⁺, ᶜu⁺, ᶜts⁺)
end
