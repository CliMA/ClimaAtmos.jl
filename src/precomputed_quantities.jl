#####
##### Precomputed quantities
#####

import Thermodynamics as TD
import ClimaCore: Geometry, Spaces, Fields

"""
    set_boundary_velocity!(Y)

Modifies `Y.f.w` so that `ᶠu³` is 0 at the surface.
"""
function set_boundary_velocity!(Y)
    uₕ_surface_data = Fields.level(Fields.field_values(Y.c.uₕ), 1)
    uₕ_surface_geom = Fields.level(Spaces.local_geometry_data(axes(Y.c.uₕ)), 1)
    uᵥ_surface_data = Fields.level(Fields.field_values(Y.f.w), 1)
    uᵥ_surface_geom = Fields.level(Spaces.local_geometry_data(axes(Y.f.w)), 1)
    @. uᵥ_surface_data = Geometry.Covariant3Vector(
        -Geometry.contravariant3(uₕ_surface_data, uₕ_surface_geom) /
        Geometry.contravariant3(one(uᵥ_surface_data), uᵥ_surface_geom),
    )
end

"""
    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, ᶠw, ᶜYc)

Uses `ᶠw` and `ᶜYc` to set the values of `ᶜu`, `ᶠu³`, and `ᶜK`, assuming that
the horizontal velocity is `ᶜYc.uₕ` and that it should be interpolated to cell
faces using `ᶜYc.ρ * ᶜJ` in the weighted interpolation.
"""
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

"""
    thermo_state(thermo_params; ρ, p, θ, e_int, q_tot, q_pt)

Constructs a thermodynamic state based on the supplied keyword arguments.
"""
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

function set_precomputed_quantities!(Y, p, t)
    (; energy_form, moisture_model) = p.atmos
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p

    # TODO: This should probably be moved to dss! (i.e., enforce_constraints!).
    set_boundary_velocity!(Y)

    (; ᶜu, ᶠu³, ᶜK, ᶜts, ᶜp) = p
    set_velocity_quantities!(ᶜu, ᶠu³, ᶜK, Y.f.w, Y.c)
    @. ᶜts = ts(Y.c, ᶜK, ᶜΦ, thermo_params, energy_form, moisture_model)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)
end
