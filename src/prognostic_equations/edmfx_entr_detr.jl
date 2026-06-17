#####
##### EDMF entrainment and detrainment parameterizations
#####

import Thermodynamics.Parameters as TDP

# Entrainment models

# Helper function for Pi-Group based models
"""
    calculate_pi_groups(
        elev_above_sfc, ref_H, ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ,
        ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰, ᶜtke
    )

Calculates non-dimensional Π-groups used in EDMF entrainment/detrainment models.

Arguments:

  - `elev_above_sfc`: Difference between cell-center height and surface elevation (ᶜz - z_sfc) [m].
  - `ref_H`: Reference pressure scale height [m].
  - `ᶜaʲ`: Updraft area fraction [-].
  - `ᶜwʲ`: Updraft physical vertical velocity [m/s].
  - `ᶜRHʲ`: Updraft relative humidity [-].
  - `ᶜbuoyʲ`: Updraft buoyancy [m/s²].
  - `ᶜw⁰`: Environment physical vertical velocity [m/s].
  - `ᶜRH⁰`: Environment relative humidity [-].
  - `ᶜbuoy⁰`: Environment buoyancy [m/s²].
  - `ᶜtke`: Turbulent kinetic energy [m²/s²].

Returns a tuple of five Π-groups: (Π₁, Π₂, Π₃, Π₄, Π₅).
Π₁: Related to buoyancy difference and velocity difference.
Π₂: Related to TKE and velocity difference.
Π₃: Related to updraft area.
Π₄: Related to relative humidity difference.
Π₅: Related to normalized elevation above surface.
Π₁ and Π₂ are scaled by empirical factors (100 and 2 respectively) to
become O(1) and then clipped to the range [-1, 1].
"""
function calculate_pi_groups(
    elev_above_sfc,
    ref_H,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
)
    FT = eltype(elev_above_sfc)
    vel_diff_sq = (ᶜwʲ - ᶜw⁰)^2 + eps(FT)
    Π₁_raw = elev_above_sfc * (ᶜbuoyʲ - ᶜbuoy⁰) / vel_diff_sq
    Π₂_raw = max(ᶜtke, 0) / vel_diff_sq
    Π₃ = sqrt(max(ᶜaʲ, 0))
    Π₄ = ᶜRHʲ - ᶜRH⁰
    Π₅ = elev_above_sfc / max(ref_H, eps(FT))

    Π₁ = min(max(Π₁_raw / FT(100), FT(-1)), FT(1))
    Π₂ = min(max(Π₂_raw / FT(2), FT(-1)), FT(1))
    return (Π₁, Π₂, Π₃, Π₄, Π₅)
end

"""
    entr_upper_area_limiter_factor(a, turbconv_params)

Return a multiplicative limiter for the entrainment rate based on the
current subdraft area `a`.

The factor smoothly damps entrainment as the area approaches the upper
bound `a_max` (preventing the area from being driven above `a_max`):

    (max(0, 1 - a / a_max))^max_area_limiter_power

The upper bound `a_max` is obtained from `turbconv_params` through
`CAP.max_area`, and `max_area_limiter_power` from `CAP.max_area_limiter_power`.

This is the entrainment counterpart of [`detr_lower_area_limiter_factor`](@ref),
which damps detrainment near the lower bound `a_min`. Together they keep
`a ∈ [a_min, a_max]` without requiring a comparison between `entr` and `detr`.
"""
@inline function entr_upper_area_limiter_factor(a::FT, turbconv_params) where {FT}
    a_max = CAP.max_area(turbconv_params)
    max_area_limiter_power = CAP.max_area_limiter_power(turbconv_params)
    a_safe = max(eps(FT), a)
    return (max(0, (1 - a_safe / a_max)))^max_area_limiter_power
end

"""
    detr_lower_area_limiter_factor(a, turbconv_params)

Return a multiplicative limiter for the detrainment rate based on the current
subdraft area `a`.

The factor smoothly damps detrainment as the area approaches the lower bound
`a_min` (preventing the area from being driven below `a_min`):

    (max(0, 1 - a_min / a))^min_area_limiter_power

The lower bound `a_min` is obtained from `turbconv_params` through
`CAP.min_area`, and `min_area_limiter_power` from `CAP.min_area_limiter_power`.

This is the detrainment counterpart of [`entr_upper_area_limiter_factor`](@ref),
which damps entrainment near the upper bound `a_max`. Together they keep
`a ∈ [a_min, a_max]` without requiring a comparison between `entr` and `detr`.
"""
@inline function detr_lower_area_limiter_factor(a::FT, turbconv_params) where {FT}
    a_min = CAP.min_area(turbconv_params)
    min_area_limiter_power = CAP.min_area_limiter_power(turbconv_params)
    a_safe = max(eps(FT), a)
    return (max(0, (1 - a_min / a_safe)))^min_area_limiter_power
end

"""
    detr_buoy_inv_time_scale(Δwʲ, Δbuoyʲ, detr_buoy_inv_tau_max)

Clipped inverse buoyancy time-scale [1/s] used by the buoyancy-driven
detrainment branch:

    τ⁻¹_buoy = min(detr_buoy_inv_tau_max,
                   |min(Δbuoyʲ, 0)| / max(eps, |Δwʲ|))

Only negative buoyancy contributes (positive buoyancy doesn't detrain), and
the rate is capped at `detr_buoy_inv_tau_max` so a vanishing `Δwʲ` doesn't
produce an unbounded rate.

Extracted as a helper so it can be reused by `detrainment_rate` (the
explicit detrainment closure) and the implicit ρa solve (where the same
buoyancy-detrainment piece appears in the `(ε − δ)` denominator).
"""
@inline function detr_buoy_inv_time_scale(Δwʲ, Δbuoyʲ, detr_buoy_inv_tau_max)
    FT = typeof(Δwʲ)
    return min(
        detr_buoy_inv_tau_max,
        abs(min(Δbuoyʲ, FT(0))) / max(eps(FT), abs(Δwʲ)),
    )
end

"""
    compute_entrainment(ᶜentr_vel_scale, ᶜarea_bounding_entr_detr, ᶜwʲ)

Total entrainment rate [1/s] as the sum of a velocity-proportional term
and the positive part of the signed area-bounding rate:

    entr = entr_vel_scale * |wʲ| + max(0, area_bounding_entr_detr)

`entr_vel_scale` [1/m] is precomputed by `entrainment_velocity_scale`, and
`area_bounding_entr_detr` [1/s] is the signed rate produced by
[`area_bounding_entr_detr`](@ref) (positive ⇒ this entrainment branch,
negative ⇒ the detrainment branch in `compute_detrainment`).
`ᶜwʲ` is the physical updraft vertical velocity [m/s].
"""
compute_entrainment(
    ᶜentr_vel_scale,
    ᶜarea_bounding_entr_detr,
    ᶜwʲ,
) =
    ᶜentr_vel_scale * abs(ᶜwʲ) +
    max(zero(ᶜarea_bounding_entr_detr), ᶜarea_bounding_entr_detr)

"""
    entrainment_velocity_scale(
        thermo_params, turbconv_params, ᶜz, z_sfc, ᶜp, ᶜρ,
        ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ, ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰, ᶜtke,
        model_option::AbstractEntrainmentModel,
    )

Velocity-scaling prefactor [1/m] for the model-specific entrainment rate.
The total entrainment rate [1/s] is assembled by `compute_entrainment` as

    entr = entr_vel_scale * abs(wʲ) + max(0, area_bounding_entr_detr)

where the second term comes from [`area_bounding_entr_detr`](@ref) (its
positive branch). `model_option` dispatches to different entrainment models,
such as `NoEntrainment`, `PiGroupsEntrainment`, or `InvZEntrainment`.

Arguments (all cell-centered):

  - `thermo_params`: Thermodynamic parameters.
  - `turbconv_params`: Turbulence convection parameters.
  - `ᶜz`: Height [m].
  - `z_sfc`: Surface elevation [m].
  - `ᶜp`: Pressure [Pa].
  - `ᶜρ`: Air density [kg/m³].
  - `ᶜaʲ`: Updraft area fraction [-].
  - `ᶜwʲ`: Updraft physical vertical velocity [m/s].
  - `ᶜRHʲ`: Updraft relative humidity [-].
  - `ᶜbuoyʲ`: Updraft buoyancy [m/s²].
  - `ᶜw⁰`: Environment physical vertical velocity [m/s].
  - `ᶜRH⁰`: Environment relative humidity [-].
  - `ᶜbuoy⁰`: Environment buoyancy [m/s²].
  - `ᶜtke`: Turbulent kinetic energy [m²/s²].
  - `model_option`: Object specifying the entrainment model.

Returns the velocity-scaling prefactor [1/m].
"""
function entrainment_velocity_scale(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
    ::NoEntrainment,
)
    return zero(eltype(thermo_params))
end

function entrainment_velocity_scale(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
    ::PiGroupsEntrainment,
)
    FT = eltype(thermo_params)

    elev_above_sfc = ᶜz - z_sfc
    # If elevation above surface is not positive, terms like 1/elev_above_sfc
    # become singular. Model assumes operation above the surface.
    if elev_above_sfc <= eps(FT)
        return 0
    end

    g = TDP.grav(thermo_params)
    ref_H = ᶜp / (ᶜρ * g) # Pressure scale height

    Π₁, Π₂, Π₃, Π₄, Π₅ = calculate_pi_groups(
        elev_above_sfc,
        ref_H,
        ᶜaʲ,
        ᶜwʲ,
        ᶜRHʲ,
        ᶜbuoyʲ,
        ᶜw⁰,
        ᶜRH⁰,
        ᶜbuoy⁰,
        ᶜtke,
    )

    entr_param_vec = CAP.entr_param_vec(turbconv_params)
    pi_sum =
        entr_param_vec[1] * abs(Π₁) +
        entr_param_vec[2] * abs(Π₂) +
        entr_param_vec[3] * abs(Π₃) +
        entr_param_vec[4] * abs(Π₄) +
        entr_param_vec[5] * abs(Π₅) +
        entr_param_vec[6]

    area_limiter_factor = entr_upper_area_limiter_factor(ᶜaʲ, turbconv_params)
    entr_vel_scale = area_limiter_factor * max(0, pi_sum) / elev_above_sfc
    return max(0, entr_vel_scale)
end

function entrainment_velocity_scale(
    thermo_params,
    turbconv_params,
    ᶜz,
    z_sfc,
    ᶜp,
    ᶜρ,
    ᶜaʲ,
    ᶜwʲ,
    ᶜRHʲ,
    ᶜbuoyʲ,
    ᶜw⁰,
    ᶜRH⁰,
    ᶜbuoy⁰,
    ᶜtke,
    ::InvZEntrainment,
)
    FT = eltype(thermo_params)
    entr_vel_scale_param = CAP.entr_coeff(turbconv_params)

    elev_above_sfc = ᶜz - z_sfc
    # If elevation above surface is not positive, terms like 1/elev_above_sfc
    # become singular. Model assumes operation above the surface.
    if elev_above_sfc <= eps(FT)
        return 0
    end

    area_limiter_factor = entr_upper_area_limiter_factor(ᶜaʲ, turbconv_params)
    entr_vel_scale = area_limiter_factor * entr_vel_scale_param / elev_above_sfc
    return max(0, entr_vel_scale)
end

"""
    area_bounding_entr_detr(turbconv_params, a)

Signed velocity-independent rate [1/s] that smoothly relaxes the updraft
area `a` back into `[a_min, a_max]`, applied unconditionally (no
dispatch on the entrainment/detrainment model). The sign convention is

    > 0  ⇒  acts as additional entrainment (drives the area up; a < a_min)
    < 0  ⇒  acts as additional detrainment (drives the area down; a > a_max)
    = 0  ⇒  inactive (a_min ≤ a ≤ a_max)

The result is

    +min_area_limiter_scale · (max(0, a_min − a)/a_min)^min_area_limiter_power
    −max_area_limiter_scale · (max(0, a − a_max)/(1 − a_max))^max_area_limiter_power.

The two ranges (`a < a_min` and `a > a_max`) are mutually exclusive, so
exactly one term is non-zero outside `[a_min, a_max]` and both are zero
inside it.

The positive part is consumed by [`compute_entrainment`](@ref) via
`max(0, area_bounding_entr_detr)`; the negative part is consumed by
[`compute_detrainment`](@ref) via `max(0, -area_bounding_entr_detr)`.
"""
@inline function area_bounding_entr_detr(turbconv_params, a)
    FT = typeof(a)
    a_min = CAP.min_area(turbconv_params)
    a_max = CAP.max_area(turbconv_params)
    min_scale = CAP.min_area_limiter_scale(turbconv_params)
    min_power = CAP.min_area_limiter_power(turbconv_params)
    max_scale = CAP.max_area_limiter_scale(turbconv_params)
    max_power = CAP.max_area_limiter_power(turbconv_params)

    # Extra entrainment for a < a_min (0 at a_min, min_scale at a = 0).
    min_relax = min_scale * (max(FT(0), a_min - a) / a_min)^min_power
    # Extra detrainment for a > a_max (0 at a_max, max_scale at a = 1).
    max_relax = max_scale * (max(FT(0), a - a_max) / (1 - a_max))^max_power

    return min_relax - max_relax
end

"""
    compute_detrainment(turbconv_params, aʲ, ρaʲ, buoy_inv_time_scale,
                        massflux_vert_div, area_bounding_entr_detr, detr_model)

Total detrainment rate [1/s] as the sum of the model-specific rate from
`detrainment_rate` (which internally applies
[`detr_lower_area_limiter_factor`](@ref) so that detrainment is damped as
`a → a_min`) with the negative part of the signed area-bounding rate:

    detr = detrainment_rate(...) + max(0, -area_bounding_entr_detr)

`area_bounding_entr_detr` is produced by [`area_bounding_entr_detr`](@ref);
its positive branch feeds [`compute_entrainment`](@ref) while the
negative branch (the only one that contributes here) drives the area back
below `a_max`.

Arguments:

  - `turbconv_params`: Turbulence convection parameters.
  - `aʲ`: Updraft area fraction [-].
  - `ρaʲ`: Updraft density-area product [kg/m³].
  - `buoy_inv_time_scale`: Clipped inverse buoyancy time scale [1/s] from
    [`detr_buoy_inv_time_scale`](@ref). The caller chooses where to evaluate
    it (centers, or faces with subsequent `ᶜinterp`) to control smoothness.
  - `massflux_vert_div`: Vertical divergence of the updraft mass flux [kg/(m³ s)].
  - `area_bounding_entr_detr`: Signed area-bounding rate [1/s] from
    [`area_bounding_entr_detr`](@ref).
  - `detr_model`: Object specifying the detrainment model.

Returns the total detrainment rate [1/s].
"""
function compute_detrainment(
    turbconv_params,
    aʲ,
    ρaʲ,
    buoy_inv_time_scale,
    massflux_vert_div,
    area_bounding_entr_detr,
    detr_model,
)

    detr = detrainment_rate(
        turbconv_params,
        aʲ,
        ρaʲ,
        buoy_inv_time_scale,
        massflux_vert_div,
        detr_model,
    )

    return detr + max(zero(area_bounding_entr_detr), -area_bounding_entr_detr)
end

"""
    detrainment_rate(turbconv_params, ᶜaʲ, ᶜρaʲ, ᶜbuoy_inv_time_scale,
                     ᶜmassflux_vert_div, detr_model::AbstractDetrainmentModel)

Model-specific detrainment rate [1/s] for a given detrainment model.

This abstract fallback returns zero; concrete subtypes of
`AbstractDetrainmentModel` should override this method to provide a
non-trivial rate.

Arguments:

  - `turbconv_params`: Turbulence convection parameters.
  - `ᶜaʲ`: Updraft area fraction [-].
  - `ᶜρaʲ`: Updraft density-area product [kg/m³].
  - `ᶜbuoy_inv_time_scale`: Clipped inverse buoyancy time scale [1/s] from
    [`detr_buoy_inv_time_scale`](@ref).
  - `ᶜmassflux_vert_div`: Vertical divergence of the updraft mass flux [kg/(m³ s)].
  - `detr_model`: Detrainment model dispatch tag.

Returns the model-specific detrainment rate [1/s] (zero for the abstract fallback).
"""
function detrainment_rate(
    turbconv_params,
    ᶜaʲ,
    ᶜρaʲ,
    ᶜbuoy_inv_time_scale,
    ᶜmassflux_vert_div,
    ::AbstractDetrainmentModel,
)
    return zero(eltype(ᶜρaʲ))
end

function detrainment_rate(
    turbconv_params,
    ᶜaʲ,
    ᶜρaʲ,
    ᶜbuoy_inv_time_scale,
    ᶜmassflux_vert_div,
    ::BuoyancyVelocityDetrainment,
)
    FT = eltype(ᶜρaʲ)
    detr_buoy_coeff = CAP.detr_buoy_coeff(turbconv_params)
    detr_massflux_vertdiv_coeff =
        CAP.detr_massflux_vertdiv_coeff(turbconv_params)

    area_limiter_factor = detr_lower_area_limiter_factor(ᶜaʲ, turbconv_params)
    detr =
        area_limiter_factor *
        (
            detr_buoy_coeff * ᶜbuoy_inv_time_scale -
            detr_massflux_vertdiv_coeff * min(ᶜmassflux_vert_div, 0) / max(eps(FT), ᶜρaʲ)
        )

    return max(0, detr)
end

function turbulent_entrainment(turbconv_params, ᶜaʲ)
    turb_entr_param_vec = CAP.turb_entr_param_vec(turbconv_params)
    return max(turb_entr_param_vec[1] * exp(-turb_entr_param_vec[2] * ᶜaʲ), 0)
end

"""
    edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model)

Add the entrainment contribution to the EDMF scalar tendencies (`mse`,
`q_tot`, and the microphysics tracers carried by the updraft).

Detrainment is **not** applied here because it is absorbed into the
analytic implicit ρa solve (see
[`solve_sgs_ρa_implicit_stage_analytic!`](@ref)); scalars are detrained
implicitly through the area divergence of the mass flux.

The entrainment rate is assembled lazily from the precomputed
`ᶜentr_vel_scaleʲs`, `ᶜarea_bounding_entr_detrʲs`, and the updraft physical
velocity via [`compute_entrainment`](@ref).
"""
edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

function edmfx_entr_detr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜturb_entrʲs, ᶜentr_vel_scaleʲs, ᶜarea_bounding_entr_detrʲs, ᶜuʲs) = p.precomputed

    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    ᶜlg = Fields.local_geometry_field(Y.c)

    for j in 1:n
        ᶜentrʲ = @. lazy(
            compute_entrainment(
                ᶜentr_vel_scaleʲs.:($$j),
                ᶜarea_bounding_entr_detrʲs.:($$j),
                get_physical_w(ᶜuʲs.:($$j), ᶜlg),
            ),
        )
        ᶜturb_entrʲ = ᶜturb_entrʲs.:($j)
        ᶜmseʲ = Y.c.sgsʲs.:($j).mse
        ᶜq_totʲ = Y.c.sgsʲs.:($j).q_tot

        @. Yₜ.c.sgsʲs.:($$j).mse += (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜmse⁰ - ᶜmseʲ)

        @. Yₜ.c.sgsʲs.:($$j).q_tot +=
            (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜq_tot⁰ - ᶜq_totʲ)

        # Auto-discovered SGS tracers (microphysics species and any
        # user-defined passive tracers)
        for χ_name in sgs_tracer_names(Y)
            ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
            ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:(1), χ_name)
            @. ᶜχʲₜ += (ᶜentrʲ .+ ᶜturb_entrʲ) * (ᶜχ⁰ - ᶜχʲ)
        end
    end
    return nothing
end

"""
    edmfx_first_interior_entr_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

Apply first-interior–level entrainment tendencies for each EDMF updraft.

This routine adds entrainment tendencies for moist static energy (`mse`) and total
humidity (`q_tot`) in the first model cell.
The entrained tracer value is taken from `sgs_scalar_first_interior_bc`.
"""
edmfx_first_interior_entr_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing
function edmfx_first_interior_entr_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)

    (; params) = p
    (; ᶜK, ᶜρʲs, ᶜturb_entrʲs, ᶜentr_vel_scaleʲs, ᶜarea_bounding_entr_detrʲs, ᶜuʲs) =
        p.precomputed

    FT = eltype(params)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    turbconv_params = CAP.turbconv_params(params)
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜaʲ_int_val = p.scratch.temp_data_level
    ᶜmse_buoyant_air_int_val = p.scratch.temp_data_level_2
    ᶜq_tot_buoyant_air_int_val = p.scratch.temp_data_level_3

    (;
        ustar,
        obukhov_length,
        buoyancy_flux,
        ρ_flux_h_tot,
        ρ_flux_q_tot,
        ustar,
        obukhov_length,
    ) =
        p.precomputed.sfc_conditions

    ᶜz_int_val = Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
    z_sfc_val =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.f).z, Fields.half))
    ᶜρ_int_val = Fields.field_values(Fields.level(Y.c.ρ, 1))

    buoyancy_flux_val = Fields.field_values(buoyancy_flux)
    ρ_flux_h_tot_val = Fields.field_values(ρ_flux_h_tot)
    ρ_flux_q_tot_val = Fields.field_values(ρ_flux_q_tot)

    ustar_val = Fields.field_values(ustar)
    obukhov_length_val = Fields.field_values(obukhov_length)
    sfc_local_geometry_val = Fields.field_values(
        Fields.local_geometry_field(Fields.level(Y.f, Fields.half)),
    )

    (; ᶜh_tot) = p.precomputed
    ᶜh_tot_int_val = Fields.field_values(Fields.level(ᶜh_tot, 1))
    ᶜK_int_val = Fields.field_values(Fields.level(ᶜK, 1))
    ᶜmse⁰ = ᶜspecific_env_mse(Y, p)
    env_mse_int_val = Fields.field_values(Fields.level(ᶜmse⁰, 1))

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    ᶜq_tot_int_val = Fields.field_values(Fields.level(ᶜq_tot, 1))
    ᶜq_tot⁰ = ᶜspecific_env_value(@name(q_tot), Y, p)
    env_q_tot_int_val = Fields.field_values(Fields.level(ᶜq_tot⁰, 1))

    for j in 1:n
        # Apply entrainment tendencies in the first model cell for moist static energy (mse)
        # and total humidity (q_tot). The entrained fluid is assumed to have a scalar value
        # given by `sgs_scalar_first_interior_bc` (mean + SGS perturbation). Since
        # `edmfx_entr_detr_tendency!` computes entrainment based on the environment–updraft
        # contrast, we supply the high-value (entrained) tracer minus the environment value
        # here to form the correct tendency.
        ᶜentrʲ = @. lazy(
            compute_entrainment(
                ᶜentr_vel_scaleʲs.:($$j),
                ᶜarea_bounding_entr_detrʲs.:($$j),
                get_physical_w(ᶜuʲs.:($$j), ᶜlg),
            ),
        )
        entr_int_val = Fields.field_values(Fields.level(ᶜentrʲ, 1))
        turb_entr_int_val = Fields.field_values(Fields.level(ᶜturb_entrʲs.:($j), 1))
        sgsʲs_ρ_int_val = Fields.field_values(Fields.level(ᶜρʲs.:($j), 1))
        sgsʲs_ρa_int_val = Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).ρa, 1))
        @. ᶜaʲ_int_val = max(
            FT(turbconv_params.surface_area),
            draft_area(sgsʲs_ρa_int_val, sgsʲs_ρ_int_val),
        )
        sgsʲs_mseₜ_int_val =
            Fields.field_values(Fields.level(Yₜ.c.sgsʲs.:($j).mse, 1))
        @. ᶜmse_buoyant_air_int_val = sgs_scalar_first_interior_bc(
            ᶜz_int_val - z_sfc_val,
            ᶜρ_int_val,
            ᶜaʲ_int_val,
            ᶜh_tot_int_val - ᶜK_int_val,
            buoyancy_flux_val,
            ρ_flux_h_tot_val,
            ustar_val,
            obukhov_length_val,
            sfc_local_geometry_val,
        )
        @. sgsʲs_mseₜ_int_val +=
            (entr_int_val + turb_entr_int_val) *
            (ᶜmse_buoyant_air_int_val - env_mse_int_val)

        sgsʲs_q_totₜ_int_val =
            Fields.field_values(Fields.level(Yₜ.c.sgsʲs.:($j).q_tot, 1))
        @. ᶜq_tot_buoyant_air_int_val = sgs_scalar_first_interior_bc(
            ᶜz_int_val - z_sfc_val,
            ᶜρ_int_val,
            ᶜaʲ_int_val,
            ᶜq_tot_int_val,
            buoyancy_flux_val,
            ρ_flux_q_tot_val,
            ustar_val,
            obukhov_length_val,
            sfc_local_geometry_val,
        )
        @. sgsʲs_q_totₜ_int_val +=
            (entr_int_val + turb_entr_int_val) *
            (ᶜq_tot_buoyant_air_int_val - env_q_tot_int_val)

    end
end

"""
    set_first_cell_entr_detr_bc!(
        ρaʲ_int, ρʲ_int, area_bounding_entr_detr_int, entr_vel_scale_int,
        buoyancy_flux, dz_int, surface_area, dt, FT,
    )

Apply surface boundary conditions for the area-bounding rate and the
entrainment velocity scale at the first model cell for a single EDMF
updraft.

When `buoyancy_flux < 0` (convectively stable): the area-bounding rate is
clipped to ≤ 0 so the lower boundary cannot drive net area growth.

When `buoyancy_flux ≥ 0` (convectively unstable): the updraft area is
nudged toward `surface_area` by overwriting the area-bounding rate with
the signed first-order step

    area_bounding_entr_detr = (1 − a / surface_area) / dt,

which is positive (entrainment) when `a < surface_area` and negative
(detrainment) when `a > surface_area`. The form `(1 − a/surface_area)/dt`
is linear in `a`, finite at `a = 0`, and compatible with the implicit
update inside the ρa solve. The entrainment velocity scale is also
replaced with the kinematic estimate `2/dz` so the total entrainment
includes the one-sided advective area flux `∂(ρaw)/∂z` at the surface.
Finally, `ρaʲ` is seeded with a tiny positive value so an initially
zero-area plume can start growing.

Arguments (all level-1 field-value slices unless noted):

  - `ρaʲ_int`                 — updraft `ρa` [kg/m³] (read/write; seeded first)
  - `ρʲ_int`                  — updraft density [kg/m³] (read-only)
  - `area_bounding_entr_detr_int` — signed area-bounding rate [1/s] (read/write)
  - `entr_vel_scale_int`      — entrainment velocity scale [1/m] (read/write)
  - `buoyancy_flux`           — surface buoyancy flux [m²/s³] (read-only scalar field values)
  - `dz_int`                  — first-cell height [m] (read-only)
  - `surface_area`            — target updraft area fraction [-] (scalar)
  - `dt`                      — timestep [s] (scalar)
  - `FT`                      — float type
"""
function set_first_cell_entr_detr_bc!(
    ρaʲ_int,
    ρʲ_int,
    area_bounding_entr_detr_int,
    entr_vel_scale_int,
    buoyancy_flux,
    dz_int,
    surface_area,
    dt,
    ::Type{FT},
) where {FT}
    # Seed a small positive area when buoyancy flux is non-negative so that
    # an initially zero-area plume can grow toward `surface_area`.
    @. ρaʲ_int += ifelse(buoyancy_flux < 0,
        FT(0),
        max(FT(0), ρʲ_int * eps(FT) - ρaʲ_int),
    )

    # Signed surface-BC area-bounding rate. When buoyancy_flux ≥ 0, drive
    # `a` toward `surface_area` via a first-order step; otherwise clip to
    # ≤ 0 so the lower boundary cannot grow the area further.
    @. area_bounding_entr_detr_int = ifelse(
        buoyancy_flux < 0,
        min(area_bounding_entr_detr_int, FT(0)),
        (FT(1) - draft_area(ρaʲ_int, ρʲ_int) / FT(surface_area)) / FT(dt),
    )
    # Replace entrainment coefficient with the kinematic estimate 2/dz when
    # buoyancy flux is positive so the total entrainment includes the advective
    # area flux ∂(ρaw)/∂z at the lower boundary (one-sided, zero below surface).
    @. entr_vel_scale_int = ifelse(
        buoyancy_flux < 0,
        entr_vel_scale_int,
        FT(2) / dz_int,
    )
    return nothing
end

# limit entrainment and detrainment rates for diagnostic EDMF
# limit rates approximately below the inverse timescale w/dz
limit_entrainment(entr::FT, a, dt) where {FT} = max(
    min(
        entr,
        FT(0.9) * (1 - a) / max(a, eps(FT)) / dt,
        FT(0.9) * 1 / dt,
    ),
    0,
)
limit_entrainment(entr::FT, a, w, dz) where {FT} =
    max(min(entr, FT(0.9) * w / dz), 0)

limit_detrainment(detr::FT, a, dt) where {FT} =
    max(min(detr, FT(0.9) * 1 / dt), 0)
limit_detrainment(detr::FT, a, w, dz, dt) where {FT} =
    limit_detrainment(max(min(detr, FT(0.9) * w / dz), 0), a, dt)

function limit_turb_entrainment(dyn_entr::FT, turb_entr, w, dz) where {FT}
    return max(min((FT(0.9) * w / dz) - dyn_entr, turb_entr), 0)
end
