#####
##### Couples the atmospheric model with a slab surface model
#####

using .SurfaceConditions: SurfaceTemperature, SlabOceanTemperature

"""
    surface_precipitation_tendency!(Yₜ, Y, p, t, temperature, microphysics_model)

Add the surface water and energy deposition from precipitation.

Only a `SlabOceanTemperature` surface with a moist microphysics model does
anything; the other methods are no-ops. Decrements `Yₜ.sfc.T` by the
column-integrated precipitation energy tendency divided by the slab heat capacity
per unit area `ρ_ocean cp_ocean depth_ocean`, and decrements `Yₜ.sfc.water` by the
sum of the surface rain and snow fluxes, so precipitation warms or cools the slab
and adds water to it with the sign convention that fluxes are positive upward.

Reads the precomputed `col_integrated_precip_energy_tendency`,
`surface_rain_flux`, and `surface_snow_flux` for equilibrium 0-moment
microphysics (and under automatic differentiation). For non-equilibrium
microphysics on plain floating-point states, evaluates the pointwise bottom-face
sedimentation water and energy fluxes directly from `(Y, p)` (converting the
level-1 velocity on the center space before slicing) so explicit-diagonal stages
(`a_{11} = 0`) do not read the previous step's stage-weighted diagnostic in
`p.precomputed`. When the stage-weighted `ManualSparseJacobian` accumulator is
active (`has_precip_stage_accumulator(p)` — direct Newton solve without Krylov or
line search at `update_jacobian_every = "solve"`, or `"stage"` when
`max_newton_iters_ode == 1`), `add_post_newton_surface_precipitation_tendency!`
adds the post-Newton Jacobian increment at each implicit stage so the slab ocean
conserves column water and energy with the atmosphere across IMEX stages
whenever `constrain_state!` leaves column water unchanged (`ρq_tot ≥ 0`). When
the accumulator is disabled (Krylov, line search, `update_jacobian_every = "dt"`,
`"stage"` with `max_newton_iters_ode > 1`, or autodiff Jacobians), `build_cache`
warns once and the slab receives only the pointwise stage flux `F(U^g)`.
"""
surface_precipitation_tendency!(Yₜ, Y, p, t, _, _) = nothing

surface_precipitation_tendency!(Yₜ, Y, p, t, ::SlabOceanTemperature, ::DryModel) = nothing

function surface_precipitation_tendency!(
    Yₜ, Y, p, t, slab::SlabOceanTemperature, microphysics_model,
)
    # Surface energy from precipitation
    pet = p.precomputed.col_integrated_precip_energy_tendency
    depth_ocean = slab.depth_ocean
    ρ_ocean = slab.ρ_ocean
    cp_ocean = slab.cp_ocean
    surface_heat_capacity_per_area = ρ_ocean * cp_ocean * depth_ocean
    @. Yₜ.sfc.T -= pet / surface_heat_capacity_per_area

    # Surface water from precipitation (rain + snow)
    P_liq = p.precomputed.surface_rain_flux
    P_snow = p.precomputed.surface_snow_flux
    @. Yₜ.sfc.water -= P_liq + P_snow
end

@inline function _slab_pointwise_precip_energy_flux(
    thp,
    J_ratio,
    int_T,
    int_Φ,
    int_uvw,
    int_ρq_lcl,
    int_ρq_icl,
    int_ρq_rai,
    int_ρq_sno,
    sfc_wₗ,
    sfc_wᵢ,
    sfc_wᵣ,
    sfc_wₛ,
    f_e_sgs,
)
    zero_FT = zero(int_T)
    f_lcl = -sfc_wₗ * J_ratio * max(zero_FT, int_ρq_lcl)
    f_icl = -sfc_wᵢ * J_ratio * max(zero_FT, int_ρq_icl)
    f_rai = -sfc_wᵣ * J_ratio * max(zero_FT, int_ρq_rai)
    f_sno = -sfc_wₛ * J_ratio * max(zero_FT, int_ρq_sno)
    e_int_liq = TD.internal_energy_liquid(thp, int_T)
    e_int_ice = TD.internal_energy_ice(thp, int_T)
    e_lcl = e_int_liq + int_Φ + sedimentation_kinetic_energy(sfc_wₗ, int_uvw)
    e_icl = e_int_ice + int_Φ + sedimentation_kinetic_energy(sfc_wᵢ, int_uvw)
    e_rai = e_int_liq + int_Φ + sedimentation_kinetic_energy(sfc_wᵣ, int_uvw)
    e_sno = e_int_ice + int_Φ + sedimentation_kinetic_energy(sfc_wₛ, int_uvw)
    return f_e_sgs + f_rai * e_rai + f_lcl * e_lcl + f_sno * e_sno + f_icl * e_icl
end

@inline function _slab_pointwise_precip_water_flux(
    J_ratio,
    int_ρq_lcl,
    int_ρq_icl,
    int_ρq_rai,
    int_ρq_sno,
    sfc_wₗ,
    sfc_wᵢ,
    sfc_wᵣ,
    sfc_wₛ,
)
    zero_FT = zero(int_ρq_rai)
    return -J_ratio * (
        sfc_wᵣ * max(zero_FT, int_ρq_rai) +
        sfc_wₗ * max(zero_FT, int_ρq_lcl) +
        sfc_wₛ * max(zero_FT, int_ρq_sno) +
        sfc_wᵢ * max(zero_FT, int_ρq_icl)
    )
end

function surface_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    t,
    slab::SlabOceanTemperature,
    ::Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
        NonEquilibriumMicrophysics2MP3,
    },
)
    if !(
        eltype(Y) === eltype(p.params) &&
        MatrixFields.has_field(Y.c, @name(ρq_lcl)) &&
        MatrixFields.has_field(Y.c, @name(ρq_icl)) &&
        MatrixFields.has_field(Y.c, @name(ρq_rai)) &&
        MatrixFields.has_field(Y.c, @name(ρq_sno))
    )
        surface_heat_capacity_per_area =
            slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
        @. Yₜ.sfc.T -=
            p.precomputed.col_integrated_precip_energy_tendency /
            surface_heat_capacity_per_area
        @. Yₜ.sfc.water -=
            p.precomputed.surface_rain_flux + p.precomputed.surface_snow_flux
        return nothing
    end
    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜT, ᶜu, ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ) = p.precomputed
    thp = CAP.thermodynamics_params(params)
    sfc_space = axes(Yₜ.sfc.water)
    sfc_lev(x) =
        Fields.Field(Fields.field_values(Fields.level(x, 1)), sfc_space)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    int_J = sfc_lev(ᶜJ)
    sfc_J = Fields.level(ᶠJ, Fields.half)
    J_ratio = @. lazy(int_J / sfc_J)

    int_ρq_lcl = sfc_lev(Y.c.ρq_lcl)
    int_ρq_icl = sfc_lev(Y.c.ρq_icl)
    int_ρq_rai = sfc_lev(Y.c.ρq_rai)
    int_ρq_sno = sfc_lev(Y.c.ρq_sno)
    int_T = sfc_lev(ᶜT)
    int_Φ = sfc_lev(ᶜΦ)
    int_uvw = if hasproperty(p.scratch, :c_lev1_uvw)
        @. p.scratch.c_lev1_uvw = Geometry.UVWVector($(Fields.level(ᶜu, 1)))
        Fields.Field(Fields.field_values(p.scratch.c_lev1_uvw), sfc_space)
    else
        @. lazy(Geometry.UVWVector($(sfc_lev(ᶜu))))
    end
    sfc_wₗ = sfc_lev(ᶜwₗ)
    sfc_wᵢ = sfc_lev(ᶜwᵢ)
    sfc_wᵣ = sfc_lev(ᶜwᵣ)
    sfc_wₛ = sfc_lev(ᶜwₛ)

    f_e_sgs = Fields.Field(Fields.field_values(p.scratch.ᶠtemp_field_level), sfc_space)
    gm = (;
        sfc_lev, J_ratio, int_T, int_uvw,
        int_ρq_lcl, int_ρq_icl, int_ρq_rai, int_ρq_sno,
        sfc_wₗ, sfc_wᵢ, sfc_wᵣ, sfc_wₛ,
    )
    set_sfc_sgs_precip_energy_flux!(
        f_e_sgs,
        Y,
        p,
        p.atmos.turbconv_model,
        gm,
    )

    surface_heat_capacity_per_area =
        slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
    @. Yₜ.sfc.water -= _slab_pointwise_precip_water_flux(
        J_ratio,
        int_ρq_lcl,
        int_ρq_icl,
        int_ρq_rai,
        int_ρq_sno,
        sfc_wₗ,
        sfc_wᵢ,
        sfc_wᵣ,
        sfc_wₛ,
    )
    @. Yₜ.sfc.T -=
        _slab_pointwise_precip_energy_flux(
            thp,
            J_ratio,
            int_T,
            int_Φ,
            int_uvw,
            int_ρq_lcl,
            int_ρq_icl,
            int_ρq_rai,
            int_ρq_sno,
            sfc_wₗ,
            sfc_wᵢ,
            sfc_wᵣ,
            sfc_wₛ,
            f_e_sgs,
        ) / surface_heat_capacity_per_area
    return nothing
end

"""
    add_post_newton_surface_precipitation_tendency!(Yₜ, Y, p, temperature, microphysics_model)

Add to `Yₜ.sfc` the post-Newton Jacobian increment `-(w_jac (ρq* - ρq^g))` of
the bottom-face sedimentation water and energy fluxes at `Y = U*`, using the
linearization state cached in `p.scratch.sfc_precip_lin_state` by
`record_stage_precipitation_linearization!`. Active only for a
`SlabOceanTemperature` surface with non-equilibrium microphysics when the
stage-weighted `ManualSparseJacobian` accumulator is enabled; a no-op otherwise.
"""
add_post_newton_surface_precipitation_tendency!(Yₜ, Y, p, _, _) = nothing
add_post_newton_surface_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    ::SlabOceanTemperature,
    _,
) = nothing
function add_post_newton_surface_precipitation_tendency!(
    Yₜ,
    Y,
    p,
    slab::SlabOceanTemperature,
    ::Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
        NonEquilibriumMicrophysics2MP3,
    },
)
    (
        eltype(Y) === eltype(p.params) &&
        has_precip_stage_accumulator(p) &&
        MatrixFields.has_field(Y.c, @name(ρq_lcl)) &&
        MatrixFields.has_field(Y.c, @name(ρq_icl)) &&
        MatrixFields.has_field(Y.c, @name(ρq_rai)) &&
        MatrixFields.has_field(Y.c, @name(ρq_sno))
    ) || return nothing

    lin = p.scratch.sfc_precip_lin_state
    sfc_space = axes(lin)
    sfc_lev(x) =
        Fields.Field(Fields.field_values(Fields.level(x, 1)), sfc_space)
    int_ρq_lcl = sfc_lev(Y.c.ρq_lcl)
    int_ρq_icl = sfc_lev(Y.c.ρq_icl)
    int_ρq_rai = sfc_lev(Y.c.ρq_rai)
    int_ρq_sno = sfc_lev(Y.c.ρq_sno)

    Δf_lcl = @. lazy(lin.w_lcl_jac * (int_ρq_lcl - lin.ρq_lcl_guess))
    Δf_icl = @. lazy(lin.w_icl_jac * (int_ρq_icl - lin.ρq_icl_guess))
    Δf_rai = @. lazy(lin.w_rai_jac * (int_ρq_rai - lin.ρq_rai_guess))
    Δf_sno = @. lazy(lin.w_sno_jac * (int_ρq_sno - lin.ρq_sno_guess))

    surface_heat_capacity_per_area =
        slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
    @. Yₜ.sfc.water -= Δf_rai + Δf_lcl + Δf_sno + Δf_icl
    @. Yₜ.sfc.T -=
        (
            Δf_rai * lin.e_rai +
            Δf_lcl * lin.e_lcl +
            Δf_sno * lin.e_sno +
            Δf_icl * lin.e_icl
        ) / surface_heat_capacity_per_area
    return nothing
end

"""
    surface_temp_tendency!(Yₜ, Y, p, t, temperature::SurfaceTemperature)
    surface_temp_tendency!(Yₜ, Y, p, t, temperature::SlabOceanTemperature)

Add the tendencies of the prognostic surface temperature `Y.sfc.T` and, when
moisture is active, the surface water content `Y.sfc.water`.

All fluxes are positive when directed upward, from the surface to the atmosphere,
and are subtracted from the tendencies, so an upward flux cools and dries the
surface. The first method is a no-op: for any `SurfaceTemperature` other than a
slab ocean, `T` is prescribed or diagnosed. For `SlabOceanTemperature`:

  - `Yₜ.sfc.T` is decremented by the sum of the net upward radiative flux at the
    surface, the upward turbulent energy flux (sensible plus latent heat), and the
    idealized ocean heat-flux divergence `Q` (Q-flux, active only when
    `slab.q_flux`), divided by the slab heat capacity per unit area
    `ρ_ocean cp_ocean depth_ocean`. The Q-flux profile follows Merlis et al. (2013),
    J. Climate 26, https://doi.org/10.1175/JCLI-D-12-00149.1.
  - `Yₜ.sfc.water` is decremented by the upward turbulent water flux (evaporation).

Both turbulent terms are dropped when `p.atmos.disable_surface_flux_tendency` is
`true`, and the radiative term when there is no radiation model. Precipitation
deposition of energy and water is handled separately by
`surface_precipitation_tendency!`. Reads `p.radiation.ᶠradiation_flux` and
`p.precomputed.sfc_conditions`; `t` is unused. Called from
`additional_tendency!`, after the microphysics tendencies.
"""
surface_temp_tendency!(Yₜ, Y, p, t, ::SurfaceTemperature) = nothing

function surface_temp_tendency!(Yₜ, Y, p, t, slab::SlabOceanTemperature)
    FT = eltype(Y)

    depth_ocean = slab.depth_ocean
    ρ_ocean = slab.ρ_ocean
    cp_ocean = slab.cp_ocean
    q_flux_enabled = slab.q_flux

    # --- ENERGY BALANCE ---
    # Denominator for temperature tendency
    surface_heat_capacity_per_area = ρ_ocean * cp_ocean * depth_ocean

    # 1. Radiative energy surface fluxes
    if !isnothing(p.atmos.radiation_mode)
        # ᶠradiation_flux is positive for net upward flux at the surface
        # (SW_up - SW_down + LW_up - LW_down)
        (; ᶠradiation_flux) = p.radiation
        sfc_rad_e_flux = Spaces.level(ᶠradiation_flux, half).components.data.:1
    else
        sfc_rad_e_flux = 0
    end

    # 2. Turbulent surface energy fluxes (sensible + latent heat) from surface to atmosphere
    if !(p.atmos.disable_surface_flux_tendency)
        turb_e_flux_sfc_to_atm =
            Geometry.WVector.(
                p.precomputed.sfc_conditions.ρ_flux_h_tot,
            ).components.data.:1
    else
        turb_e_flux_sfc_to_atm = 0
    end

    # 3. Idealized Q-fluxes (parameterization of horizontal ocean energy flux divergence),
    # following Merlis et al. (2013), "Hadley Circulation Response to Orbital Precession.
    # Part II: Subtropical Continent.", J. Climate, 26, https://doi.org/10.1175/JCLI-D-12-00149.1
    if q_flux_enabled
        ϕ₀ = slab.ϕ₀
        Q₀ = slab.Q₀
        ϕ = deg2rad.(Fields.level(Fields.coordinate_field(Y.f).lat, half))
        ϕ₀ʳ = FT(deg2rad(ϕ₀))
        Q = @. Q₀ * (1 - 2ϕ^2 / ϕ₀ʳ^2) * exp(-(ϕ^2 / ϕ₀ʳ^2)) / cos(ϕ)
    else
        Q = FT(0)
    end

    # Total energy tendency for surface temperature
    # (precipitation energy/water deposition is handled separately
    # by surface_precipitation_tendency! in both implicit and explicit paths):
    # dT/dt = -(NetRad_upward + TurbFlux_sfc_to_atm + Q_div) / HeatCapacity
    @. Yₜ.sfc.T -=
        (sfc_rad_e_flux + turb_e_flux_sfc_to_atm + Q) /
        surface_heat_capacity_per_area

    # --- WATER BALANCE (if moisture is active) ---
    if !(p.atmos.microphysics_model isa DryModel)
        # Turbulent surface water fluxes (evaporation/condensation)
        if !(p.atmos.disable_surface_flux_tendency)
            sfc_turb_w_flux =
                Geometry.WVector.(
                    p.precomputed.sfc_conditions.ρ_flux_q_tot,
                ).components.data.:1
        else
            sfc_turb_w_flux = 0
        end

        # Water tendency from turbulent fluxes only;
        # precipitation (rain + snow) is handled by surface_precipitation_tendency!.
        @. Yₜ.sfc.water -= sfc_turb_w_flux
    end

end
