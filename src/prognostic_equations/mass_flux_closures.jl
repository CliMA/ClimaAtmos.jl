#####
##### Mass flux closures for EDMFX
#####

import StaticArrays as SA
import Thermodynamics.Parameters as TDP
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    buoyancy(ρ_ref, ρ, gradᵥ_Φ)

Compute the buoyancy acceleration vector `(ρ_ref - ρ) / ρ * gradᵥ_Φ`.

# Arguments

  - `ρ_ref`: Reference density [kg/m³].
  - `ρ`: Air density [kg/m³].
  - `gradᵥ_Φ`: Vertical gradient of geopotential (`Covariant3Vector`, i.e.
    gravitational acceleration) [m/s²].

# Returns

The buoyancy acceleration as a `Covariant3Vector` [m/s²]; directed upward
(along `gradᵥ_Φ`) when `ρ < ρ_ref`.
"""
function buoyancy(ρ_ref, ρ, gradᵥ_Φ)
    result = (ρ_ref - ρ) / ρ * gradᵥ_Φ
    return result
end

"""
    vertical_buoyancy_acceleration(ρ_ref, ρ, gradᵥ_Φ, local_geometry)
    vertical_buoyancy_acceleration(ρ_diff, gradᵥ_Φ, local_geometry)

Compute the signed vertical component of the buoyancy acceleration in
physical units.

Form the buoyancy acceleration vector due to a density anomaly (`buoyancy`
in the first method, `-ρ_diff * gradᵥ_Φ` in the second) and project it onto the
local vertical direction.

# Arguments

  - `ρ_ref`: Reference density [kg/m³].
  - `ρ`: Air density [kg/m³].
  - `gradᵥ_Φ`: Vertical gradient of geopotential (`Covariant3Vector`, i.e.
    gravitational acceleration) [m/s²].
  - `local_geometry`: Local geometry used to project onto the vertical direction.
  - `ρ_diff`: Normalized density difference `(ρ - ρ_ref) / ρ` [-].

# Returns

Scalar acceleration in the vertical direction [m/s²], positive when buoyancy
acts upward.
"""
function vertical_buoyancy_acceleration(ρ_ref, ρ, gradᵥ_Φ, local_geometry)
    # Compute the full buoyancy acceleration vector (Covariant3Vector)
    buoy_vector = buoyancy(ρ_ref, ρ, gradᵥ_Φ)
    # Project onto vertical axis and return signed scalar value
    return projected_vector_data(C3, buoy_vector, local_geometry)
end
function vertical_buoyancy_acceleration(ρ_diff, gradᵥ_Φ, local_geometry)
    # Compute the full buoyancy acceleration vector (Covariant3Vector)
    buoy_vector = -1 * ρ_diff * gradᵥ_Φ
    # Project onto vertical axis and return signed scalar value
    return projected_vector_data(C3, buoy_vector, local_geometry)
end

"""
    draft_buoyancy_per_mse(thermo_params, T, ρ, q_lcl, q_icl)

Buoyancy change per unit change of a draft's moist static energy [m s⁻² / (J/kg)],
`g / (c_p T Γ)`. In a cloudy draft, the moist factor
`Γ = 1 + (L/c_p) ∂q_sat/∂T`, with the phase ramp of `liquid_fraction_ramp`,
accounts for the share of an mse change that goes into condensation. A draft
without cloud condensate (`TD.has_condensate` on `q_lcl + q_icl`, as for
`_updraft_cloud_fraction`) responds as dry air, `Γ = 1`: a moist factor there
would underestimate the stratification, and an underestimate leaves the
alternating iterates that `stable_draft_mse_response` damps, whereas an
overestimate only slows their convergence.
"""
@inline function draft_buoyancy_per_mse(thermo_params, T, ρ, q_lcl, q_icl)
    cp_d = TD.Parameters.cp_d(thermo_params)
    λ = TD.liquid_fraction_ramp(thermo_params, T)
    L = TD.latent_heat_mixed(thermo_params, T, λ)
    q_c = max(zero(T), q_lcl + q_icl)
    # Both arms evaluate, so the kernel has no branch on the condensate test.
    Γ =
        1 + ifelse(
            TD.has_condensate(thermo_params, q_c),
            L / cp_d * TD.∂q_vap_sat_∂T(thermo_params, T, ρ),
            zero(T),
        )
    return TD.Parameters.grav(thermo_params) / (cp_d * T * Γ)
end

"""
    stable_draft_mse_response(β, Δmse, C, J, dtγ)

Buoyancy change of a stably stratified draft cell [m/s²] per unit face flux
`J u³` at its bottom face within one Newton step, for the Jacobian's `u₃` block:
`dtγ β max(Δmse, 0) / (J (1 + C))`. `β` is `draft_buoyancy_per_mse`, `Δmse` the
mse step from the cell below (`mse_c − mse_{c−1}`), `J` the cell's Jacobian, and
`C = dtγ (J u³_jac)_{c−½} / J` the damping of the cell's upwinded mse at the
Jacobian updraft velocity.

The `u₃` rows omit the buoyancy derivatives, so Newton sees the buoyancy of the
previous iterate: a velocity error moves the draft's mse, whose buoyancy change
the next iterate responds to. In a stable draft (`N² > 0`) the error alternates
in sign and grows once the gain of this loop, `dtγ² N²` reduced by the damping
of the `u₃` and mse diagonals, exceeds one, as for a draft at rest under
positive buoyancy, which then swings between rest and launch. With this
response, the `u₃` block carries the Schur complement of the omitted
`u₃`–mse–buoyancy coupling where `Δmse > 0`, with the mse block replaced by its
diagonal. Where `Δmse < 0` the same term would shrink the diagonal, which is why
the buoyancy derivatives are left out of convecting drafts.
"""
@inline stable_draft_mse_response(β, Δmse, C, J, dtγ) =
    dtγ * β * max(Δmse, zero(Δmse)) / (J * (1 + C))


"""
    draft_area(ρa, ρ)

Return the draft area fraction `a = ρa / ρ` [-].

# Arguments

  - `ρa`: Area-weighted density of the subdomain, `ρ a` [kg/m³].
  - `ρ`: Density of the same subdomain [kg/m³].
"""
function draft_area(ρa, ρ)
    return ρa / ρ
end

"""
    ᶠupdraft_nh_pressure_buoyancy(params, ᶠbuoyʲ)

Return the virtual-mass (buoyancy) term `α_b ᶠbuoyʲ` of the non-hydrostatic
pressure closure for updrafts, in the units of `ᶠbuoyʲ` (for a covariant3
component, [m²/s²]).

`α_b` is `pressure_normalmode_buoy_coeff1`, so the effective buoyancy retained
in the updraft momentum equation is `(1 - α_b) ᶠbuoyʲ`.

# Arguments

  - `params`: Model parameter set.
  - `ᶠbuoyʲ`: Updraft buoyancy at cell faces, as a covariant3 or contravariant3
    component.

# Notes

Currently unused: the implicit updraft-momentum solve applies `α_b` directly
(see `initialize_implicit_problem.jl`).
"""
function ᶠupdraft_nh_pressure_buoyancy(params, ᶠbuoyʲ)
    turbconv_params = CAP.turbconv_params(params)
    # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    return α_b * ᶠbuoyʲ
end

"""
    pressure_drag_coefficient(
        α_d, scale_height, aʲ, a⁰, a_min, a_max,
        z = 0, z_sfc = 0, c_R = 0, R_max = 0,
    )

Return the coefficient `d_c` [1/m] of the pairwise form drag between an updraft
of area fraction `aʲ` and the environment of area fraction `a⁰`,

    d_c = α_d / 2 * (1 / Rʲ + 1 / R⁰),
    Rʲ = plume_radius(H, aʲ, a_min, a_max, z, z_sfc, c_R, R_max),
    R⁰ = H √clamp(a⁰, 1 - a_max, 1),

so that the drag acceleration on the updraft is `a⁰ d_c (wʲ - w⁰) |wʲ - w⁰|`,
with `a⁰ = ρ̂⁰/ρ` the environment weight of the pairwise form drag. The length scale
of each subdomain is `H √a` with `H = scale_height`, and the harmonic mean of the
two gives the sum of inverse radii. The updraft radius `Rʲ`, with its
height-growing floor, is shared with the TKE-driven turbulent entrainment. The
environment is clamped from
below at `1 - a_max`, which keeps the coefficient finite when the implicit
velocity solve is handed an extrapolated `a⁰` outside `[0, 1]`.

Shared by the implicit updraft momentum tendency (`sgs_u₃_implicit_tendency!`)
and the TKE return-to-isotropy source
(the pressure-drag term of `edmfx_tke_sources!`), so the kinetic energy the drag
removes from the updraft is the energy the TKE budget receives.
"""
@inline function pressure_drag_coefficient(
    α_d,
    scale_height,
    aʲ,
    a⁰,
    a_min,
    a_max,
    z = zero(aʲ),
    z_sfc = zero(aʲ),
    c_R = zero(aʲ),
    R_max = zero(aʲ),
)
    Rʲ = plume_radius(scale_height, aʲ, a_min, a_max, z, z_sfc, c_R, R_max)
    R⁰ = scale_height * sqrt(clamp(a⁰, 1 - a_max, one(a⁰)))
    return α_d / 2 * (1 / Rʲ + 1 / R⁰)
end

"""
    plume_radius(scale_height, aʲ, a_min, a_max, z, z_sfc, c_R, R_max)

Radius [m] of an updraft of area fraction `aʲ`, the updraft length scale of the
form drag (`pressure_drag_coefficient`) and of the TKE-driven turbulent
entrainment (`turbulent_entrainment`),

    R = H √clamp(aʲ, a_lb, a_max),   a_lb = min(max(a_min, (R_min / H)²), a_max),
    R_min(z) = min(c_R max(0, z − z_sfc), R_max),

with `H = scale_height`: the radius grows at least linearly with height above
the surface at slope `c_R` up to `R_max` (`pressure_drag_plume_radius_slope`,
`pressure_drag_plume_radius_max`), and `a_lb = a_min` when either is `0`.
"""
@inline function plume_radius(scale_height, aʲ, a_min, a_max, z, z_sfc, c_R, R_max)
    R_min = min(c_R * max(zero(z), z - z_sfc), R_max)
    a_lb = min(max(a_min, (R_min / scale_height)^2), a_max)
    return scale_height * sqrt(clamp(aʲ, a_lb, a_max))
end

"""
    ᶠupdraft_nh_pressure_drag(params, ᶠlg, ᶠu3ʲ, ᶠu3⁰)

Return the drag term of the non-hydrostatic pressure closure for updrafts,

    α_d (u₃ʲ - u₃⁰) ‖u₃ʲ - u₃⁰‖ / max(H, H_up_min),

a simplified form in which the length scale is fixed at the reference scale
height `H = R_d T_surf_ref / g`, floored by `min_updraft_top`. The result has
the units of the velocity difference times an inverse length.

# Arguments

  - `params`: Model parameter set (`α_d` is `pressure_normalmode_drag_coeff`).
  - `ᶠlg`: Face local geometry, used to take the norm of the velocity difference.
  - `ᶠu3ʲ`, `ᶠu3⁰`: Updraft and environment vertical velocity at faces, as
    covariant3 or contravariant3 components.

# Notes

Currently unused: the implicit updraft-momentum solve builds the equivalent
quadratic drag sink directly (see `initialize_implicit_problem.jl`).
"""
function ᶠupdraft_nh_pressure_drag(params, ᶠlg, ᶠu3ʲ, ᶠu3⁰)
    turbconv_params = CAP.turbconv_params(params)
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    H_up_min = CAP.min_updraft_top(turbconv_params)
    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    # We also used to have advection term here: α_a * w_up * div_w_up
    return α_d * (ᶠu3ʲ - ᶠu3⁰) * CC.Geometry._norm(ᶠu3ʲ - ᶠu3⁰, ᶠlg) /
           max(scale_height, H_up_min)
end

"""
    surface_mass_flux_coefficient(buoyancy_flux, z_i, ustar, a_s_max, c_u)

Effective surface area fraction of the convective mass flux entering the
EDMF updraft,

    a_s = a_s_max · w*³ / (w*³ + c_u · u*³),

where `w*³ = max(z_i · ⟨w'b'⟩_s, 0)` (`w*` is the Deardorff convective
velocity scale), `u*` is the friction velocity, and `c_u` is an
O(1) tuning coefficient setting the relative weight of shear vs.
buoyancy production in the blend (TOML key:
`EDMF_sfc_mass_flux_ustar_coeff`). The factor `w*³/(w*³+c_u·u*³)`
interpolates smoothly between free convection (`a_s → a_s_max`) and
shear-only conditions (`a_s → 0`). `a_s_max` is the asymptotic plume
area fraction in the free-convection limit. Used both to set the
surface mass flux magnitude (via `surface_mass_flux`) and to
specify the percentile range from which the high-tail buoyant scalar
values are sampled at the surface.
"""
@inline function surface_mass_flux_coefficient(
    buoyancy_flux,
    z_i,
    ustar,
    a_s_max,
    c_u,
)
    FT = typeof(ustar)
    w3 = max(z_i * buoyancy_flux, FT(0))
    return a_s_max * w3 / max(eps(FT), w3 + c_u * ustar^3)
end

"""
    surface_mass_flux(buoyancy_flux, ρ, z_i, ustar, a_s_max, c_u)

Surface EDMF updraft mass flux [kg/m²/s] entering the first cell:

    F_surf = a_s · ρ · w*,

with `a_s` given by `surface_mass_flux_coefficient` and
`w* = cbrt(max(z_i · ⟨w'b'⟩_s, 0))`. Returns zero in stable boundary
layers (`⟨w'b'⟩_s ≤ 0`).
"""
@inline function surface_mass_flux(buoyancy_flux, ρ, z_i, ustar, a_s_max, c_u)
    FT = typeof(ρ)
    w_star = cbrt(max(z_i * buoyancy_flux, FT(0)))
    a_s = surface_mass_flux_coefficient(buoyancy_flux, z_i, ustar, a_s_max, c_u)
    return a_s * ρ * w_star
end

"""
    condensate_rescale_ratio(q_cond, q_tot, ϵ)

Return the factor `min(1, q_tot / q_cond)` by which condensates that sum to `q_cond`
are multiplied so that they cannot exceed the total water `q_tot`, or zero where
`q_cond` or `q_tot` is not above `ϵ`. Applies alike to specific humidities and to
densities.
"""
@inline condensate_rescale_ratio(q_cond, q_tot, ϵ) =
    ifelse((q_cond > ϵ) & (q_tot > ϵ), min(one(ϵ), q_tot / q_cond), zero(ϵ))

"""
    enforce_grid_mean_microphysics_constraints!(Y, p, t)

Clip the grid-mean condensate tracers to non-negative values and rescale them
so their sum cannot exceed the available total moisture.

`ρq_lcl`, `ρq_icl`, `ρq_rai`, and `ρq_sno` are first floored at zero, then
rescaled by `condensate_rescale_ratio`, which removes condensate in cells with
no (or negative) total water.

Mutates `Y.c`; returns `nothing`. Called from
`enforce_physical_constraints!` for the 1M and 2M non-equilibrium
microphysics models.
"""
function enforce_grid_mean_microphysics_constraints!(Y, p, t)
    FT = eltype(p.params)
    # The clip below changes the condensate but not `ρq_tot`, so the surface
    # precipitation of the stage is linearized at the condensate before it.
    record_solved_precipitation_condensate!(Y, p)
    (zero_FT, ϵ) = (zero(FT), ϵ_numerics(FT))
    DataLayouts.foreach_column(Y.c) do Yc
        ρq_lcl = @. max(zero_FT, Yc.ρq_lcl)
        ρq_icl = @. max(zero_FT, Yc.ρq_icl)
        ρq_rai = @. max(zero_FT, Yc.ρq_rai)
        ρq_sno = @. max(zero_FT, Yc.ρq_sno)
        ρq_cond = @. ρq_lcl + ρq_icl + ρq_rai + ρq_sno
        ratio = @. condensate_rescale_ratio(ρq_cond, Yc.ρq_tot, ϵ)
        @. Yc.ρq_lcl = ρq_lcl * ratio
        @. Yc.ρq_icl = ρq_icl * ratio
        @. Yc.ρq_rai = ρq_rai * ratio
        @. Yc.ρq_sno = ρq_sno * ratio
    end
    return nothing
end

"""
    enforce_edmf_updraft_constraints!(Y, p, t, turbconv_model)

Clip the prognostic EDMFX updraft state to a physically admissible range and
relax degenerate updrafts toward the grid mean.  Called from
`enforce_physical_constraints!` at stage boundaries when the `filter` flag is
enabled.

For each of the `n_prognostic_mass_flux_subdomains(turbconv_model)` updrafts:

  - `Y.c.sgsʲs.:(j).ρa` is clamped to `[0, ρʲ·a_max]`, i.e. the area fraction
    `ρa / ρʲ` to `[0, a_max]`, using the updraft density `ρʲ` and `a_max` from
    turbconv params (the same bound as the `ρa` bounds-restoring term of
    `sgs_ρa_implicit_tendency!` and `initialize_implicit_stage_problem!`).
  - The covariant³ component of `Y.f.sgsʲs.:(j).u₃` is clamped to be
    non-negative (updrafts do not descend), and is set to zero where the
    face-interpolated `ρa` is below `ϵ_numerics`.
  - `mse` and `q_tot` are reset to their **grid-mean** values (`ᶜh_tot - ᶜK`
    and `specific(Y.c.ρq_tot, Y.c.ρ)`) where `ρa < ϵ_numerics`; otherwise
    `q_tot` is floored at zero and bounded by `ρq_tot / ρa`, which enforces the
    subdomain mass bound `ρaʲ q_totʲ ≤ ρ q_tot`.
  - Each auto-discovered SGS tracer (microphysics species and passive tracers)
    is treated like `q_tot`: grid-mean value where `ρa` is negligible, else
    floored at zero and bounded by `ρχ / ρa`. This block is a no-op for the 0M
    microphysics model, where the grid-mean `ρχ` fields do not exist.
  - The subdomain condensate species are finally rescaled by a common
    factor so that `q_lclʲ + q_iclʲ + q_raiʲ + q_snoʲ ≤ q_totʲ`, mirroring
    the grid-mean `enforce_grid_mean_microphysics_constraints!`.

No-op when `n_prognostic_mass_flux_subdomains(turbconv_model) == 0` (e.g.
`EDOnlyEDMFX`). Mutates `Y.c.sgsʲs` and `Y.f.sgsʲs`; returns `nothing`.
"""
function enforce_edmf_updraft_constraints!(Y, p, t, turbconv_model)
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    n == 0 && return nothing
    (; ᶜh_tot, ᶜK, ᶜρʲs) = p.precomputed
    a_max = CAP.max_area(CAP.turbconv_params(p.params))
    # The SGS tracers whose grid-mean densities are prognostic
    χ_names = unrolled_filter(
        χ_name -> MatrixFields.has_field(Y.c, get_ρχ_name(χ_name)),
        sgs_tracer_names(Y),
    )
    (; microphysics_model) = p.atmos
    drafts = unrolled_map(StaticOneTo(n)) do j
        @inline
        (Y.c.sgsʲs.:($j), Y.f.sgsʲs.:($j).u₃, ᶜρʲs.:($j))
    end
    DataLayouts.foreach_column(Y.c, drafts, ᶜh_tot, ᶜK) do Yc, drafts, ᶜh_tot, ᶜK
        (zero_FT, ϵ) = (zero(eltype(ᶜK)), ϵ_numerics(eltype(ᶜK)))
        for (ᶜsgsʲ, ᶠu₃ʲ, ᶜρʲ) in drafts
            # Clip the area fraction ρa/ρʲ to [0, a_max] using the updraft
            # density ρʲ, and the vertical velocity to non-negative values, with zero velocity
            # where the face-averaged ρa is below ϵ_numerics.
            ᶜρa = @. clamp(ᶜsgsʲ.ρa, zero_FT, ᶜρʲ * a_max)
            @. ᶜsgsʲ.ρa = ᶜρa
            @. ᶠu₃ʲ = ifelse(
                ᶠinterp(ᶜρa) < ϵ,
                C3(zero_FT),
                C3(max(ᶠu₃ʲ.components.data.:1, zero_FT)),
            )
            # Relax mse, q_tot, and the SGS tracers toward the grid mean where
            # ρa is negligible; elsewhere, floor them at zero and ensure mass
            # conservation, ρaχʲ ≤ ρχ.
            ᶜnegligible = @. ᶜρa < ϵ
            @. ᶜsgsʲ.mse = ifelse(ᶜnegligible, ᶜh_tot - ᶜK, ᶜsgsʲ.mse)
            ᶜq_totʲ = @. ifelse(
                ᶜnegligible,
                specific(Yc.ρq_tot, Yc.ρ),
                min(max(zero_FT, ᶜsgsʲ.q_tot), max(zero_FT, Yc.ρq_tot) / ᶜρa),
            )
            @. ᶜsgsʲ.q_tot = ᶜq_totʲ
            MatrixFields.unrolled_foreach(χ_names) do χ_name
                @inline
                ᶜχʲ = MatrixFields.get_field(ᶜsgsʲ, χ_name)
                ᶜρχ = MatrixFields.get_field(Yc, get_ρχ_name(χ_name))
                @. ᶜχʲ = ifelse(
                    ᶜnegligible,
                    specific(ᶜρχ, Yc.ρ),
                    min(max(zero_FT, ᶜχʲ), max(zero_FT, ᶜρχ) / ᶜρa),
                )
            end
            # Within-subdomain condensate rescaling: ensure
            # q_lclʲ+q_iclʲ+q_raiʲ+q_snoʲ ≤ q_totʲ. The bound above already
            # clipped each χʲ ≥ 0, and multiplying by a ratio ≤ 1 preserves
            # ρaχʲ ≤ ρχ.
            microphysics_model isa
            Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ||
                continue
            ᶜq_cond = @. ᶜsgsʲ.q_lcl + ᶜsgsʲ.q_icl + ᶜsgsʲ.q_rai + ᶜsgsʲ.q_sno
            ᶜratio = @. condensate_rescale_ratio(ᶜq_cond, ᶜq_totʲ, ϵ)
            @. ᶜsgsʲ.q_lcl *= ᶜratio
            @. ᶜsgsʲ.q_icl *= ᶜratio
            @. ᶜsgsʲ.q_rai *= ᶜratio
            @. ᶜsgsʲ.q_sno *= ᶜratio
        end
    end
    return nothing
end

"""
    enforce_physical_constraints!(Y, p, t, atmos)

Enforce physical consistency of the state `Y` by dispatching to the constraint
helpers selected by the active microphysics and turbulence-convection models.

  - `enforce_grid_mean_microphysics_constraints!` runs for
    `NonEquilibriumMicrophysics1M` and `NonEquilibriumMicrophysics2M`.
  - `enforce_edmf_updraft_constraints!` runs for `AbstractEDMF` when
    the `edmfx_filter` configuration flag is enabled
    (`atmos.edmfx_model.filter`); it is itself a no-op for models
    without prognostic mass-flux subdomains, such as `EDOnlyEDMFX`.

Mutates `Y`; returns `nothing`. Called from `constrain_state!` after each
timestepper stage.
"""
function enforce_physical_constraints!(Y, p, t, atmos::AtmosModel)
    # Grid-mean microphysics: non-negativity + condensate ≤ total moisture.
    if atmos.microphysics_model isa
       Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M}
        enforce_grid_mean_microphysics_constraints!(Y, p, t)
    end

    # EDMF updraft constraints: only active when the filter flag is enabled.
    # Each helper is a no-op for EDOnlyEDMFX (n_prognostic_mass_flux_subdomains == 0).
    if atmos.turbconv_model isa AbstractEDMF &&
       atmos.edmfx_model.filter
        enforce_edmf_updraft_constraints!(Y, p, t, atmos.turbconv_model)
    end

    return nothing
end
