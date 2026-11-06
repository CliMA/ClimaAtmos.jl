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
    pressure_drag_coefficient(α_d, scale_height, aʲ, a⁰, a_min, a_max)

Return the coefficient `d_c` [1/m] of the pairwise form drag between an updraft
of area fraction `aʲ` and the environment of area fraction `a⁰`,

    d_c = α_d / (2 H) (1 / √clamp(aʲ, a_min, a_max) + 1 / √clamp(a⁰, 1 - a_max, 1)),

so that the drag acceleration on the updraft is `a⁰ d_c (wʲ - w⁰) |wʲ - w⁰|`,
with `a⁰ = ρ̂⁰/ρ` the environment weight of the pairwise form. The
length scale of each subdomain is `H √a` with `H = scale_height`, and the
harmonic mean of the two gives the sum of inverse square roots. The updraft
radius may shrink to `a_min`; the environment is clamped from below at
`1 - a_max`, which keeps the coefficient finite when the implicit velocity
solve is handed an extrapolated `a⁰` outside `[0, 1]`.

Shared by the implicit updraft momentum solve
(`solve_sgs_u₃_implicit_stage_analytic!`) and the TKE return-to-isotropy source
(the pressure-drag term of `edmfx_tke_sources!`), so the kinetic energy the drag
removes from the updraft is the energy the TKE budget receives.
"""
pressure_drag_coefficient(α_d, scale_height, aʲ, a⁰, a_min, a_max) =
    α_d / (2 * scale_height) * (
        1 / sqrt(clamp(aʲ, a_min, a_max)) +
        1 / sqrt(clamp(a⁰, 1 - a_max, one(a⁰)))
    )

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
    enforce_grid_mean_microphysics_constraints!(Y, p, t)

Clip the grid-mean condensate tracers to non-negative values and rescale them
so their sum cannot exceed the available total moisture.

`ρq_lcl`, `ρq_icl`, `ρq_rai`, and `ρq_sno` are first floored at zero, then all
four are multiplied by `ratio = min(1, ρq_tot / Σ ρq_cond)`; `ratio` is zero
where either the condensate sum or `ρq_tot` is below `ϵ_numerics`, which
removes condensate in cells with no (or negative) total water.

Mutates `Y.c` and uses `p.scratch`; returns `nothing`. Called from
`enforce_physical_constraints!` for the 1M and 2M non-equilibrium
microphysics models.
"""
function enforce_grid_mean_microphysics_constraints!(Y, p, t)
    FT = eltype(p.params)
    (zero_FT, one_FT, ϵ) = (zero(FT), one(FT), ϵ_numerics(FT))
    DataLayouts.foreach_column(Y.c) do Yc
        ρq_lcl = @. max(zero_FT, Yc.ρq_lcl)
        ρq_icl = @. max(zero_FT, Yc.ρq_icl)
        ρq_rai = @. max(zero_FT, Yc.ρq_rai)
        ρq_sno = @. max(zero_FT, Yc.ρq_sno)
        ρq_cond = @. ρq_lcl + ρq_icl + ρq_rai + ρq_sno
        ratio = @. ifelse(
            (ρq_cond > ϵ) & (Yc.ρq_tot > ϵ),
            min(one_FT, Yc.ρq_tot / ρq_cond),
            zero_FT,
        )
        @. Yc.ρq_lcl = ρq_lcl * ratio
        @. Yc.ρq_icl = ρq_icl * ratio
        @. Yc.ρq_rai = ρq_rai * ratio
        @. Yc.ρq_sno = ρq_sno * ratio
    end
    return nothing
end

"""
    enforce_edmf_updraft_constraints!(Y, p, t, turbconv_model)

Clip the prognostic PROPHET (`EDMFX` in code) updraft state to a physically
admissible range and relax degenerate updrafts toward the grid mean.

For each of the `n_prognostic_mass_flux_subdomains(turbconv_model)` updrafts:

  - `Y.c.sgsʲs.:(j).ρa` is clamped to `[0, ᶜρʲ]`, i.e. the area fraction to
    `[0, 1]`.
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
`EDOnlyEDMFX`). Mutates `Y.c.sgsʲs` and `Y.f.sgsʲs`; returns `nothing`. Called
from `enforce_physical_constraints!`.
"""
function enforce_edmf_updraft_constraints!(Y, p, t, turbconv_model)
    FT = eltype(p.params)
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    n == 0 && return nothing
    (; ᶜh_tot, ᶜK, ᶜρʲs) = p.precomputed
    (zero_FT, one_FT, ϵ) = (zero(FT), one(FT), ϵ_numerics(FT))
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
        for (ᶜsgsʲ, ᶠu₃ʲ, ᶜρʲ) in drafts
            # Clip the area fraction to [0, 1] and the vertical velocity to
            # non-negative values, with zero velocity where the face-averaged
            # area fraction is negligible.
            ᶜρa = @. max(zero_FT, min(ᶜsgsʲ.ρa, ᶜρʲ))
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
            ᶜratio = @. ifelse(
                (ᶜq_cond > ϵ) & (ᶜq_totʲ > ϵ),
                min(one_FT, ᶜq_totʲ / ᶜq_cond),
                zero_FT,
            )
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
