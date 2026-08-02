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

Form the buoyancy acceleration vector due to a density anomaly ([`buoyancy`](@ref)
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
surface mass flux magnitude (via [`surface_mass_flux`](@ref)) and to
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

with `a_s` given by [`surface_mass_flux_coefficient`](@ref) and
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
[`enforce_physical_constraints!`](@ref) for the 1M and 2M non-equilibrium
microphysics models.
"""
function enforce_grid_mean_microphysics_constraints!(Y, p, t)
    FT = eltype(p.params)
    ρq_cond = p.scratch.ᶜtemp_scalar
    ratio = p.scratch.ᶜtemp_scalar_2
    @. Y.c.ρq_lcl = max(FT(0), Y.c.ρq_lcl)
    @. Y.c.ρq_icl = max(FT(0), Y.c.ρq_icl)
    @. Y.c.ρq_rai = max(FT(0), Y.c.ρq_rai)
    @. Y.c.ρq_sno = max(FT(0), Y.c.ρq_sno)

    @. ρq_cond = Y.c.ρq_lcl + Y.c.ρq_icl + Y.c.ρq_rai + Y.c.ρq_sno
    @. ratio = ifelse(
        (ρq_cond > ϵ_numerics(FT)) & (Y.c.ρq_tot > ϵ_numerics(FT)),
        min(FT(1), Y.c.ρq_tot / ρq_cond),
        FT(0),
    )
    @. Y.c.ρq_lcl *= ratio
    @. Y.c.ρq_icl *= ratio
    @. Y.c.ρq_rai *= ratio
    @. Y.c.ρq_sno *= ratio
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
    the grid-mean [`enforce_grid_mean_microphysics_constraints!`](@ref).

No-op when `n_prognostic_mass_flux_subdomains(turbconv_model) == 0` (e.g.
`EDOnlyEDMFX`). Mutates `Y.c.sgsʲs` and `Y.f.sgsʲs`; returns `nothing`. Called
from [`enforce_physical_constraints!`](@ref).

# Notes

The tracer branch reads and writes `Y.c.sgsʲs.:(1)` rather than subdomain `j`,
so with more than one updraft only the first is corrected.
"""
function enforce_edmf_updraft_constraints!(Y, p, t, turbconv_model)
    FT = eltype(p.params)
    n = n_prognostic_mass_flux_subdomains(turbconv_model)
    n == 0 && return nothing
    (; ᶜh_tot, ᶜK, ᶜρʲs) = p.precomputed
    for j in 1:n
        # clip updraft area fraction and vertical velocity to non-negative values
        @. Y.c.sgsʲs.:($$j).ρa = max(0, min(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
        @. Y.f.sgsʲs.:($$j).u₃ =
            C3(max(Y.f.sgsʲs.:($$j).u₃.components.data.:1, 0))

        # clip updraft velocity to zero when face-averaged area fraction is negligible
        @. Y.f.sgsʲs.:($$j).u₃ = ifelse(
            ᶠinterp(Y.c.sgsʲs.:($$j).ρa) < ϵ_numerics(FT),
            C3(0),
            Y.f.sgsʲs.:($$j).u₃,
        )

        # relax updraft mse and q_tot toward the grid mean when ρa is negligible
        @. Y.c.sgsʲs.:($$j).mse = ifelse(
            Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
            ᶜh_tot - ᶜK,
            Y.c.sgsʲs.:($$j).mse,
        )
        @. Y.c.sgsʲs.:($$j).q_tot = ifelse(
            Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
            specific(Y.c.ρq_tot, Y.c.ρ),
            # ensure mass conservation: ρaχʲ < ρχ
            min(
                max(0, Y.c.sgsʲs.:($$j).q_tot),
                max(0, Y.c.ρq_tot) / Y.c.sgsʲs.:($$j).ρa,
            ),
        )

        # Auto-discovered SGS tracers: relax toward grid mean when ρa is
        # negligible; enforce mass conservation bound ρaχʲ < ρχ.
        for χ_name in sgs_tracer_names(Y)
            ρχ_name = get_ρχ_name(χ_name)
            MatrixFields.has_field(Y.c, ρχ_name) || continue
            ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            @. ᶜχʲ = ifelse(
                Y.c.sgsʲs.:($$j).ρa < ϵ_numerics(FT),
                specific(ᶜρχ, Y.c.ρ),
                # ensure mass conservation: ρaχʲ < ρχ
                min(max(0, ᶜχʲ), max(0, ᶜρχ) / Y.c.sgsʲs.:($$j).ρa),
            )
        end

        # Within-subdomain condensate rescaling: ensure
        # q_lclʲ+q_iclʲ+q_raiʲ+q_snoʲ ≤ q_totʲ. The GM ↔ SGS bound above
        # already clipped each χʲ ≥ 0, so we only need the ratio rescale.
        # Multiplying by ratio ≤ 1 preserves the ρaχʲ ≤ ρχ bound.
        if p.atmos.microphysics_model isa Union{NonEquilibriumMicrophysics1M,
            NonEquilibriumMicrophysics2M}
            q_cond = p.scratch.ᶜtemp_scalar
            ratio = p.scratch.ᶜtemp_scalar_2
            @. q_cond =
                Y.c.sgsʲs.:($$j).q_lcl + Y.c.sgsʲs.:($$j).q_icl +
                Y.c.sgsʲs.:($$j).q_rai + Y.c.sgsʲs.:($$j).q_sno
            @. ratio = ifelse(
                (q_cond > ϵ_numerics(FT)) &
                (Y.c.sgsʲs.:($$j).q_tot > ϵ_numerics(FT)),
                min(FT(1), Y.c.sgsʲs.:($$j).q_tot / q_cond),
                FT(0),
            )
            @. Y.c.sgsʲs.:($$j).q_lcl *= ratio
            @. Y.c.sgsʲs.:($$j).q_icl *= ratio
            @. Y.c.sgsʲs.:($$j).q_rai *= ratio
            @. Y.c.sgsʲs.:($$j).q_sno *= ratio
        end
    end
    return nothing
end

"""
    enforce_physical_constraints!(Y, p, t, atmos)

Enforce physical consistency of the state `Y` by dispatching to the constraint
helpers selected by the active microphysics and turbulence-convection models.

  - [`enforce_grid_mean_microphysics_constraints!`](@ref) runs for
    `NonEquilibriumMicrophysics1M` and `NonEquilibriumMicrophysics2M`.
  - [`enforce_edmf_updraft_constraints!`](@ref) runs for `AbstractEDMF` when
    the `edmfx_filter` configuration flag is enabled
    (`atmos.edmfx_model.filter isa Val{true}`); it is itself a no-op for models
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
       atmos.edmfx_model.filter isa Val{true}
        enforce_edmf_updraft_constraints!(Y, p, t, atmos.turbconv_model)
    end

    return nothing
end
