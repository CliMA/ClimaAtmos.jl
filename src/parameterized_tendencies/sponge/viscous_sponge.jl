#####
##### Viscous sponge
#####

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

αₘ(s::ViscousSponge, z) = ifelse(z > s.zd, s.κ₂, zero(s.κ₂))
ζ_viscous(s::ViscousSponge, z, zmax) = sinpi((z - s.zd) / (zmax - s.zd) / 2)^2

"""
    β_viscous(s::ViscousSponge, z, zmax)

Return the viscous sponge coefficient
``β(z) = κ₂ · sin²(π (z - zd) / (2 (zmax - zd)))`` for `z > zd` and zero below [m²/s].

# Arguments

  - `s`: The `ViscousSponge` model, with lower damping height `zd` [m] and damping
    coefficient `κ₂` [m²/s].
  - `z`: Altitude [m].
  - `zmax`: Domain top height [m].
"""
β_viscous(s::ViscousSponge, z, zmax) = αₘ(s, z) * ζ_viscous(s, z, zmax)

"""
    viscous_sponge_tendency_uₕ(ᶜuₕ, s)

Return a lazy broadcast of the viscous sponge tendency `β_viscous(s, z, zmax) * ∇ₕ²uₕ` for
the cell-center horizontal velocity, or a `NullBroadcasted` when `s isa Nothing` or the
space is a single column (horizontal operators are undefined there).

The horizontal vector Laplacian is computed as `∇ₕ²u = ∇ₕ(∇ₕ·u) - ∇ₕ×(∇ₕ×u)`, projected
onto the horizontal (Covariant12) axis.
"""
function viscous_sponge_tendency_uₕ(ᶜuₕ, s)
    if s isa Nothing || axes(ᶜuₕ) isa Spaces.FiniteDifferenceSpace
        return NullBroadcasted()
    end
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜuₕ))
    zmax = Spaces.z_max(axes(ᶠz))
    axis_C12 = (Geometry.Covariant12Axis(),)
    axis_C3 = (Geometry.Covariant3Axis(),)
    # vector Laplacian: ∇²u = ∇·∇u - ∇×(∇×u)
    ᶜ∇²uₕ = @. lazy(
        wgradₕ(divₕ(ᶜuₕ)) -
        Geometry.project(axis_C12, wcurlₕ(Geometry.project(axis_C3, curlₕ(ᶜuₕ)))),
    )
    return @. lazy(β_viscous(s, ᶜz, zmax) * ᶜ∇²uₕ)
end

"""
    viscous_sponge_tendency_u₃(u₃, s)

Return a lazy broadcast of the viscous sponge tendency `β_viscous(s, z, zmax) * ∇ₕ²u₃` for
the physical component of the cell-face vertical velocity, or a `NullBroadcasted` when
`s isa Nothing`.
"""
function viscous_sponge_tendency_u₃(u₃, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶠz) = z_coordinate_fields(axes(u₃))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶠz, zmax) * wdivₕ(gradₕ(u₃.components.data.:1)))
end

"""
    viscous_sponge_tendency_ρe_tot_dry(ᶜρ, ᶜs_d, s)

Return a lazy broadcast of the dry-static-energy piece of the viscous sponge
tendency for total energy density,
`β_viscous(s, z, zmax) * ∇ₕ·(ρ ∇ₕs_d)`, or a `NullBroadcasted` when
`s isa Nothing`.
"""
function viscous_sponge_tendency_ρe_tot_dry(ᶜρ, ᶜs_d, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜs_d)))
end

"""
    viscous_sponge_tendency_ρe_tot_water(ᶜρ, ᶜh_eff_plus_Φ, ᶜq_tot_eff, s)

Return a lazy broadcast of the water-enthalpy piece of the viscous sponge
tendency for total energy density,
`β_viscous(s, z, zmax) * ∇ₕ·(ρ (h_eff + Φ) ∇ₕq_tot_eff)`, or a `NullBroadcasted`
when `s isa Nothing`.
"""
function viscous_sponge_tendency_ρe_tot_water(
    ᶜρ,
    ᶜh_eff_plus_Φ,
    ᶜq_tot_eff,
    s,
)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(
        β_viscous(s, ᶜz, zmax) *
        wdivₕ(ᶜρ * ᶜh_eff_plus_Φ * gradₕ(ᶜq_tot_eff)),
    )
end

"""
    viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, s)

Return a lazy broadcast of the viscous sponge tendency
`β_viscous(s, z, zmax) * ∇ₕ·(ρ ∇ₕχ)` for a cell-center specific tracer `ᶜχ`, or a
`NullBroadcasted` when `s isa Nothing`.
"""
function viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, s)
    s isa Nothing && return NullBroadcasted()
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜρ))
    zmax = Spaces.z_max(axes(ᶠz))
    return @. lazy(β_viscous(s, ᶜz, zmax) * wdivₕ(ᶜρ * gradₕ(ᶜχ)))
end

"""
    viscous_sponge_tendency!(Yₜ, Y, p)

Accumulate the full viscous-sponge tendency into `Yₜ` in-place. No-op when
`p.atmos.viscous_sponge isa Nothing`. Covers:

  - Horizontal (`c.uₕ`) and vertical (`f.u₃`) velocities of the grid mean,
    and — under `PrognosticEDMFX` — the per-updraft `sgsʲ.u₃`.
  - `c.ρe_tot`: split-form diffusion of dry static energy plus the
    h_eff-weighted diffusion of `q_tot_eff = q_tot - q_rai - q_sno` (rain
    and snow excluded from both `q_tot_eff` and the enthalpy weighting),
    matching the hyperdiffusion and vertical-diffusion energy treatments.
  - Total-water mass: diffuse `q_tot_eff`; apply the aggregate tendency to
    `c.ρq_tot` and `c.ρ`; distribute it proportionally to the suspended
    cloud mass species (`ρq_lcl`, `ρq_icl`).
  - Rain and snow are not sponge-diffused.
  - Passive (non-microphysics) grid-scale tracers: full independent
    diffusion.
"""
NVTX.@annotate function viscous_sponge_tendency!(Yₜ, Y, p)
    (; viscous_sponge, microphysics_model, turbconv_model) = p.atmos
    isnothing(viscous_sponge) && return nothing
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜT) = p.precomputed
    (; ᶜΦ) = p.core
    ᶜρ = Y.c.ρ

    # Velocities (grid mean and, under EDMFX, per-updraft u₃).
    vst_uₕ = viscous_sponge_tendency_uₕ(Y.c.uₕ, viscous_sponge)
    vst_u₃ = viscous_sponge_tendency_u₃(Y.f.u₃, viscous_sponge)
    @. Yₜ.c.uₕ += vst_uₕ
    @. Yₜ.f.u₃.components.data.:1 += vst_u₃
    if turbconv_model isa PrognosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        for j in 1:n
            ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
            vst_u₃ʲ = viscous_sponge_tendency_u₃(ᶠu₃ʲ, viscous_sponge)
            @. Yₜ.f.sgsʲs.:($$j).u₃.components.data.:1 += vst_u₃ʲ
        end
    end

    # Energy: dry-static-energy piece (applies in all configurations).
    ᶜs_d = @. lazy(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ))
    vst_ρe_tot_dry =
        viscous_sponge_tendency_ρe_tot_dry(ᶜρ, ᶜs_d, viscous_sponge)
    @. Yₜ.c.ρe_tot += vst_ρe_tot_dry

    # Water pieces: energy water-enthalpy flux + q_tot_eff mass diffusion +
    # proportional distribution to suspended cloud species. Reuse the shared
    # `ᶜdiffusing_water` / `ᶜsuspended_water` / `ᶜh_eff_plus_Φ!` helpers so
    # the split matches the vertical-diffusion and hyperdiffusion tendencies.
    if !(microphysics_model isa DryModel)
        FT = eltype(Y)
        ϵ_FT = eps(FT)
        ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
        ᶜq_vap, ᶜq_lcl, ᶜq_icl = ᶜsuspended_water(Y, p)
        ᶜh_eff_plus_Φ = ᶜh_eff_plus_Φ!(
            p.scratch.ᶜtemp_scalar,
            thermo_params,
            ᶜT,
            ᶜΦ,
            ᶜq_vap,
            ᶜq_lcl,
            ᶜq_icl,
        )
        vst_ρe_tot_water = viscous_sponge_tendency_ρe_tot_water(
            ᶜρ,
            ᶜh_eff_plus_Φ,
            ᶜq_tot_eff,
            viscous_sponge,
        )
        @. Yₜ.c.ρe_tot += vst_ρe_tot_water

        # Aggregate water mass tendency; reuse the generic tracer sponge on
        # `q_tot_eff`. Applied to `ρq_tot` and `ρ`, then distributed to
        # suspended cloud species (and matching number densities).
        ᶜρq_tot_diff = p.scratch.ᶜtemp_scalar_2
        vst_ρq_tot =
            viscous_sponge_tendency_tracer(ᶜρ, ᶜq_tot_eff, viscous_sponge)
        @. ᶜρq_tot_diff = vst_ρq_tot
        @. Yₜ.c.ρq_tot += ᶜρq_tot_diff
        @. Yₜ.c.ρ += ᶜρq_tot_diff
        ᶜratio = p.scratch.ᶜtemp_scalar_3
        for (ρq_name, ρn_name) in (
            (@name(c.ρq_lcl), @name(c.ρn_lcl)),
            (@name(c.ρq_icl), @name(c.ρn_icl)),
        )
            MatrixFields.has_field(Y, ρq_name) || continue
            ᶜρq = MatrixFields.get_field(Y, ρq_name)
            ᶜρqₜ = MatrixFields.get_field(Yₜ, ρq_name)
            @. ᶜratio = max(
                FT(0),
                min(FT(1), specific(ᶜρq, Y.c.ρ) / max(ᶜq_tot_eff, ϵ_FT)),
            )
            @. ᶜρqₜ += ᶜratio * ᶜρq_tot_diff
            if MatrixFields.has_field(Y, ρn_name)
                ᶜρn = MatrixFields.get_field(Y, ρn_name)
                ᶜρnₜ = MatrixFields.get_field(Yₜ, ρn_name)
                @. ᶜρnₜ +=
                    ᶜratio * max(FT(0), ᶜρn) / max(ᶜρq, ϵ_FT) *
                    ᶜρq_tot_diff
            end
        end
    end

    # Passive (non-microphysics) grid-scale tracers.
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ρχ_name in microphysics_tracer_names(Y) && return
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        vst_tracer = viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, viscous_sponge)
        @. ᶜρχₜ += vst_tracer
    end
    return nothing
end
