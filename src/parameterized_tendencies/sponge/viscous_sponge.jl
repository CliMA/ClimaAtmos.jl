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
space is a column space of one or more columns (horizontal operators are undefined
there).

The horizontal vector Laplacian is computed as `∇ₕ²u = ∇ₕ(∇ₕ·u) - ∇ₕ×(∇ₕ×u)`, projected
onto the horizontal (Covariant12) axis.
"""
function viscous_sponge_tendency_uₕ(ᶜuₕ, s)
    if s isa Nothing || iscolumn(axes(ᶜuₕ))
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
    viscous_sponge_tendency!(Yₜ, Y, p)

Accumulate the full viscous-sponge tendency into `Yₜ` in-place. No-op when
`p.atmos.viscous_sponge isa Nothing` or on a single column, where horizontal
operators are undefined. Covers:

  - Horizontal (`c.uₕ`) and vertical (`f.u₃`) velocities of the grid mean,
    and — under `PrognosticEDMFX` — the per-updraft `sgsʲ.u₃`.
  - Every grid-mean scalar, through the horizontal diffusion shared with the
    second-order closures (`add_scalar_diffusionₕ!`) with the coefficient
    `β_viscous(s, z, zmax) ρ`: the enthalpy flux
    `c_p ∇ₕT + ∇ₕΦ + (h_v + Φ) ∇ₕq_tot_eff + (h_l - h_v) ∇ₕq_lcl + (h_i - h_v) ∇ₕq_icl`
    on `c.ρe_tot`, the diffusing water `q_tot_eff = q_tot - q_rai - q_sno` on
    `c.ρq_tot` and `c.ρ`, each `suspended_cloud_species` and each passive
    tracer down its own gradient. Rain and snow do not diffuse.
"""
NVTX.@annotate function viscous_sponge_tendency!(Yₜ, Y, p)
    (; viscous_sponge, turbconv_model) = p.atmos
    isnothing(viscous_sponge) && return nothing
    iscolumn(axes(Y.c)) && return nothing

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

    # Scalars: horizontal diffusion with coefficient β(z) ρ.
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(Y.c))
    zmax = Spaces.z_max(axes(ᶠz))
    ᶜβρ = @. lazy(β_viscous(viscous_sponge, ᶜz, zmax) * Y.c.ρ)
    add_scalar_diffusionₕ!(
        Yₜ,
        Y,
        p,
        ᶜβρ,
        p.scratch.ᶜtemp_scalar,
        p.scratch.ᶜtemp_scalar_2,
    )
    return nothing
end
