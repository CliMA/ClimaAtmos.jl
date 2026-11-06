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
    ᶜ∇²uₕ = horizontal_vector_laplacian(ᶜuₕ)
    return @. lazy(β_viscous(s, ᶜz, zmax) * ᶜ∇²uₕ)
end

# The lazy vector Laplacian ∇²u = ∇·∇u - ∇×(∇×u) of a horizontal velocity Field or
# slab, projected onto the horizontal (Covariant12) axis.
@inline function horizontal_vector_laplacian(ᶜuₕ)
    axis_C12 = (Geometry.Covariant12Axis(),)
    axis_C3 = (Geometry.Covariant3Axis(),)
    return @. lazy(
        wgradₕ(divₕ(ᶜuₕ)) -
        Geometry.project(axis_C12, wcurlₕ(Geometry.project(axis_C3, curlₕ(ᶜuₕ)))),
    )
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

The faces and the centers are each updated in one fused slab loop, in which
every tendency is computed before any is stored (see the slab form of
`add_scalar_diffusionₕ!`).
"""
NVTX.@annotate function viscous_sponge_tendency!(Yₜ, Y, p)
    (; viscous_sponge, turbconv_model, microphysics_model) = p.atmos
    isnothing(viscous_sponge) && return nothing
    iscolumn(axes(Y.c)) && return nothing
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(Y.c))
    zmax = Spaces.z_max(axes(ᶠz))
    draft_indices = StaticOneTo(n_mass_flux_subdomains(turbconv_model))
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜT, ᶜq_liq, ᶜq_ice) = p.precomputed
    ρχ_names = self_diffused_tracer_names(Y, false)

    # The coordinates are loop arguments: a sliced lazy coefficient that
    # contains them would leave dynamic dispatch in the GPU kernels.
    DataLayouts.foreach_slab(Yₜ.f, Y.f, ᶠz) do Yₜf, Yf, ᶠz
        ᶠβ = @. β_viscous(viscous_sponge, ᶠz, zmax)
        ᶠ∇²ws = unrolled_map(vertical_velocities(Yf, draft_indices)) do ᶠu₃
            @inline
            ᶠw = ᶠu₃.components.data.:1
            @. ᶠβ * wdivₕ(gradₕ(ᶠw))
        end
        unrolled_foreach(vertical_velocities(Yₜf, draft_indices), ᶠ∇²ws) do ᶠu₃ₜ, ᶠ∇²w
            @inline
            ᶠwₜ = ᶠu₃ₜ.components.data.:1
            @. ᶠwₜ += ᶠ∇²w
        end
    end
    DataLayouts.foreach_slab(
        Yₜ.c, Y.c, ᶜz, ᶜT, p.core.ᶜΦ, ᶜq_liq, ᶜq_ice,
    ) do Yₜc, Yc, ᶜz, ᶜT, ᶜΦ, ᶜq_liq, ᶜq_ice
        ᶜβ = @. β_viscous(viscous_sponge, ᶜz, zmax)
        ᶜ∇²uₕ = horizontal_vector_laplacian(Yc.uₕ)
        ᶜuₕₜ = @. ᶜβ * ᶜ∇²uₕ
        ᶜβρ = @. ᶜβ * Yc.ρ
        ᶜq_water =
            diffusing_water_phases(Yc, ᶜq_liq, ᶜq_ice, microphysics_model, Val(false))
        add_scalar_diffusionₕ!(Yₜc, Yc, ᶜβρ, ᶜT, ᶜΦ, ᶜq_water, thermo_params, ρχ_names)
        @. Yₜc.uₕ += ᶜuₕₜ
    end
    return nothing
end

# The u₃ of the grid mean and of every updraft in a face Field or slab.
@inline vertical_velocities(ᶠY, draft_indices) =
    (ᶠY.u₃, unrolled_map(j -> ᶠY.sgsʲs.:($j).u₃, draft_indices)...)
