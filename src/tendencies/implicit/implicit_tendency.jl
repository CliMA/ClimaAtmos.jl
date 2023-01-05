#####
##### Implicit tendencies
#####

import Thermodynamics as TD
import LinearAlgebra: norm_sqr, dot
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operators
import ClimaCore.Spaces as Spaces

# TODO: add types and use dispatch instead of specializing with Val
#       for operators and optionally populate cache

# TODO: All of these should use dtγ instead of dt, but dtγ is not available in
# the implicit tendency function. Since dt >= dtγ, we can safely use dt for now.
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:none}) # used for mass ᶜρc === ᶜρ
    (; dt) = p.simulation
    (; ᶜdivᵥ, ᶠwinterp, ᶠinterp) = p.operators
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ = -(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³ * ᶠinterp(ᶜρc / ᶜρ)))
end
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:first_order})
    (; dt) = p.simulation
    (; ᶜdivᵥ, ᶠwinterp, ᶠupwind1) = p.operators
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ = -(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜρc / ᶜρ)))
end
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:third_order})
    (; dt) = p.simulation
    (; ᶜdivᵥ, ᶠwinterp, ᶠupwind3) = p.operators
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ = -(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind3(ᶠu³, ᶜρc / ᶜρ)))
end
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:boris_book})
    (; dt) = p.simulation
    (; ᶜdivᵥ, ᶠwinterp, ᶠupwind1, ᶠupwind3, ᶠfct_boris_book) = p.operators
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ =
        -(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜρc / ᶜρ))) - ᶜdivᵥ(
            ᶠwinterp(ᶜJ, ᶜρ) * ᶠfct_boris_book(
                ᶠupwind3(ᶠu³, ᶜρc / ᶜρ) - ᶠupwind1(ᶠu³, ᶜρc / ᶜρ),
                (ᶜρc / dt - ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜρc / ᶜρ))) /
                ᶜρ,
            ),
        )
end
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:zalesak})
    (; dt) = p.simulation
    (; ᶜdivᵥ, ᶠwinterp, ᶠupwind1, ᶠupwind3, ᶠfct_zalesak) = p.operators
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ =
        -(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜρc / ᶜρ))) - ᶜdivᵥ(
            ᶠwinterp(ᶜJ, ᶜρ) * ᶠfct_zalesak(
                ᶠupwind3(ᶠu³, ᶜρc / ᶜρ) - ᶠupwind1(ᶠu³, ᶜρc / ᶜρ),
                ᶜρc / ᶜρ / dt,
                (ᶜρc / dt - ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜρc / ᶜρ))) /
                ᶜρ,
            ),
        )
end

#=

TODO: can we delete this?

# Used for automatically computing the Jacobian ∂Yₜ/∂Y. Currently requires
# allocation because the cache is stored separately from Y, which means that
# similar(Y, <:Dual) doesn't allocate an appropriate cache for computing Yₜ.

function implicit_cache_vars(
    Y::Fields.FieldVector{T},
    p,
) where {T <: AbstractFloat}
    (; ᶜK, ᶜts, ᶜp) = p
    return (; ᶜK, ᶜts, ᶜp)
end

import ForwardDiff: Dual
function implicit_cache_vars(Y::Fields.FieldVector{T}, p) where {T <: Dual}
    ᶜρ = Y.c.ρ
    ᶜK = similar(ᶜρ)
    ᶜts = similar(ᶜρ, eltype(p.ts).name.wrapper{eltype(ᶜρ)})
    ᶜp = similar(ᶜρ)
    return (; ᶜK, ᶜts, ᶜp)
end
=#

function implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
    FT = Spaces.undertype(axes(Y.c))
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶠgradᵥ_ᶜΦ, ᶜts, ᶜp, ᶠu³, params, thermo_dispatcher) = p
    (; ᶜρ_ref, ᶜp_ref) = p
    (; energy_upwinding, tracer_upwinding, simulation) = p
    (; ᶠgradᵥ, ᶜinterp, ᶠinterp, ᶠwinterp) = p.operators
    C123 = Geometry.Covariant123Vector

    thermo_params = CAP.thermodynamics_params(params)
    dt = simulation.dt
    # TODO: can we move this to implicit tendencies?
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    @. ᶠu³[colidx] = Geometry.project(
        Geometry.Contravariant3Axis(),
        C123(ᶠwinterp(ᶜJ[colidx] * ᶜρ[colidx], ᶜuₕ[colidx])) + C123(ᶠw[colidx]),
    )
    compute_kinetic!(ᶜK[colidx], Y.c.uₕ[colidx], Y.f.w[colidx])
    thermo_state!(Y, p, ᶜinterp, colidx; time = t)
    @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])

    vertical_transport!(
        Yₜ.c.ρ[colidx],
        ᶠu³[colidx],
        ᶜρ[colidx],
        ᶜρ[colidx],
        p,
        Val(:none),
    )

    if :ρθ in propertynames(Y.c)
        vertical_transport!(
            Yₜ.c.ρθ[colidx],
            ᶠu³[colidx],
            ᶜρ[colidx],
            Y.c.ρθ[colidx],
            p,
            energy_upwinding,
        )
    elseif :ρe_tot in propertynames(Y.c)
        (; ᶜρh) = p
        @. ᶜρh[colidx] = Y.c.ρe_tot[colidx] + ᶜp[colidx]
        vertical_transport!(
            Yₜ.c.ρe_tot[colidx],
            ᶠu³[colidx],
            ᶜρ[colidx],
            ᶜρh[colidx],
            p,
            energy_upwinding,
        )
    elseif :ρe_int in propertynames(Y.c)
        (; ᶜρh) = p
        @. ᶜρh[colidx] = Y.c.ρe_int[colidx] + ᶜp[colidx]
        vertical_transport!(
            Yₜ.c.ρe_int[colidx],
            ᶠu³[colidx],
            ᶜρ[colidx],
            ᶜρh[colidx],
            p,
            energy_upwinding,
        )
        @. Yₜ.c.ρe_int[colidx] += ᶜinterp(
            dot(ᶠgradᵥ(ᶜp[colidx]), Geometry.Contravariant3Vector(ᶠu³[colidx])),
        )
    end

    Yₜ.c.uₕ[colidx] .= Ref(zero(eltype(Yₜ.c.uₕ[colidx])))

    @. Yₜ.f.w[colidx] = -(
        ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) / ᶠinterp(ᶜρ[colidx]) +
        (ᶠinterp(ᶜρ[colidx] - ᶜρ_ref[colidx])) / ᶠinterp(ᶜρ[colidx]) *
        ᶠgradᵥ_ᶜΦ[colidx]
    )
    if p.atmos.rayleigh_sponge isa RayleighSponge
        @. Yₜ.f.w[colidx] -= p.ᶠβ_rayleigh_w[colidx] * Y.f.w[colidx]
    end

    for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
        ᶜρcₜ = getproperty(Yₜ.c, ᶜρc_name)
        ᶜρc = getproperty(Y.c, ᶜρc_name)
        vertical_transport!(
            ᶜρcₜ[colidx],
            ᶠu³[colidx],
            ᶜρ[colidx],
            ᶜρc[colidx],
            p,
            tracer_upwinding,
        )
    end
    return nothing
end
