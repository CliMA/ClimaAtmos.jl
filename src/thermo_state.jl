#####
##### Thermodynamic state functions
#####

import LinearAlgebra: norm_sqr
import Thermodynamics as TD
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

function thermo_state_type(Yc::Fields.Field, ::Type{FT}) where {FT}
    pns = propertynames(Yc)
    return if (:ρq_liq in pns && :ρq_ice in pns && :ρq_tot in pns)
        TD.PhaseNonEquil{FT}
    elseif :ρq_tot in pns
        TD.PhaseEquil{FT}
    else
        TD.PhaseDry{FT}
    end
end

#= High-level wrapper =#
function thermo_state!(Y::Fields.FieldVector, p, ᶜinterp, colidx)
    ᶜts = p.ᶜts[colidx]
    ᶜK = p.ᶜK[colidx]
    ᶠw = Y.f.w[colidx]
    ᶜp = p.ᶜp[colidx]
    thermo_params = CAP.thermodynamics_params(p.params)
    td = p.thermo_dispatcher
    thermo_state!(ᶜts, Y.c[colidx], thermo_params, td, ᶜinterp, ᶜK, ᶠw, ᶜp)
end

#= High-level wrapper =#
function thermo_state!(Y::Fields.FieldVector, p, ᶜinterp)
    ᶜts = p.ᶜts
    ᶜK = p.ᶜK
    ᶠw = Y.f.w
    ᶜp = p.ᶜp
    thermo_params = CAP.thermodynamics_params(p.params)
    td = p.thermo_dispatcher
    thermo_state!(ᶜts, Y.c, thermo_params, td, ᶜinterp, ᶜK, ᶠw, ᶜp)
end

thermo_state!(
    ᶜts,
    Y::Fields.FieldVector,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp,
    K = nothing,
) = thermo_state!(ᶜts, Y.c, thermo_params, td, ᶜinterp, K, Y.f.w)

#=

    thermo_state!(ᶜts, wf, thermo_params, ᶜinterp[, K, wf])

Populate the thermodynamic state, `ᶜts`, given `Field`
`Yc` and thermodynamic parameters `thermo_params`.
Interpolation `ᶜinterp` is used to interpolate vertical
velocity when computing kinetic energy (assuming it's not
given) when using the total energy formulation.
=#
function thermo_state!(
    ᶜts,
    Yc::Fields.Field,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp,
    K = nothing,
    wf = nothing,
    ᶜp = nothing,
)
    # Sometimes we want to zero out kinetic energy
    (; energy_form, moisture_model, compressibility_model) = td
    if energy_form isa TotalEnergy
        if compressibility_model isa CompressibleFluid
            if isnothing(K)
                @assert !isnothing(wf)
                C123 = Geometry.Covariant123Vector
                K = @. norm_sqr(C123(Yc.uₕ) + C123(ᶜinterp(wf))) / 2
            end
            z = Fields.local_geometry_field(Yc).coordinates.z
            thermo_state_ρe_tot!(ᶜts, Yc, thermo_params, moisture_model, z, K)
        elseif compressibility_model isa AnelasticFluid
            @assert !isnothing(ᶜp)
            z = Fields.local_geometry_field(Yc).coordinates.z
            thermo_state_ρe_tot_anelastic!(
                ᶜts,
                Yc,
                thermo_params,
                moisture_model,
                z,
                K,
                ᶜp,
            )
        end
    elseif energy_form isa InternalEnergy
        thermo_state_ρe_int!(ᶜts, Yc, thermo_params, moisture_model)
    elseif energy_form isa PotentialTemperature
        thermo_state_ρθ!(ᶜts, Yc, thermo_params, moisture_model)
    else
        error("Could not determine energy model")
    end
    return nothing
end

function thermo_state_ρe_tot_anelastic!(
    ᶜts,
    Yc,
    thermo_params,
    ::EquilMoistModel,
    z,
    ᶜK,
    ᶜp,
)
    grav = TD.Parameters.grav(thermo_params)
    @. ᶜts = TD.PhaseEquil_peq(
        thermo_params,
        ᶜp,
        Yc.ρe_tot / Yc.ρ - ᶜK - grav * z,
        Yc.ρq_tot / Yc.ρ,
    )
    return nothing
end


thermo_state(
    Y::Fields.FieldVector,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp,
    K = nothing,
    ᶜp = nothing,
) = thermo_state(Y.c, thermo_params, td, ᶜinterp, K, Y.f.w, ᶜp)

function thermo_state(
    Yc::Fields.Field,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp,
    K = nothing,
    wf = nothing,
    ᶜp = nothing,
)
    FT = Spaces.undertype(axes(Yc))
    ts_type = thermo_state_type(Yc, FT)
    ts = similar(Yc, ts_type)
    thermo_state!(ts, Yc, thermo_params, td, ᶜinterp, K, wf, ᶜp)
    return ts
end

function thermo_state_ρθ!(ts, Yc, thermo_params, ::DryModel)
    @. ts = TD.PhaseDry_ρθ(thermo_params, Yc.ρ, Yc.ρθ / Yc.ρ)
end
function thermo_state_ρθ!(ts, Yc, thermo_params, ::EquilMoistModel)
    @. ts =
        TD.PhaseEquil_ρθq(thermo_params, Yc.ρ, Yc.ρθ / Yc.ρ, Yc.ρq_tot / Yc.ρ)
end
function thermo_state_ρθ!(ts, Yc, thermo_params, ::NonEquilMoistModel)
    @. ts = TD.PhaseNonEquil_ρθq(
        thermo_params,
        Yc.ρ,
        Yc.ρθ / Yc.ρ,
        TD.PhasePartition(Yc.ρq_tot / Yc.ρ, Yc.ρq_liq / Yc.ρ, Yc.ρq_ice / Yc.ρ),
    )
end

internal_energy(Yc, K, g, z) = Yc.ρe_tot - Yc.ρ * (K + g * z)

function thermo_state_ρe_tot!(ts, Yc, thermo_params, ::DryModel, z, K)
    g = TD.Parameters.grav(thermo_params)
    @. ts =
        TD.PhaseDry(thermo_params, internal_energy(Yc, K, g, z) / Yc.ρ, Yc.ρ)
end
function thermo_state_ρe_tot!(ts, Yc, thermo_params, ::EquilMoistModel, z, K)
    g = TD.Parameters.grav(thermo_params)
    @. ts = TD.PhaseEquil_ρeq(
        thermo_params,
        Yc.ρ,
        internal_energy(Yc, K, g, z) / Yc.ρ,
        Yc.ρq_tot / Yc.ρ,
    )
end

function thermo_state_ρe_tot!(ts, Yc, thermo_params, ::NonEquilMoistModel, z, K)
    g = TD.Parameters.grav(thermo_params)
    @. ts = TD.PhaseNonEquil(
        thermo_params,
        internal_energy(Yc, K, g, z) / Yc.ρ,
        Yc.ρ,
        TD.PhasePartition(Yc.ρq_tot / Yc.ρ, Yc.ρq_liq / Yc.ρ, Yc.ρq_ice / Yc.ρ),
    )
end

function thermo_state_ρe_int!(ts, Yc, thermo_params, ::DryModel)
    @. ts = TD.PhaseDry(thermo_params, Yc.ρe_int / Yc.ρ, Yc.ρ)
end
function thermo_state_ρe_int!(ts, Yc, thermo_params, ::EquilMoistModel)
    @. ts = TD.PhaseEquil_ρeq(
        thermo_params,
        Yc.ρ,
        Yc.ρe_int / Yc.ρ,
        Yc.ρq_tot / Yc.ρ,
    )
end
function thermo_state_ρe_int!(ts, Yc, thermo_params, ::NonEquilMoistModel)
    @. ts = TD.PhaseNonEquil(
        thermo_params,
        Yc.ρe_int / Yc.ρ,
        Yc.ρ,
        TD.PhasePartition(Yc.ρq_tot / Yc.ρ, Yc.ρq_liq / Yc.ρ, Yc.ρq_ice / Yc.ρ),
    )
end
