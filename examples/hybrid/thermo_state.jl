import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

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

function get_moisture_model(Yc::Fields.Field)
    pns = propertynames(Yc)
    if (:ρq_liq in pns && :ρq_ice in pns && :ρq_tot in pns)
        return NonEquilMoistModel()
    elseif :ρq_tot in pns
        return EquilMoistModel()
    else
        return DryModel()
    end
end

function get_energy_model(Yc::Fields.Field)
    pns = propertynames(Yc)
    if :ρθ in pns
        return PotentialTemperature()
    elseif :ρe_tot in pns
        return TotalEnergy()
    elseif :ρe_int in pns
        return InternalEnergy()
    else
        error("Could not determine energy model")
    end
end

#=

    thermo_state!(ᶜts, Y, params, ᶜinterp)

Populate the thermodynamic state, `ᶜts`, given
`FieldVector` `Y` and parameters `params`. Interpolation
`ᶜinterp` is used to interpolate vertical velocity
when computing kinetic energy (assuming it's not given)
when using the total energy formulation.
=#
function thermo_state!(ᶜts, Y::Fields.FieldVector, params, ᶜinterp, K = nothing)
    # Sometimes we want to zero out kinetic energy
    Yc = Y.c
    moisture_model = get_moisture_model(Yc)
    energy_model = get_energy_model(Yc)
    thermo_params = CAP.thermodynamics_params(params)
    if energy_model isa TotalEnergy
        if isnothing(K)
            C123 = Geometry.Covariant123Vector
            K = @. norm_sqr(C123(Yc.uₕ) + C123(ᶜinterp(Y.f.w))) / 2
        end
        z = Fields.coordinate_field(Yc).z
        thermo_state_ρe_tot!(ᶜts, Yc, thermo_params, moisture_model, z, K)
    elseif energy_model isa InternalEnergy
        thermo_state_ρe_int!(ᶜts, Yc, thermo_params, moisture_model)
    elseif energy_model isa PotentialTemperature
        thermo_state_ρθ!(ᶜts, Yc, thermo_params, moisture_model)
    else
        error("Could not determine energy model")
    end
    return nothing
end

function thermo_state(Y::Fields.FieldVector, params, ᶜinterp, K = nothing)
    FT = eltype(Y)
    ts_type = thermo_state_type(Y.c, FT)
    ts = similar(Y.c, ts_type)
    thermo_state!(ts, Y, params, ᶜinterp, K)
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
