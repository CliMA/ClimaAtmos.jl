#####
##### Thermodynamic state functions
#####

import LinearAlgebra: norm_sqr
import Thermodynamics as TD
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

time_0(f::Fields.Field) = Spaces.undertype(axes(f))(0)

function thermo_state_type(
    moisture_model::AbstractMoistureModel,
    ::Type{FT},
) where {FT}
    return if moisture_model isa DryModel
        TD.PhaseDry{FT}
    elseif moisture_model isa EquilMoistModel
        TD.PhaseEquil{FT}
    elseif moisture_model isa EquilMoistModel
        TD.PhaseNonEquil{FT}
    else
        error("Unsupported moisture model")
    end
end

#= High-level wrapper =#
function thermo_state!(
    Y::Fields.FieldVector,
    p,
    ᶜinterp,
    colidx;
    time = time_0(Y.c),
)
    ᶜts = p.ᶜts[colidx]
    ᶠw = Y.f.w[colidx]
    thermo_params = CAP.thermodynamics_params(p.params)
    td = p.thermo_dispatcher
    call_thermo_state!(ᶜts, Y.c[colidx], thermo_params, td, ᶜinterp, ᶠw; time)
end

#= High-level wrapper =#
function thermo_state!(Y::Fields.FieldVector, p, ᶜinterp; time = time_0(Y.c))
    ᶜts = p.ᶜts
    ᶠw = Y.f.w
    thermo_params = CAP.thermodynamics_params(p.params)
    td = p.thermo_dispatcher
    call_thermo_state!(ᶜts, Y.c, thermo_params, td, ᶜinterp, ᶠw; time)
end

thermo_state!(
    ᶜts,
    Y::Fields.FieldVector,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp;
    time = time_0(Y.c),
) = call_thermo_state!(ᶜts, Y.c, thermo_params, td, ᶜinterp, Y.f.w; time)

#=

    call_thermo_state!(
        ᶜts,
        Yc,
        thermo_params,
        td::ThermoDispatcher,
        ᶜinterp,
        ᶠw;
        time = time_0(Y.c)
    )

Populate the thermodynamic state, `ᶜts`, given
 - `Yc` the cell centered prognostic fields
 - `thermo_params` thermodynamic parameters
 - `td` thermodynamic dispatcher (contains energy form, moisture model etc.)
 - `ᶠw` vertical velocity
 - `ᶜinterp` interpolation `ᶜinterp` used to interpolate vertical
    velocity when computing kinetic energy when using the total energy formulation.
=#
function call_thermo_state!(
    ᶜts,
    Yc::Fields.Field,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp,
    ᶠw;
    time = time_0(Y.c),
)
    try
        base_thermo_state!(ᶜts, Yc, thermo_params, td, ᶜinterp, ᶠw; time)
    catch
        @error "Failure to establish a thermodynamic state at time $time"
        base_thermo_state!(ᶜts, Yc, thermo_params, td, ᶜinterp, ᶠw; time)
    end

end

function base_thermo_state!(
    ᶜts,
    Yc::Fields.Field,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp,
    ᶠw;
    time = time_0(Y.c),
)
    # Sometimes we want to zero out kinetic energy
    (; energy_form, moisture_model) = td
    if energy_form isa TotalEnergy
        z = Fields.local_geometry_field(Yc).coordinates.z
        thermo_state_ρe_tot!(
            ᶜts,
            Yc,
            thermo_params,
            moisture_model,
            z,
            ᶜinterp,
            ᶠw;
            time,
        )
    elseif energy_form isa InternalEnergy
        thermo_state_ρe_int!(ᶜts, Yc, thermo_params, moisture_model; time)
    elseif energy_form isa PotentialTemperature
        thermo_state_ρθ!(ᶜts, Yc, thermo_params, moisture_model; time)
    else
        error("Could not determine energy model")
    end
    return nothing
end

thermo_state(
    Y::Fields.FieldVector,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp;
    time = time_0(Y.c),
) = thermo_state(Y.c, thermo_params, td, ᶜinterp, Y.f.w; time)

function thermo_state(
    Yc::Fields.Field,
    thermo_params,
    td::ThermoDispatcher,
    ᶜinterp,
    ᶠw::Fields.Field;
    time = time_0(Y.c),
)
    FT = Spaces.undertype(axes(Yc))
    ts_type = thermo_state_type(td.moisture_model, FT)
    ts = similar(Yc, ts_type)
    call_thermo_state!(ts, Yc, thermo_params, td, ᶜinterp, ᶠw; time)
    return ts
end

function thermo_state_ρθ!(ts, Yc, thermo_params, ::DryModel; time)
    @. ts = TD.PhaseDry_ρθ(thermo_params, Yc.ρ, Yc.ρθ / Yc.ρ)
end
function thermo_state_ρθ!(ts, Yc, thermo_params, ::EquilMoistModel; time)
    @. ts =
        TD.PhaseEquil_ρθq(thermo_params, Yc.ρ, Yc.ρθ / Yc.ρ, Yc.ρq_tot / Yc.ρ)
end
function thermo_state_ρθ!(ts, Yc, thermo_params, ::NonEquilMoistModel; time)
    @. ts = TD.PhaseNonEquil_ρθq(
        thermo_params,
        Yc.ρ,
        Yc.ρθ / Yc.ρ,
        TD.PhasePartition(Yc.ρq_tot / Yc.ρ, Yc.ρq_liq / Yc.ρ, Yc.ρq_ice / Yc.ρ),
    )
end

internal_energy(Yc, K, g, z) = Yc.ρe_tot - Yc.ρ * (K + g * z)

function thermo_state_ρe_tot!(
    ts,
    Yc,
    thermo_params,
    ::DryModel,
    z,
    ᶜinterp,
    ᶠw;
    time,
)
    C123 = Geometry.Covariant123Vector
    g = TD.Parameters.grav(thermo_params)
    @. ts = TD.PhaseDry(
        thermo_params,
        internal_energy(
            Yc,
            (norm_sqr(C123(Yc.uₕ) + C123(ᶜinterp(ᶠw))) / 2),
            g,
            z,
        ) / Yc.ρ,
        Yc.ρ,
    )
end
function thermo_state_ρe_tot!(
    ts,
    Yc,
    thermo_params,
    ::EquilMoistModel,
    z,
    ᶜinterp,
    ᶠw;
    time,
)
    C123 = Geometry.Covariant123Vector
    g = TD.Parameters.grav(thermo_params)
    @. ts = TD.PhaseEquil_ρeq(
        thermo_params,
        Yc.ρ,
        internal_energy(
            Yc,
            (norm_sqr(C123(Yc.uₕ) + C123(ᶜinterp(ᶠw))) / 2),
            g,
            z,
        ) / Yc.ρ,
        Yc.ρq_tot / Yc.ρ,
    )
end

function thermo_state_ρe_tot!(
    ts,
    Yc,
    thermo_params,
    ::NonEquilMoistModel,
    z,
    ᶜinterp,
    ᶠw;
    time,
)
    C123 = Geometry.Covariant123Vector
    g = TD.Parameters.grav(thermo_params)
    @. ts = TD.PhaseNonEquil(
        thermo_params,
        internal_energy(
            Yc,
            (norm_sqr(C123(Yc.uₕ) + C123(ᶜinterp(ᶠw))) / 2),
            g,
            z,
        ) / Yc.ρ,
        Yc.ρ,
        TD.PhasePartition(Yc.ρq_tot / Yc.ρ, Yc.ρq_liq / Yc.ρ, Yc.ρq_ice / Yc.ρ),
    )
end

function thermo_state_ρe_int!(ts, Yc, thermo_params, ::DryModel; time)
    @. ts = TD.PhaseDry(thermo_params, Yc.ρe_int / Yc.ρ, Yc.ρ)
end
function thermo_state_ρe_int!(ts, Yc, thermo_params, ::EquilMoistModel; time)
    @. ts = TD.PhaseEquil_ρeq(
        thermo_params,
        Yc.ρ,
        Yc.ρe_int / Yc.ρ,
        Yc.ρq_tot / Yc.ρ,
    )
end
function thermo_state_ρe_int!(ts, Yc, thermo_params, ::NonEquilMoistModel; time)
    @. ts = TD.PhaseNonEquil(
        thermo_params,
        Yc.ρe_int / Yc.ρ,
        Yc.ρ,
        TD.PhasePartition(Yc.ρq_tot / Yc.ρ, Yc.ρq_liq / Yc.ρ, Yc.ρq_ice / Yc.ρ),
    )
end
