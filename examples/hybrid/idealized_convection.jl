#####
##### Idealized convection models
#####

import CloudMicrophysics.Microphysics1M as CM1
import CloudMicrophysics as CM
import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields

#####
##### No convection
#####

convection_cache(Y, convection_model::NoConvection) = (; convection_model)
convection_tendency!(Yₜ, Y, p, t, colidx, ::NoConvection) = nothing

#####
##### Betts-Miller scheme
#####

function convection_cache(Y, convection_model::BettsMiller)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        convection_model,
        τ_bm = FT(7200),
        rh_bm = FT(0.8),
        ᶜS_ρq_tot = similar(Y.c, FT),
        col_integrated_rain = similar(Fields.level(Y.c.ρ, 1), FT),
        col_integrated_snow = similar(Fields.level(Y.c.ρ, 1), FT),
    )
end

function convection_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    convection_model::BettsMiller,
)
    FT = Spaces.undertype(axes(Y.c))
    (;
        τ_bm,
        rh_bm,
    ) = p # assume ᶜts has been updated


    
    cape_calculation()
    if cape >= 0
        set_reference_profiles!(Y, p, Tp, zLZB, rp, deltaq, deltaT, qref, Tref, colidx)
        pq_calculation!(Y, p, zLZB, qref, deltaq, Pq, colidx)
        pt_calculation!(Y, p, )
        if (pq > 0) & (pt > 0)
            do_deep_convection()
        elseif (pt > 0)
            do_shallow_convection()
        else
            pq = 0
            set_profiles_to_full_model_values()
        end
    else
        pq = 0
        set_profiles_to_full_model_values()
    end

    @. Yₜ.c.ρq_tot[colidx] += ᶜS_ρq_tot[colidx]
    @. Yₜ.c.ρ[colidx] += ᶜS_ρq_tot[colidx]

    if :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] += ...
    end
    return nothing
end

function cape_calculation()
end

function set_reference_profiles!(Y, p, Tp, zLZB, rp, deltaq, deltaT, qref, Tref, colidx)
    FT = Spaces.undertype(axes(Y.c))
    (; rh_bm, ᶜp, ᶜts, ᶜT, params) = p
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])

    Tref[colidx] .= Tp[colidx]
    eref = @. rh_bm * ᶜp[colidx] * rp[colidx] / (rp[colidx] + R_d / R_v)
    @. rp[colidx] = R_d * eref / R_v / (ᶜp[colidx] - eref)
    @. qref[colidx] = rp[colidx] / (FT(1) + rp[colidx])
    ᶜα = @. ifelse(ᶜz[colidx] < zLZB[colidx], FT(1), FT(0))
    @. Tref[colidx] = ᶜα * Tref[colidx] + (FT(1) - ᶜα) * ᶜT[colidx]
    #@. Tref = ifelse(ᶜz < zLZB, Tref, ᶜT)
    @. qref[colidx] = ᶜα * qref[colidx] + (FT(1) - ᶜα) * (Y.c.ρq_tot[colidx] / Y.c.ρ[colidx])
    @. deltaT[colidx] = ᶜα * deltaT[colidx]
    @. deltaq[colidx] = ᶜα * deltaq[colidx]
end

```
   Calculate the precipitation rate Pq
```
function pq_calculation!(Y, p, zLZB, qref, deltaq, Pq, colidx)
    (; τ_bm, params) = p
    (; dt) = p.simulation
    @. deltaq[colidx] = -(Y.c.ρq_tot[colidx] / Y.c.ρ[colidx] - qref[colidx]) * dt / τ_bm 
    @. deltaq[colidx] = ifelse(ᶜz < zLZB, deltaq[colidx], FT(0))
    Operators.column_integral_definite!(
        Pq[colidx],
        -deltaq[colidx] * Y.c.ρ[colidx],
    )
end

```
   Calculate the humidity change that would be necessary
   to balance temperature change by latent heat release
```
function pt_calculation!(Y, p, zLZB, Tref, deltaT, Pt, colidx)
    FT = Spaces.undertype(axes(Y.c))

    (; τ_bm, ᶜts, ᶜT, params) = p
    (; dt) = p.simulation
    
    thermo_params = CAP.thermodynamics_params(params)
    cp_m = @. TD.cp_m(thermo_params, ᶜts[colidx])
    L_v = @. TD.latent_heat_vapor(thermo_params, ᶜts[colidx])
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])

    @. deltaT[colidx] = -(ᶜT[colidx] - Tref[colidx]) * dt / τ_bm 
    @. deltaT[colidx] = ifelse(ᶜz < zLZB, deltaT[colidx], FT(0))
    Operators.column_integral_definite!(
        Pt[colidx],
        deltaT[colidx] * cp_m / L_v * Y.c.ρ[colidx],
    )
end

function do_deep_convection!()
end

