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
    (; ᶜts, ᶜT, params, τ_bm, rh_bm) = p # assume ᶜts has been updated
    (; dt) = p.simulation
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])

    cape_calculation()
    if cape >= 0
        set_reference_profiles!(Y, p, Tp, zLZB, rp, deltaq, deltaT, qref, Tref, colidx)
        pq_calculation!(Y, p, zLZB, qref, deltaq, Pq, colidx)
        pt_calculation!(Y, p, )
        if (parent(Pq[colidx])[] > FT(0)) & (parent(Pt[colidx])[] > FT(0))
            do_deep_convection!()
        elseif parent(Pt[colidx])[] > FT(0)
            do_shallow_convection!()
        else
            @. Pq[colidx] = FT(0)
            @. Tref[colidx] = ᶜT[colidx]
            @. qref[colidx] = Y.c.ρq_tot[colidx] / Y.c.ρ[colidx]
            @. deltaT[colidx] = FT(0)
            @. deltaq[colidx] = FT(0)
        end
    else
        @. Pq[colidx] = FT(0)
        @. Tref[colidx] = ᶜT[colidx]
        @. qref[colidx] = Y.c.ρq_tot[colidx] / Y.c.ρ[colidx]
        @. deltaT[colidx] = FT(0)
        @. deltaq[colidx] = FT(0)
    end

    T_tend = @. Pt[colidx] / dt
    qt_tend = @. Pq[colidx] / dt

    @. Yₜ.c.ρ[colidx] += Y.c.ρ[colidx] * Pq[colidx] / dt
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * Pq[colidx] / dt

    if :ρe_tot in propertynames(Y.c)
        # TODO: add qt tendency
        @. Yₜ.c.ρe_tot[colidx] += TD.cv_m(thermo_params, ᶜts) * Pt[colidx] / dt
    end

    return nothing
end

function cape_calculation!(Y, p, colidx)
    (; rh_bm, ᶜp, ᶜts, ᶜT, params) = p
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])
    nocape = true
    cape = FT(0)
    cin = FT(0)
    pLZB = FT(0)
    kLFC = Int(0)
    kLZB = Int(0)
    @. Tp[colidx] = ᶜT[colidx]
    @. rp[colidx] = TD.shum_to_mixing_ratio(Y.c.ρq_tot / Y.c.ρ, Y.c.ρq_tot / Y.c.ρ)
    @. T_virtual[colidx] = TD.virtual_temperature(thermo_params, ᶜts)
    saturated = saturated(thermo_params, Fields.level(ᶜts[colidx], 1))
    
    cape_below_lcl!()
    cape_above_lcl!()
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
    #ᶜα = @. ifelse(ᶜz[colidx] < zLZB[colidx], FT(1), FT(0))
    @. Tref[colidx] = ifelse(ᶜz[colidx] < zLZB[colidx], Tref[colidx], ᶜT[colidx])
    @. qref[colidx] = ifelse(ᶜz[colidx] < zLZB[colidx], qref[colidx], Y.c.ρq_tot[colidx] / Y.c.ρ[colidx])
    @. deltaT[colidx] = ifelse(ᶜz[colidx] < zLZB[colidx], deltaT[colidx], FT(0))
    @. deltaq[colidx] = ifelse(ᶜz[colidx] < zLZB[colidx], deltaq[colidx], FT(0))
end

```
   Calculate the precipitation rate Pq
```
function pq_calculation!(Y, p, zLZB, qref, deltaq, Pq, colidx)
    (; τ_bm, params) = p
    (; dt) = p.simulation
    @. deltaq[colidx] = -(Y.c.ρq_tot[colidx] / Y.c.ρ[colidx] - qref[colidx]) * dt / τ_bm 
    @. deltaq[colidx] = ifelse(ᶜz <= zLZB, deltaq[colidx], FT(0))
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
    @. deltaT[colidx] = ifelse(ᶜz <= zLZB, deltaT[colidx], FT(0))
    Operators.column_integral_definite!(
        Pt[colidx],
        deltaT[colidx] * cp_m / L_v * Y.c.ρ[colidx],
    )
end

function do_deep_convection()
    if Pq[colidx] > Pt[colidx]
        do_change_time_scale_deepconv()
    else
        do_change_tref_deepconv()
    end
end

function do_change_time_scale_deepconv!(Y, p, zLZB, Pt, Pq, deltaq, colidx)
    (; τ_bm) = p

    @. deltaq[colidx] = ifelse(ᶜz <= zLZB, deltaq[colidx] * Pt[colidx] / Pq[colidx], deltaq[colidx])
    @. Pq[colidx] = Pt[colidx]
end

function do_change_tref_deepconv(Y, p, zLZB, deltaT, deltaq, Tref, colidx)
    (; τ_bm) = p
    (; dt) = p.simulation

    deltaT1 = @. -(deltaT[colidx] + L_v / cp_m * deltaq[colidx])
    @. deltaT1 = ifelse(ᶜz[colidx] <= zLZB, deltaT1, FT(0))
    Operators.column_integral_definite!(
        deltak,
        deltaT1 * Y.c.ρ[colidx]
    )
    ρ1 = @. ifelse(ᶜz[colidx] <= zLZB, Y.c.ρ, FT(0))
    Operators.column_integral_definite!(
        deltap,
        ρ1,
    )
    @. deltak = deltak / deltap
    deltak1 = @. ifelse(ᶜz[colidx] <= zLZB, deltak, FT(0))

    @. Tref[colidx] = Tref[colidx] + deltak1 * τ_bm / dt
    @. deltaT[colidx] = deltaT[colidx] + deltak1
end

function do_shallow_convection!(Y, p, colidx)
    z_top, k_top = level_of_zero_precip()

    (; ᶜT, ᶜts, params) = p
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])

    if Pq[colidx] .> 0
        change_Tref_LZB_shallowconv()
    else
        if z_top == zLZB
            @. Tref[colidx] = ifelse(ᶜz[colidx] == Fields.level(ᶜz[colidx], 1), ᶜT[colidx], Tref[colidx])
            @. qref[colidx] = ifelse(ᶜz[colidx] == Fields.level(ᶜz[colidx], 1), Y.c.ρq_tot[colidx] / Y.c.ρ[colidx], qref[colidx])
            @. deltaT[colidx] = ifelse(ᶜz[colidx] == Fields.level(ᶜz[colidx], 1), 0, deltaT[colidx])
            @. deltaq[colidx] = ifelse(ᶜz[colidx] == Fields.level(ᶜz[colidx], 1), 0, deltaq[colidx])
        else
            @. Tref[colidx] = ifelse(ᶜz[colidx] <= zLZB & ᶜz[colidx] >= z_top, ᶜT[colidx], Tref[colidx])
            @. qref[colidx] = ifelse(ᶜz[colidx] <= zLZB & ᶜz[colidx] >= z_top, Y.c.ρq_tot[colidx] / Y.c.ρ[colidx], qref[colidx])
            @. deltaT[colidx] = ifelse(ᶜz[colidx] <= zLZB & ᶜz[colidx] >= z_top, 0, deltaT[colidx])
            @. deltaq[colidx] = ifelse(ᶜz[colidx] <= zLZB & ᶜz[colidx] >= z_top, 0, deltaq[colidx])
        end
    end

    @. Pq[colidx] = FT(0)
end

function level_of_zero_precip!(Y, p)

    (; ᶜT, ᶜts, params) = p
    thermo_params = CAP.thermodynamics_params(params)
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])

    z_top = zLZB
    k_top = kLZB

    for k in kLZB:-1:1
        z_top = Fields.level(Fields.coordinate_field(Y.c[colidx]).z, k)
        k_top = k
        Pq[colidx] .-= Fields.level(deltaq[colidx] .* Y.c.ρ[colidx] .* Fields.dz_field(Y.c[colidx]), k)
        if Pq[colidx] .> 0
            break
        end
    end

    if z_top < zLZB
        @. Tref[colidx] = ifelse(ᶜz[colidx] <= zLZB & ᶜz[colidx] > z_top, ᶜT[colidx], Tref[colidx])
        @. qref[colidx] = ifelse(ᶜz[colidx] <= zLZB & ᶜz[colidx] > z_top, Y.c.ρq_tot[colidx] / Y.c.ρ[colidx], qref[colidx])
        @. deltaT[colidx] = ifelse(ᶜz[colidx] <= zLZB & ᶜz[colidx] > z_top, 0, deltaT[colidx])
        @. deltaq[colidx] = ifelse(ᶜz[colidx] <= zLZB & ᶜz[colidx] > z_top, 0, deltaq[colidx])
    end

    return z_top, k_top
end

function change_Tref_LZB_shallowconv!(Pq, colidx)
    (; τ_bm) = p
    c = Pq ./ Fields.level((Y.c.ρ[colidx] .* Fields.dz_field(Y.c[colidx])), k_top)
    @. deltaT[colidx] = ifelse(ᶜz[colidx] == z_top, deltaT[colidx] .* c, deltaT[colidx])
    @. deltaq[colidx] = ifelse(ᶜz[colidx] == z_top, deltaq[colidx] .* c, deltaq[colidx]) 

    deltaT1 = @. -(deltaT[colidx] + L_v / cp_m * deltaq[colidx])
    @. deltaT1 = ifelse(ᶜz[colidx] <= z_top, deltaT1, FT(0))
    Operators.column_integral_definite!(
        deltak,
        deltaT1 * Y.c.ρ[colidx]
    )
    ρ1 = @. ifelse(ᶜz[colidx] <= z_top, Y.c.ρ, FT(0))
    Operators.column_integral_definite!(
        deltap,
        ρ1,
    )
    @. deltak = deltak / deltap
    deltak1 = @. ifelse(ᶜz[colidx] <= z_top, deltak, FT(0))
    @. Tref[colidx] = Tref[colidx] + deltak1 * τ_bm / dt
    @. deltaT[colidx] = deltaT[colidx] + deltak1
end