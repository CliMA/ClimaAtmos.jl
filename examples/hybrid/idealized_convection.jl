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
        set_reference_profiles!()
        pq_calculation()
        pt_calculation()
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

function set_reference_profiles!(Y, p, Tp, kLZB, rp, deltaq, deltaT, qref, Tref, colidx)
    (; rh_bm, ᶜp, params) = p
    R_d = CAP.R_d(params)
    R_v = CAP.R_v(params)
    q_tot
    Tref .= Tp
    eref = @. rh_bm * ᶜp * rp / (rp + R_d / R_v)
    rp = @. R_d * eref / R_v / (ᶜp - eref)
    qref = @. rp / (FT(1) + rp)
    ᶜα = @. ifelse(ᶜz < zLZB, FT(1), FT(0))
    @. Tref = ᶜα * Tref + (FT(1) - ᶜα) * ᶜT
    @. qref = ᶜα * qref + (FT(1) - ᶜα) * q_tot
    @. deltaT = ᶜα * deltaT
    @. deltaq = ᶜα * deltaq
end