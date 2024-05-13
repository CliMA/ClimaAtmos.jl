#####
##### Subsidence forcing
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

subsidence_cache(Y, atmos::AtmosModel) = subsidence_cache(Y, atmos.subsidence)

#####
##### No subsidence
#####

subsidence_cache(Y, subsidence::Nothing) = (;)
subsidence_tendency!(Yₜ, Y, p, t, subsidence::Nothing) = nothing

#####
##### Subsidence
#####

function subsidence_cache(Y, subsidence::Subsidence)
    FT = Spaces.undertype(axes(Y.c))
    toa(f) = Spaces.level(f, Spaces.nlevels(axes(f)))
    return (;
        ᶜsubsidence = similar(Y.c, FT), # TODO: fix types
        ᶜ∇MSE_gm = similar(Y.c, FT), # TODO: fix types
        ᶜ∇q_tot_gm = similar(Y.c, FT), # TODO: fix types
        ᶜ∇q_liq_gm = similar(Y.c, FT), # TODO: fix types
        ᶜ∇q_ice_gm = similar(Y.c, FT), # TODO: fix types
        ᶜMSE_gm_toa = similar(toa(Y.c), FT),
        ᶜq_tot_gm_toa = similar(toa(Y.c), FT),
    )
end

function subsidence_tendency!(Yₜ, Y, p, t, ::Subsidence)
    moisture_model = p.atmos.moisture_model
    subsidence_profile = p.atmos.subsidence.prof
    ᶜ∇MSE_gm = p.subsidence.ᶜ∇MSE_gm
    ᶜsubsidence = p.subsidence.ᶜsubsidence
    ᶜ∇q_tot_gm = p.subsidence.ᶜ∇q_tot_gm
    ᶜK = p.precomputed.ᶜK
    ᶜh_tot = p.precomputed.ᶜh_tot
    ᶜMSE_gm_toa = p.subsidence.ᶜMSE_gm_toa
    ᶜq_tot_gm_toa = p.subsidence.ᶜq_tot_gm_toa

    toa(f) = Spaces.level(f, Spaces.nlevels(axes(f)))
    wvec = Geometry.WVector
    ∇c = Operators.DivergenceF2C()

    z = Fields.coordinate_field(axes(ᶜsubsidence))
    @. ᶜsubsidence = subsidence_profile(z.z)

    ᶜh_tot_toa = toa(ᶜh_tot)
    ᶜK_toa = toa(ᶜK)
    @. ᶜMSE_gm_toa = ᶜh_tot_toa - ᶜK_toa
    ρq_tot_toa = toa(Y.c.ρq_tot)
    ρ_toa = toa(Y.c.ρ)
    @. ᶜq_tot_gm_toa = ρq_tot_toa / ρ_toa
    RBe = Operators.RightBiasedC2F(; top = Operators.SetValue(ᶜMSE_gm_toa))
    RBq = Operators.RightBiasedC2F(; top = Operators.SetValue(ᶜq_tot_gm_toa))
    @. ᶜ∇MSE_gm = ∇c(wvec(RBe(ᶜh_tot - ᶜK)))
    @. ᶜ∇q_tot_gm = ∇c(wvec(RBq(Y.c.ρq_tot / Y.c.ρ)))

    if moisture_model isa NonEquilMoistModel
        # TODO: fix for non-equilibrium case:
        #   Y.c.q_liq -> Y.c.ρq_liq
        #   Y.c.q_ice -> Y.c.ρq_ice
        ᶜ∇q_liq_gm = p.ᶜ∇q_liq_gm
        ᶜ∇q_ice_gm = p.ᶜ∇q_ice_gm
        q_liq_gm_toa = toa(Y.c.q_liq)
        q_ice_gm_toa = toa(Y.c.q_ice)
        RBq_liq =
            Operators.RightBiasedC2F(; top = Operators.SetValue(q_liq_gm_toa))
        RBq_ice =
            Operators.RightBiasedC2F(; top = Operators.SetValue(q_ice_gm_toa))
        @. ᶜ∇q_liq_gm = ∇c(wvec(RBq_liq(Y.c.q_liq)))
        @. ᶜ∇q_ice_gm = ∇c(wvec(RBq_ice(Y.c.q_ice)))
    end

    # LS Subsidence
    @. Yₜ.c.ρe_tot -= Y.c.ρ * ᶜsubsidence * ᶜ∇MSE_gm
    @. Yₜ.c.ρq_tot -= Y.c.ρ * ᶜsubsidence * ᶜ∇q_tot_gm
    if moisture_model isa NonEquilMoistModel
        @. Yₜ.c.ρq_liq -= ᶜ∇q_liq_gm * ᶜsubsidence
        @. Yₜ.c.ρq_ice -= ᶜ∇q_ice_gm * ᶜsubsidence
    end

    return nothing
end
