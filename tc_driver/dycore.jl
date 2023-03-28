import LinearAlgebra as LA
import LinearAlgebra: ×

import ClimaAtmos as CA
import ClimaAtmos.TurbulenceConvection as TC
import ClimaAtmos.TurbulenceConvection.Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters
import Thermodynamics as TD
import ClimaCore as CC
import ClimaCore.Geometry as CCG
import OrdinaryDiffEq as ODE

import CLIMAParameters as CP

include(joinpath(@__DIR__, "dycore_variables.jl"))

#####
##### Methods
#####

function assign_thermo_aux!(state, grid, moisture_model, thermo_params)
    If = CCO.InterpolateC2F(bottom = CCO.Extrapolate(), top = CCO.Extrapolate())
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ᶜts = TC.center_aux_grid_mean_ts(state)
    p_c = TC.center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ
    @. ρ_f = If(ρ_c)

    @. aux_gm.q_tot = TD.total_specific_humidity(thermo_params, ᶜts)
    @. aux_gm.q_liq = TD.liquid_specific_humidity(thermo_params, ᶜts)
    @. aux_gm.q_ice = TD.ice_specific_humidity(thermo_params, ᶜts)
    @. aux_gm.T = TD.air_temperature(thermo_params, ᶜts)
    @. aux_gm.RH = TD.relative_humidity(thermo_params, ᶜts)
    @. aux_gm.θ_liq_ice = TD.liquid_ice_pottemp(thermo_params, ᶜts)
    @. aux_gm.h_tot =
        TD.total_specific_enthalpy(thermo_params, ᶜts, prog_gm.ρe_tot / ρ_c)
    @. p_c = TD.air_pressure(thermo_params, ᶜts)
    @. aux_gm.θ_virt = TD.virtual_pottemp(thermo_params, ᶜts)
    return
end

function compute_implicit_gm_tendencies!(
    edmf::TC.EDMFModel,
    grid::TC.Grid,
    state::TC.State,
    surf,
    param_set::APS,
)
    tendencies_gm = TC.center_tendencies_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    ρ_c = prog_gm.ρ
    tendencies_gm_uₕ = TC.tendencies_grid_mean_uₕ(state)

    TC.compute_sgs_flux!(edmf, grid, state, surf, param_set)

    ∇sgs = CCO.DivergenceF2C()
    @. tendencies_gm.ρe_tot += -∇sgs(aux_gm_f.sgs_flux_h_tot)
    @. tendencies_gm_uₕ += -∇sgs(aux_gm_f.sgs_flux_uₕ) / ρ_c
    if hasproperty(tendencies_gm, :ρq_tot)
        @. tendencies_gm.ρq_tot += -∇sgs(aux_gm_f.sgs_flux_q_tot)
    end

    return nothing
end
