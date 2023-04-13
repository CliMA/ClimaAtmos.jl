using LinearAlgebra
import OrdinaryDiffEq as ODE
import Logging
import TerminalLoggers
import LinearAlgebra as LA
import LinearAlgebra: ×
import Thermodynamics as TD
import CLIMAParameters as CP

import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaCore.Geometry: ⊗

import ..Parameters as CAP
import ..TurbulenceConvection as TC
import ..InitialConditions as ICs
import ..TurbulenceConvection.Parameters as TCP
import ..TurbulenceConvection.Parameters.AbstractTurbulenceConvectionParameters as APS

#####
##### TurbulenceConvection sgs flux tendencies and cache
#####

function assign_thermo_aux!(state, moisture_model, thermo_params)
    If = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
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

    ∇sgs = Operators.DivergenceF2C()
    @. tendencies_gm.ρe_tot += -∇sgs(aux_gm_f.sgs_flux_h_tot)
    @. tendencies_gm_uₕ += -∇sgs(aux_gm_f.sgs_flux_uₕ) / ρ_c
    if hasproperty(tendencies_gm, :ρq_tot)
        @. tendencies_gm.ρq_tot += -∇sgs(aux_gm_f.sgs_flux_q_tot)
    end

    return nothing
end

function compute_explicit_gm_tendencies!(
    edmf::TC.EDMFModel,
    state::TC.State,
    surf,
    param_set::APS,
)
    tendencies_gm = TC.center_tendencies_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    aux_tc = TC.center_aux_turbconv(state)

    # Apply precipitation tendencies
    @. tendencies_gm.ρe_tot += ρ_c * aux_tc.e_tot_tendency_precip_sinks
    if hasproperty(tendencies_gm, :ρq_tot)
        @. tendencies_gm.ρq_tot += ρ_c * aux_tc.qt_tendency_precip_sinks
    end
    if edmf.precip_model isa Microphysics1Moment
        @. tendencies_gm.ρq_rai += ρ_c * aux_tc.qr_tendency_precip_sinks
        @. tendencies_gm.ρq_sno += ρ_c * aux_tc.qs_tendency_precip_sinks
    end
    return nothing
end

#####
##### No TurbulenceConvection scheme
#####

turbconv_cache(
    Y,
    turbconv_model,
    atmos,
    param_set,
    parsed_args,
    initial_condition,
) = (; turbconv_model)

implicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, _) = nothing
explicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, _) = nothing

function turbconv_aux(atmos, edmf, Y, ::Type{FT}) where {FT}
    face_aux_vars(FT, local_geometry, atmos, edmf) = (;
        sgs_flux_h_tot = Geometry.Covariant3Vector(FT(0)),
        sgs_flux_q_tot = Geometry.Covariant3Vector(FT(0)),
        sgs_flux_uₕ = Geometry.Covariant3Vector(FT(0)) ⊗
                      Geometry.Covariant12Vector(FT(0), FT(0)),
        ρ = FT(0),
        buoy = FT(0),
        TC.face_aux_vars_edmf(FT, local_geometry, edmf)...,
    )
    # Center only
    cent_aux_vars(FT, local_geometry, atmos, edmf) = (;
        tke = FT(0),
        q_liq = FT(0),
        q_ice = FT(0),
        RH = FT(0),
        T = FT(0),
        buoy = FT(0),
        cloud_fraction = FT(0),
        θ_virt = FT(0),
        Ri = FT(0),
        θ_liq_ice = FT(0),
        q_tot = FT(0),
        h_tot = FT(0),
        TC.cent_aux_vars_edmf(FT, local_geometry, atmos)...,
    )
    fspace = axes(Y.f)
    cspace = axes(Y.c)
    aux_cent_fields =
        TC.FieldFromNamedTuple(cspace, cent_aux_vars, FT, atmos, edmf)
    aux_face_fields =
        TC.FieldFromNamedTuple(fspace, face_aux_vars, FT, atmos, edmf)
    aux = Fields.FieldVector(cent = aux_cent_fields, face = aux_face_fields)
    return aux
end

function turbconv_cache(
    Y,
    turbconv_model::TC.EDMFModel,
    atmos,
    param_set,
    parsed_args,
    initial_condition,
)
    FT = Spaces.undertype(axes(Y.c))
    imex_edmf_turbconv = parsed_args["imex_edmf_turbconv"]
    imex_edmf_gm = parsed_args["imex_edmf_gm"]
    test_consistency = parsed_args["test_edmf_consistency"]
    thermo_params = CAP.thermodynamics_params(param_set)
    surf_params = ICs.surface_params(initial_condition, thermo_params)
    edmf = turbconv_model
    @info "EDMFModel: \n$(summary(edmf))"
    cache = (;
        edmf,
        turbconv_model,
        imex_edmf_turbconv,
        imex_edmf_gm,
        test_consistency,
        surf_params,
        param_set,
        aux = turbconv_aux(atmos, edmf, Y, FT),
        Y_filtered = similar(Y),
    )
    return (; edmf_cache = cache, turbconv_model)
end

# TODO: Split update_aux! and other functions into implicit and explicit parts.

function implicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, ::TC.EDMFModel)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, surf_params, Y_filtered) = edmf_cache
    (; imex_edmf_turbconv, imex_edmf_gm, test_consistency) = edmf_cache
    thermo_params = CAP.thermodynamics_params(param_set)
    tc_params = CAP.turbconv_params(param_set)

    imex_edmf_turbconv || imex_edmf_gm || return nothing

    Y_filtered.c[colidx] .= Y.c[colidx]
    Y_filtered.f[colidx] .= Y.f[colidx]

    state = TC.tc_column_state(Y_filtered, p, Yₜ, colidx)

    grid = TC.Grid(state)
    if test_consistency
        parent(state.aux.face) .= NaN
        parent(state.aux.cent) .= NaN
    end

    assign_thermo_aux!(state, edmf.moisture_model, thermo_params)

    surf = get_surface(
        p.atmos.model_config,
        surf_params,
        grid,
        state,
        t,
        tc_params,
    )
    println("SURF")
    @show surf

    TC.affect_filter!(edmf, grid, state, tc_params, surf, t)

    TC.update_aux!(edmf, grid, state, surf, tc_params, t, Δt)

    imex_edmf_turbconv &&
        TC.compute_implicit_turbconv_tendencies!(edmf, grid, state)

    imex_edmf_gm &&
        compute_implicit_gm_tendencies!(edmf, grid, state, surf, tc_params)

    # Note: The "filter relaxation tendency" should not be included in the
    # implicit tendency because its derivative with respect to Y is
    # discontinuous, which means that including it would make the linear
    # linear equation being solved by Newton's method ill-conditioned.

    return nothing
end

function explicit_sgs_flux_tendency!(Yₜ, Y, p, t, colidx, ::TC.EDMFModel)
    (; edmf_cache, Δt) = p
    (; edmf, param_set, surf_params, Y_filtered) = edmf_cache
    (; imex_edmf_turbconv, imex_edmf_gm, test_consistency) = edmf_cache
    thermo_params = CAP.thermodynamics_params(param_set)
    tc_params = CAP.turbconv_params(param_set)

    # Note: We could also do Y_filtered .= Y further upstream if needed.
    Y_filtered.c[colidx] .= Y.c[colidx]
    Y_filtered.f[colidx] .= Y.f[colidx]

    state = TC.tc_column_state(Y_filtered, p, Yₜ, colidx)

    grid = TC.Grid(state)
    if test_consistency
        parent(state.aux.face) .= NaN
        parent(state.aux.cent) .= NaN
    end

    assign_thermo_aux!(state, edmf.moisture_model, thermo_params)

    surf = get_surface(
        p.atmos.model_config,
        surf_params,
        grid,
        state,
        t,
        tc_params,
    )

    TC.affect_filter!(edmf, grid, state, tc_params, surf, t)

    TC.update_aux!(edmf, grid, state, surf, tc_params, t, Δt)

    TC.compute_precipitation_sink_tendencies(
        p.precip_model,
        state,
        tc_params,
        Δt,
    )

    # Ensure that, when a tendency is not computed with an IMEX formulation,
    # both its implicit and its explicit components are computed here.

    TC.compute_explicit_turbconv_tendencies!(edmf, grid, state)
    imex_edmf_turbconv ||
        TC.compute_implicit_turbconv_tendencies!(edmf, grid, state)

    # TODO: incrementally disable this and enable proper grid mean terms
    compute_explicit_gm_tendencies!(edmf, state, surf, tc_params)
    imex_edmf_gm ||
        compute_implicit_gm_tendencies!(edmf, grid, state, surf, tc_params)

    # Note: This "filter relaxation tendency" can be scaled down if needed, but
    # it must be present in order to prevent Y and Y_filtered from diverging
    # during each timestep.
    Yₜ_turbconv =
        Fields.FieldVector(c = Yₜ.c.turbconv[colidx], f = Yₜ.f.turbconv[colidx])
    Y_filtered_turbconv = Fields.FieldVector(
        c = Y_filtered.c.turbconv[colidx],
        f = Y_filtered.f.turbconv[colidx],
    )
    Y_turbconv =
        Fields.FieldVector(c = Y.c.turbconv[colidx], f = Y.f.turbconv[colidx])
    Yₜ_turbconv .+= (Y_filtered_turbconv .- Y_turbconv) ./ Δt
    return nothing
end
