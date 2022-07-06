# TODO: should this live in its own module?

import StatsBase

import ClimaCore
const CC = ClimaCore
const CCO = CC.Operators

import CLIMAParameters as CP
import TurbulenceConvection.Parameters as TCP
const APS = TCP.AbstractTurbulenceConvectionParameters

""" Purely diagnostic fields for the host model """
diagnostics(state, fl) = getproperty(state, TC.field_loc(fl))

center_diagnostics_grid_mean(state) = diagnostics(state, TC.CentField())
center_diagnostics_turbconv(state) = diagnostics(state, TC.CentField()).turbconv
face_diagnostics_turbconv(state) = diagnostics(state, TC.FaceField()).turbconv
face_diagnostics_precip(state) = diagnostics(state, TC.FaceField()).precip

svpc_diagnostics_grid_mean(state) = diagnostics(state, TC.SingleValuePerColumn())
svpc_diagnostics_turbconv(state) = diagnostics(state, TC.SingleValuePerColumn()).turbconv

#=
    io_dictionary_diagnostics()

These functions return a dictionary whose
 - `keys` are the nc variable names
 - `values` are NamedTuples corresponding to
    - `dims` (`("z")`  or `("z", "t")`) and
    - `group` (`"reference"` or `"profiles"`)

This dictionary is for purely diagnostic quantities--which
are not required to compute in order to run a simulation.
=#

#! format: off
function io_dictionary_diagnostics()
    DT = NamedTuple{(:dims, :group, :field), Tuple{Tuple{String, String}, String, Any}}
    io_dict = Dict{String, DT}(
        "nh_pressure" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_diagnostics_turbconv(state).nh_pressure),
        "nh_pressure_adv" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_diagnostics_turbconv(state).nh_pressure_adv,),
        "nh_pressure_drag" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_diagnostics_turbconv(state).nh_pressure_drag,),
        "nh_pressure_b" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_diagnostics_turbconv(state).nh_pressure_b,),
        "turbulent_entrainment" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).frac_turb_entr,),
        "entrainment_sc" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).entr_sc),
        "nondim_entrainment_sc" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).ε_nondim),
        "detrainment_sc" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).detr_sc),
        "nondim_detrainment_sc" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).δ_nondim),
        "asp_ratio" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).asp_ratio),
        "massflux" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).massflux),
        "rain_flux" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_diagnostics_precip(state).rain_flux),
        "snow_flux" => (; dims = ("zf", "t"), group = "profiles", field = state -> face_diagnostics_precip(state).snow_flux,),
    )
    return io_dict
end
#! format: on

function io(surf::TC.SurfaceBase, surf_params, grid, state, Stats::NetCDFIO_Stats, t::Real)
    write_ts(Stats, "Tsurface", TC.surface_temperature(surf_params, t))
    write_ts(Stats, "shf", surf.shf)
    write_ts(Stats, "lhf", surf.lhf)
    write_ts(Stats, "ustar", surf.ustar)
    write_ts(Stats, "wstar", surf.wstar)
end
function io(io_dict::Dict, Stats::NetCDFIO_Stats, state)
    for var in keys(io_dict)
        write_field(Stats, var, vec(io_dict[var].field(state)), io_dict[var].group)
    end
end


function initialize_io(nc_filename, FT, ts_list)
    NC.Dataset(nc_filename, "a") do ds
        for var_name in ts_list
            add_ts(ds, var_name, FT)
        end
    end
    return nothing
end

function initialize_io(nc_filename, FT, io_dicts::Dict...)
    NC.Dataset(nc_filename, "a") do ds
        for io_dict in io_dicts
            for var_name in keys(io_dict)
                add_field(ds, var_name, io_dict[var_name].dims, io_dict[var_name].group, FT)
            end
        end
    end
end

#=
    compute_diagnostics!

Computes diagnostic quantities. The state _should not_ depend
on any quantities here. I.e., we should be able to shut down
diagnostics and still run, at which point we'll be able to export
the state, auxiliary fields (which the state does depend on), and
tendencies.
=#
function compute_diagnostics!(
    edmf::TC.EDMFModel,
    precip_model::TC.AbstractPrecipitationModel,
    param_set::APS,
    grid::TC.Grid,
    state::TC.State,
    diagnostics::D,
    Stats::NetCDFIO_Stats,
    surf_params,
    t::Real,
    calibrate_io::Bool,
) where {D <: CC.Fields.FieldVector}
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = TC.float_type(state)
    N_up = TC.n_updrafts(edmf)
    aux_gm = TC.center_aux_grid_mean(state)
    aux_en = TC.center_aux_environment(state)
    aux_up = TC.center_aux_updrafts(state)
    aux_up_f = TC.face_aux_updrafts(state)
    aux_tc_f = TC.face_aux_turbconv(state)
    aux_tc = TC.center_aux_turbconv(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_pr = TC.center_prog_precipitation(state)
    aux_bulk = TC.center_aux_bulk(state)
    ts_gm = TC.center_aux_grid_mean(state).ts
    ts_en = TC.center_aux_environment(state).ts
    a_up_bulk = aux_bulk.area
    kc_toa = TC.kc_top_of_atmos(grid)
    prog_gm = TC.center_prog_grid_mean(state)
    diag_tc = center_diagnostics_turbconv(diagnostics)
    diag_tc_f = face_diagnostics_turbconv(diagnostics)
    diag_tc_f_precip = face_diagnostics_precip(diagnostics)
    ρ_c = prog_gm.ρ
    p_c = aux_gm.p

    diag_tc_svpc = svpc_diagnostics_turbconv(diagnostics)
    diag_svpc = svpc_diagnostics_grid_mean(diagnostics)

    surf = get_surface(surf_params, grid, state, t, param_set)

    @. aux_gm.s = TD.specific_entropy(thermo_params, ts_gm)
    @. aux_en.s = TD.specific_entropy(thermo_params, ts_en)

    @inbounds for k in TC.real_center_indices(grid)
        @inbounds for i in 1:N_up
            aux_up[i].s[k] = if aux_up[i].area[k] > 0.0
                thermo_args = if edmf.moisture_model isa TC.EquilibriumMoisture
                    ()
                elseif edmf.moisture_model isa TC.NonEquilibriumMoisture
                    (aux_up[i].q_liq[k], aux_up[i].q_ice[k])
                end
                ts_up = TC.thermo_state_pθq(
                    param_set,
                    p_c[k],
                    aux_up[i].θ_liq_ice[k],
                    aux_up[i].q_tot[k],
                    thermo_args...,
                )
                TD.specific_entropy(thermo_params, ts_up)
            else
                FT(0)
            end
        end
    end

    # TODO(ilopezgp): Fix bottom gradient
    wvec = CC.Geometry.WVector
    m_bcs = (; bottom = CCO.SetValue(FT(0)), top = CCO.SetValue(FT(0)))
    # ∇0_bcs = (; bottom = CCO.SetDivergence(wvec(FT(0))), top = CCO.SetDivergence(wvec(FT(0))))
    ∇0_bcs = (; bottom = CCO.SetDivergence(FT(0)), top = CCO.SetDivergence(FT(0)))
    If = CCO.InterpolateC2F(; m_bcs...)
    ∇f = CCO.DivergenceC2F(; ∇0_bcs...)
    massflux_s = aux_gm_f.massflux_s
    parent(massflux_s) .= 0
    @. aux_gm_f.diffusive_flux_s = -aux_tc_f.ρ_ae_KH * ∇f(wvec(aux_en.s))
    @inbounds for i in 1:N_up
        @. massflux_s += aux_up_f[i].massflux * (If(aux_up[i].s) - If(aux_en.s))
    end

    # Mean water paths for calibration
    cent = TC.Cent(1)
    diag_svpc.lwp_mean[cent] = sum(ρ_c .* aux_gm.q_liq)
    diag_svpc.iwp_mean[cent] = sum(ρ_c .* aux_gm.q_ice)
    diag_svpc.rwp_mean[cent] = sum(ρ_c .* prog_pr.q_rai)
    diag_svpc.swp_mean[cent] = sum(ρ_c .* prog_pr.q_sno)

    #####
    ##### Cloud base, top and cover
    #####
    # TODO: write this in a mutating-free way using findfirst/findlast/map

    cloud_base_up = Vector{FT}(undef, N_up)
    cloud_top_up = Vector{FT}(undef, N_up)
    cloud_cover_up = Vector{FT}(undef, N_up)

    @inbounds for i in 1:N_up
        cloud_base_up[i] = TC.zc_toa(grid).z
        cloud_top_up[i] = FT(0)
        cloud_cover_up[i] = FT(0)

        @inbounds for k in TC.real_center_indices(grid)
            if aux_up[i].area[k] > 1e-3
                if TD.has_condensate(aux_up[i].q_liq[k] + aux_up[i].q_ice[k])
                    cloud_base_up[i] = min(cloud_base_up[i], grid.zc[k].z)
                    cloud_top_up[i] = max(cloud_top_up[i], grid.zc[k].z)
                    cloud_cover_up[i] = max(cloud_cover_up[i], aux_up[i].area[k])
                end
            end
        end
    end
    # Note definition of cloud cover : each updraft is associated with
    # a cloud cover equal to the maximum area fraction of the updraft
    # where ql > 0. Each updraft is assumed to have maximum overlap with
    # respect to itup (i.e. no consideration of tilting due to shear)
    # while the updraft classes are assumed to have no overlap at all.
    # Thus total updraft cover is the sum of each updraft's cover

    diag_tc_svpc.updraft_cloud_cover[cent] = sum(cloud_cover_up)
    diag_tc_svpc.updraft_cloud_base[cent] = minimum(abs.(cloud_base_up))
    diag_tc_svpc.updraft_cloud_top[cent] = maximum(abs.(cloud_top_up))

    cloud_top_en = FT(0)
    cloud_base_en = TC.zc_toa(grid).z
    cloud_cover_en = FT(0)
    @inbounds for k in TC.real_center_indices(grid)
        if TD.has_condensate(aux_en.q_liq[k] + aux_en.q_ice[k]) && aux_en.area[k] > 1e-6
            cloud_base_en = min(cloud_base_en, grid.zc[k].z)
            cloud_top_en = max(cloud_top_en, grid.zc[k].z)
            cloud_cover_en = max(cloud_cover_en, aux_en.area[k] * aux_en.cloud_fraction[k])
        end
    end
    # Assuming amximum overlap in environmental clouds
    diag_tc_svpc.env_cloud_cover[cent] = cloud_cover_en
    diag_tc_svpc.env_cloud_base[cent] = cloud_base_en
    diag_tc_svpc.env_cloud_top[cent] = cloud_top_en

    cloud_cover_gm = min(cloud_cover_en + sum(cloud_cover_up), 1)
    cloud_base_gm = grid.zc[kc_toa].z
    cloud_top_gm = FT(0)
    @inbounds for k in TC.real_center_indices(grid)
        if TD.has_condensate(aux_gm.q_liq[k] + aux_gm.q_ice[k])
            cloud_base_gm = min(cloud_base_gm, grid.zc[k].z)
            cloud_top_gm = max(cloud_top_gm, grid.zc[k].z)
        end
    end

    diag_svpc.cloud_cover_mean[cent] = cloud_cover_gm
    diag_svpc.cloud_base_mean[cent] = cloud_base_gm
    diag_svpc.cloud_top_mean[cent] = cloud_top_gm

    #####
    ##### Fluxes
    #####
    Ic = CCO.InterpolateF2C()
    parent(diag_tc.massflux) .= 0
    @inbounds for i in 1:N_up
        @. diag_tc.massflux += Ic(aux_up_f[i].massflux)
    end

    @inbounds for k in TC.real_center_indices(grid)
        a_up_bulk_k = a_up_bulk[k]
        diag_tc.entr_sc[k] = 0
        diag_tc.ε_nondim[k] = 0
        diag_tc.detr_sc[k] = 0
        diag_tc.δ_nondim[k] = 0
        diag_tc.asp_ratio[k] = 0
        diag_tc.frac_turb_entr[k] = 0
        if a_up_bulk_k > 0.0
            @inbounds for i in 1:N_up
                aux_up_i = aux_up[i]
                diag_tc.entr_sc[k] += aux_up_i.area[k] * aux_up_i.entr_sc[k] / a_up_bulk_k
                diag_tc.ε_nondim[k] += aux_up_i.area[k] * aux_up_i.ε_nondim[k] / a_up_bulk_k
                diag_tc.detr_sc[k] += aux_up_i.area[k] * aux_up_i.detr_sc[k] / a_up_bulk_k
                diag_tc.δ_nondim[k] += aux_up_i.area[k] * aux_up_i.δ_nondim[k] / a_up_bulk_k
                diag_tc.asp_ratio[k] += aux_up_i.area[k] * aux_up_i.asp_ratio[k] / a_up_bulk_k
                diag_tc.frac_turb_entr[k] += aux_up_i.area[k] * aux_up_i.frac_turb_entr[k] / a_up_bulk_k
            end
        end
    end

    a_bulk_bcs = TC.a_bulk_boundary_conditions(surf, edmf)
    Ifabulk = CCO.InterpolateC2F(; a_bulk_bcs...)
    a_up_bulk_f = @. Ifabulk(a_up_bulk)

    RB_precip = CCO.RightBiasedC2F(; top = CCO.SetValue(FT(0)))

    @inbounds for i in 1:N_up
        a_up_bcs = TC.a_up_boundary_conditions(surf, edmf, i)
        Ifaup = CCO.InterpolateC2F(; a_up_bcs...)
        a_up_f = @. Ifaup(aux_up[i].area)
        @inbounds for k in TC.real_face_indices(grid)
            diag_tc_f.nh_pressure[k] = 0
            diag_tc_f.nh_pressure_b[k] = 0
            diag_tc_f.nh_pressure_adv[k] = 0
            diag_tc_f.nh_pressure_drag[k] = 0
            if a_up_bulk_f[k] > 0.0
                diag_tc_f.nh_pressure[k] += a_up_f[k] * aux_up_f[i].nh_pressure[k] / a_up_bulk_f[k]
                diag_tc_f.nh_pressure_b[k] += a_up_f[k] * aux_up_f[i].nh_pressure_b[k] / a_up_bulk_f[k]
                diag_tc_f.nh_pressure_adv[k] += a_up_f[k] * aux_up_f[i].nh_pressure_adv[k] / a_up_bulk_f[k]
                diag_tc_f.nh_pressure_drag[k] += a_up_f[k] * aux_up_f[i].nh_pressure_drag[k] / a_up_bulk_f[k]
            end
        end
    end
    @. diag_tc_f_precip.rain_flux = RB_precip(ρ_c * prog_pr.q_rai * aux_tc.term_vel_rain)
    @. diag_tc_f_precip.snow_flux = RB_precip(ρ_c * prog_pr.q_sno * aux_tc.term_vel_snow)

    TC.GMV_third_m(edmf, grid, state, Val(:Hvar), Val(:θ_liq_ice), Val(:H_third_m))
    TC.GMV_third_m(edmf, grid, state, Val(:QTvar), Val(:q_tot), Val(:QT_third_m))
    TC.GMV_third_m(edmf, grid, state, Val(:tke), Val(:w), Val(:W_third_m))

    TC.compute_covariance_interdomain_src(edmf, grid, state, Val(:tke), Val(:w), Val(:w))
    TC.compute_covariance_interdomain_src(edmf, grid, state, Val(:Hvar), Val(:θ_liq_ice), Val(:θ_liq_ice))
    TC.compute_covariance_interdomain_src(edmf, grid, state, Val(:QTvar), Val(:q_tot), Val(:q_tot))
    TC.compute_covariance_interdomain_src(edmf, grid, state, Val(:HQTcov), Val(:θ_liq_ice), Val(:q_tot))

    TC.update_cloud_frac(edmf, grid, state)

    diag_tc_svpc.env_lwp[cent] = sum(ρ_c .* aux_en.q_liq .* aux_en.area)
    diag_tc_svpc.env_iwp[cent] = sum(ρ_c .* aux_en.q_ice .* aux_en.area)

    #TODO - change to rain rate that depends on rain model choice
    ρ_cloud_liq = TCP.ρ_cloud_liq(param_set)
    if (precip_model isa TC.Clima0M)
        f =
            (aux_en.qt_tendency_precip_formation .+ aux_bulk.qt_tendency_precip_formation) .* ρ_c ./ ρ_cloud_liq .*
            FT(3.6) .* 1e6
        diag_svpc.cutoff_precipitation_rate[cent] = sum(f)
    end

    lwp = sum(i -> sum(ρ_c .* aux_up[i].q_liq .* aux_up[i].area .* (aux_up[i].area .> 1e-3)), 1:N_up)
    iwp = sum(i -> sum(ρ_c .* aux_up[i].q_ice .* aux_up[i].area .* (aux_up[i].area .> 1e-3)), 1:N_up)

    plume_scale_height = map(1:N_up) do i
        TC.compute_plume_scale_height(grid, state, edmf.H_up_min, i)
    end

    diag_tc_svpc.updraft_lwp[cent] = lwp
    diag_tc_svpc.updraft_iwp[cent] = iwp
    diag_tc_svpc.Hd[cent] = StatsBase.mean(plume_scale_height)

    return
end
