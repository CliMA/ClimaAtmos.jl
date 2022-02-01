if !("." in LOAD_PATH) # for easier local testing
    push!(LOAD_PATH, ".")
end
import TurbulenceConvection
const TC = TurbulenceConvection
const tc_dir = dirname(dirname(pathof(TurbulenceConvection)))

import UnPack

import StaticArrays
import OrderedCollections
const SA = StaticArrays

import SurfaceFluxes
const SF = SurfaceFluxes
const UF = SF.UniversalFunctions

import CLIMAParameters
const CP = CLIMAParameters
const CPP = CP.Planet
const APS = CP.AbstractEarthParameterSet

import StatsBase

import ClimaCore
const CC = ClimaCore
const CCO = CC.Operators

import Thermodynamics
const TD = Thermodynamics

using Test

import NCDatasets
const NC = NCDatasets

import OrdinaryDiffEq
const ODE = OrdinaryDiffEq

import Dierckx
import Statistics
import Random

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

import ArgParse

import CloudMicrophysics
const CM = CloudMicrophysics
const CM0 = CloudMicrophysics.Microphysics_0M
const CM1 = CloudMicrophysics.Microphysics_1M

import SciMLBase
import JSON

mutable struct NetCDFIO_Stats
    root_grp::NC.NCDataset{Nothing}
    profiles_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    ts_grp::NC.NCDataset{NC.NCDataset{Nothing}}
    last_output_time::Float64
    uuid::String
    frequency::Float64
    stats_path::String
    path_plus_file::String
    vars::Dict{String, Any} # Hack to avoid https://github.com/Alexander-Barth/NCDatasets.jl/issues/135
    function NetCDFIO_Stats(namelist, grid::TC.Grid)

        # Initialize properties with valid type:
        tmp = tempname()
        root_grp = NC.Dataset(tmp, "c")
        NC.defGroup(root_grp, "profiles")
        NC.defGroup(root_grp, "timeseries")
        profiles_grp = root_grp.group["profiles"]
        ts_grp = root_grp.group["timeseries"]
        close(root_grp)

        last_output_time = 0.0
        uuid = string(namelist["meta"]["uuid"])

        frequency = namelist["stats_io"]["frequency"]

        # Setup the statistics output path
        simname = namelist["meta"]["simname"]
        casename = namelist["meta"]["casename"]
        outpath = joinpath(namelist["output"]["output_root"], "Output.$simname.$uuid")
        mkpath(outpath)

        stats_path = joinpath(outpath, namelist["stats_io"]["stats_dir"])
        mkpath(stats_path)

        path_plus_file = joinpath(stats_path, "Stats.$simname.nc")

        # Write namelist file to output directory
        open(joinpath(outpath, "namelist_$casename.in"), "w") do io
            JSON.print(io, namelist, 4)
        end

        # Remove the NC file if it exists, in case it accidentally wasn't closed
        isfile(path_plus_file) && rm(path_plus_file; force = true)

        NC.Dataset(path_plus_file, "c") do root_grp

            zf = vec(grid.zf)
            zc = vec(grid.zc)

            # Set profile dimensions
            profile_grp = NC.defGroup(root_grp, "profiles")
            NC.defDim(profile_grp, "zf", TC.n_cells(grid) + 1)
            NC.defDim(profile_grp, "zc", TC.n_cells(grid))
            NC.defDim(profile_grp, "t", Inf)
            NC.defVar(profile_grp, "zf", zf, ("zf",))
            NC.defVar(profile_grp, "zc", zc, ("zc",))
            NC.defVar(profile_grp, "t", Float64, ("t",))

            reference_grp = NC.defGroup(root_grp, "reference")
            NC.defDim(reference_grp, "zf", TC.n_cells(grid) + 1)
            NC.defDim(reference_grp, "zc", TC.n_cells(grid))
            NC.defVar(reference_grp, "zf", zf, ("zf",))
            NC.defVar(reference_grp, "zc", zc, ("zc",))

            ts_grp = NC.defGroup(root_grp, "timeseries")
            NC.defDim(ts_grp, "t", Inf)
            NC.defVar(ts_grp, "t", Float64, ("t",))
        end
        vars = Dict{String, Any}()
        return new(root_grp, profiles_grp, ts_grp, last_output_time, uuid, frequency, stats_path, path_plus_file, vars)
    end
end


function open_files(self)
    self.root_grp = NC.Dataset(self.path_plus_file, "a")
    self.profiles_grp = self.root_grp.group["profiles"]
    self.ts_grp = self.root_grp.group["timeseries"]
    vars = self.vars

    # Hack to avoid https://github.com/Alexander-Barth/NCDatasets.jl/issues/135
    vars["profiles"] = Dict{String, Any}()
    for k in keys(self.profiles_grp)
        vars["profiles"][k] = self.profiles_grp[k]
    end
    vars["timeseries"] = Dict{String, Any}()
    for k in keys(self.ts_grp)
        vars["timeseries"][k] = self.ts_grp[k]
    end
end

function close_files(self::NetCDFIO_Stats)
    close(self.root_grp)
end

#####
##### Generic field
#####

function add_field(self::NetCDFIO_Stats, var_name::String; dims, group)
    NC.Dataset(self.path_plus_file, "a") do root_grp
        profile_grp = root_grp.group[group]
        new_var = NC.defVar(profile_grp, var_name, Float64, dims)
    end
end

#####
##### Time-series data
#####

function add_ts(self::NetCDFIO_Stats, var_name::String)
    NC.Dataset(self.path_plus_file, "a") do root_grp
        ts_grp = root_grp.group["timeseries"]
        new_var = NC.defVar(ts_grp, var_name, Float64, ("t",))
    end
end

#####
##### Performance critical IO
#####

# Field wrapper
write_field(self::NetCDFIO_Stats, var_name::String, data; group) = write_field(self, var_name, vec(data); group = group)

function write_field(self::NetCDFIO_Stats, var_name::String, data::T; group) where {T <: AbstractArray{Float64, 1}}
    if group == "profiles"
        @inbounds self.vars[group][var_name][:, end] = data
    elseif group == "reference"
        NC.Dataset(self.path_plus_file, "a") do root_grp
            reference_grp = root_grp.group[group]
            var = reference_grp[var_name]
            var .= data::T
        end
    else
        error("Bad group given")
    end
end

function write_ts(self::NetCDFIO_Stats, var_name::String, data::Float64)
    @inbounds self.vars["timeseries"][var_name][end] = data::Float64
end

function write_simulation_time(self::NetCDFIO_Stats, t::Float64)
    profile_t = self.profiles_grp["t"]
    @inbounds profile_t[end + 1] = t::Float64
    ts_t = self.ts_grp["t"]
    @inbounds ts_t[end + 1] = t::Float64
end

""" Purely diagnostic fields for the host model """
diagnostics(state, fl) = getproperty(state, TC.field_loc(fl))

center_diagnostics_grid_mean(state) = diagnostics(state, TC.CentField())
center_diagnostics_turbconv(state) = diagnostics(state, TC.CentField()).turbconv
face_diagnostics_turbconv(state) = diagnostics(state, TC.FaceField()).turbconv

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
        "detrainment_sc" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).detr_sc),
        "asp_ratio" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).asp_ratio),
        "massflux" => (; dims = ("zc", "t"), group = "profiles", field = state -> center_diagnostics_turbconv(state).massflux),
    )
    return io_dict
end
#! format: on

function io(surf::TC.SurfaceBase, surf_params, grid, state, Stats::NetCDFIO_Stats, t::Real)
    write_ts(Stats, "Tsurface", TC.surface_temperature(surf_params, t))
    write_ts(Stats, "shf", surf.shf)
    write_ts(Stats, "lhf", surf.lhf)
    write_ts(Stats, "ustar", surf.ustar)
end
function io(io_dict::Dict, Stats::NetCDFIO_Stats, state)
    for var in keys(io_dict)
        write_field(Stats, var, io_dict[var].field(state); group = io_dict[var].group)
    end
end


function initialize_io(gm::TC.GridMeanVariables, Stats::NetCDFIO_Stats)
    add_ts(Stats, "Tsurface")
    add_ts(Stats, "shf")
    add_ts(Stats, "lhf")
    add_ts(Stats, "ustar")
    add_ts(Stats, "lwp_mean")
    add_ts(Stats, "iwp_mean")
    add_ts(Stats, "cloud_base_mean")
    add_ts(Stats, "cloud_top_mean")
    add_ts(Stats, "cloud_cover_mean")
    return nothing
end

# Initialize the IO pertaining to this class
function initialize_io(edmf::TC.EDMF_PrognosticTKE, Stats::NetCDFIO_Stats)
    add_ts(Stats, "env_cloud_base")
    add_ts(Stats, "env_cloud_top")
    add_ts(Stats, "env_cloud_cover")
    add_ts(Stats, "env_lwp")
    add_ts(Stats, "env_iwp")
    add_ts(Stats, "updraft_cloud_cover")
    add_ts(Stats, "updraft_cloud_base")
    add_ts(Stats, "updraft_cloud_top")
    add_ts(Stats, "updraft_lwp")
    add_ts(Stats, "updraft_iwp")
    add_ts(Stats, "rwp_mean")
    add_ts(Stats, "swp_mean")
    add_ts(Stats, "cutoff_precipitation_rate")
    add_ts(Stats, "Hd")
    return nothing
end

function initialize_io(io_dict::Dict, Stats::NetCDFIO_Stats)
    for var_name in keys(io_dict)
        add_field(Stats, var_name; dims = io_dict[var_name].dims, group = io_dict[var_name].group)
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
    edmf::TC.EDMF_PrognosticTKE,
    gm::TC.GridMeanVariables,
    grid::TC.Grid,
    state::TC.State,
    diagnostics::D,
    Stats::NetCDFIO_Stats,
    case::TC.CasesBase,
    t::Real,
) where {D <: CC.Fields.FieldVector}
    FT = eltype(grid)
    N_up = TC.n_updrafts(edmf)
    ρ0_c = TC.center_ref_state(state).ρ0
    p0_c = TC.center_ref_state(state).p0
    aux_gm = TC.center_aux_grid_mean(state)
    aux_en = TC.center_aux_environment(state)
    aux_up = TC.center_aux_updrafts(state)
    aux_up_f = TC.face_aux_updrafts(state)
    aux_tc_f = TC.face_aux_turbconv(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    prog_pr = TC.center_prog_precipitation(state)
    aux_bulk = TC.center_aux_bulk(state)
    a_up_bulk = aux_bulk.area
    kc_toa = TC.kc_top_of_atmos(grid)
    param_set = TC.parameter_set(gm)
    prog_gm = TC.center_prog_grid_mean(state)
    precip_model = edmf.precip_model
    diag_tc = center_diagnostics_turbconv(diagnostics)
    diag_tc_f = face_diagnostics_turbconv(diagnostics)
    surf = get_surface(case.surf_params, grid, state, gm, t, param_set)

    @inbounds for k in TC.real_center_indices(grid)
        ts = TC.thermo_state_pθq(param_set, p0_c[k], prog_gm.θ_liq_ice[k], prog_gm.q_tot[k])
        aux_gm.s[k] = TD.specific_entropy(ts)
        ts_en = TC.thermo_state_pθq(param_set, p0_c[k], aux_en.θ_liq_ice[k], aux_en.q_tot[k])
        aux_en.s[k] = TD.specific_entropy(ts_en)
        @inbounds for i in 1:N_up
            if aux_up[i].area[k] > 0.0
                ts_up = TC.thermo_state_pθq(param_set, p0_c[k], aux_up[i].θ_liq_ice[k], aux_up[i].q_tot[k])
                aux_up[i].s[k] = TD.specific_entropy(ts_up)
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
    write_ts(Stats, "updraft_cloud_cover", sum(cloud_cover_up))
    write_ts(Stats, "updraft_cloud_base", minimum(abs.(cloud_base_up)))
    write_ts(Stats, "updraft_cloud_top", maximum(abs.(cloud_top_up)))

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
    write_ts(Stats, "env_cloud_cover", cloud_cover_en)
    write_ts(Stats, "env_cloud_base", cloud_base_en)
    write_ts(Stats, "env_cloud_top", cloud_top_en)

    cloud_cover_gm = min(cloud_cover_en + sum(cloud_cover_up), 1)
    cloud_base_gm = grid.zc[kc_toa].z
    cloud_top_gm = FT(0)
    @inbounds for k in TC.real_center_indices(grid)
        if TD.has_condensate(aux_gm.q_liq[k] + aux_gm.q_ice[k])
            cloud_base_gm = min(cloud_base_gm, grid.zc[k].z)
            cloud_top_gm = max(cloud_top_gm, grid.zc[k].z)
        end
    end
    write_ts(Stats, "cloud_cover_mean", cloud_cover_gm)
    write_ts(Stats, "cloud_base_mean", cloud_base_gm)
    write_ts(Stats, "cloud_top_mean", cloud_top_gm)

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
        if a_up_bulk_k > 0.0
            @inbounds for i in 1:N_up
                aux_up_i = aux_up[i]
                diag_tc.entr_sc[k] += aux_up_i.area[k] * aux_up_i.entr_sc[k] / a_up_bulk_k
                diag_tc.detr_sc[k] += aux_up_i.area[k] * aux_up_i.detr_sc[k] / a_up_bulk_k
                diag_tc.asp_ratio[k] += aux_up_i.area[k] * aux_up_i.asp_ratio[k] / a_up_bulk_k
                diag_tc.frac_turb_entr[k] += aux_up_i.area[k] * aux_up_i.frac_turb_entr[k] / a_up_bulk_k
            end
        end
    end

    a_up_bulk_f = copy(diag_tc_f.nh_pressure)
    a_bulk_bcs = TC.a_bulk_boundary_conditions(surf, edmf)
    Ifa = CCO.InterpolateC2F(; a_bulk_bcs...)
    @. a_up_bulk_f = Ifa(a_up_bulk)

    a_up_bulk_f = copy(diag_tc_f.nh_pressure)
    a_up_f = copy(a_up_bulk_f)
    Ifabulk = CCO.InterpolateC2F(; a_bulk_bcs...)
    @. a_up_bulk_f = Ifabulk(a_up_bulk)
    @inbounds for i in 1:N_up
        a_up_bcs = TC.a_up_boundary_conditions(surf, edmf, i)
        Ifaup = CCO.InterpolateC2F(; a_up_bcs...)
        @. a_up_f = Ifaup(aux_up[i].area)
        @inbounds for k in TC.real_face_indices(grid)
            if a_up_bulk_f[k] > 0.0
                diag_tc_f.nh_pressure[k] += a_up_f[k] * aux_up_f[i].nh_pressure[k] / a_up_bulk_f[k]
                diag_tc_f.nh_pressure_b[k] += a_up_f[k] * aux_up_f[i].nh_pressure_b[k] / a_up_bulk_f[k]
                diag_tc_f.nh_pressure_adv[k] += a_up_f[k] * aux_up_f[i].nh_pressure_adv[k] / a_up_bulk_f[k]
                diag_tc_f.nh_pressure_drag[k] += a_up_f[k] * aux_up_f[i].nh_pressure_drag[k] / a_up_bulk_f[k]
            end
        end
    end

    TC.GMV_third_m(edmf, grid, state, Val(:Hvar), Val(:θ_liq_ice), Val(:H_third_m))
    TC.GMV_third_m(edmf, grid, state, Val(:QTvar), Val(:q_tot), Val(:QT_third_m))
    TC.GMV_third_m(edmf, grid, state, Val(:tke), Val(:w), Val(:W_third_m))

    TC.compute_covariance_interdomain_src(edmf, grid, state, Val(:tke), Val(:w), Val(:w))
    TC.compute_covariance_interdomain_src(edmf, grid, state, Val(:Hvar), Val(:θ_liq_ice), Val(:θ_liq_ice))
    TC.compute_covariance_interdomain_src(edmf, grid, state, Val(:QTvar), Val(:q_tot), Val(:q_tot))
    TC.compute_covariance_interdomain_src(edmf, grid, state, Val(:HQTcov), Val(:θ_liq_ice), Val(:q_tot))

    TC.update_cloud_frac(edmf, grid, state, gm)


    write_ts(Stats, "lwp_mean", sum(ρ0_c .* aux_gm.q_liq))
    write_ts(Stats, "iwp_mean", sum(ρ0_c .* aux_gm.q_ice))
    write_ts(Stats, "env_lwp", sum(ρ0_c .* aux_en.q_liq .* aux_en.area))
    write_ts(Stats, "env_iwp", sum(ρ0_c .* aux_en.q_ice .* aux_en.area))

    write_ts(Stats, "rwp_mean", sum(ρ0_c .* prog_pr.q_rai))
    write_ts(Stats, "swp_mean", sum(ρ0_c .* prog_pr.q_sno))
    #TODO - change to rain rate that depends on rain model choice

    # TODO: Move rho_cloud_liq to CLIMAParameters
    rho_cloud_liq = 1e3
    if (precip_model isa TC.CutoffPrecipitation)
        f =
            (aux_en.qt_tendency_precip_formation .+ aux_bulk.qt_tendency_precip_formation) .* ρ0_c ./
            TC.rho_cloud_liq .* 3.6 .* 1e6
        write_ts(Stats, "cutoff_precipitation_rate", sum(f))
    end

    lwp = sum(i -> sum(ρ0_c .* aux_up[i].q_liq .* aux_up[i].area .* (aux_up[i].area .> 1e-3)), 1:N_up)
    iwp = sum(i -> sum(ρ0_c .* aux_up[i].q_ice .* aux_up[i].area .* (aux_up[i].area .> 1e-3)), 1:N_up)
    write_ts(Stats, "updraft_lwp", lwp)
    write_ts(Stats, "updraft_iwp", iwp)
    plume_scale_height = map(1:N_up) do i
        TC.compute_plume_scale_height(grid, state, param_set, i)
    end
    write_ts(Stats, "Hd", StatsBase.mean(plume_scale_height))

    return
end

struct EarthParameterSet{NT} <: CP.AbstractEarthParameterSet
    nt::NT
end

CLIMAParameters.Planet.MSLP(ps::EarthParameterSet) = ps.nt.MSLP
CLIMAParameters.Planet.cp_d(ps::EarthParameterSet) = ps.nt.cp_d
CLIMAParameters.Planet.cp_v(ps::EarthParameterSet) = ps.nt.cp_v
CLIMAParameters.Planet.R_d(ps::EarthParameterSet) = ps.nt.R_d
CLIMAParameters.Planet.R_v(ps::EarthParameterSet) = ps.nt.R_v
CLIMAParameters.Planet.molmass_ratio(ps::EarthParameterSet) = ps.nt.molmass_ratio
# microphysics parameters
CLIMAParameters.Atmos.Microphysics_0M.τ_precip(ps::EarthParameterSet) = ps.nt.τ_precip
CLIMAParameters.Atmos.Microphysics.τ_acnv_rai(ps::EarthParameterSet) = ps.nt.τ_acnv_rai
CLIMAParameters.Atmos.Microphysics.τ_acnv_sno(ps::EarthParameterSet) = ps.nt.τ_acnv_sno
CLIMAParameters.Atmos.Microphysics.q_liq_threshold(ps::EarthParameterSet) = ps.nt.q_liq_threshold
CLIMAParameters.Atmos.Microphysics.q_ice_threshold(ps::EarthParameterSet) = ps.nt.q_ice_threshold
# entrainment/detrainment parameters
CLIMAParameters.Atmos.EDMF.c_ε(ps::EarthParameterSet) = ps.nt.c_ε # factor multiplier for dry term in entrainment/detrainment
CLIMAParameters.Atmos.EDMF.α_b(ps::EarthParameterSet) = ps.nt.α_b # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
CLIMAParameters.Atmos.EDMF.α_a(ps::EarthParameterSet) = ps.nt.α_a # factor multiplier for pressure advection
CLIMAParameters.Atmos.EDMF.α_d(ps::EarthParameterSet) = ps.nt.α_d # factor multiplier for pressure drag
CLIMAParameters.Atmos.EDMF.H_up_min(ps::EarthParameterSet) = ps.nt.H_up_min # minimum updraft top to avoid zero division in pressure drag and turb-entr
CLIMAParameters.Atmos.EDMF.c_δ(ps::EarthParameterSet) = ps.nt.c_δ # factor multiplier for moist term in entrainment/detrainment
CLIMAParameters.Atmos.EDMF.β(ps::EarthParameterSet) = ps.nt.β # sorting power for ad-hoc moisture detrainment function
CLIMAParameters.Atmos.EDMF.χ(ps::EarthParameterSet) = ps.nt.χ # fraction of updraft air for buoyancy mixing in entrainment/detrainment (0≤χ≤1)
CLIMAParameters.Atmos.EDMF.c_γ(ps::EarthParameterSet) = ps.nt.c_γ # scaling factor for turbulent entrainment rate
CLIMAParameters.Atmos.EDMF.c_λ(ps::EarthParameterSet) = ps.nt.c_λ # scaling factor for TKE in entrainment scale calculations
CLIMAParameters.Atmos.EDMF.w_min(ps::EarthParameterSet) = ps.nt.w_min # minimum updraft velocity to avoid zero division in b/w²
CLIMAParameters.Atmos.EDMF.μ_0(ps::EarthParameterSet) = ps.nt.μ_0 # dimensional scale logistic function in the dry term in entrainment/detrainment
# mixing length parameters
CLIMAParameters.Atmos.EDMF.c_m(ps::EarthParameterSet) = ps.nt.c_m # tke diffusivity coefficient
CLIMAParameters.Atmos.EDMF.c_d(ps::EarthParameterSet) = ps.nt.c_d # tke dissipation coefficient
CLIMAParameters.Atmos.EDMF.c_b(ps::EarthParameterSet) = ps.nt.c_b # static stability coefficient
CLIMAParameters.Atmos.EDMF.κ_star²(ps::EarthParameterSet) = ps.nt.κ_star² # Ratio of TKE to squared friction velocity in surface layer
CLIMAParameters.Atmos.EDMF.Pr_n(ps::EarthParameterSet) = ps.nt.Pr_n # turbulent Prandtl number in neutral conditions
CLIMAParameters.Atmos.EDMF.ω_pr(ps::EarthParameterSet) = ps.nt.ω_pr # cospectral budget factor for turbulent Prandtl number
CLIMAParameters.Atmos.EDMF.Ri_c(ps::EarthParameterSet) = ps.nt.Ri_c # critical Richardson number
CLIMAParameters.Atmos.EDMF.smin_ub(ps::EarthParameterSet) = ps.nt.smin_ub #  lower limit for smin function
CLIMAParameters.Atmos.EDMF.smin_rm(ps::EarthParameterSet) = ps.nt.smin_rm #  upper ratio limit for smin function

#! format: off
function create_parameter_set(namelist)
    TC = TurbulenceConvection
    nt = (;
        MSLP = 100000.0, # or grab from, e.g., namelist[""][...]
        cp_d = 1004.0,
        cp_v = 1859.0,
        R_d = 287.1,
        R_v = 461.5,
        molmass_ratio = 461.5/287.1,
        τ_precip = TC.parse_namelist(namelist, "microphysics", "τ_precip"; default = 1000.0),
        τ_acnv_rai = TC.parse_namelist(namelist, "microphysics", "τ_acnv_rai"; default = 2500.0),
        τ_acnv_sno = TC.parse_namelist(namelist, "microphysics", "τ_acnv_sno"; default = 100.0),
        q_liq_threshold = TC.parse_namelist(namelist, "microphysics", "q_liq_threshold"; default = 0.5e-3),
        q_ice_threshold = TC.parse_namelist(namelist, "microphysics", "q_ice_threshold"; default = 1e-6),
        c_ε = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "entrainment_factor"),
        c_div = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "entrainment_massflux_div_factor"; default = 0.0),
        α_b = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_buoy_coeff1"),
        α_a = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_adv_coeff"),
        α_d = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "pressure_normalmode_drag_coeff"),
        H_up_min = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "min_updraft_top"),
        ω_pr = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "Prandtl_number_scale"),
        c_δ = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "detrainment_factor"),
        c_gen = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "general_ent_params"),
        β = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "sorting_power"),
        χ = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "updraft_mixing_frac"),
        c_γ = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "turbulent_entrainment_factor"),
        c_λ = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "entrainment_smin_tke_coeff"),
        w_min = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "min_upd_velocity"),
        μ_0 = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "entrainment_scale"),
        γ_lim = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "area_limiter_scale"),
        β_lim = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "area_limiter_power"),
        c_m = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "tke_ed_coeff"),
        c_d = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "tke_diss_coeff"),
        c_b = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "static_stab_coeff"; default = 0.4), # this is here due to a value error in CliMAParmameters.jl
        κ_star² = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "tke_surf_scale"),
        Pr_n = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "Prandtl_number_0"),
        Ri_c = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "Ri_crit"),
        smin_ub = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "smin_ub"),
        smin_rm = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "smin_rm"),
        l_max = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "l_max"; default = 1.0e6),

        ## Stochastic parameters
        # lognormal model
        stoch_ε_lognormal_var = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "stochastic", "entr_lognormal_var"; default = 0.0),
        stoch_δ_lognormal_var = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "stochastic", "detr_lognormal_var"; default = 0.0),
        # sde model
        sde_ϵ_θ = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "stochastic", "sde_entr_theta"; default = 1.0),
        sde_ϵ_σ = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "stochastic", "sde_entr_std"; default = 0.0),
        sde_δ_θ = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "stochastic", "sde_detr_theta"; default = 1.0),
        sde_δ_σ = TC.parse_namelist(namelist, "turbulence", "EDMF_PrognosticTKE", "stochastic", "sde_detr_std"; default = 0.0),
    )
    param_set = EarthParameterSet(nt)
    if !isbits(param_set)
        @show param_set
        error("The parameter set MUST be isbits in order to be stack-allocated.")
    end
    return param_set
end
#! format: on

struct Rico end

function ForcingBase(param_set::APS)
    latitude = 18.0
    Omega = CPP.Omega(param_set)
    return TC.ForcingBase(
        TC.ForcingStandard;
        apply_coriolis = true,
        apply_subsidence = true,
        coriolis_param = 2.0 * Omega * sin(latitude * π / 180.0),
    ) #= s^{-1} =#
end

function reference_params(grid::TC.Grid, param_set::APS, namelist)
    molmass_ratio = CPP.molmass_ratio(param_set)
    Pg = 1.0154e5  #Pressure at ground
    Tg = 299.8  #Temperature at ground
    pvg = TD.saturation_vapor_pressure(param_set, Tg, TD.Liquid())
    qtg = (1 / molmass_ratio) * pvg / (Pg - pvg)   #Total water mixing ratio at surface
    return (; Pg, Tg, qtg)
end

function surface_params(grid::TC.Grid, state::TC.State, param_set; kwargs...)
    FT = eltype(grid)

    zrough = 0.00015
    cm = 0.001229
    ch = 0.001094
    cq = 0.001133
    # Adjust for non-IC grid spacing
    grid_adjust = (log(20.0 / zrough) / log(TC.zc_surface(grid) / zrough))^2
    cm = cm * grid_adjust
    ch = ch * grid_adjust
    cq = cq * grid_adjust # TODO: not yet used..
    Tsurface = 299.8

    # For Rico we provide values of transfer coefficients
    kf_surf = TC.kf_surface(grid)
    p0_f_surf = TC.face_ref_state(state).p0[kf_surf]
    ts = TD.PhaseEquil_pTq(param_set, p0_f_surf, Tsurface, FT(0)) # TODO: is this correct?
    qsurface = TD.q_vap_saturation(ts)
    kwargs = (; zrough, Tsurface, qsurface, cm, ch)
    return TC.FixedSurfaceCoeffs(FT; kwargs...)
end

#####
##### Fields
#####

##### Auxiliary fields

# Face & Center
aux_vars_ref_state(FT) = (; ref_state = (ρ0 = FT(0), α0 = FT(0), p0 = FT(0)))

# Center only
cent_aux_vars_gm(FT) = (;
    tke = FT(0),
    Hvar = FT(0),
    QTvar = FT(0),
    HQTcov = FT(0),
    q_liq = FT(0),
    q_ice = FT(0),
    RH = FT(0),
    s = FT(0),
    T = FT(0),
    buoy = FT(0),
    cloud_fraction = FT(0),
    H_third_m = FT(0),
    W_third_m = FT(0),
    QT_third_m = FT(0),
    # From RadiationBase
    dTdt_rad = FT(0), # horizontal advection temperature tendency
    dqtdt_rad = FT(0), # horizontal advection moisture tendency
    # From ForcingBase
    subsidence = FT(0), #Large-scale subsidence
    dTdt = FT(0), #Large-scale temperature tendency
    dqtdt = FT(0), #Large-scale moisture tendency
    dTdt_hadv = FT(0), #Horizontal advection of temperature
    H_nudge = FT(0), #Reference H profile for relaxation tendency
    dTdt_fluc = FT(0), #Vertical turbulent advection of temperature
    dqtdt_hadv = FT(0), #Horizontal advection of moisture
    qt_nudge = FT(0), #Reference qt profile for relaxation tendency
    dqtdt_fluc = FT(0), #Vertical turbulent advection of moisture
    u_nudge = FT(0), #Reference u profile for relaxation tendency
    v_nudge = FT(0), #Reference v profile for relaxation tendency
    ug = FT(0), #Geostrophic u velocity
    vg = FT(0), #Geostrophic v velocity
    ∇θ_liq_ice_gm = FT(0),
    ∇q_tot_gm = FT(0),
)
cent_aux_vars(FT, n_up) = (; aux_vars_ref_state(FT)..., cent_aux_vars_gm(FT)..., TC.cent_aux_vars_edmf(FT, n_up)...)

# Face only
face_aux_vars_gm(FT) = (;
    massflux_s = FT(0),
    diffusive_flux_s = FT(0),
    total_flux_s = FT(0),
    f_rad = FT(0),
    sgs_flux_θ_liq_ice = FT(0),
    sgs_flux_q_tot = FT(0),
    sgs_flux_u = FT(0),
    sgs_flux_v = FT(0),
)
face_aux_vars(FT, n_up) = (; aux_vars_ref_state(FT)..., face_aux_vars_gm(FT)..., TC.face_aux_vars_edmf(FT, n_up)...)

##### Diagnostic fields

##### Prognostic fields

# Center only
cent_prognostic_vars(FT, n_up) = (; cent_prognostic_vars_gm(FT)..., TC.cent_prognostic_vars_edmf(FT, n_up)...)
cent_prognostic_vars_gm(FT) = (; u = FT(0), v = FT(0), θ_liq_ice = FT(0), q_tot = FT(0))

# Face only
face_prognostic_vars(FT, n_up) = (; w = FT(0), TC.face_prognostic_vars_edmf(FT, n_up)...)
# TC.face_prognostic_vars_edmf(FT, n_up) = (;) # could also use this for empty model


#####
##### Methods
#####

####
#### Reference state
####

"""
    compute_ref_state!(
        state,
        grid::TC.Grid,
        param_set::PS;
        Pg::FT,
        Tg::FT,
        qtg::FT,
    ) where {PS, FT}

TODO: add better docs once the API converges

The reference profiles, given
 - `grid` the grid
 - `param_set` the parameter set
"""
function compute_ref_state!(state, grid::TC.Grid, param_set::PS; Pg::FT, Tg::FT, qtg::FT) where {PS, FT}

    p0_c = TC.center_ref_state(state).p0
    ρ0_c = TC.center_ref_state(state).ρ0
    α0_c = TC.center_ref_state(state).α0
    p0_f = TC.face_ref_state(state).p0
    ρ0_f = TC.face_ref_state(state).ρ0
    α0_f = TC.face_ref_state(state).α0

    q_pt_g = TD.PhasePartition(qtg)
    ts_g = TD.PhaseEquil_pTq(param_set, Pg, Tg, qtg)
    θ_liq_ice_g = TD.liquid_ice_pottemp(ts_g)

    # We are integrating the log pressure so need to take the log of the
    # surface pressure
    logp = log(Pg)

    # Form a right hand side for integrating the hydrostatic equation to
    # determine the reference pressure
    function rhs(logp, u, z)
        p_ = exp(logp)
        ts = TC.thermo_state_pθq(param_set, p_, θ_liq_ice_g, qtg)
        R_m = TD.gas_constant_air(ts)
        T = TD.air_temperature(ts)
        return -FT(CPP.grav(param_set)) / (T * R_m)
    end

    # Perform the integration
    z_span = (grid.zmin, grid.zmax)
    @show z_span
    prob = ODE.ODEProblem(rhs, logp, z_span)
    sol = ODE.solve(prob, ODE.Tsit5(), reltol = 1e-12, abstol = 1e-12)

    parent(p0_f) .= sol.(vec(grid.zf))
    parent(p0_c) .= sol.(vec(grid.zc))

    p0_f .= exp.(p0_f)
    p0_c .= exp.(p0_c)

    # Compute reference state thermodynamic profiles
    @inbounds for k in TC.real_center_indices(grid)
        ts = TC.thermo_state_pθq(param_set, p0_c[k], θ_liq_ice_g, qtg)
        α0_c[k] = TD.specific_volume(ts)
    end

    @inbounds for k in TC.real_face_indices(grid)
        ts = TC.thermo_state_pθq(param_set, p0_f[k], θ_liq_ice_g, qtg)
        α0_f[k] = TD.specific_volume(ts)
    end

    ρ0_f .= 1 ./ α0_f
    ρ0_c .= 1 ./ α0_c
    return nothing
end

# Compute the sum of tendencies for the scheme
function ∑tendencies!(tendencies::FV, prog::FV, params::NT, t::Real) where {NT, FV <: CC.Fields.FieldVector}
    UnPack.@unpack edmf, grid, gm, case, aux, TS = params

    state = TC.State(prog, aux, tendencies)

    Δt = TS.dt
    param_set = TC.parameter_set(gm)
    surf = get_surface(case.surf_params, grid, state, gm, t, param_set)
    force = case.Fo
    radiation = case.Rad
    en_thermo = edmf.en_thermo
    precip_model = edmf.precip_model

    TC.affect_filter!(edmf, grid, state, gm, surf, case.casename, t)
    TC.update_aux!(edmf, gm, grid, state, case, surf, param_set, t, Δt)

    tends_face = tendencies.face
    tends_cent = tendencies.cent
    parent(tends_face) .= 0
    parent(tends_cent) .= 0

    # causes division error in dry bubble first time step
    TC.compute_precipitation_formation_tendencies(grid, state, edmf, precip_model, Δt, param_set)

    TC.microphysics(en_thermo, grid, state, precip_model, Δt, param_set)
    TC.compute_precipitation_sink_tendencies(precip_model, grid, state, gm, Δt)
    TC.compute_precipitation_advection_tendencies(precip_model, edmf, grid, state, gm)

    # compute tendencies
    compute_gm_tendencies!(edmf, grid, state, surf, radiation, force, gm)
    TC.compute_up_tendencies!(edmf, grid, state, gm, surf)

    TC.compute_en_tendencies!(edmf, grid, state, param_set, Val(:tke), Val(:ρatke))
    TC.compute_en_tendencies!(edmf, grid, state, param_set, Val(:Hvar), Val(:ρaHvar))
    TC.compute_en_tendencies!(edmf, grid, state, param_set, Val(:QTvar), Val(:ρaQTvar))
    TC.compute_en_tendencies!(edmf, grid, state, param_set, Val(:HQTcov), Val(:ρaHQTcov))

    return nothing
end


function compute_gm_tendencies!(
    edmf::TC.EDMF_PrognosticTKE,
    grid::TC.Grid,
    state::TC.State,
    surf::TC.SurfaceBase,
    radiation::TC.RadiationBase,
    force::TC.ForcingBase,
    gm::TC.GridMeanVariables,
)
    tendencies_gm = TC.center_tendencies_grid_mean(state)
    kc_toa = TC.kc_top_of_atmos(grid)
    kf_surf = TC.kf_surface(grid)
    FT = eltype(grid)
    param_set = TC.parameter_set(gm)
    prog_gm = TC.center_prog_grid_mean(state)
    aux_gm = TC.center_aux_grid_mean(state)
    aux_gm_f = TC.face_aux_grid_mean(state)
    ∇θ_liq_ice_gm = TC.center_aux_grid_mean(state).∇θ_liq_ice_gm
    ∇q_tot_gm = TC.center_aux_grid_mean(state).∇q_tot_gm
    aux_en = TC.center_aux_environment(state)
    aux_up = TC.center_aux_updrafts(state)
    aux_bulk = TC.center_aux_bulk(state)
    aux_tc_f = TC.face_aux_turbconv(state)
    aux_up_f = TC.face_aux_updrafts(state)
    ρ0_f = TC.face_ref_state(state).ρ0
    p0_c = TC.center_ref_state(state).p0
    α0_c = TC.center_ref_state(state).α0
    aux_tc = TC.center_aux_turbconv(state)

    θ_liq_ice_gm_toa = prog_gm.θ_liq_ice[kc_toa]
    q_tot_gm_toa = prog_gm.q_tot[kc_toa]
    RBθ = CCO.RightBiasedC2F(; top = CCO.SetValue(θ_liq_ice_gm_toa))
    RBq = CCO.RightBiasedC2F(; top = CCO.SetValue(q_tot_gm_toa))
    wvec = CC.Geometry.WVector
    ∇c = CCO.DivergenceF2C()
    @. ∇θ_liq_ice_gm = ∇c(wvec(RBθ(prog_gm.θ_liq_ice)))
    @. ∇q_tot_gm = ∇c(wvec(RBq(prog_gm.q_tot)))

    @inbounds for k in TC.real_center_indices(grid)
        # Apply large-scale horizontal advection tendencies
        ts = TC.thermo_state_pθq(param_set, p0_c[k], prog_gm.θ_liq_ice[k], prog_gm.q_tot[k])
        Π = TD.exner(ts)

        tendencies_gm.u[k] -= force.coriolis_param * (aux_gm.vg[k] - prog_gm.v[k])
        tendencies_gm.v[k] += force.coriolis_param * (aux_gm.ug[k] - prog_gm.u[k])

        tendencies_gm.θ_liq_ice[k] -= ∇θ_liq_ice_gm[k] * aux_gm.subsidence[k]
        tendencies_gm.q_tot[k] -= ∇q_tot_gm[k] * aux_gm.subsidence[k]

        tendencies_gm.θ_liq_ice[k] += aux_gm.dTdt[k] / Π
        tendencies_gm.q_tot[k] += aux_gm.dqtdt[k]

        tendencies_gm.q_tot[k] +=
            aux_bulk.qt_tendency_precip_formation[k] +
            aux_en.qt_tendency_precip_formation[k] +
            aux_tc.qt_tendency_precip_sinks[k]
        tendencies_gm.θ_liq_ice[k] +=
            aux_bulk.θ_liq_ice_tendency_precip_formation[k] +
            aux_en.θ_liq_ice_tendency_precip_formation[k] +
            aux_tc.θ_liq_ice_tendency_precip_sinks[k]
    end
    TC.compute_sgs_tendencies!(edmf, grid, state, surf, radiation, force, gm)

    sgs_flux_θ_liq_ice = aux_gm_f.sgs_flux_θ_liq_ice
    sgs_flux_q_tot = aux_gm_f.sgs_flux_q_tot
    sgs_flux_u = aux_gm_f.sgs_flux_u
    sgs_flux_v = aux_gm_f.sgs_flux_v
    # apply surface BC as SGS flux at lowest level
    sgs_flux_θ_liq_ice[kf_surf] = surf.ρθ_liq_ice_flux
    sgs_flux_q_tot[kf_surf] = surf.ρq_tot_flux
    sgs_flux_u[kf_surf] = surf.ρu_flux
    sgs_flux_v[kf_surf] = surf.ρv_flux

    tends_θ_liq_ice = tendencies_gm.θ_liq_ice
    tends_q_tot = tendencies_gm.q_tot
    tends_u = tendencies_gm.u
    tends_v = tendencies_gm.v

    ∇θ_liq_ice_sgs = CCO.DivergenceF2C()
    ∇q_tot_sgs = CCO.DivergenceF2C()
    ∇u_sgs = CCO.DivergenceF2C()
    ∇v_sgs = CCO.DivergenceF2C()

    @. tends_θ_liq_ice += -α0_c * ∇θ_liq_ice_sgs(wvec(sgs_flux_θ_liq_ice))
    @. tends_q_tot += -α0_c * ∇q_tot_sgs(wvec(sgs_flux_q_tot))
    @. tends_u += -α0_c * ∇u_sgs(wvec(sgs_flux_u))
    @. tends_v += -α0_c * ∇v_sgs(wvec(sgs_flux_v))
    return nothing
end

mutable struct TimeStepping
    dt::Float64
    t_max::Float64
    t::Float64
    nstep::Int
    cfl_limit::Float64
    dt_max::Float64
    dt_max_edmf::Float64
    dt_io::Float64
end

function TimeStepping(namelist)
    dt = TC.parse_namelist(namelist, "time_stepping", "dt_min"; default = 1.0)
    t_max = TC.parse_namelist(namelist, "time_stepping", "t_max"; default = 7200.0)
    cfl_limit = TC.parse_namelist(namelist, "time_stepping", "cfl_limit"; default = 0.5)
    dt_max = TC.parse_namelist(namelist, "time_stepping", "dt_max"; default = 10.0)
    dt_max_edmf = 0.0

    # set time
    t = 0.0
    dt_io = 0.0
    nstep = 0

    return TimeStepping(dt, t_max, t, nstep, cfl_limit, dt_max, dt_max_edmf, dt_io)
end

function get_surface(
    surf_params::TC.FixedSurfaceCoeffs,
    grid::TC.Grid,
    state::TC.State,
    gm::TC.GridMeanVariables,
    t::Real,
    param_set::CP.AbstractEarthParameterSet,
)
    FT = eltype(grid)
    kc_surf = TC.kc_surface(grid)
    kf_surf = TC.kf_surface(grid)
    p0_f_surf = TC.face_ref_state(state).p0[kf_surf]
    p0_c_surf = TC.center_ref_state(state).p0[kc_surf]
    aux_gm = TC.center_aux_grid_mean(state)
    prog_gm = TC.center_prog_grid_mean(state)
    u_gm_surf = prog_gm.u[kc_surf]
    v_gm_surf = prog_gm.v[kc_surf]
    q_tot_gm_surf = prog_gm.q_tot[kc_surf]
    θ_liq_ice_gm_surf = prog_gm.θ_liq_ice[kc_surf]
    Tsurface = TC.surface_temperature(surf_params, t)
    qsurface = TC.surface_q_tot(surf_params, t)
    zrough = surf_params.zrough
    cm = surf_params.cm
    ch = surf_params.ch

    universal_func = UF.Businger()
    scheme = SF.FVScheme()
    z_sfc = FT(0)
    z_in = grid.zc[kc_surf].z
    ts_sfc = TC.thermo_state_pθq(param_set, p0_f_surf, Tsurface, qsurface)
    ts_in = TC.thermo_state_pθq(param_set, p0_c_surf, θ_liq_ice_gm_surf, q_tot_gm_surf)
    u_sfc = SA.SVector{2, FT}(0, 0)
    u_in = SA.SVector{2, FT}(u_gm_surf, v_gm_surf)
    vals_sfc = SF.SurfaceValues(z_sfc, u_sfc, ts_sfc)
    vals_int = SF.InteriorValues(z_in, u_in, ts_in)
    sc = SF.Coefficients{FT}(state_in = vals_int, state_sfc = vals_sfc, Cd = cm, Ch = ch, z0m = zrough, z0b = zrough)
    result = SF.surface_conditions(param_set, sc, universal_func, scheme)
    lhf = result.lhf
    shf = result.shf

    return TC.SurfaceBase{FT}(;
        cm = result.Cd,
        ch = result.Ch,
        obukhov_length = result.L_MO,
        lhf = lhf,
        shf = shf,
        ustar = result.ustar,
        ρu_flux = result.ρτxz,
        ρv_flux = result.ρτyz,
        ρθ_liq_ice_flux = shf / TD.cp_m(ts_in),
        ρq_tot_flux = lhf / TD.latent_heat_vapor(ts_in),
        bflux = result.buoy_flux,
    )
end

struct Simulation1d{IONT, G, S, GM, C, EDMF, D, TIMESTEPPING, STATS, PS}
    io_nt::IONT
    grid::G
    state::S
    gm::GM
    case::C
    edmf::EDMF
    diagnostics::D
    TS::TIMESTEPPING
    Stats::STATS
    param_set::PS
    skip_io::Bool
    adapt_dt::Bool
    cfl_limit::Float64
    dt_min::Float64
end

function Simulation1d(namelist)
    TC = TurbulenceConvection
    param_set = create_parameter_set(namelist)

    FT = Float64
    skip_io = namelist["stats_io"]["skip"]
    adapt_dt = namelist["time_stepping"]["adapt_dt"]
    cfl_limit = namelist["time_stepping"]["cfl_limit"]
    dt_min = namelist["time_stepping"]["dt_min"]

    grid = TC.Grid(FT(namelist["grid"]["dz"]), namelist["grid"]["nz"])
    Stats = skip_io ? nothing : NetCDFIO_Stats(namelist, grid)
    ref_params = reference_params(grid, param_set, namelist)

    gm = TC.GridMeanVariables(param_set)
    Fo = ForcingBase(param_set)
    Rad = TC.RadiationBase{TC.RadiationNone}()
    TS = TimeStepping(namelist)

    edmf = TC.EDMF_PrognosticTKE(namelist, grid, param_set)
    isbits(edmf) || error("Something non-isbits was added to edmf and needs to be fixed.")
    N_up = TC.n_updrafts(edmf)

    cspace = TC.center_space(grid)
    fspace = TC.face_space(grid)

    cent_prog_fields() = TC.FieldFromNamedTuple(cspace, cent_prognostic_vars(FT, N_up))
    face_prog_fields() = TC.FieldFromNamedTuple(fspace, face_prognostic_vars(FT, N_up))
    aux_cent_fields = TC.FieldFromNamedTuple(cspace, cent_aux_vars(FT, N_up))
    aux_face_fields = TC.FieldFromNamedTuple(fspace, face_aux_vars(FT, N_up))
    diagnostic_cent_fields = TC.FieldFromNamedTuple(cspace, TC.cent_diagnostic_vars_edmf(FT, N_up))
    diagnostic_face_fields = TC.FieldFromNamedTuple(fspace, TC.face_diagnostic_vars_edmf(FT, N_up))

    prog = CC.Fields.FieldVector(cent = cent_prog_fields(), face = face_prog_fields())
    aux = CC.Fields.FieldVector(cent = aux_cent_fields, face = aux_face_fields)
    diagnostics = CC.Fields.FieldVector(cent = diagnostic_cent_fields, face = diagnostic_face_fields)

    # `nothing` goes into State because OrdinaryDiffEq.jl owns tendencies.
    state = TC.State(prog, aux, nothing)
    compute_ref_state!(state, grid, param_set; ref_params...)

    Ri_bulk_crit = namelist["turbulence"]["EDMF_PrognosticTKE"]["Ri_crit"]
    surf_params = surface_params(grid, state, param_set; Ri_bulk_crit = Ri_bulk_crit)
    inversion_type = TC.CriticalRiInversion()
    case = TC.CasesBase(Rico(); inversion_type, surf_params, Fo, Rad)

    io_nt = (;
        ref_state = TC.io_dictionary_ref_state(),
        aux = TC.io_dictionary_aux(),
        diagnostics = io_dictionary_diagnostics(),
    )

    return Simulation1d(
        io_nt,
        grid,
        state,
        gm,
        case,
        edmf,
        diagnostics,
        TS,
        Stats,
        param_set,
        skip_io,
        adapt_dt,
        cfl_limit,
        dt_min,
    )
end
include("init_1d_rico.jl")

function condition_io(u, t, integrator)
    UnPack.@unpack TS, Stats = integrator.p
    TS.dt_io += TS.dt
    io_flag = false
    if TS.dt_io > Stats.frequency
        TS.dt_io = 0
        io_flag = true
    end
    return io_flag || t ≈ 0 || t ≈ TS.t_max
end

condition_every_iter(u, t, integrator) = true

function affect_io!(integrator)
    UnPack.@unpack edmf, aux, grid, io_nt, diagnostics, case, gm, Stats, skip_io = integrator.p
    skip_io && return nothing
    t = integrator.t
    state = TC.State(integrator.u, aux, integrator.du)
    param_set = TC.parameter_set(gm)
    compute_diagnostics!(edmf, gm, grid, state, diagnostics, Stats, case, t)
    write_simulation_time(Stats, t)
    io(io_nt.aux, Stats, state)
    io(io_nt.diagnostics, Stats, diagnostics)
    surf = get_surface(case.surf_params, grid, state, gm, t, param_set)
    io(surf, case.surf_params, grid, state, Stats, t)
    ODE.u_modified!(integrator, false) # We're legitamately not mutating `u` (the state vector)
end

function affect_filter!(integrator)
    UnPack.@unpack edmf, grid, gm, aux, case = integrator.p
    t = integrator.t
    param_set = TC.parameter_set(gm)
    state = TC.State(integrator.u, aux, integrator.du)
    surf = get_surface(case.surf_params, grid, state, gm, t, param_set)
    TC.affect_filter!(edmf, grid, state, gm, surf, case.casename, t)
    ODE.u_modified!(integrator, false)
end

function dt_max!(integrator)
    UnPack.@unpack gm, grid, edmf, aux, TS = integrator.p
    state = TC.State(integrator.u, aux, integrator.du)
    prog_gm = TC.center_prog_grid_mean(state)
    prog_gm_f = TC.face_prog_grid_mean(state)
    Δzc = TC.get_Δz(prog_gm.u)
    Δzf = TC.get_Δz(prog_gm_f.w)
    CFL_limit = TS.cfl_limit
    N_up = TC.n_updrafts(edmf)

    dt_max = TS.dt_max # initialize dt_max

    aux_tc = TC.center_aux_turbconv(state)
    aux_up_f = TC.face_aux_updrafts(state)
    aux_en_f = TC.face_aux_environment(state)
    KM = aux_tc.KM
    KH = aux_tc.KH

    # helper to calculate the rain velocity
    # TODO: assuming gm.W = 0
    # TODO: verify translation
    term_vel_rain = aux_tc.term_vel_rain
    term_vel_snow = aux_tc.term_vel_snow

    @inbounds for k in TC.real_face_indices(grid)
        TC.is_surface_face(grid, k) && continue
        @inbounds for i in 1:N_up
            dt_max = min(dt_max, CFL_limit * Δzf[k] / (abs(aux_up_f[i].w[k]) + eps(Float32)))
        end
        dt_max = min(dt_max, CFL_limit * Δzf[k] / (abs(aux_en_f.w[k]) + eps(Float32)))
    end
    @inbounds for k in TC.real_center_indices(grid)
        vel_max = max(term_vel_rain[k], term_vel_snow[k])
        # Check terminal rain/snow velocity CFL
        dt_max = min(dt_max, CFL_limit * Δzc[k] / (vel_max + eps(Float32)))
        # Check diffusion CFL (i.e., Fourier number)
        dt_max = min(dt_max, CFL_limit * Δzc[k]^2 / (max(KH[k], KM[k]) + eps(Float32)))
    end
    TS.dt_max_edmf = dt_max

    ODE.u_modified!(integrator, false)
end

function monitor_cfl!(integrator)
    UnPack.@unpack gm, grid, edmf, aux, TS = integrator.p
    state = TC.State(integrator.u, aux, integrator.du)
    prog_gm = TC.center_prog_grid_mean(state)
    Δz = TC.get_Δz(prog_gm.u)
    Δt = TS.dt
    CFL_limit = TS.cfl_limit
    aux_tc = TC.center_aux_turbconv(state)
    term_vel_rain = aux_tc.term_vel_rain
    term_vel_snow = aux_tc.term_vel_snow
    @inbounds for k in TC.real_center_indices(grid)
        # check stability criterion
        CFL_out_rain = Δt / Δz[k] * term_vel_rain[k]
        CFL_out_snow = Δt / Δz[k] * term_vel_snow[k]
        if TC.is_toa_center(grid, k)
            CFL_in_rain = 0.0
            CFL_in_snow = 0.0
        else
            CFL_in_rain = Δt / Δz[k] * term_vel_rain[k + 1]
            CFL_in_snow = Δt / Δz[k] * term_vel_snow[k + 1]
        end
        if max(CFL_in_rain, CFL_in_snow, CFL_out_rain, CFL_out_snow) > CFL_limit
            error("Time step is too large for rain fall velocity!")
        end
    end
    ODE.u_modified!(integrator, false)
end


function main(namelist)
    sim = Simulation1d(namelist)
    initialize_rico(sim)
    @timev begin
        open_files(sim.Stats)

        grid = sim.grid
        state = sim.state
        prog = state.prog
        aux = state.aux
        TS = sim.TS
        diagnostics = sim.diagnostics

        t_span = (0.0, sim.TS.t_max)
        params = (;
            edmf = sim.edmf,
            grid = grid,
            gm = sim.gm,
            aux = aux,
            io_nt = sim.io_nt,
            case = sim.case,
            diagnostics = diagnostics,
            TS = sim.TS,
            Stats = sim.Stats,
            skip_io = sim.skip_io,
            adapt_dt = sim.adapt_dt,
            cfl_limit = sim.cfl_limit,
            dt_min = sim.dt_min,
        )

        callback_io = ODE.DiscreteCallback(condition_io, affect_io!; save_positions = (false, false))
        callback_io = (callback_io,)
        callback_cfl = ODE.DiscreteCallback(condition_every_iter, monitor_cfl!; save_positions = (false, false))
        callback_cfl = sim.edmf.precip_model isa TC.Clima1M ? (callback_cfl,) : ()
        callback_dtmax = ODE.DiscreteCallback(condition_every_iter, dt_max!; save_positions = (false, false))
        callback_filters = ODE.DiscreteCallback(condition_every_iter, affect_filter!; save_positions = (false, false))

        callbacks = ODE.CallbackSet(callback_dtmax, callback_cfl..., callback_filters, callback_io...)

        prob = ODE.ODEProblem(∑tendencies!, state.prog, t_span, params; dt = sim.TS.dt)

        alg = ODE.Euler()
        sol = @timev ODE.solve(prob, alg;
            progress_steps = 100,
            save_start = false,
            saveat = last(t_span),
            callback = callbacks,
            progress = true,
            progress_message = (dt, u, p, t) -> t,
            )
        close_files(sim.Stats)
    end
    return sim.Stats.path_plus_file
end

function rico_namelist()

    namelist = Dict()
    namelist["meta"] = Dict()
    namelist["meta"]["uuid"] = basename(tempname())

    namelist["turbulence"] = Dict()

    namelist["turbulence"]["EDMF_PrognosticTKE"] = Dict()
    namelist["turbulence"]["EDMF_PrognosticTKE"]["surface_area"] = 0.1
    namelist["turbulence"]["EDMF_PrognosticTKE"]["max_area"] = 0.9
    # mixing_length
    namelist["turbulence"]["EDMF_PrognosticTKE"]["tke_ed_coeff"] = 0.14
    namelist["turbulence"]["EDMF_PrognosticTKE"]["tke_diss_coeff"] = 0.22
    namelist["turbulence"]["EDMF_PrognosticTKE"]["static_stab_coeff"] = 0.4
    namelist["turbulence"]["EDMF_PrognosticTKE"]["tke_surf_scale"] = 3.75
    namelist["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_scale"] = 53.0 / 13.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["Prandtl_number_0"] = 0.74
    namelist["turbulence"]["EDMF_PrognosticTKE"]["Ri_crit"] = 0.25
    namelist["turbulence"]["EDMF_PrognosticTKE"]["smin_ub"] = 0.1
    namelist["turbulence"]["EDMF_PrognosticTKE"]["smin_rm"] = 1.5
    namelist["turbulence"]["EDMF_PrognosticTKE"]["l_max"] = 1.0e6
    # entrainment
    namelist["turbulence"]["EDMF_PrognosticTKE"]["entrainment_factor"] = 0.13
    namelist["turbulence"]["EDMF_PrognosticTKE"]["detrainment_factor"] = 0.51
    # 1-layer nn parameters
    #! format: off
    namelist["turbulence"]["EDMF_PrognosticTKE"]["general_ent_params"] =
        SA.SVector(0.3038, 0.719,-0.910,-0.483,
                   0.739, 0.0755, 0.178, 0.521,
                   0.0, 0.0, 0.843,-0.340,
                   0.655, 0.113, 0.0, 0.0)
    #! format: on

    namelist["turbulence"]["EDMF_PrognosticTKE"]["entrainment_massflux_div_factor"] = 0.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["turbulent_entrainment_factor"] = 0.075
    namelist["turbulence"]["EDMF_PrognosticTKE"]["entrainment_smin_tke_coeff"] = 0.3
    namelist["turbulence"]["EDMF_PrognosticTKE"]["updraft_mixing_frac"] = 0.25
    namelist["turbulence"]["EDMF_PrognosticTKE"]["area_limiter_scale"] = 10.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["area_limiter_power"] = 3.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["entrainment_scale"] = 0.0004
    namelist["turbulence"]["EDMF_PrognosticTKE"]["sorting_power"] = 2.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["min_upd_velocity"] = 0.001
    # pressure
    namelist["turbulence"]["EDMF_PrognosticTKE"]["min_updraft_top"] = 500.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff1"] = 0.12
    namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_buoy_coeff2"] = 0.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_adv_coeff"] = 0.1
    namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_normalmode_drag_coeff"] = 10.0

    # stochastic closures
    namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"] = Dict()
    namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"]["closure"] = "none"
    namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"]["entr_lognormal_var"] = 0.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"]["detr_lognormal_var"] = 0.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"]["sde_entr_theta"] = 1.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"]["sde_entr_std"] = 0.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"]["sde_detr_theta"] = 1.0
    namelist["turbulence"]["EDMF_PrognosticTKE"]["stochastic"]["sde_detr_std"] = 0.0

    # From namelist
    namelist["grid"] = Dict()
    namelist["grid"]["dims"] = 1

    namelist["thermodynamics"] = Dict()
    namelist["thermodynamics"]["thermal_variable"] = "thetal"
    namelist["thermodynamics"]["sgs"] = "quadrature"
    namelist["thermodynamics"]["quadrature_order"] = 3
    namelist["thermodynamics"]["quadrature_type"] = "log-normal" #"gaussian" or "log-normal"

    namelist["time_stepping"] = Dict()
    namelist["time_stepping"]["dt_max"] = 12.0
    namelist["time_stepping"]["dt_min"] = 1.0
    namelist["time_stepping"]["adapt_dt"] = true
    namelist["time_stepping"]["cfl_limit"] = 0.5

    namelist["microphysics"] = Dict()
    namelist["microphysics"]["precipitation_model"] = "None"

    namelist["turbulence"]["scheme"] = "EDMF_PrognosticTKE"

    namelist["turbulence"]["EDMF_PrognosticTKE"]["updraft_number"] = 1
    namelist["turbulence"]["EDMF_PrognosticTKE"]["entrainment"] = "moisture_deficit" #{"moisture_deficit", "NN", "Linear"}
    namelist["turbulence"]["EDMF_PrognosticTKE"]["use_local_micro"] = true
    namelist["turbulence"]["EDMF_PrognosticTKE"]["constant_area"] = false
    namelist["turbulence"]["EDMF_PrognosticTKE"]["calculate_tke"] = true
    namelist["turbulence"]["EDMF_PrognosticTKE"]["mixing_length"] = "sbtd_eq"
    namelist["turbulence"]["EDMF_PrognosticTKE"]["env_buoy_grad"] = "quadratures"

    namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_closure_buoy"] = "normalmode"
    namelist["turbulence"]["EDMF_PrognosticTKE"]["pressure_closure_drag"] = "normalmode"

    namelist["output"] = Dict()
    namelist["output"]["output_root"] = "./"

    namelist["stats_io"] = Dict()
    namelist["stats_io"]["stats_dir"] = "stats"
    namelist["stats_io"]["frequency"] = 60.0
    namelist["stats_io"]["skip"] = false

    namelist["meta"]["casename"] = "Rico"

    namelist["grid"]["nz"] = 80
    namelist["grid"]["dz"] = 50.0

    namelist["time_stepping"]["adapt_dt"] = false
    namelist["time_stepping"]["t_max"] = 86400.0
    #namelist["time_stepping"]["dt_max"] = 5.0
    namelist["time_stepping"]["dt_min"] = 1.5

    namelist["microphysics"]["precipitation_model"] = "clima_1m"

    namelist["meta"]["simname"] = "Rico"
    namelist["meta"]["casename"] = "Rico"

    return namelist
end

best_mse = OrderedCollections.OrderedDict()
best_mse["qt_mean"] = 1.257616309786706
best_mse["updraft_area"] = 479.23915567257177
best_mse["updraft_w"] = 70.42491018439706
best_mse["updraft_qt"] = 15.090769083132974
best_mse["updraft_thetal"] = 133.864818058174
best_mse["v_mean"] = 0.42557572031336477
best_mse["u_mean"] = 0.5798141428320317
best_mse["tke_mean"] = 143.3811018101168
best_mse["temperature_mean"] = 0.0006164390439910221
best_mse["ql_mean"] = 316.40935576448123
best_mse["qi_mean"] = "NA"
best_mse["qr_mean"] = 735.0401217909445
best_mse["thetal_mean"] = 0.0006111966685643192
best_mse["Hvar_mean"] = 213066.6024927279
best_mse["QTvar_mean"] = 44465.50066598392

case_name = "Rico"
println("Running $case_name...")
namelist = rico_namelist()
namelist["meta"]["uuid"] = "01"
ds_tc_filename, return_code = main(namelist)

include(joinpath(tc_dir, "post_processing", "compute_mse.jl"))
computed_mse = compute_mse_wrapper(
    case_name,
    best_mse,
    ds_tc_filename;
    ds_les_filename = joinpath(PyCLES_output_dataset_path, "Rico.nc"),
    ds_scm_filename = joinpath(SCAMPy_output_dataset_path, "Rico.nc"),
    plot_comparison = true,
    t_start = 22 * 3600,
    t_stop = 24 * 3600,
)

open("computed_mse_$case_name.json", "w") do io
    JSON.print(io, computed_mse)
end

@testset "Rico" begin
    for k in keys(best_mse)
        test_mse(computed_mse, best_mse, k)
    end
    include(joinpath(tc_dir, "post_processing", "post_run_tests.jl"))
    nothing
end
