# TurbulenceConvection.jl's API

```@meta
CurrentModule = TurbulenceConvection
```

```@docs
TurbulenceConvection.Parameters
```

# NameList Parameters

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| set_src_seed | Bool |  |
| config | String |  |
| test_duals | Bool |  |
| float_type | Float64 or Float32 |  |
|  |  |  |

## Turbulence

All parameters below are wrapped in a Dict named EDMF_PrognosticTKE

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| entrainment_massflux_div_factor | Float64 |  |
| stochastic_entrainment | String |  |
| pressure_closure_buoy | String |  |
| l_max | Float64 |  |
| tke_diss_coeff | Float64 |  |
| fno_ent_n_modes | Int64 |  |
| area_limiter_scale | Float64 |  |
| nn_ent_params | Vector{Float64} |  |
| entr_dim_scale | String |  |
| fno_ent_width | Int64 |  |
| pressure_normalmode_drag_coeff | Float64 |  |
| tke_ed_coeff | Float64 |  |
| max_area | Float64 |  |
| nn_ent_biases | Bool |  |
| rf_fix_ent_params | Vector{Float64} |  |
| surface_area | Float64 |  |
| entrainment_smin_tke_coeff | Float64 |  |
| rf_opt_ent_params | Vector{Float64} |  |
| min_area | Float64 |  |
| pressure_closure_drag | String |  |
| min_upd_velocity | Float64 |  |
| entrainment | String |  |
| smin_rm | Float64 |  |
| area_limiter_power | Float64 |  |
| entrainment_scale | Float64 |  |
| pressure_normalmode_buoy_coeff1 | Float64 |  |
| smin_ub | Float64 |  |
| detr_dim_scale | String |  |
| tke_surf_scale | Float64 |  |
| static_stab_coeff | Float64 |  |
| fno_ent_params | Vector{Float64} |  |
| pressure_normalmode_adv_coeff | Float64 |  |
| updraft_number | Int64 |  |
| entr_pi_subset | Vector{Int64} |  |
| Ri_crit | Float64 |  |
| nn_arc | Vector{Int64} |  |
| detrainment_factor | Float64 |  |
| sorting_power | Float64 |  |
| entrainment_factor | Float64 |  |
| Prandtl_number_scale | Float64 |  |
| pi_norm_consts | Vector{Float64} |  |
| updraft_mixing_frac | Float64 |  |
| min_updraft_top | Float64 |  |
| Prandtl_number_0 | Float64 |  |
| pressure_normalmode_buoy_coeff2 | Float64 |  |
| linear_ent_params | Vector{Float64} |  |
| general_stochastic_ent_params | Vector{Float64} |  |
| turbulent_entrainment_factor | Float64 |  |

## Microphysics

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| precipitation_model | Unknown |  |

## Thermodynamics

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| moisture_model | String |  |
| quadrature_order | Int64 |  |
| diagnostic_covar_limiter | Float64 |  |
| sgs | String |  |
| quadrature_type | String |  |
| thermo_covariance_model | String |  |

## Time Stepping

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| t_max | Float64 |  |
| dt_min | Float64 |  |
| adapt_dt | Bool |  |
| dt_max | Float64 |  |
| cfl_limit | Float64 |  |

## Forcing

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| coriolis | Float64 |  |


## Grid

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| nz | Int64 |  |
| dz | Float64 |  |
| dims | Int64 |  |

## Stretch

All parameters below are wrapped in a Dict named stretch

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| nz | Int64 |  |
| dz | Float64 |  |
| flag | Bool |  |
| dz_toa | Float64 |  |
| z_toa | Float64 |  |

## Logging

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| truncate_stack_trace | Bool |  |

## Meta

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| simname | String |  |
| uuid | String |  |
| casename | String |  |

## IO Statistics

| Parameter Name | Type | Description |
| ----------- | ----------- | ----------- |
| calibrate_io | Bool |  |
| stats_dir | String |  |
| frequency | Float64 |  |
| skip | Bool |  |
