#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_hightop"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :ρ)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :ρ)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["single_column_nonorographic_gravity_wave"] = OrderedCollections.OrderedDict()
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρ)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρe_tot)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:f, :u₃, :components, :data, 1)] = 0
#
#! format: on
#################################
#################################
#################################
