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
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_hightop"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρ)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_life_cycle_tan2018"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρ)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_rico"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_rico"][(:c, :ρ)] = 0
all_best_mse["edmf_rico"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_rico"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_rico"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_rico"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_soares"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_soares"][(:c, :ρ)] = 0
all_best_mse["edmf_soares"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_soares"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_soares"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["edmf_soares"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_nieuwstadt"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_nieuwstadt"][(:c, :ρ)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["edmf_nieuwstadt"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["toml_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["toml_edmf_bomex"][(:c, :ρ)] = 0
all_best_mse["toml_edmf_bomex"][(:c, :ρe_tot)] = 0
all_best_mse["toml_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["toml_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["toml_edmf_bomex"][(:c, :ρq_tot)] = 0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["toml_edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_bomex_jfnk"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex_jfnk"][(:c, :ρ)] = 0
all_best_mse["edmf_bomex_jfnk"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_bomex_jfnk"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_bomex_jfnk"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
#
all_best_mse["edmf_gabls_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :ρ)] = 0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0
#
all_best_mse["single_column_nonorographic_gravity_wave"] = OrderedCollections.OrderedDict()
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρ)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρe_tot)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:f, :w, :components, :data, 1)] = 0
#
#! format: on
#################################
#################################
#################################
