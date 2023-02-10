#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0
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
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρ)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_life_cycle_tan2018"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρ)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_life_cycle_tan2018"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_arm_sgp"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_arm_sgp"][(:c, :ρ)] = 0
all_best_mse["edmf_arm_sgp"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_arm_sgp"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_arm_sgp"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_arm_sgp"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_rico"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_rico"][(:c, :ρ)] = 0
all_best_mse["edmf_rico"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_rico"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_rico"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_rico"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_soares"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_soares"][(:c, :ρ)] = 0
all_best_mse["edmf_soares"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_soares"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_soares"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_soares"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_soares_const_entr"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_soares_const_entr"][(:c, :ρ)] = 0
all_best_mse["edmf_soares_const_entr"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_soares_const_entr"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_soares_const_entr"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_soares_const_entr"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_soares_const_entr"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_soares_const_entr"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_nieuwstadt"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_nieuwstadt"][(:c, :ρ)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_nieuwstadt"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 0
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 0
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_bomex_const_entr"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex_const_entr"][(:c, :ρ)] = 0
all_best_mse["edmf_bomex_const_entr"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_bomex_const_entr"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_bomex_const_entr"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_bomex_const_entr"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_bomex_const_entr"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_bomex_const_entr"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_bomex_const_entr"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_bomex_const_entr"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_bomex_const_entr"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["compressible_edmf_bomex_jfnk"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρ)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρe_tot)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρq_tot)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["compressible_edmf_bomex_jfnk"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["compressible_edmf_bomex_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρ)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρe_tot)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρq_tot)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["compressible_edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_trmm_0_moment"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm_0_moment"][(:c, :ρ)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :q_rai)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :q_sno)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_trmm_0_moment"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0
#
all_best_mse["compressible_edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 0
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 0
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 0
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 0
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0
#
all_best_mse["compressible_edmf_gabls_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :ρ)] = 0
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :ρe_tot)] = 0
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0
#
all_best_mse["single_column_nonorographic_gravity_wave"] = OrderedCollections.OrderedDict()
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρ)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρe_tot)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_baroclinic_wave_ogw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_ogw"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_ogw"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_ogw"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_ogw"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_ogw"][(:f, :w, :components, :data, 1)] = 0
#
#! format: on
#################################
#################################
#################################
