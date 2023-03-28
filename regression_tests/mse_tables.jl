#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 3.4043366642004188e-6
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 5.894396991980018e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.00045079614526708243
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 1.37862918904251
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 2.5218684801335134
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 3.265381240484281e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.0001668122344695985
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0004618410531524941
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 1.2535698030940425
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.00100004233856698
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 2.254485341883491
#
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 7.522490778392754e-6
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.0004832031017656964
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.00591711469712799
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 1.7586784795261765
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.007451026190067161
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 8.884260028212735
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 9.077581922776559e-6
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 3.652040007955783e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.059769168415572685
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 26.40053515235962
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 81.3865557666874
#
all_best_mse["sphere_held_suarez_rhoe_hightop"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρ)] = 1.5199329083185117e-6
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρe_tot)] = 2.8159175100092787e-5
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 1)] = 0.09542961493622666
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 2)] = 4.591869705400723
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:f, :w, :components, :data, 1)] = 21.80665388722684
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 3.674769821933335e-8
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 1.0095076290487171e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.015585574784049622
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 1.412321387673362
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.0007551456606333955
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 26.335147978308584
#
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 4.8981218623483466e-8
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 9.115644103162541e-6
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.00354605834397669
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0.2899462385348985
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.0005749224337714894
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 31.905656590254917
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρ)] = 0.023985793064178387
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρe_tot)] = 0.7827727183934854
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 1)] = 190795.70312235292
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 2)] = 199088.75383201748
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρq_tot)] = 900.6347276210037
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:f, :w, :components, :data, 1)] = 156282.59880724704
#
all_best_mse["edmf_life_cycle_tan2018"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρ)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_rico"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_rico"][(:c, :ρ)] = 0.0
all_best_mse["edmf_rico"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_rico"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_rico"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_soares"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_soares"][(:c, :ρ)] = 0.0
all_best_mse["edmf_soares"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_soares"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_soares"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_nieuwstadt"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_nieuwstadt"][(:c, :ρ)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["toml_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["toml_edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :ρe_tot)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :ρq_tot)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["toml_edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_bomex_jfnk"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex_jfnk"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_bomex_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk_imex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
#
all_best_mse["edmf_gabls_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
all_best_mse["single_column_nonorographic_gravity_wave"] = OrderedCollections.OrderedDict()
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρ)] = 0.0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρe_tot)] = 0.0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["single_column_nonorographic_gravity_wave"][(:f, :w, :components, :data, 1)] = 0.0
#
#! format: on
#################################
#################################
#################################
