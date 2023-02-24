#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_held_suarez_rhoe_hightop"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρ)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:f, :w, :components, :data, 1)] = 0.0
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
all_best_mse["edmf_arm_sgp"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_arm_sgp"][(:c, :ρ)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_arm_sgp"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_rico"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_rico"][(:c, :ρ)] = 7.673390552151986e-9
all_best_mse["edmf_rico"][(:c, :ρe_tot)] = 1.95270047626101e-6
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 1)] = 1.779028876467344e-7
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 2)] = 2.669318237263653e-9
all_best_mse["edmf_rico"][(:c, :ρq_tot)] = 1.8821972224244147e-5
all_best_mse["edmf_rico"][(:c, :turbconv, :en, :ρatke)] = 0.0007929094880510628
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0011182056484553694
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0011185396749156913
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.001121559121207032
all_best_mse["edmf_rico"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.00040685999089119893
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
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["compressible_edmf_bomex_jfnk"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρ)] = 8.662649905412733e-9
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρe_tot)] = 1.7949443734587964e-5
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 1)] = 4.409950602039042e-6
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 2)] = 0.0005959104656280111
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρq_tot)] = 9.653984469133201e-5
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :en, :ρatke)] = 0.10267881189462741
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρarea)] = 0.09716812968991771
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.09712380635196455
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.09657833146375092
all_best_mse["compressible_edmf_bomex_jfnk"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.019642239467530097
#
all_best_mse["compressible_edmf_bomex_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρ)] = 8.911188830481073e-9
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρe_tot)] = 7.758331675076136e-6
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 2.839431678890034e-5
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0.009527194600684725
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρq_tot)] = 4.585789253968857e-5
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0.013284076134185372
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.012589676000028317
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.012586668842845873
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.012370163362393365
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.1561799245619159
#
all_best_mse["compressible_edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 3.2313164654232975e-30
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 6.212210206799507e-27
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 1.446288468437206e-22
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 3.0795944156526275e-22
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 6.120382984063495e-26
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 8.260890731170429e-22
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 9.982213117019765e-22
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 1.2154550877445844e-21
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 3.2944781617792076e-23
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 2.3508812057631155e-22
#
all_best_mse["edmf_trmm_0_moment"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm_0_moment"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["compressible_edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
#
all_best_mse["compressible_edmf_gabls_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :ρ)] = 9.20983119818352e-17
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :ρe_tot)] = 2.3422956509508993e-14
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 2.5303976081194453e-14
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 5.600899886480019e-14
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 2.0877699681644594e-12
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
