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
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρ)] = 5.4569129108809057e-11
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρe_tot)] = 4.644555091836496e-8
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 1)] = 1.2245092710613188e-8
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 2)] = 2.136280823329404e-6
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρq_tot)] = 2.9040087038375996e-7
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :en, :ρatke)] = 5.04698737958936e-5
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0005014992408003643
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0005015191004720635
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0004744463063085754
all_best_mse["edmf_life_cycle_tan2018"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 4.629945935044364e-5
#
all_best_mse["edmf_arm_sgp"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_arm_sgp"][(:c, :ρ)] = 1.7667389388369508e-10
all_best_mse["edmf_arm_sgp"][(:c, :ρe_tot)] = 2.8633336357067905e-7
all_best_mse["edmf_arm_sgp"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :ρq_tot)] = 2.0820978778036354e-6
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :en, :ρatke)] = 0.0007127730723281035
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρarea)] = 2.172381180654625
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2.172394945940357
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 2.1795121420077126
all_best_mse["edmf_arm_sgp"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.7054547400076115
#
all_best_mse["edmf_rico"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_rico"][(:c, :ρ)] = 6.188300006494352e-9
all_best_mse["edmf_rico"][(:c, :ρe_tot)] = 1.753120538632706e-6
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 1)] = 6.336920150450744e-6
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 2)] = 3.320547532408796e-6
all_best_mse["edmf_rico"][(:c, :ρq_tot)] = 1.6248826482007914e-5
all_best_mse["edmf_rico"][(:c, :turbconv, :en, :ρatke)] = 0.0006831275579923617
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0031539216294679913
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.003146820633655658
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.003301525154849918
all_best_mse["edmf_rico"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0009256451299803385
#
all_best_mse["edmf_soares"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_soares"][(:c, :ρ)] = 2.0868197415862133e-10
all_best_mse["edmf_soares"][(:c, :ρe_tot)] = 1.3221402666584838e-6
all_best_mse["edmf_soares"][(:c, :uₕ, :components, :data, 1)] = 8.354280791850703e-6
all_best_mse["edmf_soares"][(:c, :turbconv, :en, :ρatke)] = 0.0003711454808443144
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρarea)] = 0.028427771946245996
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.028396494585465032
all_best_mse["edmf_soares"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 3.157882714016015e-5
#
all_best_mse["edmf_nieuwstadt"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_nieuwstadt"][(:c, :ρ)] = 1.764155465556637e-9
all_best_mse["edmf_nieuwstadt"][(:c, :ρe_tot)] = 4.506464129833698e-7
all_best_mse["edmf_nieuwstadt"][(:c, :uₕ, :components, :data, 1)] = 0.00033483353820388785
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :en, :ρatke)] = 0.005614590744314549
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρarea)] = 71.82190478734476
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 71.76997427158464
all_best_mse["edmf_nieuwstadt"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0068077842291032215
#
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 1.705722338795752e-7
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 0.00015977542964798084
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 9.016695981504411e-5
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.01941486354916147
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 0.0009574436145768153
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.2560292093450055
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 1.6984949595036163
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 1.6973961883945339
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.7157037355862839
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.1934406508381025
#
all_best_mse["compressible_edmf_bomex_jfnk"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρ)] = 3.522173700154228e-8
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρe_tot)] = 3.6548141357079826e-5
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 1)] = 4.74394889069334e-6
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 2)] = 0.0013779738287187384
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρq_tot)] = 0.00021308901536764533
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :en, :ρatke)] = 0.0779234671944853
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρarea)] = 0.12582495476545907
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.12574937465739242
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.12542982292463864
all_best_mse["compressible_edmf_bomex_jfnk"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.018733230752450758
#
all_best_mse["compressible_edmf_bomex_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρ)] = 2.574138043400991e-8
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρe_tot)] = 1.8214997776285834e-5
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 6.892296210847749e-6
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0.0019142255488579871
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρq_tot)] = 0.00011298671850805217
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0.02754210204779927
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.12010110439966713
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.12003848362440463
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.120572760821411
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.2877233378941673
#
all_best_mse["compressible_edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 8.783104191928554e-13
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 1.3352538098573193e-9
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 4.52284420675064e-11
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 4.5228442071544274e-11
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 5.847864470104757e-9
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 7.136939703058327e-8
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 5.4773700121846584e-8
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 5.482579622813564e-8
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 5.139191439495801e-8
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 2.898291975274525e-8
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 2.8299106933210245e-30
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 2.9169970728742227e-27
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 2.1972976610587575e-22
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 4.557816359772174e-22
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 3.009820398221793e-26
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 2.2890441528062966e-22
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 3.012908735627665e-20
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 3.689688904990561e-20
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.108457828919043e-23
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 4.5048608748757106e-23
#
all_best_mse["edmf_trmm_0_moment"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm_0_moment"][(:c, :ρ)] = 4.480427509756449e-26
all_best_mse["edmf_trmm_0_moment"][(:c, :ρe_tot)] = 9.438350292455586e-22
all_best_mse["edmf_trmm_0_moment"][(:c, :uₕ, :components, :data, 1)] = 5.5704599233025216e-21
all_best_mse["edmf_trmm_0_moment"][(:c, :uₕ, :components, :data, 2)] = 2.980260955473406e-22
all_best_mse["edmf_trmm_0_moment"][(:c, :ρq_tot)] = 3.8086008803538145e-21
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :en, :ρatke)] = 1.855312665197817e-22
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρarea)] = 6.425182532647446e-22
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 6.396242958817969e-22
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 6.294855589636613e-22
all_best_mse["edmf_trmm_0_moment"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 5.119994854754754e-22
#
all_best_mse["compressible_edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 1.1928398820685681e-25
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 5.653033627679969e-25
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 3.822817041831643e-26
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 9.993102877824235e-26
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 1.3687489418983383e-24
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 9.216041377710569e-24
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 7.691429324639113e-24
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 7.639823843740162e-25
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 9.55903012547432e-24
#
all_best_mse["compressible_edmf_gabls_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :ρ)] = 1.1486225321584154e-16
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :ρe_tot)] = 5.3836461423369083e-14
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 3.8499403386434634e-14
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 7.744865298483949e-14
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 3.2264407932420675e-12
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
