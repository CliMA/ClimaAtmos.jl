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
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 4.021412924756646e-9
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 7.48031738273614e-7
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.001230573835677808
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.04589745047887225
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 2.6066510785552334e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 35.65557153355373
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 2.0942300316958478e-8
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 1.131278772934838e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3.005029752396643e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0068174150497760065
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 8.593085124564406e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 4.04818736970874
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 0
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 2.8522353213044766e-17
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 2.3145652648039978e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 1.1535486942427333e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 6.134943756917211e-5
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 0.00017260698425929555
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.01163008969267557
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0017033883256574393
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.001714865871979064
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0012950776505438436
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0009066996550524682
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 0
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
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 0
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 0
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 0
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0
#
#! format: on
#################################
#################################
#################################
