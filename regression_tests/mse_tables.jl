#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 2.3415174617804604e-8
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 3.687896531905108e-6
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.015135792391570714
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.5233367084407022
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.00012700009407390757
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 18.118096015737162
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
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρ)] = 1.9240276339636248e-5
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρe_tot)] = 0.001545664685093744
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :uₕ, :components, :data, 1)] = 0.15381717531231345
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :uₕ, :components, :data, 2)] = 7.481308056958331
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρq_tot)] = 0.029533897165121902
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:f, :w, :components, :data, 1)] = 44.88205900600961
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
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0
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
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0
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
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0
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
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0
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
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0
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
all_best_mse["compressible_edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 0.0
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
#! format: on
#################################
#################################
#################################
