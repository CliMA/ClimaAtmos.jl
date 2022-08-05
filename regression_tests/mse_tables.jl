#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 1.8443988202345924e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 6.324573167225068e-9
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.007389776324004594
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.3212692876633791
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 8.318620955383354
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 4.246122341500424e-9
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 8.427328181835132e-7
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0013280117592054644
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.04346560764725502
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 2.9656750499018924e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 36.02785581195702
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 5.593356654367151e-7
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 9.598830843930712e-6
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 8.609941569873989e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.046322046929675134
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.3319686299101685
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 2.5730268885919375e-8
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 1.3992122220669325e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3.116089652282274e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0072260271080074965
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 9.743377452798853e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 4.187428955536203
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 6.277819950700959e-9
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 1.1217787432959677e-7
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0006968820660912892
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.01563623202244828
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 2.92363185514007
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 1.6268100221432348e-8
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 2.2951272392218787e-6
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0.0009117225643443209
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0.013212278564959526
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 2.4745736750594602
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 3.0764983619928447e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 1.562938281562446e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 9.144068376073185e-5
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 0.00022917750743425947
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.015409050252635595
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.002329909360460263
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.002344979843411391
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0017868410256354568
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.008343994820359845
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 1.6691240601771484e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 1.1154746509782044e-11
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 1.1154746507889584e-11
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 9.803431662060278e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 2.0379662046611414e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 6.5951757617115724e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 6.5940185303181666e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 6.2828087684239064e-9
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 6.745827947213308e-9
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 1.901154285150297e-10
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 7.60789932473901e-10
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 7.888968746393095e-10
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 5.046608616099677e-9
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 1.0243384923114229e-7
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 1.8085656444583298e-7
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 1.7203705999508184e-7
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 2.9551896965161287e-7
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 1.7320395096049777e-7
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 0.11981655635627154
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.20543656614452674
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 4.926888764419212
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 8.50032654518012
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 3.282277731315681e-5
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 0.06899554058935968
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.21674773274526996
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 4.4632729650903435
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 10.081131490997622
#
#! format: on
#################################
#################################
#################################
