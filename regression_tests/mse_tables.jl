#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 0.0014160167797316022
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 0.0006785155545801942
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 1.0414821005891117
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 19.641956774316224
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 64.06908717701455
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 0.003101683183546888
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 1.5420405610532801
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 133.8827739277012
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 1697.2744391129359
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 60.27545568649035
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 31063.345588122716
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 0.009658099661916242
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 0.0819908906026302
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 58.53216480446473
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 60194.67876088351
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 49362.99084935701
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.00393354078275072
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.07133770822179655
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 147.29198860359278
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 15924.444234337836
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.2319449134153841
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 28794.924027545498
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 0.005089655050263376
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 0.12063509205027237
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 6.385687800824341
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 364.49716835554966
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 1568.5382607340387
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 0.005783913816720229
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 2.038134438731988
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 7.734961880580791
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 785.6911273364601
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 1922.1261575133015
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 1.1950886455990952e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 1.1098972792192258e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.0004706268042856439
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 8.791965003270256e-5
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.010513470751784494
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.11509778467789707
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.11624770589740116
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.07628191185627063
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.1986205964380295
#
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 3.0939494648664425e-5
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 0.01925029787268622
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.0002551077896303604
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.008198524326962603
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 0.12438594020333466
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 1.8495115120746723
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 1.3220243258676938
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 1.3375746741121528
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.8199446863241217
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 1.1880011200295955
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 4.224972879255073e-7
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 7.814120748549269e-8
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 7.81412074823259e-8
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 5.3083776392569455e-6
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0.00044756279215381903
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 8.982040285217178e-5
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 9.002326961542482e-5
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 9.252880233918873e-5
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 2.34153275805621e-5
#
all_best_mse["compressible_edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 2.323280596479405e-8
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 4.7615988844529175e-7
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 3.06186349634114e-6
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 3.0618634963192573e-6
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 6.81794164316145e-6
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0.006162437510461917
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0520571660747721
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.05203262712296469
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.049294430552673685
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 2.062455440574323
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 1.0616394077174069e-6
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0.00017055271856745112
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0.0002621800200454611
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 2.0702034380022222e-5
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0.0012430046371800724
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0017613159686027727
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0020165639135234646
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0008852586309558245
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 0.00385834647179069
#
all_best_mse["compressible_edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 1.719468889452787e-11
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 1.0205757372356625e-9
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 1.9995295051394103e-9
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 5.6033035905900835e-8
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 4.244150244269476e-8
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 8.520020088098778e-6
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 1.6067527399461942e-8
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 3.3243495690452557e-10
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 4.6708271098049195e-6
all_best_mse["compressible_edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 6889.0
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 2.827742925728304e-7
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 4.944300598721989e-7
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 4.081727630121925e-6
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 1.7893976631737016e-5
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 6.007154159742451e-11
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 7.90942061764338e-8
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 9.281809725932665e-7
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 2.3322891140919147e-6
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0015960927930047594
#
#! format: on
#################################
#################################
#################################
