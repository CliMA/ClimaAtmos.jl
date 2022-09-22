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
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0.0
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
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 6.1005941889145366e-9
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 3.6147449288250086e-6
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 5.209012942066034e-6
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.0008870059639948838
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 2.2979869717303335e-5
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.009646161773108276
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.008790268219271225
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.008798067005380716
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.008362797239014091
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.015939017060384524
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
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 1.0944896365999254e-8
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 2.6690957663835314e-7
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 1.202198245020051e-7
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 1.2021982450238426e-7
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 2.4204093395001364e-6
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0.0008117430893245465
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0009020955562455605
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0009021278767540738
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0009081318907653152
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0006499759652456001
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
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 1.690660184952545e-11
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 1.0160571925648023e-9
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 1.986933787122021e-9
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 5.5999522186158147e-8
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 4.247534848203897e-8
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 8.203205111308456e-6
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 1.5869379574132405e-8
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 5.169446771056764e-10
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 4.552554312409156e-6
all_best_mse["compressible_edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 0.004253906713943052
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 2.827742925728304e-7
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 4.944300598721989e-7
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 4.081727630121925e-6
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 1.7893976631737016e-5
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 8.594108353858545e-11
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 1.1232940371213663e-7
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 9.786880624905995e-7
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 2.4719912917168085e-6
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 1.3695372955163653e-5
#
#! format: on
#################################
#################################
#################################
