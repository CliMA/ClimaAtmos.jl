#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 1.4872195411869964e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 5.691492642213542e-9
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.006441619348953592
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.31157572259134936
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 7.755543065279825
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 4.501805784676515e-9
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 1.0378174953305407e-6
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.001229646833346352
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.04509844529622248
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 3.513704923386486e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 36.33058108236622
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 7.056394074287637e-7
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 1.2328592627061116e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.00010440967343407822
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.04562470960985503
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.4189863615381206
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 2.1107977959107045e-8
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 9.873491232147725e-7
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3.2179784938613176e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.006691412083092768
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 7.141990719057408e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 4.207743738519047
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 4.919979310209457e-9
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 8.526725426842912e-8
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0005895892885837914
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.014078405102092283
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 2.7221574293952524
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 1.4180717938002013e-8
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 1.922737444328935e-6
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0.000784261929391553
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0.012374839431819338
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 2.4220064683086586
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
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 1.7155812387713146e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 1.2384969282619718e-11
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 1.238496928544619e-11
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 9.748244466190768e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 3.0911963872882134e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 8.170562192650643e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 8.169187333207332e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 7.784512713474964e-9
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 8.201174837724625e-9
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
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 1.182173783922847e-21
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 3.715347519130266e-21
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 6.055813559241208e-21
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 2.1520015318016604e-20
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 1.6390613078045193e-6
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 0.0031630681392908334
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.01389681639739545
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0.05172358125187535
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.04058133439837814
#
#! format: on
#################################
#################################
#################################
