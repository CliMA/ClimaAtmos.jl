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
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 0.08604733991036712
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.02967001512038287
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 1.434962931134244
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 0.6304610095096791
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 12.984712550246906
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 753.3587034246002
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 810.8206589258766
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 31.570140895377566
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 3.5802551787979917
#
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 0.0002026753628882803
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 0.1131198407474154
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.020835657130529338
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.70367190924349
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 0.7384877974987911
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 16.759962512886368
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 2536.230442271806
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2704.9051667236786
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 134.69105319774428
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 5.549590353054816
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0.00789091004097318
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0.04133709159069168
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0.04133709159069437
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0.06558468877834182
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 15.514552092764653
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 2.3304045071402086
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2.322952752765473
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 2.271724593388835
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 1.0185066762484298
#
all_best_mse["compressible_edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 6.304055514330139e-6
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0.008855477752800777
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0.03505443673234349
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0.035054436732343326
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0.07883251951403353
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 13.084771212250521
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 1.537930489971088
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 1.5300869871217677
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.6006575966880232
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 2.9125097592768823
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 0.0061532662009211704
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0.18404149039455936
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0.29826167084159055
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 0.13444778062693344
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 22.99193893968716
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 8.746614420953167
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 10.061883515571617
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 4.3048564953425315
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 1.0589854959767202
#
all_best_mse["compressible_edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 2.0466637614081424e-10
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 2.5610983886792655e-8
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 2.8785899024735278e-9
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 8.082293700419961e-8
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 8.934360414606153e-7
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0.000329059018658868
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0023049503336083645
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.002581434161142903
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0008195558517435869
all_best_mse["compressible_edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 1.696341117812424e17
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 2.8530113678996726e-7
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 4.976156833407761e-7
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 4.1924081225019216e-6
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 1.7936127727491848e-5
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 6.014730080897641e-11
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 7.915674415979225e-8
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 9.23478177124616e-7
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 2.321197675522893e-6
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0015930307229294965
#
#! format: on
#################################
#################################
#################################
