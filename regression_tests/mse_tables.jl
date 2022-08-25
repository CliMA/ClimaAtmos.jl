#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 1.6193573169208953e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 5.794432690544718e-9
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.0052961737378325715
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.2856815462764704
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 6.957987065930084
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 4.329258333233513e-9
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 9.352547534007653e-7
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0014573739605400694
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.046279574860332344
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 3.2809699936140647e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 37.47881884168757
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 1.0540531930506877e-6
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 1.6479850411795196e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0006089219530418976
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 1.4327093799483166
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 3.1477667218306444
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 1.008574490686853e-7
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 3.447703103549545e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.005183137972018408
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.7771387614169974
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 2.4719041998861164e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 6.3519623066828546
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 6.639342599624274e-9
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 1.118138182095686e-7
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0006746573435171093
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.014597642126239123
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 3.188673657097964
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 1.7198435477405778e-8
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 2.6354668436730466e-6
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0.0009629275043504053
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0.015243579327998148
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 3.0699616990859373
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
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 4.481309456215971e-14
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 4.92352554029087e-13
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 4.923525527738833e-13
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 2.4494333881963007e-13
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 3.502394014764265e-10
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 3.0349461742046674e-10
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 3.034140805806489e-10
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 2.9287879929668245e-10
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 3.3824368613000977e-10
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 4.128691267630159e-15
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 1.8851082932363716e-14
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 7.20895479279743e-14
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 7.883295084687389e-14
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 2.816671570667443e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 3.3660553414560265e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 8.337273447946508e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 9.283206947395169e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.6864846874739997e-12
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 2.4745516465347836e-13
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 1.8711190181319758e-16
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 4.4172261486579754e-13
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 2.7566317385055272e-12
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 1.2284531527859584e-11
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 4.0037094639860584e-12
#
#! format: on
#################################
#################################
#################################
