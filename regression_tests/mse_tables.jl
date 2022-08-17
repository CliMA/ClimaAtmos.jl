#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 1.6809275386800948e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 6.262375765923583e-9
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.005848683042811358
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.33720431316147
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 7.58846246682216
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 3.791803395333096e-9
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 7.129459043178872e-7
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0012773046132519982
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.04058715617543107
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 2.5524901161530087e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 34.39483489419334
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 0.0
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
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 4.6643303671829996e-9
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 7.986314717737867e-8
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0005123580120576744
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.014815985887276211
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 2.9196460681211165
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 1.6571910948917722e-8
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 2.320213200956851e-6
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0.0009642264833738182
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0.014393560048110174
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 2.5413488484983864
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
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 1.885109594812877e-14
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 7.209727706937545e-14
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 7.883385322388729e-14
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 2.816674211914342e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 3.3661227756901374e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 8.333829223739702e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 9.279063933589412e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.6864875679316793e-12
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 2.4745277425294757e-13
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 2.31669141627373e-20
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 5.530777000234495e-17
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 3.083858688471818e-16
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 1.3870892916453092e-15
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 1.1684973737743386e-8
#
#! format: on
#################################
#################################
#################################
