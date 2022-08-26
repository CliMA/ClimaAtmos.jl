#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 1.9912442018013036e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 6.7498895073101735e-9
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.007869308823135057
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.3454671642133014
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 7.103516664934748
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 7.214882653823208e-9
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 1.984117830262579e-6
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.004193123325482877
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.19047845598925991
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 7.133806741225566e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 36.15274388089732
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 8.08338211796272e-7
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 1.3634467277126793e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.00011401072130306769
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.057496700599122846
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.0052814783016264
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 3.486425692160363e-8
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 2.091911219585347e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3.237833160953532e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.00789616794383664
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 1.4807089878932122e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 3.2437879768204434
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 6.735514443291634e-9
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 1.184739199222293e-7
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0014869224372780774
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.045901610603803426
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 2.71531501466558
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 2.741175432907876e-8
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 3.92079340056383e-6
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0.0019351430726498643
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0.035257178132433904
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 2.4880569885840105
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 2.0250012071270722e-6
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 9.762032352964688e-7
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 1.2749351788665655e-5
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 1.5013841065860593e-5
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.001104231759637265
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 9.059993287810685e-5
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 9.07787784069867e-5
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 8.409793779766647e-5
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0003781571556749615
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 2.9347839743405894e-13
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 3.378372565527843e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 3.3783725651283225e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 1.458647776167507e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 6.439917813252555e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 3.6905097081599082e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 3.6889268637962293e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 3.53750713739801e-9
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 3.5713032927108936e-9
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 2.8426891030143104e-14
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 2.248026302133111e-12
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 6.533014964268834e-13
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 2.6867757868689337e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 1.815072561577868e-10
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 1.9096442644786527e-10
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2.2827141721561552e-10
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.682858823202261e-11
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 1.8759830619198362e-11
#
#! format: on
#################################
#################################
#################################
