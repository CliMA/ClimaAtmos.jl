#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 1.7563656705661133e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 6.091197487939133e-9
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.006958410778063318
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.31787808095084025
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 6.496064543896283
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 6.205095478743542e-9
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 1.703963169339027e-6
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.004115404517617186
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.20162071701870093
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 6.0817600941395845e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 35.110096863059646
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 1.22867428242943e-6
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 2.090167963455944e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.00015027797457774748
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.0728012216982626
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.0241138901168292
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 1.8947854405608098e-8
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 9.947259651566052e-7
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3.0724984060387124e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0065247388091823075
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 7.0090429411384865e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 3.2362238009269158
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 7.31167152568938e-9
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 1.263078102743982e-7
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0015372633215436809
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.04559199335867039
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 2.6939850729596886
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 1.9446969006265396e-8
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 2.724489803546018e-6
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0.0013374296171284848
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0.03414028945840483
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 2.333700611917761
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 3.4611981459542376e-6
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 1.9268777291865985e-6
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 1.5908592457783354e-5
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 2.5392295932652522e-5
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.0018286549942124362
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.005189808793776857
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.005239739133657465
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0034317309229414186
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.013047610757633032
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 1.0941097708156699e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 7.762699609848757e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 7.762699612667596e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 5.8011452421404e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 5.658126318107902e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 1.1909801504262066e-8
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 1.1906850737497804e-8
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.1171477758409191e-8
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 9.151671056670677e-9
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 2.8427918857957776e-14
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 2.24800015497414e-12
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 6.532737535697031e-13
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 2.687009688145396e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 1.8151137440194669e-10
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 1.909734110983464e-10
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2.2828240824116268e-10
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.6827884795052192e-11
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 1.8760297171005383e-11
#
#! format: on
#################################
#################################
#################################
