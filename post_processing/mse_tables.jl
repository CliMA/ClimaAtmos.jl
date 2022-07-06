#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 97.46992071870785
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 74.83347475265151
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 16676.24524181918
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 1.1389683068904928e6
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 1.6885841245217498e6
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 120.22894081358288
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 2251.7928553848787
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 28369.03682316624
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 192422.61276535122
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 2458.450386578533
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 34126.95854423084
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 2.0188949212765943e-6
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 1.5471093351926498e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.00013592439661370078
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.06684697154929595
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.4005280132321167
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 2.351711789911278e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 5.96720216899784e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0001698440766726454
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.09272082481899332
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0003631561661509332
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 2.4338192630325604
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 108.37474839525592
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 2176.3585770756927
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 23911.622533988502
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 1.5735326621095007e6
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 2.690914723268757e6
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 109.20686832383264
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 41600.86903454714
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 26059.76439335406
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 1.8331848600099196e6
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 7.612222515922438e6
#
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρ)] = 118.35622830360559
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρe_int)] = 26886.034863602963
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 20725.795116556255
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 138732.35981088184
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρq_tot)] = 1498.8102656510948
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:f, :w, :components, :data, 1)] = 31540.544154951614
#
all_best_mse["edmf_single_column"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_single_column"][(:c, :ρ)] = 9.879929344257052e-5
all_best_mse["edmf_single_column"][(:c, :ρe_tot)] = 0.04689738654710128
all_best_mse["edmf_single_column"][(:c, :uₕ, :components, :data, 1)] = 15.529738096530483
all_best_mse["edmf_single_column"][(:c, :uₕ, :components, :data, 2)] = 70.43751321872755
all_best_mse["edmf_single_column"][(:c, :ρq_tot)] = 0.3399935206903438
all_best_mse["edmf_single_column"][(:c, :turbconv, :en, :ρatke)] = 30.105292925928545
all_best_mse["edmf_single_column"][(:c, :turbconv, :up, 1, :ρarea)] = 3.4347079829677534
all_best_mse["edmf_single_column"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 3.444553703302168
all_best_mse["edmf_single_column"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 3.1517880563572938
all_best_mse["edmf_single_column"][(:f, :turbconv, :up, 1, :ρaw)] = 1.9603603016726063
#
#! format: on
#################################
#################################
#################################
