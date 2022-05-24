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
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe)] = 0.0
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
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρe_int)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_single_column"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_single_column"][(:c, :ρ)] = 0.0
all_best_mse["edmf_single_column"][(:c, :ρθ)] = 0.0
all_best_mse["edmf_single_column"][(:c, :ρθ_liq_ice)] = 0.0
all_best_mse["edmf_single_column"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_single_column"][(:c, :u)] = 0.0
all_best_mse["edmf_single_column"][(:c, :v)] = 0.0
all_best_mse["edmf_single_column"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_single_column"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_single_column"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_single_column"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_single_column"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0
#
#! format: on
#################################
#################################
#################################
