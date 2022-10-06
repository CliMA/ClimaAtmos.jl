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
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρ)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 5.82391618280529e-8
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 1.136834407446956e-7
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 1.242808088963971e-6
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 4.058279973630046e-7
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 7.479913635873812e-5
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.005372997310609317
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.005571065733130005
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0006027388204744691
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :w)] = 1.1504772893360285e-6
#
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :w)] = 0.0
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w)] = 0.0
#
all_best_mse["compressible_edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w)] = 0.0
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 0.01457726702084752
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0.03134918071589366
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0.049933122706592406
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 0.2256442434454288
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 20.11475415048433
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 5.074030999989727
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 5.090923782392945
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 7.342699763888589
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :w)] = 3.4991218802696444
#
all_best_mse["compressible_edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["compressible_edmf_trmm"][(:f, :turbconv, :up, 1, :w)] = 0.0
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 0.0
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
all_best_mse["single_column_nonorographic_gravity_wave"] = OrderedCollections.OrderedDict()
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρ)] = 0.0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρe_tot)] = 0.0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["single_column_nonorographic_gravity_wave"][(:f, :w, :components, :data, 1)] = 0.0
#
#! format: on
#################################
#################################
#################################
