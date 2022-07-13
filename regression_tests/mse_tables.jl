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
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 0.4486210800499763
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 135.01929865825923
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3351.649013763883
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 12406.312347614145
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 4204.00137686495
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 10740.274918718373
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 9.286674560192182e-7
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 1.6119658983356977e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.00013289567946652065
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.06444476656133503
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.3248732589464254
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.1312268835792963
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 6.335117972467446
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 347.06437018282975
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 185697.17625458966
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 40.22136313735683
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 250571.14757836433
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 1.1467816461106903
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 26.65879192238779
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 5428.069800247032
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 167390.9313022588
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 546519.1246310811
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 1.340853747189132
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 212.60977658157142
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 6095.615212968916
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 322425.5696469427
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 673644.5019367539
#
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρ)] = 0.830777719386451
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρe_int)] = 700.0620362328294
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 4949.722717032724
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 27186.538241856208
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:c, :ρq_tot)] = 5078.586284931179
all_best_mse["sphere_held_suarez_rhoe_int_equilmoist"][(:f, :w, :components, :data, 1)] = 26722.22974591826
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 1.197866291290118e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 7.212557314398987e-6
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 5.939476233400226e-5
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 8.935908377309965e-5
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.005358185626437062
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0004415434494837931
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0004422494764266972
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.00041116483626962077
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0003549339565099078
#
all_best_mse["edmf_bomex_compressible"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex_compressible"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_bomex_compressible"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 5.801101557432303e-16
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 2.4295260790609472e-15
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 2.429526168179799e-15
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 4.44733544445686e-15
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 1.0394145433848629e-11
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 4.834510527190875e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 4.8314248378157e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 4.332832900883245e-12
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 1.537971022081113e-12
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 1.2701589520110754e-15
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 6.181260833950158e-15
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 6.406827113481417e-15
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 4.029956463517367e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 4.600678935252865e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 4.149197870693629e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 4.392701740923239e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 2.574262639832351e-11
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 4.4605832765016905e-12
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
all_best_mse["edmf_gabls_compressible"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls_compressible"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls_compressible"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_gabls_compressible"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_gabls_compressible"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_gabls_compressible"][(:c, :turbconv, :en, :ρatke)] = 0.0
#
#! format: on
#################################
#################################
#################################
