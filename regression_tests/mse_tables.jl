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
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 0.448620714626634
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 135.02040663362553
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3351.6820258767175
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 12407.45378693149
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 4204.030410601962
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 10742.198098265013
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 7.258524200136768e-7
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 1.2500089289158106e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.00010751329261232227
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.05642857181709198
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.319297977970253
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.13122869175858676
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 6.3349934596597475
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 347.0636503628998
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 185699.33351966174
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 40.22013051600165
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 250586.01251349467
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 1.1467827272750006
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 26.65882531243973
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 5428.103598315671
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 167390.4658710822
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 546535.0833567405
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
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 3.5770314733967734e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 1.969619536436527e-5
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 9.427961755172214e-5
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 0.00026599536798665633
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.01711767741221038
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0011562315756006149
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0011577528963136136
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0010896237336891917
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0004393743293610541
#
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 3.5770314733967734e-5
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 1.969619536436527e-5
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 9.427961755172214e-5
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 0.00026599536798665633
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.01711767741221038
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0011562315756006149
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0011577528963136136
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0010896237336891917
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0004393743293610541
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 0.0
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 8.084738282518901e-13
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 4.520213107109297e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 4.52021311322686e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 4.5579437032818015e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 1.2858845351725932e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 5.0814901854917914e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 5.080560053692433e-9
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 4.785247996054736e-9
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 4.215761371403777e-9
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 1.6643481250366245e-14
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 2.9097804129750215e-14
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 3.517138754956788e-14
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 9.781454454950363e-13
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 6.047795268818538e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 2.9554056338476193e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2.8291792183947724e-12
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.082407908422984e-11
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 3.2396964485239273e-12
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
#! format: on
#################################
#################################
#################################
