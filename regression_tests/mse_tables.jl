#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
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
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_first_upwind_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_first_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.0017821204694714434
all_best_mse["sphere_first_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.6871822419689623
all_best_mse["sphere_first_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 682.3936137574441
all_best_mse["sphere_first_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.0942483557437482
all_best_mse["sphere_first_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 1939.7056539585942
#
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.004751630169357323
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3.474857749714428
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 3728.3114349104862
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.21969390354691137
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 1.4657019655263304
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 12260.1309595868
#
all_best_mse["sphere_third_upwind_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_third_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 1.6421197970397946e-5
all_best_mse["sphere_third_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.013102850528390374
all_best_mse["sphere_third_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 2.9703276232775884
all_best_mse["sphere_third_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.0005854251936381209
all_best_mse["sphere_third_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.012153779050851618
all_best_mse["sphere_third_upwind_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 22.3482917708477
#
all_best_mse["sphere_third_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_third_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 2.5231351363644347e-6
all_best_mse["sphere_third_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0009066941701081075
all_best_mse["sphere_third_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.7178068730479484
all_best_mse["sphere_third_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.00016028752777675817
all_best_mse["sphere_third_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.001997646500010075
all_best_mse["sphere_third_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 17.880742574157928
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_held_suarez_rhoe_hightop"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0.0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["sphere_ssp_first_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_first_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0.0017399740735526
all_best_mse["sphere_ssp_first_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 4.738843627298765
all_best_mse["sphere_ssp_first_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 672.5354606318706
all_best_mse["sphere_ssp_first_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0.06322622254618894
all_best_mse["sphere_ssp_first_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 10.223547747059369
all_best_mse["sphere_ssp_first_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 2187.750822965008
#
all_best_mse["sphere_ssp_first_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_first_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0.0008152518744962339
all_best_mse["sphere_ssp_first_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 6.220607745015805
all_best_mse["sphere_ssp_first_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 880.2288257032595
all_best_mse["sphere_ssp_first_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0.15678292995763946
all_best_mse["sphere_ssp_first_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 9.05627827337842
all_best_mse["sphere_ssp_first_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 4336.522554389444
#
all_best_mse["sphere_ssp_third_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_third_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 7.211023528316708e-6
all_best_mse["sphere_ssp_third_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.04196335488295072
all_best_mse["sphere_ssp_third_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 3.6985044015725963
all_best_mse["sphere_ssp_third_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0.0019114304860219373
all_best_mse["sphere_ssp_third_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.13660557200616227
all_best_mse["sphere_ssp_third_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 34.80700215711332
#
all_best_mse["sphere_ssp_third_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_third_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 1.1559368820244655e-5
all_best_mse["sphere_ssp_third_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.037665637142400094
all_best_mse["sphere_ssp_third_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 5.680105869463573
all_best_mse["sphere_ssp_third_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0.0014808408204548817
all_best_mse["sphere_ssp_third_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.08241932024252487
all_best_mse["sphere_ssp_third_tracer_energy_upwind_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 34.71777398349536
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρ)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρe_tot)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρq_tot)] = 0.0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_life_cycle_tan2018"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρ)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_rico"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_rico"][(:c, :ρ)] = 0.0
all_best_mse["edmf_rico"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_rico"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_rico"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_rico"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_soares"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_soares"][(:c, :ρ)] = 0.0
all_best_mse["edmf_soares"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_soares"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_soares"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_nieuwstadt"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_nieuwstadt"][(:c, :ρ)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["toml_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["toml_edmf_bomex"][(:c, :ρ)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :ρe_tot)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :ρq_tot)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["toml_edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_bomex_jfnk"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex_jfnk"][(:c, :ρ)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_bomex_jfnk"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
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
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
#
all_best_mse["edmf_gabls_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :ρ)] = 0.0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_gabls_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0.0
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
