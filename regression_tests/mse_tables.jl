#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 2.71126446364522e-6
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 4.702137094894253e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.00038172205311410673
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 1.0985790838323866
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 2.2989306474185507
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 2.204676997520461e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.00011672346805568363
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0003182735019527393
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.8707740165278731
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0007233597422719805
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 2.0455869204612562
#
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 1.4183909876221031e-5
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.0010635544513791135
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.011523034579315108
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 3.7515302052792854
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.016716272174029653
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 12.741820346848382
#
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0.0038855324275963504
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 3.4549457429228307
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 3128.5639536047897
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.2018948943886324
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 1.419854529955954
all_best_mse["sphere_first_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 10158.91626380804
#
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_zalesak_upwind_tracer_energy_ssp_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 8.051306640663881e-6
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 2.6718939910653286e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.04913649648528998
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 20.971347685871134
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 88.2977523802861
#
all_best_mse["sphere_held_suarez_rhoe_hightop"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρ)] = 3.513364507637638e-6
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρe_tot)] = 6.408325840460303e-5
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 1)] = 0.21371825948333645
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 2)] = 10.168550163210218
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:f, :w, :components, :data, 1)] = 28.985154127099772
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 3.815631699695594e-8
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 1.0548678793606548e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.016932217161509795
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 1.4944306445426079
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.000808613711012811
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 26.360477930531108
#
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 4.6395142511439875e-8
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 8.843651478540541e-6
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0.0038802401760174156
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0.2977155677597123
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0.0005307512427151349
all_best_mse["sphere_ssp_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :w, :components, :data, 1)] = 28.303593039336945
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρ)] = 0.025209484289452196
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρe_tot)] = 0.706985402104515
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 1)] = 172777.88613649554
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :uₕ, :components, :data, 2)] = 230280.26354493856
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:c, :ρq_tot)] = 764.9151977120991
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw"][(:f, :w, :components, :data, 1)] = 158419.1226193996
#
all_best_mse["edmf_life_cycle_tan2018"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρ)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_life_cycle_tan2018"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
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
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
all_best_mse["edmf_rico"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_rico"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_soares"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_soares"][(:c, :ρ)] = 0.0
all_best_mse["edmf_soares"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_soares"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_soares"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
all_best_mse["edmf_soares"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_nieuwstadt"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_nieuwstadt"][(:c, :ρ)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_nieuwstadt"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
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
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
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
all_best_mse["toml_edmf_bomex"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
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
all_best_mse["edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
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
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
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
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρae_tot)] = 0.0
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
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρ)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρe_tot)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["single_column_nonorographic_gravity_wave"][(:f, :w, :components, :data, 1)] = 0
#
#! format: on
#################################
#################################
#################################
