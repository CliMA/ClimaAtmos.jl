#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 0.0019266493785521678
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 0.00016743740428730962
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 3.0358815541257536
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 17.234748885146566
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 103.63215202906565
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 8.187244818941813e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.009630646066931056
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 1.8880586513131714
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 10.81127277582394
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.14761337100532068
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 161.42458715446583
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 6.989740454693192e-6
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 0.00011927219477160017
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0009890173961931558
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 2.281495962502547
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.480002099636575
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 4.867205045344034e-6
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0.00029792554640804846
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.0007991530043693412
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 1.6437665430342274
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0020796455954267914
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 1.5705576017671041
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 0.0016670306466678626
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 0.03421067802321651
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 10.557469584668189
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 429.88577210203545
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 3894.39998563097
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρ)] = 7.611663332239786e-7
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρe_tot)] = 0.00013889025136598846
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :uₕ, :components, :data, 1)] = 0.0084872593734733
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :uₕ, :components, :data, 2)] = 0.08216427577298152
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:c, :ρq_tot)] = 0.00556934170456881
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky"][(:f, :w, :components, :data, 1)] = 0.6635334317894421
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
all_best_mse["edmf_arm_sgp"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_arm_sgp"][(:c, :ρ)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_arm_sgp"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_arm_sgp"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
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
all_best_mse["edmf_soares_const_entr"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_soares_const_entr"][(:c, :ρ)] = 0.0
all_best_mse["edmf_soares_const_entr"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_soares_const_entr"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_soares_const_entr"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_soares_const_entr"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_soares_const_entr"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_soares_const_entr"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
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
all_best_mse["edmf_nieuwstadt_anelastic"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_nieuwstadt_anelastic"][(:c, :ρ)] = 0.0
all_best_mse["edmf_nieuwstadt_anelastic"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_nieuwstadt_anelastic"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_nieuwstadt_anelastic"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_nieuwstadt_anelastic"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_nieuwstadt_anelastic"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_nieuwstadt_anelastic"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 8.389155952895876e-11
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 1.3215166486636658e-7
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 1.9508592340011728e-7
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 6.935369921125629e-5
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 7.181996448278168e-7
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.0009648971585796823
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.002938854545817133
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.002936122523805853
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.00299594683816082
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.00021285399362626982
#
all_best_mse["edmf_bomex_const_entr"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex_const_entr"][(:c, :ρ)] = 7.303420980299365e-6
all_best_mse["edmf_bomex_const_entr"][(:c, :ρe_tot)] = 0.017714731750917014
all_best_mse["edmf_bomex_const_entr"][(:c, :uₕ, :components, :data, 1)] = 0.0029452459430747723
all_best_mse["edmf_bomex_const_entr"][(:c, :uₕ, :components, :data, 2)] = 0.24201832110677093
all_best_mse["edmf_bomex_const_entr"][(:c, :ρq_tot)] = 0.09825123306277952
all_best_mse["edmf_bomex_const_entr"][(:c, :turbconv, :en, :ρatke)] = 2.941559112317659
all_best_mse["edmf_bomex_const_entr"][(:c, :turbconv, :up, 1, :ρarea)] = 80.64487556401033
all_best_mse["edmf_bomex_const_entr"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 80.48573838960421
all_best_mse["edmf_bomex_const_entr"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 86.17444295485588
all_best_mse["edmf_bomex_const_entr"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 19.000923893508663
#
all_best_mse["compressible_edmf_bomex_jfnk"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρ)] = 1.0565822880981106e-8
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρe_tot)] = 1.2539284657062895e-5
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 1)] = 2.2318036711329667e-5
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :uₕ, :components, :data, 2)] = 0.005143971566462438
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :ρq_tot)] = 7.211411043509399e-5
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :en, :ρatke)] = 0.06163311939936017
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρarea)] = 0.19451106105970362
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.19443444638313676
all_best_mse["compressible_edmf_bomex_jfnk"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.19347206750911528
all_best_mse["compressible_edmf_bomex_jfnk"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.036246730291864065
#
all_best_mse["compressible_edmf_bomex_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρ)] = 1.8807663077870667e-7
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρe_tot)] = 0.00010078827530134102
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 5.797191695938595e-5
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 0.01914822886533277
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :ρq_tot)] = 0.0006282965350282932
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 0.17952643885522734
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.3227099280154998
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.32227991324621813
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.33710148218016905
all_best_mse["compressible_edmf_bomex_jfnk_imex"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.1930135752556164
#
all_best_mse["compressible_edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 4.94943767046459e-19
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 3.260700738903667e-16
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 2.362907406294391e-16
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 2.36290646082065e-16
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 2.83458049384649e-15
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 4.5035230259845327e-14
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 1.3851972297733464e-14
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 1.3814680033377236e-14
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.3827665651723259e-14
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 2.462238937138891e-14
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
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["edmf_trmm_0_moment"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm_0_moment"][(:c, :ρ)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :ρe_tot)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :q_rai)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :q_sno)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :ρq_tot)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :en, :ρatke)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.0
all_best_mse["edmf_trmm_0_moment"][(:f, :turbconv, :up, 1, :w, :components, :data, 1)] = 0.0
#
all_best_mse["compressible_edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 4.13744371961237e-19
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 1.910202298637745e-18
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 4.139971330010298e-17
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 7.537165709576993e-17
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 5.879206765848933e-19
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 1.712274832067366e-17
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 2.027386299463601e-17
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2.03414894481046e-17
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 1.8880493457958592e-17
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 1.5943309076909205e-21
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 4.122049101063128e-18
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 3.7419318282042987e-16
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 1.6996863652272792e-14
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 6.140241980149347e-16
#
all_best_mse["compressible_edmf_gabls_jfnk_imex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :ρ)] = 1.8569966417804604e-16
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :ρe_tot)] = 7.950006149168756e-14
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 1)] = 1.1596802080291102e-13
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :uₕ, :components, :data, 2)] = 6.349244070022194e-13
all_best_mse["compressible_edmf_gabls_jfnk_imex"][(:c, :turbconv, :en, :ρatke)] = 1.002671719656322e-12
#
all_best_mse["single_column_nonorographic_gravity_wave"] = OrderedCollections.OrderedDict()
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρ)] = 1.1177356006213208e-13
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :ρe_tot)] = 1.6381561682156425e-12
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 1)] = 4.859812643333078e-11
all_best_mse["single_column_nonorographic_gravity_wave"][(:c, :uₕ, :components, :data, 2)] = 4.859812643333078e-11
all_best_mse["single_column_nonorographic_gravity_wave"][(:f, :w, :components, :data, 1)] = 0.0378564465920737
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :ρ)] = 3.1563996886342856e-8
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :ρe_tot)] = 5.466235873485249e-7
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :uₕ, :components, :data, 1)] = 0.08778536800884275
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :uₕ, :components, :data, 2)] = 1.2565345517485456
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:c, :ρq_tot)] = 0.0006571528700382288
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_nogw"][(:f, :w, :components, :data, 1)] = 5.735401696560568
#
all_best_mse["sphere_baroclinic_wave_ogw"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_ogw"][(:c, :ρ)] = 0.19349899352688188
all_best_mse["sphere_baroclinic_wave_ogw"][(:c, :ρe_tot)] = 2.652587639547247
all_best_mse["sphere_baroclinic_wave_ogw"][(:c, :uₕ, :components, :data, 1)] = 1637.6961729499349
all_best_mse["sphere_baroclinic_wave_ogw"][(:c, :uₕ, :components, :data, 2)] = 26147.995331909864
all_best_mse["sphere_baroclinic_wave_ogw"][(:f, :w, :components, :data, 1)] = 43754.04112124818
#
#! format: on
#################################
#################################
#################################
