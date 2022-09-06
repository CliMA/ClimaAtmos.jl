#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_held_suarez_rhotheta"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρ)] = 5.014098226783069e-7
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :ρθ)] = 3.893279962084074e-8
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 1)] = 0.02036496487585247
all_best_mse["sphere_held_suarez_rhotheta"][(:c, :uₕ, :components, :data, 2)] = 0.8658456414948728
all_best_mse["sphere_held_suarez_rhotheta"][(:f, :w, :components, :data, 1)] = 11.112625785292687
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρ)] = 8.203019985535946e-8
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρe_tot)] = 1.3757347074912629e-5
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0.002886335946915362
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.1039735113555077
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0005381521087770225
all_best_mse["sphere_held_suarez_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 35.66664568902862
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 1.3664102147961434e-5
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 0.00022970994309211626
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0014283375407292886
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.7611497259258098
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :w, :components, :data, 1)] = 1.0607610854702236
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 6.527582573161811e-7
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 2.8323126383152583e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 9.99061562419188e-5
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0.043187754098884645
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0.0001636807798770303
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :w, :components, :data, 1)] = 3.3409583371605374
#
all_best_mse["sphere_held_suarez_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρ)] = 7.582356016895943e-8
all_best_mse["sphere_held_suarez_rhoe"][(:c, :ρe_tot)] = 1.0975395533652792e-6
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0.0038196380507885446
all_best_mse["sphere_held_suarez_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0.05886469746603826
all_best_mse["sphere_held_suarez_rhoe"][(:f, :w, :components, :data, 1)] = 2.9527655934081927
#
all_best_mse["sphere_held_suarez_rhoe_int"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρ)] = 2.1273274341228773e-7
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :ρe_int)] = 2.8268543801946478e-5
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 1)] = 0.010883630791663844
all_best_mse["sphere_held_suarez_rhoe_int"][(:c, :uₕ, :components, :data, 2)] = 0.10492192862608105
all_best_mse["sphere_held_suarez_rhoe_int"][(:f, :w, :components, :data, 1)] = 3.000724371823683
#
all_best_mse["edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_bomex"][(:c, :ρ)] = 7.277440198728416e-26
all_best_mse["edmf_bomex"][(:c, :ρe_tot)] = 3.093542595116656e-8
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 4.845830990297261e-9
all_best_mse["edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 1.7447317024660328e-7
all_best_mse["edmf_bomex"][(:c, :ρq_tot)] = 2.3558587264531958e-7
all_best_mse["edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.00014250335004930046
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 6.191644934550981e-6
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 6.20866771178554e-6
all_best_mse["edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 5.469465360061898e-6
all_best_mse["edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 1.4346719966585518e-5
#
all_best_mse["compressible_edmf_bomex"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_bomex"][(:c, :ρ)] = 6.599479381523193e-9
all_best_mse["compressible_edmf_bomex"][(:c, :ρe_tot)] = 2.5196251754955896e-6
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 1)] = 8.372438942129584e-7
all_best_mse["compressible_edmf_bomex"][(:c, :uₕ, :components, :data, 2)] = 6.320317772170901e-5
all_best_mse["compressible_edmf_bomex"][(:c, :ρq_tot)] = 1.7274168978978445e-5
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :en, :ρatke)] = 0.002009543512035439
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρarea)] = 0.00012125869679749017
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.00012246916536951484
all_best_mse["compressible_edmf_bomex"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 8.02950223608129e-5
all_best_mse["compressible_edmf_bomex"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0001434423893485371
#
all_best_mse["edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_dycoms_rf01"][(:c, :ρ)] = 1.1022674703665491e-27
all_best_mse["edmf_dycoms_rf01"][(:c, :ρe_tot)] = 3.619207440075361e-14
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 4.741626964525123e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 4.741626975315181e-12
all_best_mse["edmf_dycoms_rf01"][(:c, :ρq_tot)] = 8.387583658170248e-13
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 1.295251208773202e-10
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 2.6709291833714617e-11
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2.6630940658415517e-11
all_best_mse["edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 2.6168376952468262e-11
all_best_mse["edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 1.986697384363966e-11
#
all_best_mse["compressible_edmf_dycoms_rf01"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρ)] = 1.444643240470877e-9
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρe_tot)] = 2.3449691373865413e-8
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 1)] = 2.8409700462480686e-7
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :uₕ, :components, :data, 2)] = 2.840970046240428e-7
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :ρq_tot)] = 1.381446989572235e-7
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :en, :ρatke)] = 0.0008885366471066216
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρarea)] = 0.0002901500758129882
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 0.00029012719755629836
all_best_mse["compressible_edmf_dycoms_rf01"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 0.00028808425882019886
all_best_mse["compressible_edmf_dycoms_rf01"][(:f, :turbconv, :up, 1, :ρaw)] = 0.0005648053720315534
#
all_best_mse["edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_trmm"][(:c, :ρ)] = 3.5019248937164176e-25
all_best_mse["edmf_trmm"][(:c, :ρe_tot)] = 6.9288717467139e-25
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 1.3103441061138357e-22
all_best_mse["edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 3.2516190082729803e-22
all_best_mse["edmf_trmm"][(:c, :ρq_tot)] = 5.437070314585189e-25
all_best_mse["edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 1.3434002189317145e-21
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 1.9613367922770104e-20
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 2.3818710116308115e-20
all_best_mse["edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 8.38067816251022e-23
all_best_mse["edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 2.378060290015406e-23
#
all_best_mse["compressible_edmf_trmm"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_trmm"][(:c, :ρ)] = 1.6472174576860418e-25
all_best_mse["compressible_edmf_trmm"][(:c, :ρe_tot)] = 2.6588676562659324e-25
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 1)] = 6.1720903100486845e-25
all_best_mse["compressible_edmf_trmm"][(:c, :uₕ, :components, :data, 2)] = 1.6334020925840564e-24
all_best_mse["compressible_edmf_trmm"][(:c, :ρq_tot)] = 2.540143237764355e-25
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :en, :ρatke)] = 3.914900098026658e-24
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρarea)] = 3.580568481056891e-24
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaθ_liq_ice)] = 5.195358666515997e-24
all_best_mse["compressible_edmf_trmm"][(:c, :turbconv, :up, 1, :ρaq_tot)] = 4.690126044165065e-24
all_best_mse["compressible_edmf_trmm"][(:f, :turbconv, :up, 1, :ρaw)] = 7.459883120343098e-18
#
all_best_mse["edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["edmf_gabls"][(:c, :ρ)] = 8.158177890004773e-28
all_best_mse["edmf_gabls"][(:c, :ρe_tot)] = 5.709839664149002e-25
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 2.66387662133111e-24
all_best_mse["edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 2.3474119854319e-23
all_best_mse["edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 4.28839095308812e-23
#
all_best_mse["compressible_edmf_gabls"] = OrderedCollections.OrderedDict()
all_best_mse["compressible_edmf_gabls"][(:c, :ρ)] = 7.844573963213689e-28
all_best_mse["compressible_edmf_gabls"][(:c, :ρe_tot)] = 1.2706608639316409e-25
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 1)] = 2.022046299786329e-25
all_best_mse["compressible_edmf_gabls"][(:c, :uₕ, :components, :data, 2)] = 6.282566080341559e-25
all_best_mse["compressible_edmf_gabls"][(:c, :turbconv, :en, :ρatke)] = 2.4046512312221718e-24
#
#! format: on
#################################
#################################
#################################
