#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["baroclinic_wave"] = OrderedCollections.OrderedDict()
all_best_mse["baroclinic_wave"][(:c, :ρ)] = 0.0
all_best_mse["baroclinic_wave"][(:c, :ρe)] = 0.0
all_best_mse["baroclinic_wave"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["baroclinic_wave"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["baroclinic_wave"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["held_suarez"] = OrderedCollections.OrderedDict()
all_best_mse["held_suarez"][(:c, :ρ)] = 0.0
all_best_mse["held_suarez"][(:c, :ρe)] = 0.0
all_best_mse["held_suarez"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["held_suarez"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["held_suarez"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["held_suarez_topo"] = OrderedCollections.OrderedDict()
all_best_mse["held_suarez_topo"][(:c, :ρ)] = 0.0
all_best_mse["held_suarez_topo"][(:c, :ρe)] = 0.0
all_best_mse["held_suarez_topo"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["held_suarez_topo"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["held_suarez_topo"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["equil_baroclinic_wave"] = OrderedCollections.OrderedDict()
all_best_mse["equil_baroclinic_wave"][(:c, :ρ)] = 0.0
all_best_mse["equil_baroclinic_wave"][(:c, :ρe)] = 0.0
all_best_mse["equil_baroclinic_wave"][(:c, :ρq_tot)] = 0.0
all_best_mse["equil_baroclinic_wave"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["equil_baroclinic_wave"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["equil_baroclinic_wave"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["equil_held_suarez"] = OrderedCollections.OrderedDict()
all_best_mse["equil_held_suarez"][(:c, :ρ)] = 0.0
all_best_mse["equil_held_suarez"][(:c, :ρe)] = 0.0
all_best_mse["equil_held_suarez"][(:c, :ρq_tot)] = 0.0
all_best_mse["equil_held_suarez"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["equil_held_suarez"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["equil_held_suarez"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["equil_idealized_aqua_grayrad"] = OrderedCollections.OrderedDict()
all_best_mse["equil_idealized_aqua_grayrad"][(:c, :ρ)] = 0.0
all_best_mse["equil_idealized_aqua_grayrad"][(:c, :ρe)] = 0.0
all_best_mse["equil_idealized_aqua_grayrad"][(:c, :ρq_tot)] = 0.0
all_best_mse["equil_idealized_aqua_grayrad"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["equil_idealized_aqua_grayrad"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["equil_idealized_aqua_grayrad"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["equil_idealized_aqua_allskyrad"] = OrderedCollections.OrderedDict()
all_best_mse["equil_idealized_aqua_allskyrad"][(:c, :ρ)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad"][(:c, :ρe)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad"][(:c, :ρq_tot)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["equil_idealized_aqua_allskyrad_topo"] = OrderedCollections.OrderedDict()
all_best_mse["equil_idealized_aqua_allskyrad_topo"][(:c, :ρ)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_topo"][(:c, :ρe)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_topo"][(:c, :ρq_tot)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_topo"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_topo"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_topo"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["equil_idealized_aqua_allskyrad_edmf"] = OrderedCollections.OrderedDict()
all_best_mse["equil_idealized_aqua_allskyrad_edmf"][(:c, :ρ)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf"][(:c, :ρe)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf"][(:c, :ρq_tot)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["equil_idealized_aqua_allskyrad_edmf_topo"] = OrderedCollections.OrderedDict()
all_best_mse["equil_idealized_aqua_allskyrad_edmf_topo"][(:c, :ρ)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf_topo"][(:c, :ρe)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf_topo"][(:c, :ρq_tot)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf_topo"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf_topo"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["equil_idealized_aqua_allskyrad_edmf_topo"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["radiative_equilibrium_gray"] = OrderedCollections.OrderedDict()
all_best_mse["radiative_equilibrium_gray"][(:c, :ρ)] = 0.0
all_best_mse["radiative_equilibrium_gray"][(:c, :ρe)] = 0.0
all_best_mse["radiative_equilibrium_gray"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["radiative_equilibrium_gray"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["radiative_equilibrium_gray"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["radiative_equilibrium_allsky"] = OrderedCollections.OrderedDict()
all_best_mse["radiative_equilibrium_allsky"][(:c, :ρ)] = 0.0
all_best_mse["radiative_equilibrium_allsky"][(:c, :ρe)] = 0.0
all_best_mse["radiative_equilibrium_allsky"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["radiative_equilibrium_allsky"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["radiative_equilibrium_allsky"][(:f, :w, :components, :data, 1)] = 0.0
#
all_best_mse["radiative_convective_equilibrium"] = OrderedCollections.OrderedDict()
all_best_mse["radiative_convective_equilibrium"][(:c, :ρ)] = 0.0
all_best_mse["radiative_convective_equilibrium"][(:c, :ρe)] = 0.0
all_best_mse["radiative_convective_equilibrium"][(:c, :uₕ, :components, :data, 1)] = 0.0
all_best_mse["radiative_convective_equilibrium"][(:c, :uₕ, :components, :data, 2)] = 0.0
all_best_mse["radiative_convective_equilibrium"][(:f, :w, :components, :data, 1)] = 0.0
#
#! format: on
#################################
#################################
#################################
