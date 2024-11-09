#################################
################################# MSE tables
#################################
#! format: off
#
all_best_mse = OrderedCollections.OrderedDict()
#
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_hightop_sponge"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :ρ)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_res"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :ρ)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:c, :ρq_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_equilmoist_allsky_gw_raw_zonallyasymmetric"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["deep_sphere_baroclinic_wave_rhoe_equilmoist"] = OrderedCollections.OrderedDict()
all_best_mse["deep_sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρ)] = 0
all_best_mse["deep_sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρe_tot)] = 0
all_best_mse["deep_sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["deep_sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["deep_sphere_baroclinic_wave_rhoe_equilmoist"][(:c, :ρq_tot)] = 0
all_best_mse["deep_sphere_baroclinic_wave_rhoe_equilmoist"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["diagnostic_edmfx_aquaplanet"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_aquaplanet"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet"][(:c, :ρq_tot)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet"][(:c, :sgs⁰, :ρatke)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet"][(:f, :u₃, :components, :data, 1)] = 0
#
all_best_mse["single_column_hydrostatic_balance_ft64"] = OrderedCollections.OrderedDict()
all_best_mse["single_column_hydrostatic_balance_ft64"][(:c, :ρ)] = 0
all_best_mse["single_column_hydrostatic_balance_ft64"][(:c, :ρe_tot)] = 0
all_best_mse["single_column_hydrostatic_balance_ft64"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["single_column_hydrostatic_balance_ft64"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["single_column_hydrostatic_balance_ft64"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["single_column_hydrostatic_balance_ft64"][(:c, :ρq_tot)] = 0
#
all_best_mse["box_hydrostatic_balance_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["box_hydrostatic_balance_rhoe"][(:c, :ρ)] = 0
all_best_mse["box_hydrostatic_balance_rhoe"][(:c, :ρe_tot)] = 0
all_best_mse["box_hydrostatic_balance_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["box_hydrostatic_balance_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["box_hydrostatic_balance_rhoe"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["box_hydrostatic_balance_rhoe"][(:c, :ρq_tot)] = 0
#
all_best_mse["box_density_current_test"] = OrderedCollections.OrderedDict()
all_best_mse["box_density_current_test"][(:c, :ρ)] = 0
all_best_mse["box_density_current_test"][(:c, :ρe_tot)] = 0
all_best_mse["box_density_current_test"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["box_density_current_test"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["box_density_current_test"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["box_density_current_test"][(:c, :ρq_tot)] = 0
#
all_best_mse["rcemipii_box_diagnostic_edmfx"] = OrderedCollections.OrderedDict()
all_best_mse["rcemipii_box_diagnostic_edmfx"][(:c, :ρ)] = 0
all_best_mse["rcemipii_box_diagnostic_edmfx"][(:c, :ρe_tot)] = 0
all_best_mse["rcemipii_box_diagnostic_edmfx"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["rcemipii_box_diagnostic_edmfx"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["rcemipii_box_diagnostic_edmfx"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["rcemipii_box_diagnostic_edmfx"][(:c, :ρq_tot)] = 0
#
all_best_mse["les_isdac_box"] = OrderedCollections.OrderedDict()
all_best_mse["les_isdac_box"][(:c, :ρ)] = 0
all_best_mse["les_isdac_box"][(:c, :ρe_tot)] = 0
all_best_mse["les_isdac_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["les_isdac_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["les_isdac_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["les_isdac_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["plane_agnesi_mountain_test_uniform"] = OrderedCollections.OrderedDict()
all_best_mse["plane_agnesi_mountain_test_uniform"][(:c, :ρ)] = 0
all_best_mse["plane_agnesi_mountain_test_uniform"][(:c, :ρe_tot)] = 0
all_best_mse["plane_agnesi_mountain_test_uniform"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["plane_agnesi_mountain_test_uniform"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["plane_agnesi_mountain_test_uniform"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["plane_agnesi_mountain_test_uniform"][(:c, :ρq_tot)] = 0
#
all_best_mse["plane_agnesi_mountain_test_stretched"] = OrderedCollections.OrderedDict()
all_best_mse["plane_agnesi_mountain_test_stretched"][(:c, :ρ)] = 0
all_best_mse["plane_agnesi_mountain_test_stretched"][(:c, :ρe_tot)] = 0
all_best_mse["plane_agnesi_mountain_test_stretched"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["plane_agnesi_mountain_test_stretched"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["plane_agnesi_mountain_test_stretched"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["plane_agnesi_mountain_test_stretched"][(:c, :ρq_tot)] = 0
#
all_best_mse["plane_density_current_test"] = OrderedCollections.OrderedDict()
all_best_mse["plane_density_current_test"][(:c, :ρ)] = 0
all_best_mse["plane_density_current_test"][(:c, :ρe_tot)] = 0
all_best_mse["plane_density_current_test"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["plane_density_current_test"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["plane_density_current_test"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["plane_density_current_test"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_hydrostatic_balance_rhoe_ft64"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_hydrostatic_balance_rhoe_ft64"][(:c, :ρ)] = 0
all_best_mse["sphere_hydrostatic_balance_rhoe_ft64"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_hydrostatic_balance_rhoe_ft64"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_hydrostatic_balance_rhoe_ft64"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_hydrostatic_balance_rhoe_ft64"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_hydrostatic_balance_rhoe_ft64"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_baroclinic_wave_rhoe"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_hightop"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_hightop"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_aquaplanet_rhoe_nonequilmoist_allsky"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_aquaplanet_rhoe_nonequilmoist_allsky"][(:c, :ρ)] = 0
all_best_mse["sphere_aquaplanet_rhoe_nonequilmoist_allsky"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_aquaplanet_rhoe_nonequilmoist_allsky"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_aquaplanet_rhoe_nonequilmoist_allsky"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_aquaplanet_rhoe_nonequilmoist_allsky"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_aquaplanet_rhoe_nonequilmoist_allsky"][(:c, :ρq_tot)] = 0
#
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean"] = OrderedCollections.OrderedDict()
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean"][(:c, :ρ)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean"][(:c, :ρe_tot)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean"][(:c, :ρq_tot)] = 0
#
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean_ft64"] = OrderedCollections.OrderedDict()
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean_ft64"][(:c, :ρ)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean_ft64"][(:c, :ρe_tot)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean_ft64"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean_ft64"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean_ft64"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["aquaplanet_rhoe_equil_clearsky_tvinsol_0M_slabocean_ft64"][(:c, :ρq_tot)] = 0
#
all_best_mse["rcemipii_sphere_diagnostic_edmfx"] = OrderedCollections.OrderedDict()
all_best_mse["rcemipii_sphere_diagnostic_edmfx"][(:c, :ρ)] = 0
all_best_mse["rcemipii_sphere_diagnostic_edmfx"][(:c, :ρe_tot)] = 0
all_best_mse["rcemipii_sphere_diagnostic_edmfx"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["rcemipii_sphere_diagnostic_edmfx"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["rcemipii_sphere_diagnostic_edmfx"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["rcemipii_sphere_diagnostic_edmfx"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_baroclinic_wave_rhoe_topography_dcmip_rs"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_topography_dcmip_rs"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_topography_dcmip_rs"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_topography_dcmip_rs"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_topography_dcmip_rs"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_topography_dcmip_rs"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_topography_dcmip_rs"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_topography_dcmip"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_topography_dcmip"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_topography_dcmip"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_topography_dcmip"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_topography_dcmip"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_topography_dcmip"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_topography_dcmip"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_held_suarez_rhoe_equilmoist_topography_dcmip"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_held_suarez_rhoe_equilmoist_topography_dcmip"][(:c, :ρ)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_topography_dcmip"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_topography_dcmip"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_topography_dcmip"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_topography_dcmip"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_held_suarez_rhoe_equilmoist_topography_dcmip"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_dcmip200"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_dcmip200"][(:c, :ρ)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_dcmip200"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_dcmip200"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_dcmip200"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_dcmip200"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_dcmip200"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_earth"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_earth"][(:c, :ρ)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_earth"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_earth"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_earth"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_earth"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_ssp_baroclinic_wave_rhoe_equilmoist_earth"][(:c, :ρq_tot)] = 0
#
all_best_mse["mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky"] = OrderedCollections.OrderedDict()
all_best_mse["mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky"][(:c, :ρ)] = 0
all_best_mse["mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky"][(:c, :ρe_tot)] = 0
all_best_mse["mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["mpi_sphere_aquaplanet_rhoe_equilmoist_clearsky"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_test_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_test_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_test_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_test_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_test_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_test_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_test_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_gabls_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_gabls_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_gabls_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_gabls_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_gabls_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_gabls_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_gabls_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_bomex_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_bomex_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_bomex_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_bomex_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_bomex_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_bomex_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_bomex_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_bomex_stretched_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_bomex_stretched_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_bomex_stretched_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_bomex_stretched_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_bomex_stretched_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_bomex_stretched_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_bomex_stretched_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_dycoms_rf01_explicit_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_dycoms_rf01_explicit_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_explicit_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_explicit_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_explicit_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_explicit_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_explicit_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_dycoms_rf01_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_dycoms_rf01_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf01_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_dycoms_rf02_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_dycoms_rf02_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf02_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf02_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf02_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf02_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_dycoms_rf02_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_rico_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_rico_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_rico_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_rico_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_rico_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_rico_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_rico_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_trmm_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_trmm_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_trmm_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_trmm_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_trmm_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_trmm_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_trmm_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_trmm_stretched_box"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_trmm_stretched_box"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_trmm_stretched_box"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_trmm_stretched_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_trmm_stretched_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_trmm_stretched_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_trmm_stretched_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_trmm_box_0M"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_trmm_box_0M"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_trmm_box_0M"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_trmm_box_0M"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_trmm_box_0M"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_trmm_box_0M"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_trmm_box_0M"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_adv_test_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_adv_test_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_adv_test_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_adv_test_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_adv_test_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_adv_test_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_adv_test_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_simpleplume_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_simpleplume_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_simpleplume_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_simpleplume_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_simpleplume_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_simpleplume_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_simpleplume_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_gabls_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_gabls_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_gabls_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_gabls_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_gabls_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_gabls_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_gabls_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_bomex_pigroup_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_bomex_pigroup_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_bomex_pigroup_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_bomex_pigroup_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_pigroup_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_bomex_pigroup_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_pigroup_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_bomex_fixtke_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_bomex_fixtke_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_bomex_fixtke_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_bomex_fixtke_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_fixtke_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_bomex_fixtke_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_fixtke_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_bomex_stretched_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_bomex_stretched_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_bomex_stretched_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_bomex_stretched_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_stretched_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_bomex_stretched_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_stretched_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_bomex_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_bomex_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_bomex_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_bomex_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_bomex_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_bomex_column_implicit"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_bomex_column_implicit"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_bomex_column_implicit"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_bomex_column_implicit"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_column_implicit"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_bomex_column_implicit"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_column_implicit"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_dycoms_rf01_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_dycoms_rf01_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_dycoms_rf01_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_dycoms_rf01_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_dycoms_rf01_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_dycoms_rf01_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_dycoms_rf01_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_rico_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_rico_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_rico_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_rico_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_rico_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_rico_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_rico_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_trmm_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_trmm_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_trmm_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_trmm_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_trmm_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_trmm_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_trmm_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_trmm_column_0M"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_trmm_column_0M"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_trmm_column_0M"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_trmm_column_0M"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_trmm_column_0M"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_trmm_column_0M"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_trmm_column_0M"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_gcmdriven_column"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_gcmdriven_column"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_gcmdriven_column"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_gcmdriven_column"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_gcmdriven_column"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_gcmdriven_column"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_gcmdriven_column"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_bomex_box"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_bomex_box"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_bomex_box"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_bomex_box"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_box"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_bomex_box"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_bomex_box"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_aquaplanet"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_aquaplanet"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_aquaplanet"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_aquaplanet"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_aquaplanet"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_aquaplanet"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_aquaplanet"][(:c, :ρq_tot)] = 0
#
all_best_mse["sphere_baroclinic_wave_rhoe_gpu"] = OrderedCollections.OrderedDict()
all_best_mse["sphere_baroclinic_wave_rhoe_gpu"][(:c, :ρ)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_gpu"][(:c, :ρe_tot)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_gpu"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_gpu"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_gpu"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["sphere_baroclinic_wave_rhoe_gpu"][(:c, :ρq_tot)] = 0
#
all_best_mse["diagnostic_edmfx_aquaplanet_gpu"] = OrderedCollections.OrderedDict()
all_best_mse["diagnostic_edmfx_aquaplanet_gpu"][(:c, :ρ)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet_gpu"][(:c, :ρe_tot)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet_gpu"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet_gpu"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet_gpu"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["diagnostic_edmfx_aquaplanet_gpu"][(:c, :ρq_tot)] = 0
#
all_best_mse["prognostic_edmfx_aquaplanet_gpu"] = OrderedCollections.OrderedDict()
all_best_mse["prognostic_edmfx_aquaplanet_gpu"][(:c, :ρ)] = 0
all_best_mse["prognostic_edmfx_aquaplanet_gpu"][(:c, :ρe_tot)] = 0
all_best_mse["prognostic_edmfx_aquaplanet_gpu"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_aquaplanet_gpu"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["prognostic_edmfx_aquaplanet_gpu"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["prognostic_edmfx_aquaplanet_gpu"][(:c, :ρq_tot)] = 0
#
all_best_mse["central_gpu_hs_rhoe_equil_55km_nz63_0M"] = OrderedCollections.OrderedDict()
all_best_mse["central_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :ρ)] = 0
all_best_mse["central_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :ρe_tot)] = 0
all_best_mse["central_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["central_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["central_gpu_hs_rhoe_equil_55km_nz63_0M"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["central_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :ρq_tot)] = 0
#
all_best_mse["central_cloud_diag_gpu_hs_rhoe_equil_55km_nz63_0M"] = OrderedCollections.OrderedDict()
all_best_mse["central_cloud_diag_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :ρ)] = 0
all_best_mse["central_cloud_diag_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :ρe_tot)] = 0
all_best_mse["central_cloud_diag_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["central_cloud_diag_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["central_cloud_diag_gpu_hs_rhoe_equil_55km_nz63_0M"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["central_cloud_diag_gpu_hs_rhoe_equil_55km_nz63_0M"][(:c, :ρq_tot)] = 0
#
all_best_mse["gpu_aquaplanet_dyamond"] = OrderedCollections.OrderedDict()
all_best_mse["gpu_aquaplanet_dyamond"][(:c, :ρ)] = 0
all_best_mse["gpu_aquaplanet_dyamond"][(:c, :ρe_tot)] = 0
all_best_mse["gpu_aquaplanet_dyamond"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["gpu_aquaplanet_dyamond"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["gpu_aquaplanet_dyamond"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["gpu_aquaplanet_dyamond"][(:c, :ρq_tot)] = 0
#
all_best_mse["target_gpu_implicit_baroclinic_wave_4process"] = OrderedCollections.OrderedDict()
all_best_mse["target_gpu_implicit_baroclinic_wave_4process"][(:c, :ρ)] = 0
all_best_mse["target_gpu_implicit_baroclinic_wave_4process"][(:c, :ρe_tot)] = 0
all_best_mse["target_gpu_implicit_baroclinic_wave_4process"][(:c, :uₕ, :components, :data, 1)] = 0
all_best_mse["target_gpu_implicit_baroclinic_wave_4process"][(:c, :uₕ, :components, :data, 2)] = 0
all_best_mse["target_gpu_implicit_baroclinic_wave_4process"][(:f, :u₃, :components, :data, 1)] = 0
all_best_mse["target_gpu_implicit_baroclinic_wave_4process"][(:c, :ρq_tot)] = 0
#
#! format: on
#################################
#################################
#################################
