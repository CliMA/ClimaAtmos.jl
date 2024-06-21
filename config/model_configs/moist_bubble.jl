dt_save_state_to_disk: "10secs"
initial_condition: "MoistBuoyantBubbleProfile"
x_max: 1000.0
x_elem: 10
y_max: 1000.0
y_elem: 10
z_max: 1000.0
z_elem: 40
t_end: "900.0secs"
dt: "0.05secs"
hyperdiffusion: "CAM_SE"
moist: "equil"
z_stretch: false
ode_algo: "ARS343"
config: "box"
job_id: "box_rising_thermal_test"
netcdf_interpolation_num_points: [40, 40, 40]
check_conservation: true
diagnostics:
  - short_name: [ts, ta, thetaa, ha, ua, va, wa, hur, hus, cl, clw, cli, hussfc, evspsbl]
    period: 30secs
