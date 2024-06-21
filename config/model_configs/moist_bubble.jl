dt_save_state_to_disk: "10secs"
initial_condition: "MoistBuoyantBubbleProfile"
x_max: 20000.0
x_elem: 100
y_max: 20000.0
y_elem: 100
z_max: 10000.0
z_elem: 50
t_end: "900.0secs"
dt: "0.1secs"
hyperdiffusion: "CAM_SE"
moist: "equil"
z_stretch: false
ode_algo: "SSP33ShuOsher"
config: "box"
job_id: "box_rising_thermal_test"
netcdf_interpolation_num_points: [40, 40, 40]
check_conservation: true
diagnostics:
  - short_name: [ts, ta, thetaa, ha, ua, va, wa, hur, hus, cl, clw, cli, hussfc, evspsbl]
    period: 30secs
