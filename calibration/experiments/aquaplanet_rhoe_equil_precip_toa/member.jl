import ClimaCalibrate as CAL
model_interface = CAL.env_model_interface() 
include(model_interface)
member = CAL.env_member_number()
iter = CAL.env_iteration()
CAL.run_forward_model(CAL.set_up_forward_model(member, iter, "/glade/u/home/nefrathe/clima/ClimaAtmos.jl/calibration/experiments/aquaplanet_rhoe_equil_precip_toa"))
