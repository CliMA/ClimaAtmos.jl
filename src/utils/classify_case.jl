#TODO - do we want to change anything here now?
is_baro_wave(parsed_args) = all((
    parsed_args["config"] == "sphere",
    parsed_args["forcing"] == nothing,
    parsed_args["surface_setup"] == nothing,
    parsed_args["perturb_initstate"] == true,
))

is_solid_body(parsed_args) = all((
    parsed_args["config"] == "sphere",
    parsed_args["forcing"] == nothing,
    parsed_args["rad"] == nothing,
    parsed_args["perturb_initstate"] == false,
))

is_column_without_edmfx(parsed_args) = all((
    parsed_args["config"] == "column",
    parsed_args["turbconv"] == nothing,
    parsed_args["forcing"] == nothing,
    parsed_args["turbconv"] == nothing,
))
