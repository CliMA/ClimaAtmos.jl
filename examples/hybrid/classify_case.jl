is_baro_wave(parsed_args) = all((
    parsed_args["config"] == "sphere",
    parsed_args["forcing"] == nothing,
    parsed_args["rad"] == nothing,
    parsed_args["perturb_initstate"] == true,
))

is_solid_body(parsed_args) = all((
    parsed_args["config"] == "sphere",
    parsed_args["forcing"] == nothing,
    parsed_args["rad"] == nothing,
    parsed_args["perturb_initstate"] == false,
))

is_column_without_edmf(parsed_args) = all((
    parsed_args["config"] == "column",
    parsed_args["turbconv"] == nothing,
    parsed_args["forcing"] == nothing,
    parsed_args["turbconv"] != "edmf",
))

is_column_edmf(parsed_args) = all((
    parsed_args["config"] == "column",
    parsed_args["energy_name"] == "rhoe",
    parsed_args["forcing"] == nothing,
    parsed_args["turbconv"] == "edmf",
    parsed_args["rad"] == nothing,
))
