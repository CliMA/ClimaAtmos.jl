is_baro_wave(parsed_args) = all((
    parsed_args["config"] == "sphere",
    parsed_args["forcing"] == nothing,
    parsed_args["rad"] == nothing,
))

is_column_radiative_equilibrium(parsed_args) = all((
    parsed_args["config"] == "column",
    parsed_args["turbconv"] == nothing,
    parsed_args["forcing"] == nothing,
    parsed_args["rad"] != nothing,
))

is_column_edmf(parsed_args) = all((
    parsed_args["config"] == "column",
    parsed_args["energy_name"] == "rhotheta",
    parsed_args["forcing"] == nothing,
    parsed_args["turbconv"] == "edmf",
    parsed_args["rad"] == nothing,
))
