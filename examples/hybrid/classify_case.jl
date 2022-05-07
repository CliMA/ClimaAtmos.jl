is_baro_wave(parsed_args) =
    all((parsed_args["config"] == "sphere", parsed_args["forcing"] == nothing))

is_column_radiative_equilibrium(parsed_args) = all((
    parsed_args["config"] == "column",
    parsed_args["forcing"] == nothing,
    parsed_args["rad"] != nothing,
))
