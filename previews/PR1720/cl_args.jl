const ca_dir = joinpath(@__DIR__, "..", "..")
include(joinpath(ca_dir, "src", "utils", "cli_options.jl"))
using PrettyTables
s = argparse_settings();
flags = map(arg -> arg.dest_name, s.args_table.fields)
flag_types = map(arg -> string(arg.arg_type), s.args_table.fields)
flag_help = map(arg -> arg.help, s.args_table.fields)
data = hcat(flags, flag_types, flag_help)

pretty_table(
    data;
    title = "Command line arguments",
    header = ["flag", "type", "help msg"],
    alignment = :l,
    crop = :none,
)
nothing
