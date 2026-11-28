const ca_dir = joinpath(@__DIR__, "..", "..")
const output_file = joinpath(@__DIR__, "config.md")
const input_file = joinpath(@__DIR__, "config_no_table.md")
import YAML
# Use OrderedCollections to preserve YAML order for docs
import OrderedCollections: OrderedDict
using PrettyTables

function make_table_from_config_file(config_file, title)
    config = YAML.load_file(config_file; dicttype = OrderedDict{String, Any})
    config_types = []
    config_helps = []
    config_names = []
    for (k, v) in config
        push!(config_types, typeof(v["value"]))
        push!(config_helps, v["help"])
        push!(config_names, k)
    end
    data = hcat(config_names, config_types, config_helps)
    pretty_table(
        String,
        data;
        title = title,
        header = ["Argument", "Type", "Description"],
        alignment = :l,
        backend = Val(:markdown),
    )
end
default_configs = joinpath(ca_dir, "config", "default_configs")
default_config_file = joinpath(default_configs, "default_config.yml")
open(output_file, "w") do config_md
    open(input_file) do f
        while !eof(f)
            s = readline(f)
            write(config_md, s)
            write(config_md, "\n")
        end
    end
    table = make_table_from_config_file(
        default_config_file,
        "Default configuration",
    )
    write(config_md, table)
end

nothing
