const ca_dir = joinpath(@__DIR__, "..", "..")
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
        data;
        title = title,
        header = ["Argument", "Type", "Description"],
        alignment = :l,
        crop = :none,
    )
end
default_configs = joinpath(ca_dir, "config", "default_configs")
default_config_file = joinpath(default_configs, "default_config.yml")

make_table_from_config_file(default_config_file, "Default configuration")

nothing
