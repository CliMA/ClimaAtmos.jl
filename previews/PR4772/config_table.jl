const ca_dir = joinpath(@__DIR__, "..", "..")
const output_file = joinpath(@__DIR__, "configuration_options.md")
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
        column_labels = [["Argument", "Type", "Description"]],
        alignment = fill(:l, size(data, 2)),
        backend = :markdown,
    )
end
default_configs = joinpath(ca_dir, "config", "default_configs")
default_config_file = joinpath(default_configs, "default_config.yml")
open(output_file, "w") do config_md
    write(
        config_md,
        """
        # Configuration options

        Every configuration argument accepted in YAML configuration files, with
        its type and default behavior, generated from
        `config/default_configs/default_config.yml`. See
        [Creating custom configurations](configuration.md) for how to use them.

        """,
    )
    table = make_table_from_config_file(
        default_config_file,
        "Default Configuration",
    )
    write(config_md, table)
end

nothing
