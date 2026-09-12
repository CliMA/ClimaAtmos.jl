import ClimaAtmos as CA
using PrettyTables

# Print all available diagnostics to an ASCII table

short_names = []
long_names = []
units = []
comments = []
standard_names = []
for d in values(CA.Diagnostics.ALL_DIAGNOSTICS)
    push!(short_names, d.short_name)
    push!(long_names, d.long_name)
    push!(units, d.units)
    push!(comments, d.comments)
    push!(standard_names, d.standard_name)
end
data = hcat(short_names, long_names, units, comments, standard_names)
tf = TextTableFormat(;
    horizontal_lines_at_data_rows = collect(1:size(data, 1)),
)
pretty_table(
    data;
    auto_wrap = true,
    line_breaks = true,
    fixed_data_column_widths = [10, 15, 8, 32, 15],
    column_labels = [["Short name", "Long name", "Units", "Comments", "Standard name"]],
    table_format = tf,
)
