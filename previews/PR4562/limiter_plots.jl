import ClimaAtmos as CA
import CairoMakie as MK

FT = Float32

# Example tendency
force(q) = q

# Max allowed source/sink amounts
max_q_pos = FT(5)
max_q_neg = FT(2)

# tracer range
q_range = range(FT(-20), FT(20), 100)

#Plotting
fig = MK.Figure()
ax = MK.Axis(
    fig[1, 1];
    xlabel = "tracer specific humidity [-]",
    ylabel = "limited source term [1/s]",
)

MK.lines!(
    q_range,
    CA.tendency_limiter.(force.(q_range), max_q_pos, max_q_neg),
    label = "tendency_limiter",
)
MK.hlines!([max_q_pos, -max_q_neg], label = "Allowed tracer sources")

MK.axislegend(ax; position = :lc)
MK.save("assets/limiters_plot.png", fig) # hide
