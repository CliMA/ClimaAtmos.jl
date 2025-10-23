import ClimaAtmos as CA
import CairoMakie as MK

FT = Float32

# Two example forces
force1(q) = q
force2(q) = -q

# Max allowed source amount
max_q1 = FT(5)
max_q2 = FT(2)

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
    CA.triangle_inequality_limiter.(force1.(q_range), max_q1, max_q2),
    label = "Positive force",
)
MK.lines!(
    q_range,
    CA.triangle_inequality_limiter.(force2.(q_range), max_q1, max_q2),
    label = "Negative force",
)
MK.hlines!([max_q1, -max_q2], label = "Allowed tracer sources")

MK.axislegend(ax; position = :lc)
MK.save("assets/limiters_plot.png", fig) # hide
