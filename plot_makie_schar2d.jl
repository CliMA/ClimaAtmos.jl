using CairoMakie
using ClimaCoreMakie

FT = Float32

# Domain Parameters
# Agnesi
#xmin = 0.0
#xmax = 180000.0
#zmax = 25000.0
#xlow = (xmax/2) - 10000.0
#xhigh = (xmax/2) + 10000.0
#zhigh = 10000.0

# DC
xmin = 0.0
xmax = 60000
zmax = 25000
xlow = (xmax/2) - 10000.0
xhigh = (xmax/2) + 10000.0
zhigh = 25000

### Zoomed-in Horizontal Velocity Plot
f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
(; ᶜinterp) = p.operators;
C123 = Geometry.Covariant123Vector
var = Fields.coordinate_field(Y.f).z
Axis(gaa[1, 1], aspect=2, title = "Surface Elevation")
limits!(xlow-1000, xhigh+1000, 0 , 1500)
paa = fieldcontourf!(var)
Colorbar(gaa[1, 2], paa, label = "u [m/s]")
fig_png = joinpath("./plots/testimage_z.png")
CairoMakie.save(fig_png, f)


f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
(; ᶜinterp) = p.operators;
C123 = Geometry.Covariant123Vector
u_bar = @. C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))
var = @. Geometry.project(Geometry.UAxis(), u_bar)
Axis(gaa[1, 1], aspect=2, title = "Horizontal Velocity")
limits!(xlow, xhigh, 0 , 10000)
#paa = fieldcontourf!(var, levels = FT(-10):0.01:FT(10))
paa = fieldcontourf!(var)
Colorbar(gaa[1, 2], paa, label = "u [m/s]")
fig_png = joinpath("./plots/testimage_u.png")
CairoMakie.save(fig_png, f)

### Full-domain Horizontal Velocity Plot
f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
C123 = Geometry.Covariant123Vector
u_bar = @. C123(Y.c.uₕ) + C123(ᶜinterp(Y.f.w))
var = @. Geometry.project(Geometry.UAxis(), u_bar)
Axis(gaa[1, 1], aspect=2, title = "Horizontal Velocity")
limits!(0, xmax, 0 , zmax)
#paa = fieldcontourf!(var .- Geometry.UVector.(Float32.(20)), levels = FT(-0.10):0.01:FT(0.10))
paa = fieldcontourf!(var)
Colorbar(gaa[1, 2], paa, label = "u [m/s]")
fig_png = joinpath("./plots/testimage_u_full.png")
CairoMakie.save(fig_png, f)

### Zoomed-in Vertical Velocity Plot
f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
var = @. Geometry.project(Geometry.WAxis(), u_bar)
Axis(gaa[1, 1], aspect=2, title = "Vertical Velocity")
limits!(xlow, xhigh, 0 , 10000)
#paa = fieldcontourf!(var, levels = FT(-0.015):0.002:FT(0.015))
paa = fieldcontourf!(var)
Colorbar(gaa[1, 2], paa, label = "w [m/s]")
fig_png = joinpath("./plots/testimage_w.png")
CairoMakie.save(fig_png, f)


f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
var = @. Geometry.project(Geometry.WAxis(), u_bar)
Axis(gaa[1, 1], aspect=2, title = "Vertical Velocity")
limits!(xlow, xhigh, 0 , 10000)
paa = fieldcontourf!(var, levels = FT(-1.5):0.15:FT(1.5))
Colorbar(gaa[1, 2], paa, label = "w [m/s]")
fig_png = joinpath("./plots/testimage_w_clim.png")
CairoMakie.save(fig_png, f)

f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
var = @. Geometry.project(Geometry.WAxis(), u_bar)
Axis(gaa[1, 1], aspect=2, title = "Vertical Velocity")
#paa = fieldcontourf!(var, levels = FT(-0.015):0.002:FT(0.015))
limits!(0, xmax, 0 , zmax)
paa = fieldcontourf!(var)
Colorbar(gaa[1, 2], paa, label = "w [m/s]")
fig_png = joinpath("./plots/testimage_w_full.png")
CairoMakie.save(fig_png, f)

f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
thermo_params = CAP.thermodynamics_params(params)
ᶜts = p.ᶜts
θ = TD.virtual_pottemp.(thermo_params, ᶜts)
Axis(gaa[1, 1], aspect=2, title = "Virtual Potential Temperature")
limits!(0, xmax, 0 , zmax)
paa = fieldcontourf!(θ)
Colorbar(gaa[1, 2], paa, label = "θ [K]")
fig_png = joinpath("./plots/testimage_pottemp.png")
CairoMakie.save(fig_png, f)

f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
thermo_params = CAP.thermodynamics_params(params)
ᶜts = p.ᶜts
θ = TD.air_temperature.(thermo_params, ᶜts)
Axis(gaa[1, 1], aspect=2, title = "Temperature")
limits!(0, xmax, 0 , zmax)
paa = fieldcontourf!(θ)
Colorbar(gaa[1, 2], paa, label = "T [K]")
fig_png = joinpath("./plots/testimage_temp.png")
CairoMakie.save(fig_png, f)

f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
thermo_params = CAP.thermodynamics_params(params)
ᶜts = p.ᶜts
θ = TD.air_pressure.(thermo_params, ᶜts)
Axis(gaa[1, 1], aspect=2, title = "Air Pressure")
limits!(0, xmax, 0 , zmax)
paa = fieldcontourf!(θ)
Colorbar(gaa[1, 2], paa, label = "p [Pa]")
fig_png = joinpath("./plots/testimage_pressure.png")
CairoMakie.save(fig_png, f)

f = Figure(; font = "CMU Serif")
gaa = f[1, 1] = GridLayout()
thermo_params = CAP.thermodynamics_params(params)
Axis(gaa[1, 1], aspect=2, title = "Total Energy")
limits!(0, xmax, 0 , zmax)
paa = fieldcontourf!(Y.c.ρe_tot)
Colorbar(gaa[1, 2], paa, label = "Total Energy")
fig_png = joinpath("./plots/testimage_e_tot.png")
CairoMakie.save(fig_png, f)

