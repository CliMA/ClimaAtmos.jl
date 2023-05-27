import Random
Random.seed!(1234)
import ClimaAtmos as CA
config = CA.AtmosPerfConfig();
integrator = CA.get_integrator(config);
Y₀ = deepcopy(integrator.u);
CA.benchmark_step!(integrator, Y₀);
