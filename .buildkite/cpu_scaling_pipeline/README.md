# Instructions to trigger CPU scaling build

1). Visit url: https://buildkite.com/clima/climaatmos-dot-jl-cpu-scaling

2). Click on "New Build"

3). Set branch to the PR branch

4). Click on "Create Build"

The CPU scaling build is useful for understanding the CPU scaling performance implications of merging a PR.
For higher-resolution simulations, with horizontal resolutions of 13 km or finer, memory footprint constraints 
dictated use of at most (16) message passing interface (MPI) ranks per node. To maintain consistency
in the CPU scaling studies, we use (16) MPI ranks per node for all simulations, although this results in 
under-utilization of the available computing resources at lower resolutions.
