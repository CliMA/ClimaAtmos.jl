---
description: Fuse kernels in a function by applying `foreach_point` construct
agent: build
---

Look inside a function $1 and use the `foreach_point` construct to fuse the kernels in the function. We can fuse only kernels that are composed of pointwise operations. Load the `apply-foreach-point` skill to recive instructions how to use the construct.

Try to follow the workflow belowe and heed the advice section and respect the constraints. 

The goal is to reduce number of kernels in tartget function and reduce its execution time on a GPU.

<workflow>

You need to perform the work in an iterative fasion using the tools described in the `run-and-time` skill. The workflow should be as follows:

1. Load all the required skills
2. Identify location of $1 function (you can use the semantic codebase search for this)
3. Create a 'model' as described in the `run-and-time` skill for the function $1. If it already exists, use that. 
4. Run the `quickprof_harness.jl` on the model to establish baseline runtime and register usage.
5. Fuse all kernels that we can with the `foreach_point` construct. If the register usage is around 140 or less, try to use the 'Local Caching' to move extra computation inside.
6. Always verfy that with quickprof_harness.jl. 
7. Iterate to find the option with the lowest execution time.
8. If Local Caching does not appear to lead to improvment. Stop and give user the best performing version for inspection

</workflow>

<advice>
 - If you encounter compilation problem with julia, debug it using a subagent. Try to preserve your own context window.
 - Try to make minimal modifications. Avoid large refactorings. We focus on incremental gains.
 - You are free to suggests edits to this this command and relevant skills. Do that even when unprompted by the user (do it of your own initiative).
</advice>

<contraints>
 - Do not modify anything outside the ClimaAtmos git repository
 - If you detect multiple GPUs are visable by CUDA.jl abort and inform a user. This means we are not running under resource reservation and are able to interfere with other users.
 - If you detect mistakes in the instructions or something is not clear, abort and inform the user.
 - Always set `CLIMACOMMS_DEVICE=CUDA` before running any harnesses. This is required to run on a GPU.
</contraints>
