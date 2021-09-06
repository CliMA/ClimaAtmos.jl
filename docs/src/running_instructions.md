# Running instructions

We store conventional benchmark simulation examples in the test folder, where they have access to standardized
initial conditions. These benchmarks are used and reused in several tests ranging from unit, over regression, to complex
validation tests. This guarantees that the same code piece can be efficiently reused.

Run all the test cases with:

```
$ julia --project test/runtests.jl
```
