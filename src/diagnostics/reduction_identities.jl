# This file defines group identities (technically _monoid identities_) and is included in
# Diagnostics.jl.
#
# Several diagnostics require performing reductions, such as taking the maximum or the
# average. Since it is not feasible to store all the lists of all the intermediate values,
# we aggregate the results in specific storage areas (e.g., we take
# max(max(max(max(t1, t2), t3), t4), t5) instead of max(t1, t2, t3, t4, t5)
# In this, it is convenient to preallocate the space where we want to accumulate the
# intermediate. However, we have to fill the space with something that does not affect the
# reduction. This, by definition, is the identity of the operation. The identity of the operation
# + is 0 because x + 0 = x for every x.
#
# We have to know the identity for every operation we want to support. Of course, users are
# welcome to define their own by adding new methods to identity_of_reduction.

identity_of_reduction(::Val{max}) = -Inf
identity_of_reduction(::Val{min}) = +Inf
identity_of_reduction(::Val{+}) = 0
identity_of_reduction(::Val{*}) = 1
