# Calling functions when using Ripple

# Functions inheriting the caller's block shapes

Ripple represents a block shape as a `ripple_block_t`, returned by calling
`ripple_set_block_shape()`. This value is used with other ripple constructs such
as `ripple_id()`, `ripple_get_block_size()` and `ripple_broadcast()`. It can
also be passed by value as a function's argument. The result is that one or more
block shapes of the caller become available to the callee. You can pass the
block size at *any* argument position and use it inside the function to do
additional processing.

Passing the block shape is only allowed when the function being called is
defined in the same compilation unit as the caller. Otherwise, an error is
reported.

In the following example, we separated the processing in three functions:
`load_and_square()` loads a 1D tensor and squares the value, `add_and_store`
adds a 1D tensor to the value pointed by `out` and `my_method()` composes them.
In order for the first two function to agree on the same tensor size, we pass the
block size as argument.

```C
float load_and_square(float *arr, ripple_block_t BS) {
  return arr[ripple_id(BS, 0)] * arr[ripple_id(BS, 0)];
}

float add_and_store(ripple_block_t BS, float *out, float value) {
  out[ripple_id(BS, 0)] += value;
}

float my_method(float *in, float *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  float *val = load_and_square(in, BS);
  add_and_store(BS, out, val);
}
```

# Function call vectorization
Ripple also supports the call of scalar library functions.
There are three scenarios, when a function call depends upon a `ripple_id()`:
- the scalar library function is defined in the same module as the caller.
  In this case, the callee needs to be inlined.
- The scalar function has a vector implementation known to Ripple through its
 [vector bitcode library](../ripple-spec/bc-lib.md) mechanism.
In this case, Ripple calls that vector implementation.
- the scalar library function is known but its definition is external
to the current module, or it isn't inlined.
In this case, the function calls become sequentialized.
While the call per se is sequentialized,
code around the calls remain vectorized.

Some distributions of Ripple include a math library.
Users who wish to use math functions (such as sqrt, exp, etc.) should
include `ripple_math.h`.
One advantage of `ripple_math.h` is that
it also defines 16-bit floating point versions of the math function.
The naming convention for float16 functions in Ripple
follows the existing one in the C standard library,
based on a suffix, where the suffix is `f16`.
For instance, for sqrt, we have:

  ----------------------------
  | type     | function name |
  |----------|---------------|
  | double   | sqrt          |
  | float    | sqrtf         |
  | _Float16 | sqrtf16       |
  ----------------------------

---
*Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear*
