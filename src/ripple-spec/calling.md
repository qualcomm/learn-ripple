# Calling functions when using Ripple

# Passing block shapes to functions

A `ripple_block_t` argument has two distinct uses. A function can receive and
use the block shape through specialization local to the compilation unit.
Alternatively, the first argument of an externally declared scalar function
can supply only the shape used to select a vector candidate, as described in
[Selecting a candidate with a block shape](./bc-lib.md#selecting-a-candidate-with-a-block-shape).

## Functions that use the block shape

Ripple represents a block shape as a `ripple_block_t`, returned by calling
`ripple_set_block_shape()`. This value is used with other ripple constructs such
as `ripple_id()`, `ripple_get_block_size()` and `ripple_broadcast()`. It can
also be passed by value as a function's argument. The result is that one or more
block shapes of the caller become available to the callee. You can pass the
block size at *any* argument position and use it inside the function to do
additional processing.

Passing the block shape is only allowed when the function being called is
defined in the same compilation unit as the caller. Specialization remains
local to that compilation unit, so a definition supplied by another
compilation unit or only at link time cannot be used. Ripple reports an error
when the definition is unavailable.

## Shape compatibility

The current Ripple implementation forms a single execution-mask shape for a
specialized call. If any non-block argument has a non-scalar Ripple shape,
those argument shapes determine the mask and must be broadcast-compatible.
The shapes represented by `ripple_block_t` arguments do not participate in
that compatibility check.

If every non-block argument has scalar Ripple shape, Ripple currently derives
the mask shape by broadcasting the block shapes passed to the function.
Consequently, multiple block shapes must be broadcast-compatible. This is a
known specialization implementation limitation needed to form one mask; it is
not an inherent restriction on using multiple block descriptors independently
and is expected to be addressed in a future Ripple version.

In the following example, we separate the processing into three functions:
`load_and_square()` loads a 1D tensor and squares the value, `add_and_store`
adds a 1D tensor to the value pointed by `out` and `my_method()` composes them.
For the first two functions to agree on the same tensor size, we pass the block
shape as an argument.

```C
float load_and_square(float *arr, ripple_block_t BS) {
  return arr[ripple_id(BS, 0)] * arr[ripple_id(BS, 0)];
}

void add_and_store(ripple_block_t BS, float *out, float value) {
  out[ripple_id(BS, 0)] += value;
}

void my_method(float *in, float *out) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  float val = load_and_square(in, BS);
  add_and_store(BS, out, val);
}
```

# Function call vectorization
Ripple also supports the call of scalar library functions.
There are three scenarios, when a function call depends upon a `ripple_id()`:
- the scalar library function is defined in the same compilation unit as the
  caller.
  In this case, the callee needs to be inlined.
- The scalar function has a vector implementation known to Ripple through its
  [`ripple_vector_candidate` declaration or definition](./bc-lib.md).
  In this case, Ripple calls that vector implementation, which is compiled and
  linked like any other C/C++ function.
- the scalar library function is known but its definition is external
  to the current compilation unit, or it isn't inlined.
  In this case, the function calls become sequentialized. While the call per
  se is sequentialized, code around the calls remain vectorized.

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
