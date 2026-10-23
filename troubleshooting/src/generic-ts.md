# Generic errors

Ripple introduces a slightly new SPMD representation,
in which the shape of an operation is propagated from its operands.
This varies from the traditional SPMD abstraction used for GPUs,
in which an "ambient" block shape is associated with a function
every time the function is called.

This section goes through errors users have encountered.
For each error, we explain the issue and present a solution.

## Link-time errors
Undefined symbols reported by the linker traditionally happen when users fail to
point clang to a library that contains the symbol definitions.
In Ripple, there are two more sources for potential missing symbols,
explained below.

### Missing ripple_* symbols
__Issue__: clang complains about any of the following symbols being undefined:
- ripple_id
- ripple_set_block_shape
- ripple_get_block_size
- ripple_parallel
- ripple_parallel_full

__Likely cause 1__: Ripple is not activated through command-line options,
and hence clang does not interpret the aforementioned symbols.

__Fix to likely cause 1__: Use clang with the `-fenable-ripple` flag.
For example:
```bash
$ clang -fenable-ripple -O2 [other options] my_prog.cpp
```

### Function calls not being vectorized
When a call to a scalar function has a shape,
Ripple tries to find a vector equivalent of the scalar function
that can be used with said shape.
Ripple considers only declarations and definitions marked with the
`ripple_vector_candidate` function attribute. The candidate must be declared
in the translation unit that contains the scalar call, normally by including
the vector library's header:

```C
typedef float v32f32 __attribute__((vector_size(128)));

__attribute__((ripple_vector_candidate))
v32f32 ripple_ew_pure_my_function(v32f32);
```

Compile the caller with `-fenable-ripple`. Supply an external candidate's
definition through the normal object, static-library, shared-library, or LTO
linking flow.

If the attributed declaration is missing or incompatible with the call's
shape, Ripple will not select the candidate and may create sequential scalar
calls. If the selected vector candidate or the fallback scalar function has no
linked definition, the linker will report an undefined symbol.

For C++ functions, also check the namespace. A scalar function and its
candidate must use the same named namespace. Ripple treats the global and
anonymous namespaces as equivalent for matching.

With the current AArch64 frontend lowering, the ABI may obscure vector
parameter and return types in LLVM IR. If an `ew` or
`uniform_<tensor_shape>` candidate is not selected, use explicit
`ret_<tensor_shape>` and `argX_<tensor_shape>` components for the return value
and every tensor argument. An elementwise candidate should still retain `ew`;
the explicit signature supplies the shape information that cannot be inferred
reliably on this target.

## Non-deterministic SIMD writes
__Effect__: Compilation error.

To avoid unintended concurrent write hazards,
Ripple doesn't allow users to write SIMD code in which
several block elements explicitly write to the same memory location,
as for instance
```C
  void bad_write(float A[8], float B[8]) {
    ripple_block_t B = ripple_set_block_shape(VEC, 8);
    size_t v = ripple_id(B, 0);
    A[0] = B[v]; // error! 8 different elements are written to A[0]
  }
```
__Solution__: Explicitly choose a value to be written out,
as in the following `good_write` example functions.

```C
  void bad_write(float A[8], float B[8]) {
    ripple_block_t B = ripple_set_block_shape(VEC, 8);
    size_t v = ripple_id(B, 0);
    A[0] = ripple_slice(B[v], 3); // We are writing B[3] to A[0]
  }
```
Or in this overly simple case,
```C
  void bad_write(float A[8], float B[8]) {
    A[0] = B[3]; // We are writing B[3] to A[0]
  }
```

### Interplay with the automatic broadcast rule
In the case when the shape of the right-hand-side (the "read")
of an assignment is compatible but smaller than its left-hand-side (the "write"),
the automatic broadcast rule applies:
the read is first broadcasted to match the shape of the write,
and there is no issue.
The only problem, in SIMD processing elements, comes when the write has
lower dimension than the read.
This is illustrated in the following example:
```C
void auto_bcast_example(float a, float B[8]) {
    ripple_block_t B = ripple_set_block_shape(VEC, 8);
    size_t v = ripple_id(B, 0);
    B[v] = a; // a gets auto-broadcasted to [8] before the write to B[v]
}
```

### Interplay with scalar expansion
Ripple includes a convenient mechanism to propagate shapes through scalar
temporary values (cf Ripple Manual's _Implicit Scalar Expansion_ section).
For instance, in the following function,
scalar variable `tmp` gets automatically expanded to a 8-element vector.

```C
void auto_expand_example(float A[8], float B[8]) {
    ripple_block_t B = ripple_set_block_shape(VEC, 8);
    size_t v = ripple_id(B, 0);
    float tmp = 2 * B[v]; // the [8] shape of B[v] is propagated through tmp
    A[v] = tmp;
}
```

This automatic expansion mechanism is __only valid for temporary scalars__.
Pointer-based, arrays and data structure writes are typically
not subject to automatic expansion.
In these cases,
it is the programmer's responsibility to ensure that
the shape of a SIMD write has enough dimensions to accept the incoming value.

The following example illustrates a pointer-based write that can't be expanded:
```C
void no_auto_expand(float * a, float B[8]) {
    ripple_block_t B = ripple_set_block_shape(VEC, 8);
    size_t v = ripple_id(B, 0);
    // *a has a [1] "scalar" shape,
    // but it is not a temporary scalar in 'no_auto_expand()`
    *a = B[v]; // error
}
```

## Control shape vs value shape
__Effect__: Correctness (unexpected results)

In Ripple, shape is propagated through values, not through control.
In other terms, the shape of a statement that lies in a control block
(for instance the "then" branch of an "if-then-else" statement)
is not influenced by the shape of the conditional.
This is illustrated in the following `cond_shape()` example,
in which a scalar computation is performed
under the control of a vector conditional.
`cond_shape` sets all the elements of A to 1 if any of `B[0..7]` is positive.

```C
void cond_shape(int A[8], float B[8]) {
    ripple_block_t B = ripple_set_block_shape(VEC, 8);
    size_t v = ripple_id(B, 0);
    int x = 0;
    if (B[v] > 0) { // conditional of shape [8]
        x = 1; // scalar computation -> shape not influenced by the conditional
    }
    A[v] = x; // x (0 or 1) is broadcast here to A[0..7]
}
```
__Problem__: intuitively, we may think that `A[v]` contains the result
of checking if `B[v] > 0`.

__Solution__ 1: Avoid creating statements with smaller-dimensional shapes
controlled by conditionals with higher-dimensional shapes,
as these can be counter-intuitive.

In the `cond_shape` example, if we want `A[v]` to contain
the result of whether `B[v]` is positive,
we need to give x an explicit `[8]` shape, as follows:

```C
void cond_shape_fixed(int A[8], float B[8]) {
    ripple_block_t B = ripple_set_block_shape(VEC, 8);
    size_t v = ripple_id(B, 0);
    int x = ripple_broadcast(B, 0b1, 0); // 0 --> [0 0 0 0 0 0 0 0]
    if (B[v] > 0) { // [8]-shaped conditional
        x = 1; // [8]-shaped computation controlled by a [8]-shaped conditional.
    }
    A[v] = x; // the shape of x is already [8] -> elementwise write to A[v]
}
```
`ripple_broadcast` is often useful to explicitly adjust the shape
of computations that would otherwise have too small a shape.
Some extra `ripple_broadcast`s in the code are the price we pay for the
ability to mix scalar, vector and tensor computations in the same function.

__Solution__ 2: If what you want is the semantics of `cond_shape`,
explicitly express that `x` should be an `or` redution of the values of `B[v]`,
as in the `cond_shape_explicit` code below:

```C
void cond_shape(int A[8], float B[8]) {
    ripple_block_t B = ripple_set_block_shape(VEC, 8);
    size_t v = ripple_id(B, 0);
    int x = ripple_reduceor(0b1, B[v] > 0);
    A[v] = x; // x is broadcast here to A[0..7]
}
```

## Using block sizes that don't match hardware vector sizes
__Effect__: compiler error or incorrect code.

Not all compiler backends are designed to gracefully support the lowering of
target-independent code that doesn't exactly match full vector computations.
Assume for instance that our target SIMD machine is 512 bits wide,
and that its backend was production-tested only with full vectors.
The following code, `fit_my_loop`, could cause a lowering issue
for the target's LLVM lowering backend.

```C
void fit_my_loop(char in[67], char out[67]) {
  // Here we make the block size with the data, as opposed to the SIMD hardware
  ripple_block_t B = ripple_set_block_shape(VEC, 67);
  size_t v = ripple_id(B, 0);
  out[v] = - in[v];
}
```

__Problem__: compiler internal error or incorrect results when using block sizes
that do not match the targeted SIMD hardware's vector size.
In the example above, the hardware target's vector size is 64 bytes.
However, the user requests the computation to be performed
on a block (i.e. a vector) of 67 elements.
The targeted compiler backend may not be good at lowering that to 64-byte vector
code.

__Solution__: Add the `-mllvm -ripple-pad-to-target-simd` option to your
clang compilation command line.
This will activate a Ripple behavior, which produces explicit full vector
computations.
This way, even a target that can only lower full vector code
will work with Ripple.
An added bonus is that the performance behavior of the code
becomes more predictable as we increase the Ripple block size.

## Incorrect `ripple_to_vec` and `vec_to_ripple` handling in c++ object members

We have noticed issues that appear when using aligned vectors with C++ object
member variables.
This issue seems to come from the underlying clang/LLVM compiler,
but it can affects the use of `ripple_to_vec` and `vec_to_ripple`
when applied to C++ member variables.

__Problem__: Some conversions of aligned C++ object members are incorrect.
__Solution__: Don't apply the `ripple_to_vec` and `vec_to_ripple` conversions
to aligned C++ member variables.

```
