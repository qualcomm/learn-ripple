# Ripple vector bitcode libraries
Consider the case when Ripple determines that
a function call statement should have a non-scalar shape.
Ripple's role is to render code that corresponds to a SIMD execution
of that (scalar) function.
As detailed [here](./calling.md),
the behavior of Ripple varies as a function of the information available:

- **If no SIMD version of the scalar function is known to Ripple**
  - If the scalar function definition is known, Ripple proceeds to create
  a SIMD equivalent to the (scalar) function, with its arguments expanded
  to their SIMD shape.The expanded version of the function call will be
  a call to the SIMD equivalent function.
  - If no such SIMD version of the function is known to Ripple,
  Ripple calls the function sequentially for all the elements of the block.

- **If a SIMD version of the function is available**

  Ripple will use the SIMD implementation directly (by calling the SIMD function).
  This requires the SIMD version to be declared in a vector "bitcode library".
  We describe this process in the next section.


# Creating a bitcode library for Ripple
## Custom library
Ripple currently relies on a naming convention to recognize
the relationship between a scalar function and its SIMD equivalent.

Consider a toy example in which we want to
create a vector elementwise multiplication function.
Let us call the scalar version, to be used in the Ripple source,
`elemprod`.

```C
extern "C" {
float elemprod(float a, float b);
}
```
To make it accessible to Ripple programmers,
we would declare it in a C header, say `elem_ops.h`,
and use it as in the example below:

```C
#include <ripple.h>
#include <elem_ops.h>

#define VECTOR_PE 0

float foo(float * A, float * B, float * C, size_t n) {
  ripple_block_t block = ripple_set_block_shape(VECTOR_PE, 8);
  ripple_parallel(block, 0);
  for (size_t i = 0; i < n; ++i) {
    A[i] = elemprod(B[i], C[i]);
  }
}
```
Ripple would infer shapes and determine that the shapes of `B[i]`
and `C[i]` are [8].
It would then look for a function:
- whose name indicates that it is a vectorized version of `elemprod`, and
- which takes vectors as its arguments.
It would also check if such function can be used to implement
a vector version of `elemprod` for the shape [8].

Ripple looks for such a function in the current module,
but also in special "bitcode" libraries provided to it through a
clang command-line flag.

### Naming convention
The vector implementation of a scalar function `foo` needs to be
named as follows:
```C
typedef float v8f32 __attribute__((vectorsize(32)));
v8f32 ripple_<attributes>_<signature>_elemprod(v8f32 a, v8f32 b);
```

**\<attributes\>**

A subset of the following strings, separated by `_` (underscores):

- `ew` specifies that the function is elementwise.
Ripple uses this information to adjust the calls to, say,
 `ripple_ew_elemprod()` to the requested shape ([8]).
 To do so successfully, Ripple may also need a masked definition
 of the vector function (see below).
- `pure` specifies that the function does not have side-effects
(i.e., it does not read or write to any external memory element).
Ripple uses that to avoid using masks in the case of elementwise functions.
- `mask` specifies that the function is a masked implementation of `elemprod`.
If the function call is controlled by a block-shaped conditional,
the mask corresponding to this conditional is applied to the vector function.
Masks are also used to adjust the effective size of the vector parameters,
when that size is smaller than the provided implementation
(for instance if the call's shape was [4]),
or if there are leftover elements
(for instance if the call's shape was [12] and it is elementwise).

**\<signature\>**

Optionally encodes tensor specifications for arguments and return values:

- `uniform_<tensor_shape>` — All arguments and return value share the same tensor shape.
- Individual shapes:
    - `argX_<tensor_shape>` — Shape of argument at index X (0-based).
    - `ret_<tensor_shape>` — Shape of the return value.


**Tensor Shape Format**

```
t<dim0>x<dim1>x...<dimN><type>
```

- `t` — Prefix for tensor.
- `<dim0>x<dim1>x...<dimN>` — Dimensions.
- `<type>` — Element type (e.g., i16 for 16-bit integer, f32 for 32-bit float).

For example, `t32x1x3i16` represents a tensor Tensor[32][1][3] of 16-bit integers.

Tensors with only trivial dimensions (size 1) are considered scalars, e.g.,
t1f32 or t1x1f32. Scalars do not require a signature; Ripple will pass them as
normal scalar values.

If no match for a vectorized function is found, the function call is rendered
as a sequence of calls.

In our case, the product is elementwise and pure.
We could implement a version for the Intel AVX (R) ISA as follows,
in `elem_ops.cc`:

```C
#include <immintrin.h>
extern "C" {
// note: v8f32 == __m256
typedef float v8f32 __attribute__((vectorsize(32)));

inline __attribute__((used, always_inline, weak, visibility("hidden")))
v8f32 ripple_ew_pure_elemprod(v8f32 a, v8f32 b) {
  return _mm256_mul_ps(a, b);
}
}
```
It is important to declare `ripple_ew_pure_elemprod` in a `"C"` linkage region,
in order to preserve the function name when a C++ compiler is used.
C++ mangles function names with operand types,
which prevents Ripple from matching the
scalar function name with its vector equivalent.

We also recommend that you use the set of attributes used here
for `ripple_ew_pure_elemprod`,
in order to minimize compiler warnings and enable inlining of the vector
function (if desired).

### Automatic 1D Shape Recognition

For targets whose ABI has well-defined vector argument types, Ripple can infer
1D tensor shapes directly from the type system. For unconventional vector
sizes, the ABI may dictate passing by pointer. In such cases, Ripple may not be
able to infer the correct tensor shape, so a manually specified tensor shape
through signature annotation is required.

### Signature Example: Matrix–Vector (Vector + 2D Tensor)

Shapes (row-major):

- `arg0` = `Tensor[8]` of f32 (vector)
- `arg1` = `Tensor[4][8]` of f32 (matrix with 4 rows × 8 cols)
- return = `Tensor[4]` of f32 (vector)

Signature encoding:

- `arg0_t8f32`
- `arg1_t4x8f32`
- `ret_t4f32`

Because this is not **elementwise**, omit ew. Add pure because there is no
side-effects in the matrix-vector operation.

```C
typedef float t4f32 __attribute__((vectorsize(16)));    // 4 lanes of f32
typedef float t8f32 __attribute__((vectorsize(32)));    // 8 lanes of f32
typedef float t4x8f32 __attribute__((vectorsize(128))); // 4x8 block (conceptual)

t4f32 ripple_pure_arg0_t8f32_arg1_t4x8f32_ret_t4f32_matvec(t8f32 vec, t4x8f32 matrix);

// Ripple can often infer 1D tensor shapes; the following declaration is equivalent and less verbose:
t4f32 ripple_pure_arg1_t4x8f32_matvec(t8f32 vec, t4x8f32 matrix);
```

#### Optimal implementation size

When in doubt about the size of vector parameters for a vector implementation,
we recommend to use a size for which the computation is optimally efficient.

### Function Without Argument

Ripple selects external functions based on the tensor shapes of the arguments at
the call site inside a ripple-enabled function. If a function does not have any
tensor argument, the scalar function call is not modified; ripple will ignore
it.

In order to allow for external functions taking no tensor argument, ripple
defines a special API using the block shape value. This API is often useful to
access constants or hardware accumulators/storage.

The API is as follows:

- The scalar definition has a unique `ripple_block_t` argument:
    ```c
    extern float myFunction(ripple_block_t BS);
    ```
- The ripple function definition follows the regular naming convention
and takes `void` as argument (or none in C++):
    ```c
    typedef float t4f32 __attribute__((vectorsize(16))); // 4 lanes of f32
    t4f32 ripple_ret_t4f32_myFunction(void);
    ```

**Usage example:**

```c
float foo(float * A, size_t n) {
  ripple_block_t block = ripple_set_shape(0, 4);
  ripple_parallel(block, 0);
  for (size_t i = 0; i < n; ++i) {
    A[i] = myFunction(block);  // Ripple transforms this to call ripple_ret_t4f32_myFunction()
  }
}
```

When calling this function from within a ripple-enabled region with an active
block shape, Ripple will transform the call by removing the `ripple_block_t`
argument and invoking the external vector function, which returns a tensor
matching the function's specified return shape.

### Compilation

Now the library source file can be turned into a bitcode representation using
```bash
clang elem_ops.cc -c -emit-llvm -o elem_ops.bc
```
Bitcode libraries have the `.bc` extension.
Of course, if the `elem_ops.cc` functions are implemented using Ripple, you also need to pass the `-fenable-ripple` flag in the command above.

### Declaration to Ripple
For Ripple to have access to the `elem_ops` bitcode library,
we need to pass its path to the command-line when compiling `foo.cc`:
```bash
clang foo.cc -fenable-ripple -fripple-lib elem_ops.bc
```

## Ripple standard vector library
As a target-agnostic compiler pass, Ripple allows us to write code that
is not specific to a target.
However, developers needing to exploit target-specific features,
such as special SIMD instructions or target-optimized math functions can
use bitcode libraries to access these.

Target-specific APIs presented in this manual are implemented as a bitcode
library, included by default by clang.
It is somehow a "standard" or "default" bitcode library.

### Multi-lib organization

The Ripple standard bitcode library is organized as a multi-lib,
i.e. there may be different versions of a function per processor and ISA version.

---
*Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear*
