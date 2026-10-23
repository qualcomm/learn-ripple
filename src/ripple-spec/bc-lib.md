# Ripple vector libraries

When a scalar function call has a non-scalar shape, Ripple can replace it with
a SIMD implementation supplied by a vector library. A vector implementation is
an ordinary C or C++ function marked with the `ripple_vector_candidate`
attribute. It can be declared in a header, defined in the same translation unit
or another one, and linked from an object or library like any other function.

Ripple discovers vector candidates from attributed declarations and
definitions, then uses the function name and type to match them to shaped
scalar calls.

In C and C++, the attribute can use GNU-style syntax:

```C
__attribute__((ripple_vector_candidate))
```

In C++11 and later, or C23 and later, it can also use standard attribute
syntax with the `clang` namespace:

```C++
[[clang::ripple_vector_candidate]]
```

Compile translation units that use the attribute with `-fenable-ripple`.

## Declaring a vector candidate

Consider a scalar elementwise multiplication function named `elemprod`. The
vector-library header, `elem_ops.h`, declares the scalar function and its
vector implementation:

```C
float elemprod(float a, float b);

typedef float v8f32 __attribute__((vector_size(32)));

__attribute__((ripple_vector_candidate))
v8f32 ripple_ew_pure_elemprod(v8f32 a, v8f32 b);
```

Including this header makes the candidate visible to Ripple while it
compiles a caller. The definition can live in another translation unit and be
linked into the program with the rest of the library.

For example, `foo.cc` uses the scalar function in a Ripple loop:

```C++
#include <cstddef>
#include <ripple.h>
// Declares elemprod and its Ripple vector candidate.
#include <elem_ops.h>

#define VECTOR_PE 0

void foo(float *A, float *B, float *C, std::size_t n) {
  ripple_block_t block = ripple_set_block_shape(VECTOR_PE, 8);
  ripple_parallel(block, 0);
  for (std::size_t i = 0; i < n; ++i)
    A[i] = elemprod(B[i], C[i]);
}
```

Ripple infers that `B[i]` and `C[i]` have shape [8], finds the attributed
candidate for `elemprod`, and replaces the scalar call with a call to
`ripple_ew_pure_elemprod` when its type and shape are compatible.

## Naming convention

The vector implementation of a scalar function `foo` is named as follows:

```C
ripple_<attributes>_<signature>_foo
```

The function must also have the `ripple_vector_candidate` attribute. The name
describes how Ripple may use the implementation; the attribute tells Ripple
which declarations and definitions it is allowed to inspect.

### Attributes encoded in the name

The optional attributes are separated by underscores:

- `ew` specifies that the function is elementwise. Ripple can adapt an
  elementwise implementation to a requested shape. This may require a masked
  implementation. Its operands follow Ripple's
  [implicit broadcasting rules](./spmd.md#implicit-broadcasting).
- `pure` specifies that the function has no side effects: it does not read or
  write externally visible memory. Ripple can use this information to avoid
  masks in some elementwise cases.
- `mask` specifies a masked implementation. Ripple uses masks for calls under
  block-shaped conditionals and when adapting the implementation to smaller or
  leftover shapes.

### Signature encoded in the name

The optional signature describes tensor shapes for arguments and return
values:

- `uniform_<tensor_shape>` means all arguments and the return value have the
  same tensor shape.
- `argX_<tensor_shape>` describes argument `X`, using a zero-based index.
- `ret_<tensor_shape>` describes the return value.

A tensor shape has this format:

```text
t<dim0>x<dim1>x...<dimN><type>
```

For example, `t32x1x3i16` represents a `Tensor[32][1][3]` of 16-bit
integers. Dimensions of size one are trivial, so values such as `t1f32` and
`t1x1f32` are scalar and do not require a signature component.

On Hexagon and x86-64, Ripple can infer one-dimensional tensor shapes from the
function type when the vector size is valid for the target ABI. For unusual
vector sizes that the ABI passes indirectly, encode the shape explicitly in
the function name.

On AArch64, explicitly encoding the shape of every tensor argument and the
return value is highly recommended. Use `argX_<tensor_shape>` for each tensor
argument and `ret_<tensor_shape>` for the return value. See
[AArch64 shape signatures](#aarch64-shape-signatures) for details.

For example, a matrix-vector operation with these shapes:

- argument 0: `Tensor[8]` of `float`
- argument 1: `Tensor[4][8]` of `float`
- return value: `Tensor[4]` of `float`

can be declared as:

```C
typedef float t4f32 __attribute__((vector_size(16)));
typedef float t8f32 __attribute__((vector_size(32)));
typedef float t4x8f32 __attribute__((vector_size(128)));

__attribute__((ripple_vector_candidate))
t4f32 ripple_pure_arg0_t8f32_arg1_t4x8f32_ret_t4f32_matvec(
    t8f32 vector, t4x8f32 matrix);
```

Because the operation is not elementwise, its name omits `ew`. If Ripple can
infer the one-dimensional shapes, the shorter
`ripple_pure_arg1_t4x8f32_matvec` name is equivalent.

When choosing the vector size of an implementation, prefer a size for which
the target executes the operation efficiently.

## Providing the implementation

Vector candidates use ordinary C/C++ definitions. The corresponding
`elem_ops.cc` can provide both the scalar implementation and its vector
candidate:

```C++
#include <immintrin.h>
#include <elem_ops.h>

float elemprod(float a, float b) {
  return a * b;
}

__attribute__((ripple_vector_candidate))
v8f32 ripple_ew_pure_elemprod(v8f32 a, v8f32 b) {
  return _mm256_mul_ps(a, b);
}
```

Regular scalar calls use `elemprod`. Calls on Ripple-shaped operands use
`ripple_ew_pure_elemprod`; because it is an `ew` candidate, Ripple adapts the
implementation to the elementwise call shape. Both functions can be defined
and linked as part of the same library.

Compile the implementation and caller into object files, then include both in
the application's normal link step. In this example, `main.o` represents the
rest of the application:

```bash
clang++ --target=x86_64-unknown-linux-gnu -O2 -fenable-ripple -mavx -I. \
  -c elem_ops.cc -o elem_ops.o
clang++ --target=x86_64-unknown-linux-gnu -O2 -fenable-ripple -mavx -I. \
  -c foo.cc -o foo.o
clang++ --target=x86_64-unknown-linux-gnu -O2 -mavx \
  main.o foo.o elem_ops.o -o app
```

The implementation can also be packaged in a static or shared library. Its
symbol visibility and linkage follow the usual C/C++ rules.

### C++ linkage and namespaces

Ripple supports C++ symbol name mangling, so candidates may use C++ linkage.
For functions in a named namespace, the scalar function and its vector
candidate must be in the same namespace. Ripple matches the namespace, scalar
function name, and compatible function type; a candidate in a different named
namespace is not selected.

```C++
namespace math {
float add(float a, float b);

[[clang::ripple_vector_candidate]]
v8f32 ripple_ew_pure_add(v8f32 a, v8f32 b);
} // namespace math
```

For candidate matching, Ripple treats the global namespace and an anonymous
namespace as equivalent. A global scalar function can therefore map to a
candidate in an anonymous namespace, and an anonymous-namespace scalar
function can map to a global candidate. Neither form matches a scalar function
in a named namespace; that candidate must be declared in the same named
namespace. This matching rule does not change C++ linkage: an
anonymous-namespace candidate still has internal linkage and must be defined
in the translation unit that uses it.

Use `extern "C"` when the library exposes a C ABI.

## Header-only libraries

A candidate definition can be `static inline` in a header. Because the
definition and its attribute are present in each consuming translation unit,
Ripple can select it and the normal optimizer can inline it.

```C++
typedef float v8f32 __attribute__((vector_size(32)));

[[clang::ripple_vector_candidate]]
static inline v8f32 ripple_ew_pure_elemprod(v8f32 a, v8f32 b) {
  return a * b;
}
```

Internal-linkage candidates, including `static` functions and functions in an
anonymous namespace, are local to their translation unit. External candidates
may be defined in another translation unit.

The C and C++ `inline` keyword does not require the compiler to inline a
function. To bypass the normal inlining heuristics, a header-only candidate can
also use Clang's `always_inline` attribute:

```C++
[[clang::ripple_vector_candidate, clang::always_inline]]
static inline v8f32 ripple_ew_pure_elemprod(v8f32 a, v8f32 b) {
  return a * b;
}
```

The GNU-style spelling is
`__attribute__((ripple_vector_candidate, always_inline))`. Clang will always
attempt to inline an `always_inline` function, although inlining can still be
impossible in some cases. `always_inline` is a compiler extension; C and C++ do
not currently define a standard force-inline attribute.

## LTO

Vector candidates also work with full LTO and ThinLTO. Put the attributed
declaration in the header visible to the caller and compile the participating
translation units with the usual `-flto` or `-flto=thin` options. Ripple makes
the candidate selection from the declarations and definitions visible during
compilation. The compiler and linker can then optimize and resolve those
symbols across translation units.

For example:

```bash
clang++ --target=x86_64-unknown-linux-gnu -O2 -fenable-ripple -mavx -flto -I. \
  -c elem_ops.cc -o elem_ops.o
clang++ --target=x86_64-unknown-linux-gnu -O2 -fenable-ripple -mavx -flto -I. \
  -c foo.cc -o foo.o
clang++ --target=x86_64-unknown-linux-gnu -O2 -mavx -flto \
  main.o foo.o elem_ops.o -o app
```

Compile `main.o` and the application's other LTO objects for the same target
with compatible target options. Replace `-flto` with `-flto=thin` throughout
to use ThinLTO.

## Selecting a candidate with a block shape

Ripple normally selects a vector candidate from the tensor shapes of the
scalar call's arguments. There is also a special form for an externally
declared scalar function whose call has no non-scalar tensor arguments. Its
first argument can be a `ripple_block_t` that supplies the shape used to select
the vector candidate:

```C
extern float read_value(ripple_block_t block, int index);

typedef float v4f32 __attribute__((vector_size(16)));

__attribute__((ripple_vector_candidate))
v4f32 ripple_ret_t1x4f32_read_value(int index);
```

The block shape is part of the scalar interface only. Ripple removes that
argument when it replaces the call, so the vector candidate receives `index`
but not `block`. The block must therefore be used only to describe the call's
shape; the candidate cannot inspect it or use it in its implementation.

The `ripple_block_t` must be the first source-level argument. Additional
arguments are allowed, but they must have scalar Ripple shapes and are passed
unchanged to the candidate. This includes vector-typed values that have a
scalar Ripple shape. The attributed candidate declaration must be visible
while compiling the caller, and its definition can be linked from another
object or library.

This external-candidate form is distinct from passing block shapes into a
function that uses them. In that case Ripple specializes the scalar function,
the block-shape arguments may appear at any positions, and the scalar
definition must be available in the same compilation unit as the caller. See
[Passing block shapes to functions](./calling.md#passing-block-shapes-to-functions)
for the specialization rules.

## Standard vector libraries

A Ripple distribution may provide target-specific candidates for operations
such as optimized math functions or special SIMD instructions. These
candidates follow the same rules as custom libraries: their headers expose
attributed declarations or definitions, and their symbols are supplied through
the target's libraries.

## Advanced: inspecting LLVM IR

When debugging generated LLVM IR, an attributed C/C++ function carries the
`ripple-vector-candidate` LLVM function attribute. This spelling uses hyphens,
whereas the source-level `ripple_vector_candidate` spelling uses underscores.

### AArch64 shape signatures

With the current AArch64 frontend lowering, the ABI may represent some vector
parameters or return values without preserving their original type in LLVM IR.
As a result, shape inference from an `ew` candidate or a
`uniform_<tensor_shape>` signature may not always identify the intended
function type.

For AArch64 vector libraries, explicitly describe the return value and every
tensor argument with `ret_<tensor_shape>` and `argX_<tensor_shape>`. Keep `ew`
when the operation is elementwise, but do not rely on it alone to communicate
the shapes. For example:

```C
__attribute__((ripple_vector_candidate))
t4f32 ripple_ew_pure_ret_t4f32_arg0_t4f32_arg1_t4f32_myadd(
    t4f32 a, t4f32 b);
```

---
*Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear*
