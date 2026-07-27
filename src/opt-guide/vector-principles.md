# Vector optimization principles
In this section, we discuss general optimization principles when targeting a
machine that has a vector engine.

Consider optimization as trying to avoid doing expensive things:
- *Utilization* is about putting all the vector lanes available to contribution.
- *Coalescing* is about performing loads and stores efficiently,
  in as few chunks of contiguous data as possible.
- *Alignment* is about loading and storing a data chunk
  that exactly matches one of the "hardware chunks" that the memory is made of.
- *Register reuse* is about avoiding unnecessary loads and stores
  by doing all the work needed on a given vector
  while the data is in the (vector) registers.

## Utilization
Utilization is defined by how many vector lanes are active in the computation.
It is recommended to grow the number of vector lanes to fully utilize at least
one HVX vector.

### Underutilization
The number of hardware vector lanes utilized by our code depends upon
the type of the tensor elements we are manipulating.
Let `elem_size` be the number of bits in our tensor elements,
and `vec_bits` the number of bits in our vector engine.
For example, for HVX, `vec_bits = 1024`.
It is given by `#v = ceiling((block_size x elem_size)  / vec_bits)`.

Hence, to reach full utilization on HVX,
it is recommended to have a total block size of
- 128 if we work on 8-bit data
- 64 if we work on 16-bit data
- 32 if we work on 32-bit data, etc.

For AVX512, `vec_bits = 512`, resulting in half the number of vector lanes for
any specific data type.

This usually means that a good Ripple block size is one that is at least as big
as the number of vector lanes offered by the underlying hardware vector engine.
Using more can be beneficial, because it exposes more parallel computations,
which can be used by the compiler to get better performance
(typically by improving instruction-level parallelism,
like pipeline or VLIW parallelism).

### Overutilization
If the number of active vector registers in a Ripple block of vector lanes
goes beyond the size of the vector register file
(e.g. HVX defines 32 vector registers per thread),
the compiler may have to save
registers to a slower memory and load them back later,
which significantly lowers performance.

Beware that compilers often introduce temporary computations,
which also take up register space and may lead to those extraneous store/loads.

## Coalescing
In vector hardware, loads and stores happen by chunks of contiguous memory
(typically of the native vector size).
Because they are slower than other instructions,
the amount of loads and stores in a program
tend to have a significant impact on performance.
Hence, reducing the number of loads and stores is an important step in the optimization of SIMD programs.
This impacts how we should load and store the data we manipulate.

When we only load and store contiguous data chunks that correspond to
full native SIMD vectors, the number of loads and stores is minimized.
Such loads and stores are called _coalesced_.
While we depict the general goal here, this manual contains a section about specific ways to [obtain coalescing](./coalescing.md)

We want to load contiguous memory elements to contiguous processing elements
(i.e. vector lanes).
If we are using a 1-dimensional block shape, this means that
only the rightmost dimension of a tensor's access function must depend upon
the vector lane id, and that in that access function,
the vector lane id is not multiplied by anything (i.e. its coefficient is 1).

For example:
```C
v = ripple_id(ripple_set_block_shape(0, 32), 0);
x = A[i][k][j + v];
```

If `v` is multiplied by a constant (other than 1),
this constant becomes a _stride_,
meaning that some elements of the tensor are skipped, which is less efficient
than a coalesced load.

If `v` is involved in other dimensions than the rightmost in a tensor access,
this results in a stride as well
(as large as the slice defined by the dimensions to the right of that
involving `v`).
So if the semantics of our computations allow for it, we only use `v` in the
rightmost dimension of our tensor references.
If we don't, the code will be correct but slow.

Ripple lays out block elements into vector lanes in column-major order,
as explained [here](../ripple-spec/spmd.md#mapping-software-processing-elements-to-hardware-processing-elements).

## Alignment
Hardware memory space is typically partitioned regularly as
a large set of fixed-size chunks.
An example of this is cache lines.
Because of this partitioning,
the most efficient memory transfer (load or store) that can be done is by
transferring exactly a set of full chunks.
We have seen above that a condition for this to happen is for such
load or store to be coalesced.
For a vector engine to load exactly one chunk,
the start address of a coalesced load or store also has to start
at the starting address of a chunk.
We call such a load or store "aligned".
The basic rule to have an aligned load or store is for its start address
(the address accessed by coordinate `(0)` or `(0, 0)`) to be a multiple of the
hardware vector size (e.g. 1024 bits, i.e. 128 bytes, for HVX).

### Specifying Alignment

For best result, we advise the use of the ripple API to specify tensor
alignment, as well as using them as close to the load/store instruction as
possible. We provide
`ripple_ptr_alignment` and `ripple_ptr_alignment_slice` functions to specify the
pointer alignment from a tensor of pointers.

- `ripple_ptr_alignment_slice(Tensor_of_Pointers, Alignment_in_Bytes)`

    This construct indicates that the element `(0, 0 ... 0, 0)` is aligned by the provided alignment.

- `ripple_ptr_alignment_slice(Tensor_of_Pointers, Alignment_in_Bytes, Slice_Index0, ...)`

    This constructs is similar to `ripple_ptr_alignment_slice`, with the addition of letting you specify the slice indices to extract the pointer which is aligned. Non-provided indices is assumed to be zero.

#### Considerations

By using this API, you are specifying alignment constraints that will be
followed by the compiler. If, at runtime, the pointer is not aligned to
the provided value, the behavior will be hardware defined (e.g., an hardware interrupt may be raised, or
the values may be loaded/stored by ignoring the pointer bits that are not aligned).

### Using the Ripple Vector Alignment API

#### Tensors with Alignment Hint

```c
#define VECTOR_PE 0

void function_with_aligned_ptrs(size_t size, float *in, float *out) {
    ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, 4);

    ripple_parallel(BS, 0);
    for (size_t i = 0; i < size; ++i) {

        // Indicates that in and out values are aligned to every 4 float values
        *ripple_ptr_alignment(&out[i], 4 * sizeof(float)) = *ripple_ptr_alignment(&in[i], 4 * sizeof(float)) * in[i];

        // Notice that you only need to specify the alignment once for it to apply to expressions of the same value (the second in[i] assumes the same alignment as the first one)
    }
}
```

#### Tensors with Multiple Alignment Hints

Sometimes you may want to process multiple vectors at once (pairs). To indicate multiple alignment within a tensor, you can use multiple alignment calls by providing the slicing indices:

```c
#define VECTOR_PE 0

void function_with_aligned_ptrs(size_t size, float *in, float *out) {
    // Process pairs of 4 values
    ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, 4, 2);

    ripple_parallel(BS, 0, 1);
    for (size_t i = 0; i < size; ++i) {
        // *in* and *out* pointers at tensor indices [0][0] and [0][1] are aligned to 4 float values
        ripple_ptr_alignment(&in[i], 4 * sizeof(float));
        ripple_ptr_alignment(&out[i], 4 * sizeof(float));
        ripple_ptr_alignment_slice(&in[i], 4 * sizeof(float), 0, 1);
        ripple_ptr_alignment_slice(&out[i], 4 * sizeof(float), 0, 1);

        // The 4 alignment hints apply to the following load/store
        out[i] = in[i] * in[i];
    }
}
```

### Using Other Alignment Hints

Ripple is based on clang, which natively supports a general alignment mechanism, through the `__builtin_assume_aligned()` function.
This function lets us indicate that some pointers are aligned on a certain number of bytes.
The compiler uses said indications to infer alignment of vector loads and stores occurring as a result of using Ripple.

**In certain conditions, these hints cannot effectively be used by the compiler. We encourage to use the `ripple_ptr_aligned` API as close to the load/store instructions as possible for best results**.

The following example illustrates the use of `__builtin_assume_aligned`, where we indicate that `a`, `b`, and
`c` are aligned on a 128-byte boundary.
The compiler is capable to calculate that, given the 128-byte alignments, all the vector loads from `a` and `b` and the vector store to `c` in the `i` loop will be aligned as well.

```C
#define VECTOR 0
void vadd(int16_t * c, int16_t * a, int16_t * b, size_t n) {
  a = (int16_t *) __builtin_assume_aligned(a, 128);
  b = (int16_t *) __builtin_assume_aligned(b, 128);
  c = (int16_t *) __builtin_assume_aligned(c, 128);
  ripple_block_t BS = ripple_set_block_shape(0, 64);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}
```

## Register reuse
Register reuse is about avoiding unnecessary loads and stores
between memory and the registers.
If we need to use data (say a vector X of data) several times in our program,
and we can perform all the computations that use the data right after
loading X from memory to the registers,
then we know that X won't need to be
evicted from the registers to make room for other values,
and hence X won't need to be loaded again.
Loads (and stores) are expensive
(they have a much higher access latency than the registers),
hence by not having those extra loads, we are saving time, i.e.,
optimizing our program.

Because Ripple's underlying compiler (LLVM) manages registers for us,
it is hard to precisely control register use.
The compiler may introduce loads and stores
when it cannot allocate more registers at once,
and conversely, it is able to remove superfluous loads and stores from
an input program in some cases.
A good general heuristic to keep data in registers is to decompose
our computations into
work subsets that don't use more registers than provided by the hardware.
In the realm of loop transformations, register tiling and loop fission are known
to reduce the number of registers consumed by each work subset.

---
*Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear*
