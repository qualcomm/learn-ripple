# Coalescing tips
__Performance impact__: High.

In this section, we look at various ways to obtain coalesced (i.e., stride-one access) loads from memory and stores to memory.
Coalescing often has a drastic impact on the performance of the
loads and stores in a Ripple program.

# Make clearly linear access functions
In order to optimize vector loads and stores, the compiler performs
a static analysis of the access patterns.
When the analysis detects memory access patterns that depend linearly upon
the Ripple indices,
it is able to analyze memory strides along all block dimensions.
Whenever stride-1 accesses are detected
(even if it's along only one of the dimensions),
the compiler generates coalesced vector loads or stores.
```C
x = A[...][...][ripple_id(BS, 0)]; // Coalesced access of A
```

The general user guidance here is that whenever a coalesced access exists,
make it clear to the compiler in the access functions.
The purpose of this section is to show common pitfalls, in which coalescing is
not achieved, and ways to work around it.

## Avoid dividing `ripple_id()`
Since the vector lane coordinates are integers, using a division results in
an integer division.

Integer divisions are non-linear operations, hence they should not be used
on a Ripple index.
Using an integer division results in the computation of a vector of addresses,
which is then used in a scatter/gather operation.
These are much more expensive than standard vector loads/stores.

For example, consider vectorizing the following piece of sequential code,
where we know that iterations of `w` are independent.
We store every element of `a` in `y`,
and the even elements of `b`, for every two consecutive iterations, in `z`.

```C
uint8_t foo(int W, uint8_t a[restrict 1][W], uint8_t b[restrict 1][W]) {
  for (int w = 0; w < W; ++w) {
    uint8_t y = a[w];
    uint8_t z = b[2 * (w / 2)];
    // Some other code below
    ...
  }
```

To vectorize along `w`, we take chunks of `nv0` iterations of `w`
and map the chunk elements to `v0`.
Basically, we create a loop (let's call it `u`) that goes over the chunks,
and map the values of `w` within chunks to `v0`.
To occupy one vector with 8-bit element computations,
let's start with a one-dimensional block of size 128.

```C
uint8_t foo(int W, , uint8_t a[restrict 1][W], uint8_t b[restrict 1][W]) {
  ripple_block_t BS = ripple_set_block_shape(0, 128);
  size_t v0 = ripple_id(BS, 0);
  size_t nv0 = ripple_get_block_size(BS, 0);
  for (int u = 0; u < W; u += nv0) {
    uint8_t y = a[u + v0];
    uint8_t z = b[2 * ( (u + v0) / 2)];
    // Some other code below
    ...
  }
```

We find ourselves with
- a contiguous access for `a[u + v0]`,
- but an access with an integer division in `b[2 * ((u + v0)/2)]`.
This latter access is very slow and it should be avoided if possible,
because it results in a gather (the most general but slowest kind of load).

Returning to the original function,
one fairly easy way to get rid of this integer division in the reference to `b`
is to strip-mine `w`.
We basically expose the `w/2` expression as a loop counter by rewriting the loop
with `w = 2*w_o + w_i`, and `0 <= w_i < 2`, resulting in the following code:

```C
for (int w_o = 0; w_o < W / 2; ++w_o) {
  for (int w_i = 0; (w_i < 2) & (2 * w_o + w_i < W); ++w_i) {
    uint8_t y = a[2 * w_o + w_i];
    uint8_t z = b[2 * w_o  + (w_i / 2)];
    // Some other code below
    ...
  }
}
```

Since `w_i / 2` is always equal to 0, we can simplify this as:
```C
for (int w_o = 0; w_o < W / 2; ++w_o) {
  for (int w_i = 0; (w_i < 2) & (2 * w_o + w_i < W); ++w_i) {
    uint8_t y = a[2 * w_o + w_i];
    uint8_t z = b[2 * w_o];
    // Some other code below
    ...
  }
}
```

Now, if we map `v0` along `w_o`,
we have stride-2 accesses for `a[2 * (w_o  + w_i / 2)]` and `b[2 * w_o + w_i]`
 (because of the "2" factor for `w_o`):

Small-stride loads and stores are cheaper than scatter/gathers
```C
ripple_block_t BS = ripple_set_block_shape(0, 128);
size_t v0 = ripple_id(BS, 0);
size_t nv0 = ripple_get_block_size(BS, 0);
for (int u_o = 0; u_o < W / 2; u_o += nv0) {
  for (int w_i = 0; w_i < 2 & (2 * u_o + w_i < W); ++w_i) {
    uint8_t y = a[2 * u_o + 2 * v0 + w_i];
    uint8_t z = b[2 * u_o + 2 * v0];
    // Some other code below
    ...
  }
}
```
We can further improve the situation by moving to a 2-dimensional grid.
In order to get contiguous memory access for `a`,
we can map `w_i` to the contiguous dimension `v0`,
and map `w_o` (previously mapped to `v0`) to `v1`.
Since `w_i` only takes values 0 and 1, a useful block size along `v0` is 2.
Since `v0` takes all values of `w_i`, the `w_i` loop "disappears",
leaving only the `(2 * u_o + w_i < W)` conditional:

```C
ripple_block_t BS = ripple_set_block_shape(0, 2, 64);
size_t v0 = ripple_id(BS, 0);
size_t v1 = ripple_id(BS, 1);
size_t nv1 = ripple_get_block_size(BS, 1);
for (int u_o = 0; u_o < W / 2; u_o += nv1) {
  if (2 * u_o + v0 < W) {
    uint8_t y = a[2 * u_o + 2 * v1 + v0];
    uint8_t z = b[2 * u_o + 2 * v1];
    // Some other code below
    ...
  }
}
```
`a` is now accessed contiguously, since `2 * v1 + v0` corresponds to the way
lanes are laid out in a full vector.
Note that we load a `2x64` tensor out of `a`, and a `1x64` tensor out of `b`.
Operations in the "code below" section that involve both `a` and `b` will
trigger a broadcast of `b` to a `2x64` shape.

# Small-stride loads and stores
Strided loads (stores, *mutatis mutandis*) are typically slow,
since they can require up to as many loads (stores)
as the number of elements in the strided operation.
They will happen for instance if we distribute vector lanes
along columns of a tensor,
since the elements of a column are separated by a full row.
In this case, we recommend either changing the vectorization strategy,
or modifying the tensor's layout.

Besides choosing a better way to vectorize our code, we can optimize strided
loads (stores) when the strides between elements or chunks are smaller than
a vector-width (e.g. 128 bytes on HVX).
Developers working with complex numbers, coordinate systems and RGB images often
find themselves handling large vectors of tuples,
from which they often need to extract one element at a time (for each tuple).

For instance, if we have a vector of complex numbers,
and we want to take its norm,
we could naively load the real and imaginary parts, as follows:
```C
float norm(float vec[32][2]) {
  ripple_block_t BS = ripple_set_block_shape(VECTOR, 32);
  size_t v = ripple_id(BS, 0);
  float real = vec[v][0];
  float imag = vec[v][1];
  return ripple_reduceadd(0b1, real * imag);
}
```

The problem with the above code is that the loads of `vec[v][0]` and `vec[v][1]`
are both strided loads, which are generally inefficient.
To simplify,
let's assume that our machine's hardware vector can contain 32 floats.
A more efficient approach than the strided loads is to load the whole `vec`
into two registers, and _then_ rearrange the data inside the registers,
as follows:

```C

float norm(float vec[32][2]) {
  ripple_block_t BS = ripple_set_block_shape(VECTOR, 32, 2);
  size_t v0 = ripple_id(BS, 0);
  size_t v1 = ripple_id(BS, 1);
  float all_data = vec[2 * v0 + v1];;
  return ripple_reduceadd(0b01, ripple_reducemul(0b10, all_data));
}
```

Thus, with 4 additional lines of code, we can go down from 64 loads
(assuming each scalar in the strided loads have to be loaded separately)
to 2 (vector) loads.

In this example, we could also have used the `shuffle_pair` interface to stay closer to the original idea to have a `real` and
`imag` blocks, as follows:

```C
size_t take_real_from_pair(size_t i, size_t n) {
  return 2 * i;
}

size_t take_imag_from_pair(size_t i, size_t n) {
  return 2 * i + 1;
}

float norm(float vec[32][2]) {
  ripple_block_t BS = ripple_set_block_shape(VECTOR, 32);
  size_t v0 = ripple_id(BS, 0);
  size_t v1 = ripple_id(BS, 1);
  float all_data = vec[v0][v1];
  float first = *(vec + v0);
  float second = *(vec + 32 + v0);
  float real =
    ripple_shuffle_pair(first, second, take_real_from_pair);
  float imag =
    ripple_shuffle_pair(first, second, take_imag_from_pair);
  return ripple_reduceadd(0b01, real * imag);
}
```

To do so, we had to use two different (but simpler) shuffle functions.

In C++, the `ripple/zip.h` header provides a few functions
to help load and store vectors of fixed-size tuples and turn them
into tuples of vectors.
In particular, the `rzip::shuffle_unzip` shuffle function
can express all the necessary in-vector data reordering.

# Data reordering
Some famous data layout reorderings, like data tiling,
can speed up algorithms by laying out data
that are used together (a tile) consecutively in memory.
That way, a single tile can be loaded with a small number of coalesced loads.

## Tradeoff
When considering data reordering in the process of optimizing a program,
it is important to consider the cost of the data reorganization itself,
as they can involve a lot of loads and stores.
If the data reordering does not result in enough gain
to offload the cost of the data reordering,
then the reordering is not worthwhile.

---
*Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear*
