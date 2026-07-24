# The Ripple API specification

## C API (SPMD)
```C
/// \brief returns a block shape for processing element with id \p pe_id
ripple_block_t ripple_set_block_shape(int pe_id, size_t ... shape);

/// \brief size of the block associated to \p BS along dimension \p dim
size_t ripple_get_block_size(ripple_block_t block_shape, int dim);

/// \brief SPMD coordinate of the block \p BS along dimension \p dim
size_t ripple_id(ripple_block_t block_shape, int dim);

/// \brief reduction along dimensions defined by the bitfield dims,
///        using the operator "add".
/// 'TYPE' can be any one of (u?)int(8|16|32|64) and _Float16/float/double.
///
/// Although illustrated for "add" here,
/// reductions are also defined for the following operators:
///
/// add, mul, max, min, and, or, xor.
///
/// Floating-point types additionally support 'minimum' and 'maximum'.
/// These differ from 'min' and 'max' in their handling of NaN values.
///
/// The following table summarizes behavior in the presence of signaling (sNaN)
/// and quiet (qNaN) NaNs:
///
/// +----------------------+---------------------------+--------------------------------+
/// | Float Comparison     | min / max                 | minimum / maximum              |
/// +======================+===========================+================================+
/// | NUM vs qNaN          | NUM                       | qNaN                           |
/// +----------------------+---------------------------+--------------------------------+
/// | NUM vs sNaN          | qNaN                      | qNaN                           |
/// +----------------------+---------------------------+--------------------------------+
/// | qNaN vs sNaN         | qNaN                      | qNaN                           |
/// +----------------------+---------------------------+--------------------------------+
/// | sNaN vs sNaN         | qNaN                      | qNaN                           |
/// +----------------------+---------------------------+--------------------------------+
/// | +0.0 vs -0.0         | +0.0(max)/-0.0(min)       | +0.0(max)/-0.0(min)            |
/// +----------------------+---------------------------+--------------------------------+
/// | NUM vs NUM           | larger(max)/smaller(min)  | larger(max)/smaller(min)       |
/// +----------------------+---------------------------+--------------------------------+
///
TYPE ripple_reduceadd(int dims, TYPE to_reduce);

/// \brief Explicit broadcast of \p to_broadcast
///        along dimensions of the block shape that are set in the bitfield.
/// 'TYPE' can be any one of (u?)int(8|16|32|64) and _Float16/float/double.
/// The use of this function is typically rare, because of implicit broadcast.
/// For example:
///
/// // Dimension 0 size 32 (bit mask 1) and dimension 1 size 64 (bit mask 2)
/// ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, 32, 64)
/// int src = 42;                                   // Scalar source
/// dst1 = ripple_broadcast(BS, 0b10, src);  // promoted to Tensor[64][1]
/// dst2 = ripple_broadcast(BS, 0b01, src);  // promoted to Tensor[1][32]
/// dst3 = ripple_broadcast(BS, 0b01, dst1); // promoted to Tensor[64][32]
/// dst4 = ripple_broadcast(BS, 0b11, src);  // promoted to Tensor[64][32]
///
TYPE ripple_broadcast(ripple_block_t block_shape, uint64_t dims, TYPE to_broadcast);

/// \brief Explicit pointer broadcast of \p to_broadcast
///        along dimensions defined by the bitfield dims of
///        the given block shape.
/// 'TYPE' can be any one of (u?)int(8|16|32|64)
///        and _Float16/float/double.
TYPE *ripple_broadcast_ptr(ripple_block_t block_shape, uint64_t dims, TYPE *to_broadcast);

/// \brief Extracts a slice of value \p src along the indices \p dims.
/// In \p dims, the -1 value expresses "non-slicing" along the dimension,
/// in the sense that all elements of \p src are maintained along the dimension
/// where -1 is used.
/// For example, assuming the shape of \p src is 8x4:
/// dst = ripple_slice(src, -1, 0) extracts the first 8x1 sub-block of \p src
/// dst = ripple_slice(src, 1, 0) extracts the scalar at position (0, 1)
/// in \p src
/// etc.
/// @param src the value out of which we want a slice
/// @param dims the slicing indices.
///             Their number should match the dimension of \p src,
///             and the values of \p dim must be static constants.
TYPE ripple_slice(TYPE src, int ... dims);

/// \brief `ripple_slice()` for pointers.
TYPE ripple_slice_ptr(TYPE * src, int ... dims);

/// \brief shuffle where the source index is defined as a function of the
/// destination index by function \p src_index_fn
/// defined for all scalar types.
/// 'TYPE' can be any one of (u?)int(8|16|32|64) and _Float16/float/double.
/// When the block is multi-dimensional, this represents a shuffle from
/// a flattened view of \p to_shuffle to a flattened view of the result.
/// The parameters of \p src_index_fn are the flattened destination index
/// and the total block size.
/// If we use the C row-major convention, dimensions are laid out
/// from right (dimension 0) to left in vectors.
/// Hence, in 2 dimension, we'll have flattened(v0, v1) = v1*s0 + v0,
/// where v0 and v1 are the coordinates in the PE block
/// and s0 is the block shape along dimension 0.
/// TYPE is defined for the following types:
/// i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64
TYPE ripple_shuffle(TYPE to_shuffle, size_t(*src_index_fn)(size_t, size_t) );

/// \brief creates a block whose elements are picked from \p src1 and \p src2 using \p src_index_fn.
/// \p src_index_fn basically indexes in the flattened concatenation of \p src1 and \p src2.
/// \param src1 the first source block
/// \param src2 the second source block
/// \param src_index_fn a function that takes
// the flat index of the returned block and the total block size, 
// and returns the index into the [src1|src2] concatenated source 
// array where the destination value should be taken from.
/// TYPE is defined for the following types:
/// i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64
TYPE ripple_shuffle_pair(TYPE src1, TYPE src2, size_t(*src_index_fn)(size_t, size_t) );

/// \brief Computes the saturated addition of two values. 'TYPE'
/// can be any one of (u?)int(8|16|32|64)_t.
/// \param x The first value of type TYPE.
/// \param y The second value of type TYPE.
/// \return The result of the saturated addition of x and y.
TYPE ripple_add_sat(TYPE x, TYPE y);

/// \brief Similar to `ripple_add_sat`, but performs subtraction.
TYPE ripple_sub_sat(TYPE x, TYPE y);

/// \brief Similar to `ripple_add_sat`, but performs left shift.
TYPE ripple_shl_sat(TYPE x, TYPE y);

/// \brief Converts a 1-d Ripple block of T to a <N_EL x T> C vector.
/// \param N_EL the number of elements of the output vector.
///             Would normally match the total Ripple block size,
///             but can be bigger.
/// \param T the vector's element type - you can pass it as a parameter 
///          because ripple_to_vec is a macro.
/// \param PE_ID the processing element ID
///              for which this conversion is made.
/// \param x the Ripple (block) value that needs to be converted
///          to a C  vector
T __attribute__(vector_size(sizeof(T) * N_EL))
ripple_to_vec(size_t N_EL, T, int PE_ID, T x);

/// \brief Same as ripple_to_vec,
/// for a 2-d block made of the 2 first dimensions of the Ripple block.
/// Assumes that the Ripple block shape has at least 2 dimensions.
T __attribute__(vector_size(sizeof(T) * N_EL))
ripple_to_vec_2d(size_t N_EL, T, int PE_ID, T x);

/// \brief Same as ripple_to_vec,
/// for a 3-d block made of the 3 first dimensions of the Ripple block.
/// Assumes that the Ripple block shape has at least 3 dimensions.
T __attribute__(vector_size(sizeof(T) * N_EL))
ripple_to_vec_3d(size_t N_EL, T, int PE_ID, T x);

/// \brief  Converts a <N_EL x T> vector to a 1-d Ripple block of type T.
/// \param N_EL the number of elements of the output vector.
///             Would normally match the total Ripple block size,
///             but can be bigger.
/// \param T the vector's element type - you can pass it as a parameter 
///          because ripple_to_vec is a macro.
/// \param PE_ID the processing element ID
///              for which this conversion is made.
/// \param x the C vector that needs to be converted
///          to a Ripple (block-) value
T vec_to_ripple(size_t N_EL, T, int pe_id, 
  T __attribute__(vector_size(sizeof(T) * N_EL))x);

/// \brief Same as vec_to_ripple,
/// for a 2-d block made of the 2 first dimensions of the Ripple block.
/// Assumes that the Ripple block shape has at least 2 dimensions.
T vec_to_ripple_2d(size_t N_EL, T, int pe_id, 
  T __attribute__(vector_size(sizeof(T) * N_EL))x);

/// \brief Same as vec_to_ripple,
/// for a 3-d block made of the 3 first dimensions of the Ripple block.
/// Assumes that the Ripple block shape has at least 3 dimensions.
T vec_to_ripple_3d(size_t N_EL, T, int pe_id, 
  T __attribute__(vector_size(sizeof(T) * N_EL))x);

/// \brief Provides the information that the pointer element `(0, 0 ... 0, 0)`
/// in this tensor is aligned at the provided alignment (in bytes).
/// \param ptr A pointer or tensor of pointers
/// \param Alignment The alignment in number of bytes
/// \return Returns ptr
Pointer_TYPE ripple_ptr_alignment(Pointer_TYPE ptr, size_t Alignment)

/// \brief Similar to ripple_ptr_alignment, to indicate a pointer alignment
/// hint, but additionally allows to specify the slicing indices instead of
/// assuming that every slicing index is zero. All non-provided slicing index
/// will be zero.
/// \param ptr A pointer or tensor of pointers
/// \param Alignment The alignment in number of bytes
/// \return Returns ptr
/// \see ripple_slice
/// \see ripple_assume_aligned
Pointer_TYPE ripple_ptr_alignment_slice(Pointer_TYPE ptr, size_t Alignment, SliceIdx [, SliceIdx]*)
```


## C API (loop annotations)
The loop annotation feature is basically syntactic sugar on top of SPMD.
Hence, all the SPMD functions inter-operate with the loop annotation functions
below.

```C
/// Followed by a loop with simple initialization and upper bound,
/// tells the compiler to distribute the processing elements identified by
/// \p pe_id along dimension \p dims of the block of said processing elements.
/// The list of dimensions in \p dims is ordered.
/// This generates both the full-vector loop and the epilogue.
void ripple_parallel(int pe_id, int ... dims);

/// Followed by a loop with simple initialization and upper bound,
/// tells the compiler to distribute the processing elements identified by
/// \p pe_id along dimension \p dims of the block of said processing elements.
/// The list of dimensions in \p dims is ordered.
/// This generates the full-vector loop only.
void ripple_parallel_full(int pe_id, int ... dims);
```


# C++ API
Some of the C functions above are available in a different form in C++,
which often results in different syntax.
```C++
/// \brief reduction along dimensions defined by the bitfield \p dims,
///        using the operator "add".
/// Although illustrated for "add" here,
/// reductions are also defined for the following operators:
///
/// add, max, min, and, or.
template <typename T> T ripple_reduceadd(int dims, T to_reduce);

/// \brief shuffle where the source index is defined by function \p src_index_fn
///        defined for all scalar types (int/uint 8 to 64 and float 16 to 64)
///
/// When the block is multi-dimensional, this represents a shuffle from
/// a flattened view of \p to_shuffle to a flattened view of the result.
/// If we use the C row-major convention, dimensions are laid out
/// from right (dimension 0) to left in vectors.
/// Hence, in 2 dimension, we'll have flattened(v0, v1) = v1*s0 + v0,
/// where v0 and v1 are the coordinates in the PE block
/// and s0 is the block shape along dimension 0.

/// \brief shuffle where the source index is defined by function \p src_index_fn
///        defined for all scalar types (int/uint 8 to 64 and float 16 to 64)
template <typename T>
T ripple_shuffle(T to_shuffle, std::function<size_t(size_t, size_t)>);

/// \brief Converts a <N_EL x T> vector to a 1-d Ripple block of T
/// for processing element PE_ID.
/// Ripple block size assumed to be less than or equal to N_EL.
template <size_t N_EL, typename T, int PE_ID = 0>
T __attribute__((vector_size(sizeof(T) * N_EL))) ripple_to_vec(T x);

/// \brief Same as ripple_to_vec,
/// for a 2-d block made of the 2 first dimensions of the Ripple block.
/// Assumes that the Ripple block shape has at least 2 dimensions.
template <size_t N_EL, typename T, int PE_ID = 0>
T __attribute__((vector_size(sizeof(T) * N_EL))) ripple_to_vec_2d(T x);

/// \brief Same as ripple_to_vec,
/// for a 3-d block made of the 3 first dimensions of the Ripple block.
/// Assumes that the Ripple block shape has at least 3 dimensions.
template <size_t N_EL, typename T, int PE_ID = 0>
T __attribute__((vector_size(sizeof(T) * N_EL))) ripple_to_vec_3d(T x);

/// \brief  Converts a <N_EL x T> vector to a 1-d Ripple block of type T.
/// \param N_EL the number of elements of the input vector.
///             Would normally match the total Ripple block size,
///             but can be bigger.
/// \param T the vector's element type.
/// \param PE_ID the processing element ID
///              for which this conversion is made.
/// \param x the C vector that needs to be converted
///          to a Ripple (block-) value
template <size_t N_EL, typename T, int PE_ID = 0>
T vec_to_ripple(T __attribute__((vector_size(sizeof(T) * N_EL))) x);

/// \brief Same as ripple_to_vec,
/// for a 2-d block made of the 2 first dimensions of the Ripple block.
/// Assumes that the Ripple block shape has at least 2 dimensions.
template <size_t N_EL, typename T, int PE_ID = 0>
T vec_to_ripple_2d(T __attribute__((vector_size(sizeof(T) * N_EL))) x);

/// \brief Same as ripple_to_vec,
/// for a 3-d block made of the 3 first dimensions of the Ripple block.
/// Assumes that the Ripple block shape has at least 3 dimensions.
template <size_t N_EL, typename T, int PE_ID = 0>
T vec_to_ripple_3d(T __attribute__((vector_size(sizeof(T) * N_EL))) x);
```

# Target-specific vector functions
Ripple comes with a library of functions that get vectorized to functions that
are optimized to the target architecture.
For Hexagon, please consult the 
[Ripple Hexagon optimization guide section](https://qualcomm.github.io/learn-ripple/opt/hexagon/hvx-opt.html#functions-optimized-for-hvx).


# Examples

## `ripple_reduce*`

A reduction collapses the elements of a block
along one or more of its dimensions.
For example, the reduction of a 1-dimensional block vector with the "max" operator can be written without
Ripple as follows:

```C
int32_t x[32];
int32_t maximum = 0;
for (size_t i = 0; i < 32; ++i) {
  maximum = max(x[i], maximum);
}
```
Or, it can be written with Ripple as follows:
```C
int32_t x[32];
ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, 32);
int32_t maximum =
  ripple_reducemax(0b1, x[ripple_id(BS, 0)]);
```

The `0b1` argument is a bit field, with bit `0` set to
true (1).
This means that we reduce the block along dimension 1.
If this were a 32x16 two-dimensional block, for example, the same call to `ripple_reducemax` would
collapse the block along dimension `0` as well (using max),
returning a 16-element, 1-dimensional block.

### Average / mean
We demonstrate use of `ripple_reduceadd` through average computations.
The following example, a naive version of average computation,
uses a one-dimensional block.

```C
#define VECTOR_PE 0
#define VECTOR_SIZE 32

float avg(size_t n, float * A) {
    ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, VECTOR_SIZE);
    float sum = 0;
    ripple_parallel(BS, 0);
    for (size_t i = 0; i < n; ++i) {
        sum += ripple_reduceadd(0b1, A[i]);
    }
    return sum / n;
}
```
The first argument of `riiple_reduceadd` is a bitfield, describing the block dimensions
along which the reduction is being done.
Here it says that we are reducing along dimension 0
(the only dimension available).

This implementation is naive because reductions are typically
more computationally expensive (`log(VECTOR_SIZE)` instructions)
than data-parallel computations,
and we can perform _most_ of the sum using data-parallel computations
(one instruction).
To sum all the elements faster, we can start by decomposing `A` into blocks,
and sum up all the blocks with each other.
The result of that sum is a vector,
which we can then sum up to a scalar using `ripple_reduceadd`,
as shown in the following code.

```C
#define VECTOR_PE 0
#define VECTOR_SIZE 32

float avg(size_t n, float * A) {
    ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, VECTOR_SIZE);
    assert(n >= VECTOR_SIZE);
    float partial_sum = 0;
    ripple_parallel(BS, 0);
    for (size_t i = 0; i < n; ++i) {
        // partial_sum gets expanded automatically to a 1-d block of floats
        partial_sum += A[i];
    }
    return ripple_reduceadd(0b1, partial_sum) /n;
}
```


### Block slice extraction

In the following example,
we show one possible way to extract a vector block from a two-dimensional block.
In Ripple, scalar, vector and matrix computations can coexist.
While automatic broadcast promotes lower-dimensional blocks (including scalars)
to higher-dimensional ones (e.g. vectors, matrices and tensors),
reductions take higher-dimensional blocks as their input
and return lower-dimensional blocks.
In the following example,
we use that idea to take the 2nd row of a `32x4` block, by zeroing out the other
rows and reducing the `32x4` matrix block along dimension 1.

```C
#define VECTOR_PE 0
#define N 1024

void extract_2nd_row_from_32x4_blocks(float input[N], float output[N/4]) {
  ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, 32, 4); // set up a 32 x 4 block shape
  size_t v0 = ripple_id(BS, 0);
  size_t v1 = ripple_id(BS, 1);
  size_t block_size0 = ripple_get_block_size(BS, 0);
  size_t block_size = block_size0 * ripple_get_block_size(BS, 1);

  assert N % block_size == 0; // no vector loop epilogue
  size_t n_blocks = N / block_size;
  for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
    // coalesced load into a 2d block
    float block = input[block_size * block_idx + block_size0 * v1 + v0];
    float zeroed_out = v1 == 2 ? block : 0;
    // add-reduction of a 2-d block of size 32x4 along dimension 1
    // gives a 1-d block of size 32
    float reduced = ripple_reduceadd(0b10, zeroed_out);
    // coalesced store of the 1-d block
    output[block_size0 * block_idx + v0] = reduced;
  }
}
```
Note that a more efficient way to extract a slice is to use the `ripple_slice`
API, also presented in this section.

The following other reduction functions are also
available:
- `ripple_reducemax`
- `ripple_reducemin`
- `ripple_reduceand`
- `ripple_reduceor`
- `ripple_reducexor`

## `ripple_slice`
We can multiply the two `128x1` slices along dimension 1 of a `128x2` value
(called `input` here, representing a double HVX vector of `uint8_t`) as follows:

```C
uint8_t slice0 = ripple_slice(input, -1, 0); // input is 128x2, slice0 is 128x1
uint8_t slice1 = ripple_slice(input, -1, 1); // slice1 is 128x1
uint8_t mul = slice0 * slice1; // mul is 128x1
```

## `ripple_shuffle`
`ripple_shuffle` modifies the ordering of the elements of a one-dimensional
block.
To do so, we need to define, for each element index `k`,
the index of the element in the input block
that needs to go to index `k` in the output.

In C, this is expressed using a "shuffle function"
(sometimes also called "source function"),
which (exclusively) takes `k` and the block size,
and returns the index to be used in the input.
In the example below, we choose a one-dimensional block of 64 elements
to represent a 2-dimensional tile that we want to transpose.

```C
// represents the inversion of rows and columns indices of a 8x8 matrix
size_t transpose_8x8(size_t k, size_t block_size) {
  return  8 * (k % 8) + (k / 8);
}

// @brief loads a "flat tile" and transposes it
static int16_t transpose_tile(int16_t * tile_addr, size_t v) {
  int16_t tile_element = tile_addr[v];
  return ripple_shuffle(tile_element, transpose_8x8);
}

void permute(int16_t A[N][N][8][8]) {
  ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, 64);
  size_t v = ripple_id(BS, 0);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      // load a tile, i.e. a contiguous set of 64 elements
      int16_t transposed_1 = transpose_tile(&A[i][j][0][0], v);
      int16_t transposed_2 = transpose_tile(&A[j][i][0][0], v);
      (&A[j][i][0][0])[v] = transposed_1;
      if (i != j)
        (&A[i][j][0][0])[v] = transposed_2;
    }
  }
}

```
Note that the source function (here `transpose_8x8`) can be arbitrarily complex
without affecting the performance of the program.
Because source functions only depend upon the ripple_id and the block size,
the compiler is able to instantiate the shuffle indices corresponding to each
call to `ripple_shuffle`, resulting in zero runtime overhead for shuffling.
Accessing an out-of-block index is invalid, resulting in an error.
Source functions can express any static reordering of the elements of the block.

Notice that `transpose_tile` does not have a call to `ripple_set_block_shape()`.
This is because v is passed by value and is a Tensor. Either the function
`transpose_tile` is inlined, or it will be cloned and vectorized for the shapes
that `v` takes at the call site (i.e., here Tensor[64]).

While C requires declaring a separate function to define a shuffle,
in C++ a _non-capturing_ lambda can be used to express a reordering,
as in the following version.

```C
// @brief loads a "flat tile" and transposes it
static int16_t transpose_tile(int16_t * tile_addr, size_t v) {
  int16_t tile_element = tile_addr[v];
  auto transpose_8x8 = [](size_t k, size_t block_size) -> size_t {
    return i * (k % 8) + (k / 8);
  }
  return ripple_shuffle(tile_element, transpose_8x8);
}

void permute(int16_t A[N][N][8][8]) {
  auto BS = ripple_set_block_shape(VECTOR_PE, 64);
  size_t v = ripple_id(BS, 0);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      // load a "flat tile"
      int16_t transposed_1 = transpose_tile(&A[i][j][0][0], v);
      int16_t transposed_2 = transpose_tile(&A[j][i][0][0], v);
      (&A[j][i][0][0])[v] = transposed_1;
      if (i != j)
        (&A[i][j][0][0])[v] = transposed_2;
    }
  }
}
```
An example for `ripple_shuffle_pair` can be found in the
[coalescing optimization guide](../opt-guide/coalescing.md).

## `ripple_add_sat / ripple_sub_sat`
`ripple_add_sat` is used to express saturated additions.
The saturation behavior is determined by the type of the parameters passed to
`ripple_add_sat.`
If they are unsigned, `ripple_add_sat` will saturate at the maximum unsigned value;
if they are signed, `ripple_sat_add` will saturate at the maximum
signed value.

Similarly for `ripple_sub_sat`, which will saturate at 0 if the arguments
are unsigned and saturate at the minimum signed value if the arguments are signed.

For instance, the following code will print `0 ` 128 times.
```C
ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, 128);
uint8_t x = ripple_id(BS, 0);
uint8_t y = 2 * ripple_id(BS, 0);
printf("x-y=%i ", ripple_sub_sat(x, y));
```

And the following code will print `126 `, followed by `127 ` 127 times.
```C
#include <stdint.h> // for int8_t, INT8_C
ripple_block_t BS = ripple_set_block_shape(VECTOR_PE, 128);
int8_t x = ripple_id(BS, 0);
printf("x-y=%i ", ripple_add_sat(x, INT8_C(126)));
```

## ripple_parallel_full
`ripple_parallel_full(pe_id, dim)` is a variant of `ripple_parallel`
in which the developer indicates that the number of iterations in the loop
is an exact multiple of the block size along `dim`.
As a result, `ripple_parallel_full` does not generate an epilogue.
This can be useful to save the mask computation associated with the epilogue.

The following example illustrates this with a simple vector addition:
```C
void vadd(int n, float A[n], float B[n], float C[n]) {
   ripple_block_t BS = ripple_set_block_shape(HVX_PE, 32);
   assert (n % ripple_get_block_size(BS, 0) == 0);
   ripple_parallel_full(BS, 0);
   for(size_t i = 0; i < n; ++i) {
     C[i] = A[i] + B[i];
   }
 }
```

## `ripple_to_vec` / `vec_to_ripple`

The following function implements an integer-vector norm on a sequence of pairs represented in a `VectorPair` (64 elements),
unzips the pairs into a pair of
vectors containing the elements of each pair, and returns their elementwise multiplication with each other as a `Vector` (32 elements).

```C++
#include <ripple.h>
#include <ripple/zip.h>

typedef int32_t __attribute__((vector_size(128))) Vector;
typedef int32_t __attribute__((vector_size(256))) VectorPair;

Vector norm(VectorPair pair) {
  auto BS = ripple_set_block_shape(0, 32, 2);
  int32_t x = vec_to_ripple_2d<64, int32_t>(BS, pair);
  // This "zip" shuffle turned a sequence of pairs into a pair of sequences
  int32_t even_odds = ripple_shuffle(x, rzip::shuffle_unzip<2, 0, 0>);
  int32_t evens = ripple_slice(even_odds, -1, 0); // 32x1 shape
  int32_t odds = ripple_slice(even_odds, -1, 1); // 32x1 shape
  return ripple_to_vec<32, int32_t>(BS, evens * odds);
}
```
---
Hexagon is a registered trademark of Qualcomm Incorporated.

---
*Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause-Clear*
